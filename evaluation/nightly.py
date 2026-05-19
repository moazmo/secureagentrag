"""Nightly regression eval driver.

Loads the golden Q/A set from ``evaluation/golden_set.jsonl``, runs the
project's standard Ragas pipeline, and compares the new metric scores to
the rolling baseline in ``evaluation/baseline.json``. A regression of more
than 5 percentage points on faithfulness or context_precision marks the
result and the GitHub Action surfaces it as a failure + opens an issue.

Run manually:

    uv run python -m evaluation.nightly --no-fail-on-regression
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.graph import run_rag_pipeline
from ingestion.metadata import UserContext
from utils.logging import get_logger

logger = get_logger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = _ROOT / "evaluation" / "golden_set.jsonl"
BASELINE_PATH = _ROOT / "evaluation" / "baseline.json"
RESULT_DIR = _ROOT / "evaluation" / "results"

REGRESSION_THRESHOLD_PP = 0.05  # 5 percentage points


def _load_golden() -> list[dict[str, Any]]:
    if not GOLDEN_PATH.exists():
        logger.warning("golden_set_missing", path=str(GOLDEN_PATH))
        return []
    out: list[dict[str, Any]] = []
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _load_baseline() -> dict[str, float]:
    if not BASELINE_PATH.exists():
        return {"faithfulness": 0.0, "context_precision": 0.0, "answer_relevancy": 0.0}
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


async def _run_golden_set(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Run each golden question through the pipeline and collect responses."""
    responses: list[dict[str, Any]] = []
    for item in items:
        try:
            user = UserContext(
                user_id=item.get("user_id", "eval"),
                org_id=item.get("org_id", "acme_corp"),
                roles=item.get("roles", ["admin"]),
                clearance_level=item.get("clearance_level", 3),
            )
            state = await run_rag_pipeline(
                query=item["question"],
                user_context=user,
                thread_id=f"eval-{item.get('id', 'q')}",
            )
            responses.append(
                {
                    "question": item["question"],
                    "ground_truth": item.get("ground_truth", ""),
                    "answer": state.get("generation", ""),
                    "contexts": [d["text"] for d in state.get("relevant_documents", [])]
                    or [d["text"] for d in state.get("documents", [])],
                }
            )
        except Exception as exc:
            logger.error("eval_item_failed", question=item.get("question"), error=str(exc))
            continue
    return {"responses": responses}


def _compute_scores(responses: list[dict[str, Any]]) -> dict[str, float]:
    """Compute Ragas faithfulness / context_precision / answer_relevancy.

    Falls back to a simple lexical-overlap heuristic when Ragas is not
    installed so the nightly job still produces a comparable number.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness

        ds = Dataset.from_list(
            [
                {
                    "question": r["question"],
                    "answer": r["answer"],
                    "contexts": r["contexts"],
                    "ground_truth": r["ground_truth"],
                }
                for r in responses
                if r["answer"]
            ]
        )
        result = evaluate(ds, metrics=[faithfulness, context_precision, answer_relevancy])
        scores: dict[str, float] = {}
        for metric_name in ("faithfulness", "context_precision", "answer_relevancy"):
            try:
                scores[metric_name] = float(result[metric_name])
            except Exception:
                scores[metric_name] = 0.0
        return scores
    except ImportError:
        logger.warning("ragas_unavailable_using_lexical_overlap")
        return _lexical_scores(responses)


def _lexical_scores(responses: list[dict[str, Any]]) -> dict[str, float]:
    """Cheap fallback when Ragas is not installed.

    Faithfulness = fraction of answer tokens present in any context.
    Context-precision = fraction of contexts whose token set intersects the
    answer. Answer-relevancy = fraction of ground-truth tokens in the answer.
    """
    f, p, r = 0.0, 0.0, 0.0
    n = 0
    for item in responses:
        ans = set(item["answer"].lower().split())
        ctxs = [set(c.lower().split()) for c in item["contexts"]] or [set()]
        gt = set(item["ground_truth"].lower().split())
        if not ans:
            continue
        any_ctx = set().union(*ctxs)
        f += len(ans & any_ctx) / max(len(ans), 1)
        p += sum(bool(c & ans) for c in ctxs) / max(len(ctxs), 1)
        r += len(gt & ans) / max(len(gt), 1)
        n += 1
    if n == 0:
        return {"faithfulness": 0.0, "context_precision": 0.0, "answer_relevancy": 0.0}
    return {
        "faithfulness": round(f / n, 3),
        "context_precision": round(p / n, 3),
        "answer_relevancy": round(r / n, 3),
    }


def _detect_regression(scores: dict[str, float], baseline: dict[str, float]) -> dict[str, Any]:
    regressed_metric = ""
    delta = 0.0
    for metric in ("faithfulness", "context_precision"):
        new = scores.get(metric, 0.0)
        old = baseline.get(metric, 0.0)
        d = new - old
        if d < -REGRESSION_THRESHOLD_PP and d < delta:
            delta = d
            regressed_metric = metric
    return {
        "regression_detected": bool(regressed_metric),
        "regressed_metric": regressed_metric,
        "regression_delta": round(delta, 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-fail-on-regression", action="store_true")
    args = parser.parse_args(argv)

    items = _load_golden()
    if not items:
        print("No golden questions. Create evaluation/golden_set.jsonl first.")
        return 0

    print(f"Running {len(items)} golden questions...")
    bundle = asyncio.run(_run_golden_set(items))
    scores = _compute_scores(bundle["responses"])
    baseline = _load_baseline()
    regression = _detect_regression(scores, baseline)

    result = {
        "timestamp": datetime.now(UTC).isoformat(),
        "n_questions": len(items),
        "n_answered": sum(1 for r in bundle["responses"] if r["answer"]),
        "scores": scores,
        "baseline": baseline,
        **regression,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "latest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))

    if regression["regression_detected"] and not args.no_fail_on_regression:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
