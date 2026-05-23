"""Calibrate confidence + faithfulness thresholds against the labeled gold set.

What this script does
---------------------
1. Loads ``evaluation/golden_set.jsonl`` (~50 labelled questions covering NIST
   AI RMF facts, ACME synthetic RBAC corpus, RBAC negatives, out-of-scope,
   prompt-injection probes, bilingual queries, and adversarial / unsupported
   claim probes). Each row carries an ``expected_confidence_band`` and an
   ``expected_faithfulness_band`` ("high" / "medium" / "low") plus an
   ``expected_outcome`` ("answer" / "refuse" / "block").
2. Runs each row through the live RAG pipeline (``run_rag_pipeline``) under
   its own ``UserContext`` so RBAC paths are exercised end-to-end. Forces the
   NLI faithfulness gate ON (``SAR_FAITHFULNESS_GATE_ENABLED=true``) so the
   ``faithfulness_ratio`` signal is non-trivial.
3. Sweeps thresholds across ``[0.0, 1.0]`` in 0.05 steps and picks the value
   that maximises Youden's J (TPR - FPR) — the "best" cutoff separating
   high-quality answers from refusals / blocks / off-topic responses. The
   same sweep runs independently for the confidence and faithfulness signals.
4. Persists the chosen thresholds + full sweep curves to
   ``evaluation/calibration.json`` plus a timestamped copy under
   ``evaluation/results/calibration_<ts>.json``. ``config/settings.py`` reads
   ``calibration.json`` on import to override the hard-coded defaults.
5. Computes Ragas-style scores (lexical fallback when the ragas package is
   unavailable) so ``evaluation/baseline.json`` can be refreshed with a
   measured baseline instead of the legacy hand-picked numbers.

Why not Ragas for threshold calibration?
----------------------------------------
The thresholds gate ``needs_human_review`` and the synthesizer's NLI signal —
both produced by the project's own agents. Calibrating those against
human-labelled bands is purer than chaining Ragas in the middle. Ragas still
runs (when available) to write a measured baseline, but is not on the
critical path for picking the thresholds.

Usage
-----
::

    # Full calibration (live services required — Qdrant + Ollama)
    uv run python -m scripts.calibrate_thresholds

    # Cap to N samples for a quick re-run
    uv run python -m scripts.calibrate_thresholds --limit 10

    # Skip the live pipeline and recompute thresholds from a previous JSON
    uv run python -m scripts.calibrate_thresholds --from-results \\
        evaluation/results/calibration_<ts>.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Force the NLI faithfulness gate on for the duration of this run. Must be set
# before ``config.settings`` is imported so pydantic-settings picks it up.
os.environ.setdefault("SAR_FAITHFULNESS_GATE_ENABLED", "true")
# Calibration runs the corrective-RAG loop + the per-sentence faithfulness
# gate, which adds ~30-90s on top of normal synthesis. Raise the pipeline
# SLO deadline well above the default 60s so genuine outputs aren't masked
# by timeout-state fallbacks. Still safe — the script is a one-shot bench,
# not a user-facing path.
os.environ.setdefault("SAR_REQUEST_TIMEOUT_S", "600")
# RAG fusion does an extra LLM call per row plus N parallel searches. Switch
# it off for calibration so we measure single-pass quality and stay within
# tolerable runtime.
os.environ.setdefault("SAR_RAG_FUSION_ENABLED", "false")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.graph import run_rag_pipeline  # noqa: E402
from ingestion.metadata import UserContext  # noqa: E402
from utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

GOLDEN_PATH = _ROOT / "evaluation" / "golden_set.jsonl"
CALIB_PATH = _ROOT / "evaluation" / "calibration.json"
BASELINE_PATH = _ROOT / "evaluation" / "baseline.json"
RESULTS_DIR = _ROOT / "evaluation" / "results"

# Band -> expected signal range used to label each gold row as positive
# (good answer expected) or negative (refusal / low-quality expected).
_POSITIVE_BANDS = {"high", "medium"}
_NEGATIVE_BANDS = {"low"}

# Threshold sweep granularity.
_SWEEP_STEP = 0.05


def _load_golden(path: Path) -> list[dict[str, Any]]:
    """Read the gold set, returning one dict per non-empty JSONL line."""
    rows: list[dict[str, Any]] = []
    if not path.exists():
        logger.error("golden_set_missing", path=str(path))
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


async def _run_one(row: dict[str, Any]) -> dict[str, Any]:
    """Run a single gold row through the pipeline and capture its signals."""
    user = UserContext(
        user_id=row.get("user_id", "calib"),
        org_id=row.get("org_id", "acme_corp"),
        roles=row.get("roles", ["viewer"]),
        clearance_level=int(row.get("clearance_level", 1)),
    )
    start = time.perf_counter()
    try:
        state = await run_rag_pipeline(
            query=row["question"],
            user_context=user,
            thread_id=f"calib-{row.get('id', 'q')}",
        )
    except Exception as exc:
        logger.error("calib_row_failed", id=row.get("id"), error=str(exc))
        return {
            "id": row.get("id", ""),
            "category": row.get("category", ""),
            "expected_confidence_band": row.get("expected_confidence_band", "low"),
            "expected_faithfulness_band": row.get("expected_faithfulness_band", "low"),
            "expected_outcome": row.get("expected_outcome", "refuse"),
            "confidence_score": 0.0,
            "faithfulness_ratio": 0.0,
            "blocked": True,
            "answer": "",
            "ground_truth": row.get("ground_truth", ""),
            "contexts": [],
            "latency_ms": (time.perf_counter() - start) * 1000,
            "error": str(exc),
        }
    return {
        "id": row.get("id", ""),
        "category": row.get("category", ""),
        "expected_confidence_band": row.get("expected_confidence_band", "low"),
        "expected_faithfulness_band": row.get("expected_faithfulness_band", "low"),
        "expected_outcome": row.get("expected_outcome", "refuse"),
        "confidence_score": float(state.get("confidence_score", 0.0)),
        "faithfulness_ratio": float(state.get("faithfulness_ratio", 1.0)),
        "blocked": not state.get("security_passed", True)
        or not state.get("guardrails_passed", True),
        "answer": state.get("generation", ""),
        "ground_truth": row.get("ground_truth", ""),
        "contexts": [d["text"] for d in state.get("relevant_documents", [])]
        or [d["text"] for d in state.get("documents", [])],
        "latency_ms": (time.perf_counter() - start) * 1000,
    }


async def _run_pipeline(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    """Run every gold row sequentially; bounded by ``limit`` when provided."""
    if limit is not None:
        rows = rows[:limit]
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows, start=1):
        logger.info("calib_row_start", index=i, total=len(rows), id=row.get("id"))
        result = await _run_one(row)
        out.append(result)
        logger.info(
            "calib_row_done",
            id=result["id"],
            confidence=round(result["confidence_score"], 3),
            faithfulness=round(result["faithfulness_ratio"], 3),
            blocked=result["blocked"],
            latency_ms=round(result["latency_ms"], 0),
        )
    return out


def _band_to_label(band: str) -> int:
    """Return 1 for positive bands ("high"/"medium"), 0 for "low"."""
    band = (band or "low").lower()
    if band in _POSITIVE_BANDS:
        return 1
    if band in _NEGATIVE_BANDS:
        return 0
    return 0


def _sweep_thresholds(pairs: list[tuple[float, int]]) -> dict[str, Any]:
    """Sweep ``threshold ∈ [0, 1]`` and pick the value maximising Youden's J.

    Args:
        pairs: list of ``(predicted_score, gold_label)`` tuples where the
            label is 1 for positive (expected high-quality) and 0 for negative
            (expected refusal / low quality).

    Returns:
        Dict with ``chosen_threshold``, ``chosen_metrics``, ``curve`` (full
        sweep), and ``n_pos`` / ``n_neg`` counts.
    """
    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = sum(1 for _, y in pairs if y == 0)
    curve: list[dict[str, float]] = []
    best: dict[str, float] | None = None

    t = 0.0
    while t <= 1.0 + 1e-9:
        tp = sum(1 for s, y in pairs if s >= t and y == 1)
        fp = sum(1 for s, y in pairs if s >= t and y == 0)
        fn = n_pos - tp
        tn = n_neg - fp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        tpr = recall
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        j = tpr - fpr
        point = {
            "threshold": round(t, 3),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tpr": round(tpr, 4),
            "fpr": round(fpr, 4),
            "j": round(j, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        curve.append(point)
        if best is None or j > best["j"] or (j == best["j"] and f1 > best["f1"]):
            best = point
        t += _SWEEP_STEP

    return {
        "chosen_threshold": best["threshold"] if best else 0.5,
        "chosen_metrics": best or {},
        "curve": curve,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_total": len(pairs),
    }


def _calibrate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the calibration JSON payload from per-row pipeline results."""
    # Exclude rows that errored out — they tell us nothing about thresholds.
    usable = [r for r in results if "error" not in r]

    conf_pairs = [
        (r["confidence_score"], _band_to_label(r["expected_confidence_band"])) for r in usable
    ]
    faith_pairs = [
        (r["faithfulness_ratio"], _band_to_label(r["expected_faithfulness_band"]))
        for r in usable
        # Drop blocked rows from the faithfulness sweep — they never reach the
        # synthesizer so the ratio defaults to 1.0 and pollutes the curve.
        if not r["blocked"]
    ]

    conf_calib = _sweep_thresholds(conf_pairs)
    faith_calib = _sweep_thresholds(faith_pairs)

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "golden_set_path": str(GOLDEN_PATH.relative_to(_ROOT)),
        "n_rows_total": len(results),
        "n_rows_usable": len(usable),
        "confidence": conf_calib,
        "faithfulness": faith_calib,
    }


def _lexical_baseline(results: list[dict[str, Any]]) -> dict[str, float]:
    """Cheap Ragas-style scores from token overlap; used when ragas is missing.

    Faithfulness = mean(answer ∩ context / answer) across answered rows.
    Context-precision = mean(context_i ∩ answer > 0) across contexts.
    Answer-relevancy = mean(ground_truth ∩ answer / ground_truth).
    """
    f, p, r = 0.0, 0.0, 0.0
    n = 0
    for item in results:
        if item.get("blocked") or not item.get("answer"):
            continue
        ans = set(item["answer"].lower().split())
        ctxs = [set(c.lower().split()) for c in item.get("contexts", [])] or [set()]
        gt = set((item.get("ground_truth") or "").lower().split())
        if not ans:
            continue
        any_ctx = set().union(*ctxs) if ctxs else set()
        f += len(ans & any_ctx) / max(len(ans), 1)
        p += sum(bool(c & ans) for c in ctxs) / max(len(ctxs), 1)
        r += len(gt & ans) / max(len(gt), 1) if gt else 0.0
        n += 1
    if n == 0:
        return {"faithfulness": 0.0, "context_precision": 0.0, "answer_relevancy": 0.0}
    return {
        "faithfulness": round(f / n, 3),
        "context_precision": round(p / n, 3),
        "answer_relevancy": round(r / n, 3),
    }


def _measured_baseline(results: list[dict[str, Any]]) -> dict[str, float]:
    """Compute baseline metrics — Ragas if available, lexical otherwise."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness

        ds = Dataset.from_list(
            [
                {
                    "question": r["id"],
                    "answer": r["answer"],
                    "contexts": r["contexts"] or [""],
                    "ground_truth": r.get("ground_truth", ""),
                }
                for r in results
                if r.get("answer") and not r.get("blocked")
            ]
        )
        out = evaluate(ds, metrics=[faithfulness, context_precision, answer_relevancy])
        return {
            "faithfulness": round(float(out["faithfulness"]), 3),
            "context_precision": round(float(out["context_precision"]), 3),
            "answer_relevancy": round(float(out["answer_relevancy"]), 3),
        }
    except Exception as exc:
        logger.warning("ragas_unavailable_using_lexical", error=str(exc))
        return _lexical_baseline(results)


def _persist(
    calib: dict[str, Any], baseline: dict[str, float], results: list[dict[str, Any]]
) -> None:
    """Write calibration + baseline JSON files; mirror to results dir."""
    CALIB_PATH.write_text(json.dumps(calib, indent=2), encoding="utf-8")
    logger.info(
        "calibration_persisted",
        path=str(CALIB_PATH),
        confidence_threshold=calib["confidence"]["chosen_threshold"],
        faithfulness_threshold=calib["faithfulness"]["chosen_threshold"],
    )

    # Refresh baseline.json with measured Ragas / lexical scores. Preserve the
    # original ``_note`` field if present so the historical context stays.
    prior_baseline = {}
    if BASELINE_PATH.exists():
        try:
            prior_baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        except Exception:
            prior_baseline = {}
    prior_baseline.update(baseline)
    prior_baseline["_calibrated_at"] = calib["timestamp"]
    BASELINE_PATH.write_text(json.dumps(prior_baseline, indent=2), encoding="utf-8")
    logger.info("baseline_refreshed", path=str(BASELINE_PATH), **baseline)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(datetime.now(UTC).timestamp())
    snapshot = {
        "timestamp": calib["timestamp"],
        "baseline": baseline,
        "calibration": calib,
        "per_row": results,
    }
    (RESULTS_DIR / f"calibration_{ts}.json").write_text(
        json.dumps(snapshot, indent=2), encoding="utf-8"
    )


def _from_results(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Re-load per-row pipeline results from a previous calibration snapshot."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("per_row", []), data.get("calibration", {})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Cap how many gold rows to run.")
    parser.add_argument(
        "--from-results",
        type=str,
        default="",
        help="Recompute thresholds from a previous calibration JSON without "
        "hitting the live pipeline.",
    )
    args = parser.parse_args(argv)

    if args.from_results:
        per_row, _ = _from_results(Path(args.from_results))
        if not per_row:
            print(f"[calibrate] no per_row found in {args.from_results}", file=sys.stderr)
            return 2
        calib = _calibrate(per_row)
        baseline = _measured_baseline(per_row)
        _persist(calib, baseline, per_row)
        print(json.dumps({"calibration": calib, "baseline": baseline}, indent=2))
        return 0

    rows = _load_golden(GOLDEN_PATH)
    if not rows:
        print(f"[calibrate] {GOLDEN_PATH} is empty or missing.", file=sys.stderr)
        return 2
    print(f"[calibrate] running {len(rows)} gold rows through live pipeline...")
    results = asyncio.run(_run_pipeline(rows, args.limit))

    calib = _calibrate(results)
    baseline = _measured_baseline(results)
    _persist(calib, baseline, results)

    print("\n=== CALIBRATION RESULTS ===")
    print(f"Confidence threshold:   {calib['confidence']['chosen_threshold']:.2f}")
    print(
        f"  precision={calib['confidence']['chosen_metrics']['precision']:.3f}  "
        f"recall={calib['confidence']['chosen_metrics']['recall']:.3f}  "
        f"f1={calib['confidence']['chosen_metrics']['f1']:.3f}  "
        f"J={calib['confidence']['chosen_metrics']['j']:.3f}"
    )
    print(f"Faithfulness threshold: {calib['faithfulness']['chosen_threshold']:.2f}")
    print(
        f"  precision={calib['faithfulness']['chosen_metrics']['precision']:.3f}  "
        f"recall={calib['faithfulness']['chosen_metrics']['recall']:.3f}  "
        f"f1={calib['faithfulness']['chosen_metrics']['f1']:.3f}  "
        f"J={calib['faithfulness']['chosen_metrics']['j']:.3f}"
    )
    print(f"\nBaseline (measured):  {baseline}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
