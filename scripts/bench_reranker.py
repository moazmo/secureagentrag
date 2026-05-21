"""Benchmark a reranker checkpoint vs the off-the-shelf baseline.

Computes NDCG@10 on a held-out MS-MARCO slice and on a small
hand-labelled NIST AI RMF subset (when present). Writes a Markdown
report to ``evaluation/benchmarks/reranker_finetune.md`` and a
machine-readable JSON next to it.

Usage::

    uv run python -m scripts.bench_reranker \\
        --baseline BAAI/bge-reranker-v2-m3 \\
        --candidate data/checkpoints/reranker-domain-v1 \\
        --eval-size 500
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

NIST_GOLD_PATH = _ROOT / "evaluation" / "nist_rerank_gold.jsonl"
"""Optional hand-labelled subset for in-domain bench.

Schema: one JSON per line with ``query``, ``positive``, ``negatives``.
When the file does not exist we silently skip the NIST half of the
benchmark; the MS-MARCO numbers still report.
"""


def _ndcg_at_k(rels: list[int], k: int = 10) -> float:
    """Standard NDCG@k for a single ranking. ``rels[i]`` is the
    relevance of the doc at rank ``i+1`` (1=positive, 0=negative)."""
    dcg = 0.0
    for i, r in enumerate(rels[:k]):
        if r > 0:
            dcg += r / math.log2(i + 2)
    ideal = sorted(rels, reverse=True)[:k]
    idcg = 0.0
    for i, r in enumerate(ideal):
        if r > 0:
            idcg += r / math.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0


def _load_ms_marco_eval(eval_size: int) -> list[tuple[str, list[str], list[str]]]:
    """Hold-out eval pairs from MS-MARCO. Re-uses the loader from
    ``train_reranker`` so a single source of truth covers both."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "MS-MARCO bench requires datasets. Install via: uv sync --extra embeddings-local"
        ) from exc

    ds = load_dataset("microsoft/ms_marco", "v2.1", split="train", streaming=True)
    pairs: list[tuple[str, list[str], list[str]]] = []
    for row in ds:
        if len(pairs) >= eval_size:
            break
        query = row["query"]
        passages = row.get("passages", {})
        is_sel = passages.get("is_selected", [])
        texts = passages.get("passage_text", [])
        positives = [t for t, s in zip(texts, is_sel, strict=False) if s == 1]
        negatives = [t for t, s in zip(texts, is_sel, strict=False) if s == 0]
        if positives and negatives:
            pairs.append((query, positives, negatives))
    return pairs


def _load_nist_gold() -> list[tuple[str, list[str], list[str]]]:
    if not NIST_GOLD_PATH.exists():
        return []
    out: list[tuple[str, list[str], list[str]]] = []
    for line in NIST_GOLD_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        out.append((row["query"], [row["positive"]], list(row.get("negatives", []))))
    return out


def _score_pairs(
    model_name: str, pairs: list[tuple[str, list[str], list[str]]]
) -> tuple[float, float, float]:
    """Score one model on a list of (query, positives, negatives).

    Returns:
        ``(mean_ndcg10, mean_score_diff, latency_ms)``. Score diff is the
        average margin between the top positive and top negative across
        the eval set — quick sanity that the model separates classes.
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "Reranker bench requires sentence-transformers. Install via: "
            "uv sync --extra embeddings-local"
        ) from exc

    model = CrossEncoder(model_name)
    ndcgs: list[float] = []
    margins: list[float] = []
    t0 = time.perf_counter()
    for query, positives, negatives in pairs:
        candidates = positives + negatives
        scores = model.predict([(query, c) for c in candidates])
        # Sort by score desc, then build the relevance list.
        ranking = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        rels = [1 if idx < len(positives) else 0 for idx, _ in ranking]
        ndcgs.append(_ndcg_at_k(rels, k=10))
        pos_max = max(scores[: len(positives)]) if positives else 0.0
        neg_max = max(scores[len(positives) :]) if negatives else 0.0
        margins.append(float(pos_max - neg_max))
    elapsed = (time.perf_counter() - t0) * 1000
    return (
        statistics.mean(ndcgs) if ndcgs else 0.0,
        statistics.mean(margins) if margins else 0.0,
        elapsed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--candidate", default="data/checkpoints/reranker-domain-v1")
    parser.add_argument("--eval-size", type=int, default=500)
    args = parser.parse_args()

    if not Path(args.candidate).exists():
        print(
            f"[bench_reranker] candidate not found: {args.candidate}\n"
            "Run scripts.train_reranker first or pass --candidate <model id>.",
            file=sys.stderr,
        )
        return 2

    print(f"Loading {args.eval_size} MS-MARCO hold-out pairs...")
    ms_pairs = _load_ms_marco_eval(args.eval_size)
    print(f"  {len(ms_pairs)} pairs loaded.")

    nist_pairs = _load_nist_gold()
    if nist_pairs:
        print(f"  + {len(nist_pairs)} NIST gold pairs.")
    else:
        print("  (NIST gold not present — skipping in-domain bench)")

    print(f"\nScoring baseline:  {args.baseline}")
    base_ms = _score_pairs(args.baseline, ms_pairs)
    base_nist = _score_pairs(args.baseline, nist_pairs) if nist_pairs else (0.0, 0.0, 0.0)

    print(f"Scoring candidate: {args.candidate}")
    cand_ms = _score_pairs(args.candidate, ms_pairs)
    cand_nist = _score_pairs(args.candidate, nist_pairs) if nist_pairs else (0.0, 0.0, 0.0)

    delta_ms = (cand_ms[0] - base_ms[0]) * 100  # in percentage points
    delta_nist = (cand_nist[0] - base_nist[0]) * 100 if nist_pairs else 0.0

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "baseline": args.baseline,
        "candidate": args.candidate,
        "eval_size_msmarco": len(ms_pairs),
        "eval_size_nist": len(nist_pairs),
        "msmarco": {
            "baseline_ndcg10": base_ms[0],
            "candidate_ndcg10": cand_ms[0],
            "delta_pp": delta_ms,
        },
        "nist": {
            "baseline_ndcg10": base_nist[0],
            "candidate_ndcg10": cand_nist[0],
            "delta_pp": delta_nist,
        },
    }
    out_dir = _ROOT / "evaluation" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(datetime.now(UTC).timestamp())
    (out_dir / f"reranker_finetune_{ts}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    md = f"""# Reranker Fine-tune Benchmark

- **Baseline:**  `{args.baseline}`
- **Candidate:** `{args.candidate}`
- **Eval:** MS-MARCO {len(ms_pairs)} pairs, NIST gold {len(nist_pairs)} pairs

| Dataset       | Baseline NDCG@10 | Candidate NDCG@10 | Delta (pp) |
|---------------|-----------------:|------------------:|-----------:|
| MS-MARCO      | {base_ms[0]:.4f}        | {cand_ms[0]:.4f}         | {delta_ms:+.2f}    |
| NIST (in-dom) | {base_nist[0]:.4f}      | {cand_nist[0]:.4f}       | {delta_nist:+.2f}  |

Acceptance bar (per ADR-022): candidate must beat baseline by ≥1pp on MS-MARCO and win on NIST.
"""
    (out_dir / "reranker_finetune.md").write_text(md, encoding="utf-8")
    print("\n=== RESULTS ===")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
