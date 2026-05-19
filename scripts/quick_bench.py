"""Short-form benchmark — 5 simple + 5 complex English queries against
the live local stack. Skips Arabic so the run completes in minutes
instead of hours on consumer hardware.

Usage:
    uv run python -m scripts.quick_bench
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.graph import run_rag_pipeline  # noqa: E402
from ingestion.metadata import UserContext  # noqa: E402

USER = UserContext(
    user_id="bench",
    org_id="acme_corp",
    roles=["admin", "engineer", "finance_manager", "analyst", "viewer"],
    clearance_level=3,
)

QUERIES = {
    "simple": [
        "What are the four NIST AI RMF functions?",
        "What is the GOVERN function?",
        "What is the MAP function?",
        "What is the MEASURE function?",
        "What is the MANAGE function?",
    ],
    "complex": [
        "How do GOVERN and MANAGE differ in the NIST AI RMF lifecycle?",
        "Summarise the trustworthiness characteristics defined in NIST AI RMF.",
        "Explain the relationship between MAP outcomes and MEASURE inputs.",
        "What role does context play across the four AI RMF functions?",
        "Describe how risk prioritisation flows through MAP -> MEASURE -> MANAGE.",
    ],
}


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = round((q / 100.0) * (len(s) - 1))
    return s[k]


async def _run() -> dict:
    print("warmup...")
    await run_rag_pipeline(query="warmup", user_context=USER, thread_id="bench-warmup")

    out: dict = {"by_type": {}}
    for qtype, queries in QUERIES.items():
        latencies: list[float] = []
        confidences: list[float] = []
        relevance: list[float] = []
        retries: list[int] = []
        for q in queries:
            t0 = time.perf_counter()
            state = await run_rag_pipeline(
                query=q, user_context=USER, thread_id=f"bench-{qtype}-{int(t0)}"
            )
            dt_ms = (time.perf_counter() - t0) * 1000
            latencies.append(dt_ms)
            confidences.append(state.get("confidence_score", 0.0))
            relevance.append(state.get("relevance_ratio", 0.0))
            retries.append(state.get("retry_count", 0))
            print(
                f"  [{qtype}] {dt_ms:7.0f}ms  conf={state.get('confidence_score', 0):.2f}  "
                f"rel={state.get('relevance_ratio', 0):.2f}  retries={state.get('retry_count', 0)}"
            )
        out["by_type"][qtype] = {
            "n": len(latencies),
            "p50_ms": int(statistics.median(latencies)),
            "p90_ms": int(_pct(latencies, 90)),
            "p99_ms": int(_pct(latencies, 99)),
            "mean_ms": int(statistics.mean(latencies)),
            "min_ms": int(min(latencies)),
            "max_ms": int(max(latencies)),
            "mean_confidence": round(statistics.mean(confidences), 3),
            "mean_relevance": round(statistics.mean(relevance), 3),
            "mean_retries": round(statistics.mean(retries), 2),
        }

    out["timestamp"] = datetime.now(UTC).isoformat()
    out["model"] = "qwen3:8b"
    out["embedding"] = "bge-m3"
    return out


def main() -> int:
    bench = asyncio.run(_run())
    out_path = _ROOT / "data" / "benchmarks" / f"quick_bench_{int(time.time())}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bench, indent=2), encoding="utf-8")
    print()
    print(json.dumps(bench, indent=2))
    print(f"\nSaved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
