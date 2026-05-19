"""Quick cloud-only benchmark for README numbers.

Runs 3 queries via the configured cloud provider and reports latency.
Designed to complete in under 2 minutes.
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

QUERIES = [
    "What are the four NIST AI RMF functions?",
    "How do GOVERN and MANAGE differ in the NIST AI RMF lifecycle?",
    "Summarise the trustworthiness characteristics defined in NIST AI RMF.",
]


async def _run() -> None:
    from config.settings import settings

    cloud = settings.cloud_provider or "groq"
    print(f"Running {len(QUERIES)} queries via {cloud}...")

    latencies: list[float] = []
    confidences: list[float] = []

    for q in QUERIES:
        t0 = time.perf_counter()
        state = await run_rag_pipeline(
            query=q,
            user_context=USER,
            thread_id=f"bench-cloud-{int(t0)}",
            prefer_cloud=True,
        )
        dt_ms = (time.perf_counter() - t0) * 1000
        latencies.append(dt_ms)
        confidences.append(state.get("confidence_score", 0.0))
        print(f"  {dt_ms:7.0f}ms  {q[:50]}...")

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "provider": cloud,
        "queries": len(QUERIES),
        "mean_ms": round(statistics.mean(latencies), 1),
        "p50_ms": round(sorted(latencies)[len(latencies) // 2], 1),
        "p90_ms": round(sorted(latencies)[int(len(latencies) * 0.9)], 1),
        "mean_confidence": round(statistics.mean(confidences), 3),
    }

    out_dir = Path("data/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cloud_bench_{int(datetime.now(UTC).timestamp())}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nCloud ({cloud}) results:")
    print(f"  mean latency: {report['mean_ms']:.0f}ms")
    print(f"  p50 latency:  {report['p50_ms']:.0f}ms")
    print(f"  p90 latency:  {report['p90_ms']:.0f}ms")
    print(f"  mean confidence: {report['mean_confidence']:.3f}")
    print(f"  saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(_run())
