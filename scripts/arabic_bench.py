"""Quick Arabic benchmark via cloud provider.

Runs Arabic queries through the configured cloud provider to avoid
Ollama serialization slowness on Arabic text.

Usage:
    uv run python -m scripts.arabic_bench
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
    "ما هي الوظائف الأربع لإطار إدارة مخاطر الذكاء الاصطناعي NIST؟",
    "اشرح الفرق بين وظيفتي الحوكمة GOVERN والإدارة MANAGE في إطار NIST AI RMF.",
    "ما هي خصائص الجدارة بالثقة المحددة في إطار إدارة مخاطر الذكاء الاصطناعي؟",
]


async def _run() -> None:
    from config.settings import settings

    cloud = settings.cloud_provider or "groq"
    print(f"Running {len(QUERIES)} Arabic queries via {cloud}...")

    latencies: list[float] = []
    confidences: list[float] = []

    for q in QUERIES:
        t0 = time.perf_counter()
        state = await run_rag_pipeline(
            query=q,
            user_context=USER,
            thread_id=f"bench-arabic-{int(t0)}",
            prefer_cloud=True,
        )
        dt_ms = (time.perf_counter() - t0) * 1000
        latencies.append(dt_ms)
        confidences.append(state.get("confidence_score", 0.0))
        print(f"  {dt_ms:7.0f}ms  conf={state.get('confidence_score', 0):.2f}  {q[:40]}...")

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
    out_path = out_dir / f"arabic_bench_{int(datetime.now(UTC).timestamp())}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nArabic ({cloud}) results:")
    print(f"  mean latency: {report['mean_ms']:.0f}ms")
    print(f"  p50 latency:  {report['p50_ms']:.0f}ms")
    print(f"  p90 latency:  {report['p90_ms']:.0f}ms")
    print(f"  mean confidence: {report['mean_confidence']:.3f}")
    print(f"  saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(_run())
