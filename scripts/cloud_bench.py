"""Cloud-vs-local benchmark comparison.

Runs the same query set against both local Ollama and a configured cloud
provider (Groq/OpenAI/Anthropic), producing a side-by-side latency table.

Usage:
    uv run python -m scripts.cloud_bench

Requires:
    - Ollama + Qdrant running with docs ingested
    - Cloud provider API key in .env (SAR_GROQ_API_KEY, etc.)
    - SAR_CLOUD_PROVIDER set to the provider name
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
    "Explain the relationship between MAP outcomes and MEASURE inputs.",
    "What role does context play across the four AI RMF functions?",
]


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = round((q / 100.0) * (len(s) - 1))
    return s[k]


async def _bench_provider(queries: list[str], provider: str, thread_prefix: str) -> dict:
    """Run queries against a specific provider and return metrics."""
    from config.settings import settings

    # Temporarily flip the default provider
    original = settings.default_provider
    settings.default_provider = provider

    latencies: list[float] = []
    confidences: list[float] = []
    tokens_in: list[int] = []
    tokens_out: list[int] = []

    try:
        for q in queries:
            t0 = time.perf_counter()
            state = await run_rag_pipeline(
                query=q,
                user_context=USER,
                thread_id=f"{thread_prefix}-{int(t0)}",
            )
            dt_ms = (time.perf_counter() - t0) * 1000
            latencies.append(dt_ms)
            confidences.append(state.get("confidence_score", 0.0))
            usage = state.get("synth_usage", {})
            tokens_in.append(usage.get("prompt_tokens", 0))
            tokens_out.append(usage.get("completion_tokens", 0))
            print(f"  [{provider:8s}] {dt_ms:7.0f}ms  {q[:50]}...")
    finally:
        settings.default_provider = original

    return {
        "provider": provider,
        "count": len(latencies),
        "mean_ms": statistics.mean(latencies) if latencies else 0.0,
        "p50_ms": _pct(latencies, 50),
        "p90_ms": _pct(latencies, 90),
        "p99_ms": _pct(latencies, 99),
        "min_ms": min(latencies) if latencies else 0.0,
        "max_ms": max(latencies) if latencies else 0.0,
        "mean_confidence": statistics.mean(confidences) if confidences else 0.0,
        "mean_tokens_in": statistics.mean(tokens_in) if tokens_in else 0.0,
        "mean_tokens_out": statistics.mean(tokens_out) if tokens_out else 0.0,
    }


async def _run() -> dict:
    print("warmup (local)...")
    await run_rag_pipeline(query="warmup", user_context=USER, thread_id="bench-warmup")

    print("\n--- Local (Ollama) ---")
    local_results = await _bench_provider(QUERIES, "ollama", "bench-local")

    print("\n--- Cloud ---")
    from config.settings import settings

    cloud = settings.cloud_provider or "groq"
    cloud_results = await _bench_provider(QUERIES, cloud, "bench-cloud")

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "queries": len(QUERIES),
        "local": local_results,
        "cloud": cloud_results,
    }

    out_dir = Path("data/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cloud_bench_{int(datetime.now(UTC).timestamp())}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== RESULTS ===")
    print(f"{'Metric':<20} {'Local (Ollama)':>18} {'Cloud (' + cloud + ')':>18}")
    print("-" * 58)
    for key in ["mean_ms", "p50_ms", "p90_ms", "p99_ms", "mean_confidence"]:
        local_val = local_results[key]
        cloud_val = cloud_results[key]
        print(f"{key:<20} {local_val:>18.1f} {cloud_val:>18.1f}")
    speedup = local_results["mean_ms"] / max(cloud_results["mean_ms"], 1)
    print(f"\nCloud is ~{speedup:.1f}x faster on average")
    print(f"Results saved to {out_path}")

    return report


if __name__ == "__main__":
    asyncio.run(_run())
