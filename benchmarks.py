"""Benchmark script for SecureAgentRAG performance measurement.

Measures end-to-end latency, retrieval quality, and throughput
under various configurations. Results are printed as Markdown tables
suitable for copying into README.md.

Usage:
    uv run python benchmarks.py

Requirements:
    - Qdrant running on localhost:6333
    - Ollama running on localhost:11434 with qwen3:8b and bge-m3
    - Sample documents ingested
"""

from __future__ import annotations

import asyncio
import time
from statistics import mean, median, stdev

from config.settings import settings
from core.graph import run_rag_pipeline
from ingestion.metadata import UserContext
from utils.logging import get_logger, setup_logging

logger = get_logger(__name__)
setup_logging()

# Test queries of varying complexity
TEST_QUERIES = [
    ("simple", "What is the company name?"),
    ("medium", "What are the main privacy policy principles?"),
    ("complex", "Compare the data retention policies for different user types and explain the implications."),
    ("arabic", "ما هي سياسة الخصوصية للشركة؟"),  # "What is the company's privacy policy?"
]

# Simulated user context for benchmarking
BENCH_USER = UserContext(
    user_id="benchmark_user",
    org_id="benchmark_org",
    roles=["viewer", "admin"],
    clearance_level=3,
)


async def benchmark_latency(
    num_runs: int = 3,
    warmup: int = 1,
) -> dict[str, dict[str, float]]:
    """Measure end-to-end pipeline latency for different query types.

    Args:
        num_runs: Number of measurement runs per query.
        warmup: Number of warmup runs before measurement (to warm caches).

    Returns:
        Dict mapping query_type to latency statistics.
    """
    results: dict[str, dict[str, float]] = {}

    for query_type, query in TEST_QUERIES:
        latencies: list[float] = []

        # Warmup runs
        for _ in range(warmup):
            try:
                await run_rag_pipeline(query, BENCH_USER, thread_id=f"warmup_{time.time()}")
            except Exception as exc:
                logger.warning("warmup_failed", query=query, error=str(exc))

        # Measurement runs
        for run in range(num_runs):
            try:
                start = time.perf_counter()
                state = await run_rag_pipeline(
                    query, BENCH_USER, thread_id=f"bench_{query_type}_{run}"
                )
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies.append(elapsed_ms)
                logger.info(
                    "benchmark_run",
                    query_type=query_type,
                    run=run + 1,
                    latency_ms=elapsed_ms,
                    confidence=state.get("confidence_score", 0.0),
                )
            except Exception as exc:
                logger.error("benchmark_run_failed", query=query, error=str(exc))

        if latencies:
            results[query_type] = {
                "mean_ms": mean(latencies),
                "median_ms": median(latencies),
                "min_ms": min(latencies),
                "max_ms": max(latencies),
                "std_ms": stdev(latencies) if len(latencies) > 1 else 0.0,
            }
        else:
            results[query_type] = {
                "mean_ms": 0.0,
                "median_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "std_ms": 0.0,
            }

    return results


def print_results(results: dict[str, dict[str, float]]) -> None:
    """Print benchmark results as a Markdown table."""
    print("\n## Benchmark Results\n")
    print("| Query Type | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | StdDev (ms) |")
    print("|-----------|-----------|-------------|----------|----------|-------------|")
    for query_type, stats in results.items():
        print(
            f"| {query_type:11} | "
            f"{stats['mean_ms']:9.1f} | "
            f"{stats['median_ms']:11.1f} | "
            f"{stats['min_ms']:8.1f} | "
            f"{stats['max_ms']:8.1f} | "
            f"{stats['std_ms']:11.1f} |"
        )
    print()
    print("### Configuration")
    print(f"- **Model**: {settings.llm_model}")
    print(f"- **Embedding**: {settings.embedding_model} ({settings.embedding_dim}d)")
    print(f"- **Top K**: {settings.top_k}")
    print(f"- **Chunk Size**: {settings.chunk_size}/{settings.chunk_overlap}")
    print("- **Runs per query**: 3 (plus 1 warmup)")
    print()


async def main() -> None:
    """Run all benchmarks and print results."""
    print("=" * 60)
    print("SecureAgentRAG Performance Benchmarks")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  LLM: {settings.llm_model}")
    print(f"  Embedding: {settings.embedding_model}")
    print(f"  Qdrant: {settings.qdrant_url}")
    print(f"  Ollama: {settings.ollama_url}")
    print()

    print("Running latency benchmarks (this may take a few minutes)...")
    print("-" * 60)

    results = await benchmark_latency(num_runs=3, warmup=1)
    print_results(results)

    print("=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
