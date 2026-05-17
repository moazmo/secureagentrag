"""Benchmark script for SecureAgentRAG pipeline performance.

Measures end-to-end latency, per-node timing, and retrieval quality
across different query types. Requires Ollama and Qdrant to be running.

Usage:
    uv run python -m evaluation.benchmark

Output:
    Prints benchmark results table and saves JSON report to data/benchmarks/
"""

from __future__ import annotations

import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from config.settings import settings
from core.graph import run_rag_pipeline
from ingestion.metadata import UserContext
from utils.logging import get_logger

logger = get_logger(__name__)

# Predefined benchmark queries by type
BENCHMARK_QUERIES: dict[str, list[str]] = {
    "simple": [
        "What is the main topic of the document?",
        "Who is the author mentioned in the text?",
        "What date was the report published?",
    ],
    "complex": [
        "Compare the advantages and disadvantages described across all documents.",
        "What are the key findings and how do they relate to each other?",
        "Summarize the timeline of events described in the sources.",
    ],
    "arabic": [
        "ما هو الموضوع الرئيسي للمستند؟",
        "من هو المؤلف المذكور في النص؟",
        "ما هي النتائج الرئيسية الموضحة في المصادر؟",
    ],
}

ADMIN_USER = UserContext(
    user_id="benchmark_user",
    org_id="benchmark_org",
    roles=["admin", "analyst"],
    clearance_level=3,
)


def _ensure_corpus() -> bool:
    """Check if there's ingested data to query against."""
    try:
        from retrieval.qdrant_client import QdrantManager

        qdrant = QdrantManager()
        count = qdrant.get_document_count()
        return count > 0
    except Exception:
        return False


async def _warmup() -> None:
    """Run a single warmup query to populate caches."""
    logger.info("benchmark_warmup_start")
    try:
        await run_rag_pipeline(
            query="What is this document about?",
            user_context=ADMIN_USER,
            thread_id="benchmark-warmup",
        )
    except Exception as exc:
        logger.warning("benchmark_warmup_failed", error=str(exc))
    logger.info("benchmark_warmup_complete")


async def _run_single_benchmark(query: str, query_type: str) -> dict:
    """Run a single benchmark query and collect metrics."""
    start = time.perf_counter()
    try:
        state = await run_rag_pipeline(
            query=query,
            user_context=ADMIN_USER,
            thread_id=f"benchmark-{datetime.now(UTC).isoformat()}",
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "query": query,
            "query_type": query_type,
            "latency_ms": elapsed_ms,
            "success": True,
            "security_passed": state.get("security_passed", False),
            "relevance_ratio": state.get("relevance_ratio", 0.0),
            "retry_count": state.get("retry_count", 0),
            "confidence_score": state.get("confidence_score", 0.0),
            "citation_count": len(state.get("citations", [])),
            "generation_length": len(state.get("generation", "")),
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.error("benchmark_query_failed", query=query, error=str(exc))
        return {
            "query": query,
            "query_type": query_type,
            "latency_ms": elapsed_ms,
            "success": False,
            "error": str(exc),
        }


async def run_benchmarks(runs_per_type: int = 3) -> dict:
    """Run the full benchmark suite.

    Args:
        runs_per_type: Number of queries to run per type.

    Returns:
        Benchmark results dictionary with statistics.
    """
    if not _ensure_corpus():
        print(
            "ERROR: No documents found in Qdrant. Please ingest documents first:\n"
            "  uv run python -m app.main  # Use the Upload tab to add documents\n"
            "Or use sample documents:\n"
            '  uv run python -c "from tests.conftest import *; ..."'
        )
        return {"error": "No documents ingested"}

    await _warmup()

    all_results: list[dict] = []
    type_results: dict[str, list[dict]] = {}

    for query_type, queries in BENCHMARK_QUERIES.items():
        type_results[query_type] = []
        for query in queries[:runs_per_type]:
            result = await _run_single_benchmark(query, query_type)
            all_results.append(result)
            type_results[query_type].append(result)
            print(f"  {query_type}: {result['latency_ms']:.0f}ms", end="")
            if not result["success"]:
                print(" [FAILED]", end="")
            print()

    # Compute statistics per type
    stats: dict[str, dict] = {}
    for query_type, results in type_results.items():
        successful = [r for r in results if r["success"]]
        if not successful:
            stats[query_type] = {"error": "All queries failed"}
            continue

        latencies = [r["latency_ms"] for r in successful]
        latencies.sort()
        stats[query_type] = {
            "count": len(successful),
            "mean_ms": round(statistics.mean(latencies), 1),
            "median_ms": round(statistics.median(latencies), 1),
            "min_ms": round(min(latencies), 1),
            "max_ms": round(max(latencies), 1),
            "p90_ms": round(
                latencies[int(len(latencies) * 0.9)] if len(latencies) > 1 else latencies[0], 1
            ),
            "stddev_ms": round(statistics.stdev(latencies) if len(latencies) > 1 else 0.0, 1),
            "avg_relevance_ratio": round(
                statistics.mean([r["relevance_ratio"] for r in successful]), 3
            ),
            "avg_confidence": round(
                statistics.mean([r["confidence_score"] for r in successful]), 3
            ),
        }

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "configuration": {
            "model": settings.llm_model,
            "embedding_model": settings.embedding_model,
            "top_k": settings.top_k,
            "chunk_size": settings.chunk_size,
            "runs_per_type": runs_per_type,
        },
        "summary": stats,
        "raw_results": all_results,
    }

    # Save report
    report_dir = Path("data/benchmarks")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"benchmark_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


def _print_results(report: dict) -> None:
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 70)
    print("SECUREAGENTRAG BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Timestamp: {report['timestamp']}")
    print(f"Model: {report['configuration']['model']}")
    print(f"Embedding: {report['configuration']['embedding_model']}")
    print()

    print(
        f"{'Type':<10} {'Count':>6} {'Mean':>10} {'Median':>10} {'P90':>10} {'Min':>10} {'Max':>10} {'StdDev':>8}"
    )
    print("-" * 76)

    for query_type, stats in report["summary"].items():
        if "error" in stats:
            print(f"{query_type:<10} {stats['error']}")
            continue
        print(
            f"{query_type:<10} "
            f"{stats['count']:>6} "
            f"{stats['mean_ms']:>8.0f}ms "
            f"{stats['median_ms']:>8.0f}ms "
            f"{stats['p90_ms']:>8.0f}ms "
            f"{stats['min_ms']:>8.0f}ms "
            f"{stats['max_ms']:>8.0f}ms "
            f"{stats['stddev_ms']:>6.0f}ms"
        )
        print(
            f"           "
            f"  relevance={stats['avg_relevance_ratio']:.2f} "
            f"confidence={stats['avg_confidence']:.2f}"
        )

    print()
    print("Report saved to: data/benchmarks/")
    print("=" * 70)


async def main() -> None:
    """Main benchmark entry point."""
    print("Starting SecureAgentRAG Benchmark Suite...")
    print("Note: Ensure Ollama and Qdrant are running before starting.\n")

    report = await run_benchmarks(runs_per_type=3)
    if "error" not in report:
        _print_results(report)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
