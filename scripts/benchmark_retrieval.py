"""Retrieval benchmark: dense-only vs sparse-only vs hybrid on real corpus.

Usage:
    uv run python -m scripts.benchmark_retrieval

Output:
    evaluation/benchmarks/retrieval_benchmark_<timestamp>.json
"""

from __future__ import annotations

import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from config.settings import settings
from ingestion.metadata import UserContext
from retrieval.embeddings import EmbeddingService
from retrieval.hybrid_search import HybridSearcher
from retrieval.qdrant_client import QdrantManager
from retrieval.sparse_embeddings import SparseEmbeddingService
from utils.logging import get_logger

logger = get_logger(__name__)

# Queries known to have clear answers in the NIST AI RMF
BENCHMARK_QUERIES: list[dict] = [
    {
        "query": "What are the four functions of the AI Risk Management Framework?",
        "keywords": ["govern", "map", "measure", "manage"],
    },
    {
        "query": "How does NIST define trustworthy AI?",
        "keywords": ["trustworthy", "valid", "reliable", "safe", "fair"],
    },
    {
        "query": "What is the purpose of the Govern function?",
        "keywords": ["govern", "culture", "risk management", "policies"],
    },
    {
        "query": "What does the Map function involve?",
        "keywords": ["map", "context", "categorization", "stakeholders"],
    },
    {
        "query": "How should AI risks be measured and evaluated?",
        "keywords": ["measure", "metrics", "evaluation", "testing"],
    },
    {
        "query": "What is the Manage function responsible for?",
        "keywords": ["manage", "respond", "risk", "monitoring"],
    },
    {
        "query": "What are the key characteristics of trustworthy AI systems?",
        "keywords": ["trustworthy", "characteristics", "valid", "reliable", "safe"],
    },
    {
        "query": "How does the AI RMF address bias and fairness?",
        "keywords": ["fair", "bias", "equity", "harm"],
    },
    {
        "query": "What is the AI RMF Playbook?",
        "keywords": ["playbook", "guidance", "implementation", "actions"],
    },
    {
        "query": "What are AI impacts and how are they assessed?",
        "keywords": ["impact", "assessment", "stakeholders", "society"],
    },
]

ADMIN_USER = UserContext(
    user_id="benchmark_user",
    org_id="acme_corp",
    roles=["admin", "analyst"],
    clearance_level=3,
)


def _score_result(text: str, keywords: list[str]) -> float:
    """Score a result by how many keywords it contains (case-insensitive)."""
    text_lower = text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    return matches / len(keywords) if keywords else 0.0


async def _run_single(
    searcher: HybridSearcher,
    query: str,
    keywords: list[str],
    mode: str,
    top_k: int = 10,
) -> dict:
    """Run a single query in a specific mode and collect metrics."""
    start = time.perf_counter()

    try:
        if mode == "dense":
            results = await searcher.search_dense_only(
                query=query,
                user_context=ADMIN_USER,
                top_k=top_k,
            )
        elif mode == "sparse":
            results = await searcher.search_sparse_only(
                query=query,
                user_context=ADMIN_USER,
                top_k=top_k,
            )
        else:  # hybrid
            results = await searcher.search(
                query=query,
                user_context=ADMIN_USER,
                top_k=top_k,
            )

        elapsed_ms = (time.perf_counter() - start) * 1000

        texts = [r.text for r in results]
        scores = [_score_result(t, keywords) for t in texts]
        avg_score = statistics.mean(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0
        top3_score = statistics.mean(scores[:3]) if len(scores) >= 3 else avg_score
        top5_score = statistics.mean(scores[:5]) if len(scores) >= 5 else avg_score

        return {
            "query": query,
            "mode": mode,
            "latency_ms": round(elapsed_ms, 2),
            "result_count": len(results),
            "avg_keyword_score": round(avg_score, 3),
            "max_keyword_score": round(max_score, 3),
            "top3_score": round(top3_score, 3),
            "top5_score": round(top5_score, 3),
            "success": True,
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.error("benchmark_query_failed", query=query, mode=mode, error=str(exc))
        return {
            "query": query,
            "mode": mode,
            "latency_ms": round(elapsed_ms, 2),
            "success": False,
            "error": str(exc),
        }


async def run_benchmark() -> dict:
    """Run the full retrieval benchmark suite."""
    qdrant = QdrantManager()
    embeddings = EmbeddingService()
    sparse = SparseEmbeddingService()
    searcher = HybridSearcher(qdrant, embeddings, sparse_service=sparse)

    all_results: list[dict] = []
    mode_results: dict[str, list[dict]] = {"dense": [], "sparse": [], "hybrid": []}

    print(f"Running retrieval benchmark with {len(BENCHMARK_QUERIES)} queries...")
    print(f"Collection: {qdrant.collection_name} ({qdrant.get_document_count()} docs)")
    print()

    for item in BENCHMARK_QUERIES:
        query = item["query"]
        keywords = item["keywords"]
        print(f"Query: {query[:60]}...")

        for mode in ("dense", "sparse", "hybrid"):
            result = await _run_single(searcher, query, keywords, mode, top_k=10)
            all_results.append(result)
            mode_results[mode].append(result)
            status = "OK" if result["success"] else "FAIL"
            print(
                f"  {mode:8}  {result['latency_ms']:>8.1f}ms  "
                f"top3={result.get('top3_score', 0):.2f}  "
                f"top5={result.get('top5_score', 0):.2f}  "
                f"max={result.get('max_keyword_score', 0):.2f}  [{status}]"
            )
        print()

    # Aggregate stats per mode
    stats: dict[str, dict] = {}
    for mode, results in mode_results.items():
        successful = [r for r in results if r["success"]]
        if not successful:
            stats[mode] = {"error": "All queries failed"}
            continue

        latencies = [r["latency_ms"] for r in successful]
        stats[mode] = {
            "count": len(successful),
            "mean_latency_ms": round(statistics.mean(latencies), 1),
            "median_latency_ms": round(statistics.median(latencies), 1),
            "min_latency_ms": round(min(latencies), 1),
            "max_latency_ms": round(max(latencies), 1),
            "avg_top3_score": round(statistics.mean([r["top3_score"] for r in successful]), 3),
            "avg_top5_score": round(statistics.mean([r["top5_score"] for r in successful]), 3),
            "avg_max_score": round(
                statistics.mean([r["max_keyword_score"] for r in successful]), 3
            ),
        }

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "configuration": {
            "embedding_model": settings.embedding_model,
            "sparse_backend": settings.sparse_backend,
            "collection": qdrant.collection_name,
            "document_count": qdrant.get_document_count(),
        },
        "summary": stats,
        "raw_results": all_results,
    }

    # Save report
    report_dir = Path("evaluation/benchmarks")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = (
        report_dir / f"retrieval_benchmark_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


def _print_summary(report: dict) -> None:
    """Print benchmark summary table."""
    print("=" * 80)
    print("RETRIEVAL BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Timestamp: {report['timestamp']}")
    print(f"Collection: {report['configuration']['collection']}")
    print(f"Documents: {report['configuration']['document_count']}")
    print(f"Sparse backend: {report['configuration']['sparse_backend']}")
    print()

    print(
        f"{'Mode':<10} {'Count':>6} {'Mean':>10} {'Median':>10} {'Top3':>8} {'Top5':>8} {'Max':>8}"
    )
    print("-" * 70)

    for mode, s in report["summary"].items():
        if "error" in s:
            print(f"{mode:<10} {s['error']}")
            continue
        print(
            f"{mode:<10} "
            f"{s['count']:>6} "
            f"{s['mean_latency_ms']:>8.1f}ms "
            f"{s['median_latency_ms']:>8.1f}ms "
            f"{s['avg_top3_score']:>8.3f} "
            f"{s['avg_top5_score']:>8.3f} "
            f"{s['avg_max_score']:>8.3f}"
        )

    print()
    print("Report saved to: evaluation/benchmarks/")
    print("=" * 80)


async def main() -> None:
    """Main benchmark entry point."""
    report = await run_benchmark()
    if "error" not in report:
        _print_summary(report)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
