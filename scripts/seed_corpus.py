"""Bulk-ingest the bundled sample documents into Qdrant for live demos.

Drops the existing collection (so test runs are clean), then ingests each
sample doc with sensible per-document RBAC metadata so multi-user RBAC
demos work out of the box.

Usage:
    uv run python -m scripts.seed_corpus
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ingestion.metadata import IngestRequest, SensitivityLevel  # noqa: E402
from ingestion.pipeline import IngestionPipeline  # noqa: E402
from retrieval.embeddings import EmbeddingService  # noqa: E402
from retrieval.hybrid_search import BM25Index  # noqa: E402
from retrieval.qdrant_client import QdrantManager  # noqa: E402

# (path, sensitivity, allowed_roles)
SEEDS: list[tuple[str, SensitivityLevel, list[str]]] = [
    ("sample_docs/sample_english.txt", SensitivityLevel.HIGH, ["admin", "analyst"]),
    (
        "sample_docs/sample_internal_report.pdf",
        SensitivityLevel.MEDIUM,
        ["analyst", "viewer", "admin"],
    ),
    ("sample_docs/sample_arabic.txt", SensitivityLevel.LOW, ["viewer", "analyst", "admin"]),
    ("sample_docs/sample_mixed.txt", SensitivityLevel.LOW, ["viewer", "analyst", "admin"]),
]


async def main() -> int:
    qdrant = QdrantManager()
    embeddings = EmbeddingService()
    bm25 = BM25Index()
    pipeline = IngestionPipeline(qdrant, embeddings, bm25_index=bm25)

    # Wipe collection so re-runs are clean.
    print(f"Dropping collection '{qdrant.collection_name}'...")
    qdrant.delete_collection()
    qdrant.ensure_collection()

    total_chunks = 0
    for path, sensitivity, roles in SEEDS:
        if not Path(path).exists():
            print(f"  skip (not found): {path}")
            continue
        req = IngestRequest(
            file_path=path,
            user_id="seed_script",
            org_id="acme_corp",
            sensitivity_level=sensitivity,
            roles=roles,
        )
        print(f"Ingesting {path}  sensitivity={sensitivity.value}  roles={roles}")
        result = await pipeline.ingest_document(req)
        print(f"  -> {result.num_chunks} chunks, status={result.status}")
        if result.errors:
            for err in result.errors:
                print(f"     ! {err}")
        total_chunks += result.num_chunks

    print()
    print(f"Done. {total_chunks} total chunks across {len(SEEDS)} sources.")
    info = qdrant.get_collection_info()
    if info:
        print(f"Collection points: {info['points_count']}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
