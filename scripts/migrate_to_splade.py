"""Migration script: add Qdrant native sparse vectors to an existing collection.

Scrolls all points from the configured collection, generates sparse vectors
for the chunk text, and re-upserts each point with both its original dense
vector and the new sparse vector.

Usage::

    uv run python -m scripts.migrate_to_splade --collection documents

The backend (``bm25`` or ``splade``) is controlled by ``SAR_SPARSE_BACKEND``.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.settings import settings  # noqa: E402
from retrieval.qdrant_client import QdrantManager  # noqa: E402
from retrieval.sparse_embeddings import SparseEmbeddingService  # noqa: E402
from utils.logging import get_logger  # noqa: E402

logger = get_logger("migrate_to_splade")

BATCH_SIZE = 64


async def migrate_collection(collection_name: str | None = None) -> int:
    """Migrate a single collection to include sparse vectors.

    Args:
        collection_name: Collection to migrate. Defaults to
            ``settings.qdrant_collection``.

    Returns:
        Number of points migrated.
    """
    qdrant = QdrantManager(collection_name=collection_name)
    sparse = SparseEmbeddingService()
    collection = collection_name or settings.qdrant_collection
    sparse_name = getattr(settings, "sparse_vector_name", "sparse")

    logger.info("migration_started", collection=collection, backend=sparse.backend)

    # Check collection exists
    info = qdrant.get_collection_info()
    if info is None:
        logger.error("collection_not_found", collection=collection)
        return 0

    total_points = info.get("points_count", 0)
    if total_points == 0:
        logger.info("collection_empty", collection=collection)
        return 0

    logger.info(
        "collection_info",
        collection=collection,
        points_count=total_points,
    )

    # Scroll all points with dense vectors and payload
    offset: str | None = None
    migrated = 0
    while True:
        results, next_offset = qdrant.client.scroll(
            collection_name=collection,
            limit=BATCH_SIZE,
            offset=offset,
            with_vectors=True,
            with_payload=True,
        )

        if not results:
            break

        point_ids = []
        dense_vectors = []
        texts = []
        payloads = []

        for point in results:
            point_ids.append(str(point.id))
            # Dense vector is stored under the default (empty string) name
            vec = point.vector
            if isinstance(vec, dict):
                dense_vectors.append(vec.get("", []))
            else:
                dense_vectors.append(vec)
            payload = point.payload or {}
            texts.append(payload.get("text", ""))
            payloads.append(payload)

        # Generate sparse vectors for this batch
        try:
            sparse_vectors = sparse.embed_texts(texts)
        except Exception as exc:
            logger.error("sparse_generation_failed", error=str(exc), batch_size=len(texts))
            raise

        # Re-upsert with both dense and sparse vectors
        from qdrant_client.http.models import PointStruct

        points = []
        for pid, dense, spvec, payload in zip(
            point_ids, dense_vectors, sparse_vectors, payloads, strict=False
        ):
            points.append(
                PointStruct(
                    id=pid,
                    vector={"": dense, sparse_name: spvec},
                    payload=payload,
                )
            )

        qdrant.client.upsert(collection_name=collection, points=points)
        migrated += len(points)
        logger.info(
            "batch_migrated",
            batch_size=len(points),
            total_migrated=migrated,
            collection=collection,
        )

        if next_offset is None:
            break
        offset = next_offset

    logger.info(
        "migration_completed",
        collection=collection,
        total_migrated=migrated,
        backend=sparse.backend,
    )
    return migrated


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate existing Qdrant collections to include native sparse vectors."
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Collection name to migrate (default: SAR_QDRANT_COLLECTION)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["bm25", "splade"],
        help="Sparse backend override (default: SAR_SPARSE_BACKEND)",
    )
    args = parser.parse_args()

    if args.backend:
        settings.sparse_backend = args.backend

    return await migrate_collection(collection_name=args.collection)


if __name__ == "__main__":
    try:
        code = asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("migration_interrupted")
        code = 130
    sys.exit(code)
