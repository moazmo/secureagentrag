"""Hybrid search combining dense retrieval (Qdrant) and sparse retrieval
(Qdrant native sparse vectors) with Reciprocal Rank Fusion.

The sparse path replaces the legacy ``rank_bm25`` pickle-based index.
Sparse vectors are stored in Qdrant alongside dense vectors and searched
with the same RBAC payload filters, eliminating the need for a post-fusion
RBAC re-check.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from utils.logging import get_logger

if TYPE_CHECKING:
    from ingestion.metadata import UserContext

logger = get_logger(__name__)


class SearchResult(BaseModel):
    """Represents a single search result from the hybrid retrieval pipeline.

    Attributes:
        id: Point ID from the vector store.
        text: Chunk text content.
        score: Fused relevance score.
        metadata: Payload metadata from the vector store.
        source: Origin of the result — "dense", "sparse", or "hybrid".
    """

    id: str
    text: str
    score: float = 0.0
    metadata: dict = Field(default_factory=dict)
    source: str = "hybrid"


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF).

    Combines results from different retrieval methods into a single ranked list.
    Formula: RRF_score(d) = sum(1 / (k + rank_i(d))) for each ranking list.

    Args:
        rankings: List of ranked lists, each containing (doc_id, score) tuples.
        k: RRF constant (default 60) to dampen high-rank contributions.

    Returns:
        Fused ranked list of (doc_id, rrf_score) tuples, sorted descending.
    """
    fused_scores: dict[str, float] = {}

    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0.0
            fused_scores[doc_id] += 1.0 / (k + rank)

    # Sort by fused score descending
    fused_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return fused_results


class HybridSearcher:
    """Orchestrates hybrid search combining dense (Qdrant) and sparse (Qdrant native) retrieval.

    Uses Reciprocal Rank Fusion to combine results from both retrieval methods
    with RBAC filtering applied natively by Qdrant on **both** paths.

    Args:
        qdrant_manager: Qdrant vector store manager instance.
        embedding_service: Embedding service for query vectorization.
        sparse_service: Optional sparse embedding service. When ``None`` or
            when sparse generation fails, search degrades to dense-only.
    """

    def __init__(
        self,
        qdrant_manager: QdrantManager,
        embedding_service: EmbeddingService,
        sparse_service: SparseEmbeddingService | None = None,
    ) -> None:
        """Initialize the hybrid searcher with its dependencies.

        Args:
            qdrant_manager: QdrantManager instance for dense retrieval.
            embedding_service: EmbeddingService for query embedding.
            sparse_service: SparseEmbeddingService for query sparse vector.
        """
        self._qdrant = qdrant_manager
        self._embedder = embedding_service
        self._sparse = sparse_service

    async def search(
        self,
        query: str,
        user_context: UserContext,
        top_k: int = 10,
        use_sparse: bool = True,
        extra_filter: Any = None,
    ) -> list[SearchResult]:
        """Perform hybrid search combining dense and sparse retrieval with RBAC.

        Implements graceful degradation: if dense search fails, falls back to
        sparse-only search. If both fail, returns empty results.

        Args:
            query: User's search query.
            user_context: Authenticated user context for RBAC filtering.
            top_k: Maximum number of final results to return.
            use_sparse: Whether to include sparse vector results in fusion.
            extra_filter: Optional additional Qdrant filter.

        Returns:
            List of SearchResult objects ranked by fused relevance score.
        """
        dense_results = []
        dense_ranking: list[tuple[str, float]] = []
        embeddings_failed = False

        # Multi-tenancy: scope to tenant-specific collection when enabled.
        tenant_qdrant = self._qdrant.for_org(user_context.org_id)

        # Step 1: Dense search
        try:
            query_embedding = await self._embedder.embed_text(query)
            dense_results = tenant_qdrant.search_with_rbac(
                query_embedding=query_embedding,
                user_context=user_context,
                top_k=top_k * 2,
                extra_filter=extra_filter,
            )
            dense_ranking = [(str(point.id), point.score) for point in dense_results]
        except Exception as exc:
            embeddings_failed = True
            logger.warning(
                "dense_search_degraded",
                error=str(exc),
                query_len=len(query),
                fallback="sparse_only",
            )

        rankings: list[list[tuple[str, float]]] = []
        if dense_ranking:
            rankings.append(dense_ranking)

        # Step 2: Sparse search via Qdrant native sparse vectors (RBAC-filtered)
        sparse_ranking: list[tuple[str, float]] = []
        if use_sparse and self._sparse is not None:
            try:
                sparse_vector = self._sparse.embed_text(query)
                sparse_results = tenant_qdrant.search_sparse_with_rbac(
                    sparse_vector=sparse_vector,
                    user_context=user_context,
                    top_k=top_k * 2,
                    extra_filter=extra_filter,
                )
                sparse_ranking = [(str(point.id), point.score) for point in sparse_results]
                if sparse_ranking:
                    rankings.append(sparse_ranking)
            except Exception as exc:
                logger.warning("sparse_search_failed", error=str(exc), query_len=len(query))

        if not rankings:
            if embeddings_failed:
                logger.error(
                    "search_fully_degraded",
                    query_len=len(query),
                    reason="embedding_service_and_sparse_unavailable",
                )
            return []

        # Step 3: RRF fusion
        fused = reciprocal_rank_fusion(rankings)

        # Step 4: Build SearchResult objects
        dense_map: dict[str, dict] = {}
        for point in dense_results:
            doc_id = str(point.id)
            payload = point.payload or {}
            dense_map[doc_id] = {
                "text": payload.get("text", ""),
                "metadata": {k: v for k, v in payload.items() if k != "text"},
            }

        # Fetch any sparse-only results from Qdrant (already RBAC-authorized)
        sparse_map: dict[str, dict] = {}
        sparse_only_ids = [doc_id for doc_id, _ in sparse_ranking if doc_id not in dense_map]
        if sparse_only_ids:
            try:
                retrieved = tenant_qdrant.client.retrieve(
                    collection_name=tenant_qdrant.collection_name,
                    ids=sparse_only_ids,
                )
                for point in retrieved:
                    payload = point.payload or {}
                    sparse_map[str(point.id)] = {
                        "text": payload.get("text", ""),
                        "metadata": {k: v for k, v in payload.items() if k != "text"},
                    }
            except Exception as exc:
                logger.warning("sparse_only_retrieve_failed", error=str(exc))

        # Step 5: Assemble final results
        results: list[SearchResult] = []
        for doc_id, score in fused:
            info = dense_map.get(doc_id) or sparse_map.get(doc_id)
            if info is None:
                continue

            source = "hybrid" if len(rankings) > 1 else ("sparse" if embeddings_failed else "dense")
            results.append(
                SearchResult(
                    id=doc_id,
                    text=info["text"],
                    score=score,
                    metadata=info["metadata"],
                    source=source,
                )
            )

            if len(results) >= top_k:
                break

        logger.info(
            "hybrid_search_completed",
            query_len=len(query),
            dense_count=len(dense_results),
            sparse_count=len(sparse_ranking),
            fused_count=len(fused),
            rbac_filtered_count=len(results),
            degraded=embeddings_failed,
            user_id=user_context.user_id,
        )
        return results

    async def search_dense_only(
        self,
        query: str,
        user_context: UserContext,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Perform dense-only search (no sparse) with RBAC filtering.

        Args:
            query: User's search query.
            user_context: Authenticated user context for RBAC filtering.
            top_k: Maximum number of results to return.

        Returns:
            List of SearchResult objects from dense retrieval only.
        """
        try:
            tenant_qdrant = self._qdrant.for_org(user_context.org_id)
            query_embedding = await self._embedder.embed_text(query)

            results = tenant_qdrant.search_with_rbac(
                query_embedding=query_embedding,
                user_context=user_context,
                top_k=top_k,
            )

            search_results: list[SearchResult] = []
            for point in results:
                payload = point.payload or {}
                search_results.append(
                    SearchResult(
                        id=str(point.id),
                        text=payload.get("text", ""),
                        score=point.score,
                        metadata={k: v for k, v in payload.items() if k != "text"},
                        source="dense",
                    )
                )

            return search_results

        except Exception as exc:
            logger.error("dense_only_search_failed", error=str(exc), query_len=len(query))
            return []

    async def search_sparse_only(
        self,
        query: str,
        user_context: UserContext,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Perform sparse-only search (no dense) with RBAC filtering.

        Args:
            query: User's search query.
            user_context: Authenticated user context for RBAC filtering.
            top_k: Maximum number of results to return.

        Returns:
            List of SearchResult objects from sparse retrieval only.
        """
        if self._sparse is None:
            logger.warning("sparse_only_search_no_service", query_len=len(query))
            return []

        try:
            tenant_qdrant = self._qdrant.for_org(user_context.org_id)
            sparse_vector = self._sparse.embed_text(query)

            results = tenant_qdrant.search_sparse_with_rbac(
                sparse_vector=sparse_vector,
                user_context=user_context,
                top_k=top_k,
            )

            search_results: list[SearchResult] = []
            for point in results:
                payload = point.payload or {}
                search_results.append(
                    SearchResult(
                        id=str(point.id),
                        text=payload.get("text", ""),
                        score=point.score,
                        metadata={k: v for k, v in payload.items() if k != "text"},
                        source="sparse",
                    )
                )

            return search_results

        except Exception as exc:
            logger.error("sparse_only_search_failed", error=str(exc), query_len=len(query))
            return []


if TYPE_CHECKING:
    from retrieval.embeddings import EmbeddingService
    from retrieval.qdrant_client import QdrantManager
    from retrieval.sparse_embeddings import SparseEmbeddingService
