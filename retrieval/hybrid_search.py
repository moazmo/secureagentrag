"""Hybrid search combining dense retrieval (Qdrant) and sparse retrieval (BM25) with Reciprocal Rank Fusion."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from config.settings import settings
from utils.file_lock import FileLock
from utils.logging import get_logger

if TYPE_CHECKING:
    from ingestion.metadata import UserContext

logger = get_logger(__name__)

try:
    from rank_bm25 import BM25Okapi

    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False
    logger.warning("rank_bm25_not_installed", detail="BM25 search will be unavailable")


class SearchResult(BaseModel):
    """Represents a single search result from the hybrid retrieval pipeline.

    Attributes:
        id: Point ID from the vector store.
        text: Chunk text content.
        score: Fused relevance score.
        metadata: Payload metadata from the vector store.
        source: Origin of the result — "dense", "bm25", or "hybrid".
    """

    id: str
    text: str
    score: float = 0.0
    metadata: dict = Field(default_factory=dict)
    source: str = "hybrid"


class BM25Index:
    """BM25 sparse retrieval index with optional disk persistence.

    Uses simple whitespace tokenization for building the BM25Okapi index.
    Can save/load index from disk to survive restarts.
    """

    def __init__(self, index_path: str | None = None) -> None:
        """Initialize BM25 index, optionally loading from disk.

        Args:
            index_path: Path to save/load the index. Defaults to
                settings.bm25_index_path or "data/bm25_index.pkl".
        """
        self._corpus: list[list[str]] = []
        self._doc_ids: list[str] = []
        self._bm25: BM25Okapi | None = None
        self._index_path = Path(
            index_path
            if index_path is not None
            else getattr(settings, "bm25_index_path", "data/bm25_index.pkl")
        )
        self._lock = FileLock(
            lock_path=self._index_path.with_suffix(".lock"),
            timeout=10.0,
        )

        # Try loading existing index from disk
        if self._index_path.exists():
            self.load_index()

    def build_index(self, documents: list[str], doc_ids: list[str]) -> None:
        """Tokenize documents and build the BM25Okapi index (FULL REBUILD).

        This REPLACES any existing index. Use ``add_documents`` instead when
        ingesting new docs incrementally so previously indexed chunks are
        preserved.

        Args:
            documents: List of document texts to index.
            doc_ids: Corresponding document IDs (must match length of documents).

        Raises:
            ValueError: If documents and doc_ids lengths don't match.
            RuntimeError: If rank_bm25 is not installed.
        """
        if not _BM25_AVAILABLE:
            raise RuntimeError("rank_bm25 is not installed. Install with: pip install rank-bm25")

        if len(documents) != len(doc_ids):
            raise ValueError(f"Length mismatch: documents={len(documents)}, doc_ids={len(doc_ids)}")

        self._doc_ids = list(doc_ids)
        self._corpus = [doc.lower().split() for doc in documents]
        self._bm25 = BM25Okapi(self._corpus)

        logger.info("bm25_index_built", num_documents=len(documents))
        self.save_index()

    def add_documents(self, documents: list[str], doc_ids: list[str]) -> None:
        """Append documents to the existing BM25 index, rebuild BM25Okapi.

        ``rank-bm25`` has no incremental update path — it precomputes per-corpus
        IDF statistics — so we extend the in-memory corpus and reconstruct the
        Okapi object. For typical document counts (10s-100s of thousands of
        chunks) this stays fast enough to run on every ingestion.

        Args:
            documents: New document texts to add.
            doc_ids: Corresponding point IDs (already unique).

        Raises:
            ValueError: If lengths don't match.
            RuntimeError: If rank_bm25 is not installed.
        """
        if not _BM25_AVAILABLE:
            raise RuntimeError("rank_bm25 is not installed. Install with: pip install rank-bm25")
        if len(documents) != len(doc_ids):
            raise ValueError(f"Length mismatch: documents={len(documents)}, doc_ids={len(doc_ids)}")
        if not documents:
            return

        existing = set(self._doc_ids)
        new_pairs = [(t, i) for t, i in zip(documents, doc_ids, strict=False) if i not in existing]
        if not new_pairs:
            logger.debug("bm25_add_noop", reason="all_doc_ids_already_indexed")
            return

        new_texts, new_ids = zip(*new_pairs, strict=False)
        self._doc_ids.extend(new_ids)
        self._corpus.extend(t.lower().split() for t in new_texts)
        self._bm25 = BM25Okapi(self._corpus)

        logger.info(
            "bm25_index_extended",
            added=len(new_pairs),
            total=len(self._doc_ids),
        )
        self.save_index()

    def save_index(self) -> bool:
        """Persist the BM25 index to disk with cross-process locking.

        Returns:
            True if saved successfully, False otherwise.
        """
        if not self.is_built():
            logger.debug("bm25_save_skipped", reason="index_not_built")
            return False

        try:
            with self._lock.acquire():
                self._index_path.parent.mkdir(parents=True, exist_ok=True)
                data = {
                    "doc_ids": self._doc_ids,
                    "corpus": self._corpus,
                }
                # Atomic write: temp file then rename
                temp_path = self._index_path.with_suffix(".tmp")
                with open(temp_path, "wb") as f:
                    pickle.dump(data, f)
                temp_path.replace(self._index_path)
            logger.info("bm25_index_saved", path=str(self._index_path))
            return True
        except Exception as exc:
            logger.error("bm25_save_failed", error=str(exc))
            return False

    def load_index(self) -> bool:
        """Load the BM25 index from disk with cross-process locking.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if not self._index_path.exists():
            return False

        try:
            with self._lock.acquire(), open(self._index_path, "rb") as f:
                data = pickle.load(f)

            self._doc_ids = data.get("doc_ids", [])
            self._corpus = data.get("corpus", [])

            if self._corpus and _BM25_AVAILABLE:
                self._bm25 = BM25Okapi(self._corpus)
                logger.info(
                    "bm25_index_loaded",
                    path=str(self._index_path),
                    num_documents=len(self._doc_ids),
                )
                return True
            return False
        except Exception as exc:
            logger.error("bm25_load_failed", error=str(exc))
            return False

    def search(
        self,
        query: str,
        top_k: int = 20,
        rbac_filter: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        """Search the BM25 index for relevant documents.

        Optionally applies RBAC filtering by doc_id if a filter dict is provided.
        The filter should contain 'allowed_doc_ids' — a set of doc_ids the user
        is permitted to see (from Qdrant RBAC-filtered results).

        Args:
            query: Search query string.
            top_k: Maximum number of results to return.
            rbac_filter: Optional dict with 'allowed_doc_ids' for RBAC filtering.

        Returns:
            List of (doc_id, score) tuples sorted by BM25 score descending.
        """
        if not self.is_built():
            logger.warning("bm25_search_on_empty_index")
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)  # type: ignore[union-attr]

        # Pair scores with doc_ids and sort descending
        scored_docs = [
            (self._doc_ids[i], float(scores[i])) for i in range(len(scores)) if scores[i] > 0
        ]
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Apply RBAC filtering if provided
        if rbac_filter and "allowed_doc_ids" in rbac_filter:
            allowed = set(rbac_filter["allowed_doc_ids"])
            scored_docs = [(doc_id, score) for doc_id, score in scored_docs if doc_id in allowed]
            logger.debug("bm25_rbac_filtered", before=len(scores), after=len(scored_docs))

        return scored_docs[:top_k]

    def is_built(self) -> bool:
        """Check if the BM25 index has been built.

        Returns:
            True if index contains documents, False otherwise.
        """
        return self._bm25 is not None and len(self._corpus) > 0


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
    """Orchestrates hybrid search combining dense (Qdrant) and sparse (BM25) retrieval.

    Uses Reciprocal Rank Fusion to combine results from both retrieval methods
    with RBAC filtering applied on the dense search.

    Args:
        qdrant_manager: Qdrant vector store manager instance.
        embedding_service: Embedding service for query vectorization.
        bm25_index: Optional pre-built BM25 index for sparse retrieval.
    """

    def __init__(
        self,
        qdrant_manager: QdrantManager,
        embedding_service: EmbeddingService,
        bm25_index: BM25Index | None = None,
    ) -> None:
        """Initialize the hybrid searcher with required dependencies.

        Args:
            qdrant_manager: QdrantManager instance for dense retrieval.
            embedding_service: EmbeddingService for query embedding.
            bm25_index: Optional BM25Index for sparse retrieval.
        """
        self._qdrant = qdrant_manager
        self._embedder = embedding_service
        self._bm25 = bm25_index

    async def search(
        self,
        query: str,
        user_context: UserContext,
        top_k: int = 10,
        use_bm25: bool = True,
        extra_filter: Any = None,
    ) -> list[SearchResult]:
        """Perform hybrid search combining dense and sparse retrieval with RBAC.

        Implements graceful degradation: if dense search (Qdrant/embedding)
        fails, falls back to BM25-only search. If both fail, returns empty
        results with a logged warning.

        Args:
            query: User's search query.
            user_context: Authenticated user context for RBAC filtering.
            top_k: Maximum number of final results to return.
            use_bm25: Whether to include BM25 results in fusion.

        Returns:
            List of SearchResult objects ranked by fused relevance score.
        """
        dense_results = []
        dense_ranking: list[tuple[str, float]] = []
        embeddings_failed = False

        # Multi-tenancy: when SAR_MULTI_TENANT_COLLECTIONS=true, scope this
        # search to the tenant-specific collection. Falls back to the default
        # collection when the flag is off so existing single-tenant setups
        # are unaffected.
        tenant_qdrant = self._qdrant.for_org(user_context.org_id)

        # Step 1: Try embedding + dense search
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
                fallback="bm25_only",
            )

        rankings: list[list[tuple[str, float]]] = []
        if dense_ranking:
            rankings.append(dense_ranking)

        # Step 2: BM25 search if available (search FULL index, not filtered)
        # The key insight of hybrid search: BM25 can surface documents that
        # dense retrieval missed (e.g., exact keyword matches). We apply
        # RBAC filtering AFTER fusion, not before, to preserve this benefit.
        bm25_ranking: list[tuple[str, float]] = []
        if use_bm25 and self._bm25 is not None and self._bm25.is_built():
            try:
                bm25_ranking = self._bm25.search(query, top_k=top_k * 2, rbac_filter=None)
                rankings.append(bm25_ranking)
            except Exception as exc:
                logger.warning("bm25_search_failed", error=str(exc), query_len=len(query))

        # If dense failed and BM25 is empty, we have nothing
        if not rankings:
            if embeddings_failed:
                logger.error(
                    "search_fully_degraded",
                    query_len=len(query),
                    reason="embedding_service_and_bm25_unavailable",
                )
            return []

        # Step 3: RRF fusion
        fused = reciprocal_rank_fusion(rankings)

        # Step 4: Build SearchResult objects
        dense_map: dict[str, dict] = {}
        dense_doc_ids: set[str] = set()
        for point in dense_results:
            doc_id = str(point.id)
            dense_doc_ids.add(doc_id)
            payload = point.payload or {}
            dense_map[doc_id] = {
                "text": payload.get("text", ""),
                "metadata": {k: v for k, v in payload.items() if k != "text"},
            }

        # Step 5: RBAC filtering AFTER fusion
        # Dense results already have RBAC applied by Qdrant.
        # BM25 results may include unauthorized docs — filter them out.
        # We keep docs that either came from dense (already RBAC-filtered)
        # or we can verify exist in Qdrant with proper metadata.
        allowed_doc_ids: set[str] | None = None
        if settings.enable_rbac and dense_doc_ids:
            # Only allow doc_ids that passed Qdrant's RBAC filter
            # OR fetch metadata for BM25-only results and check RBAC
            allowed_doc_ids = dense_doc_ids.copy()

            # For BM25-only results, fetch from Qdrant and apply RBAC
            bm25_only_ids = [doc_id for doc_id, _ in bm25_ranking if doc_id not in dense_doc_ids]
            if bm25_only_ids:
                try:
                    rbac_filter = tenant_qdrant.build_rbac_filter(user_context)
                    # Scroll Qdrant to find which BM25 doc_ids are authorized
                    filter_results = tenant_qdrant.client.scroll(
                        collection_name=tenant_qdrant.collection_name,
                        scroll_filter=rbac_filter,
                        limit=len(bm25_only_ids) * 2,
                        with_payload=False,
                    )[0]
                    rbac_allowed_ids = {str(p.id) for p in filter_results}
                    allowed_doc_ids.update(rbac_allowed_ids)
                except Exception:
                    # If RBAC check fails, fall back to dense-only doc_ids
                    pass

        # Step 6: Build final results, filtering unauthorized BM25 docs
        results: list[SearchResult] = []
        for doc_id, score in fused:
            # RBAC check: skip BM25-only docs that aren't authorized
            if allowed_doc_ids is not None and doc_id not in allowed_doc_ids:
                continue

            info = dense_map.get(doc_id)
            if info is None:
                # BM25-only result — fetch text from Qdrant
                try:
                    point_results = tenant_qdrant.client.retrieve(
                        collection_name=tenant_qdrant.collection_name,
                        ids=[doc_id],
                    )
                    if point_results:
                        p = point_results[0]
                        payload = p.payload or {}
                        info = {
                            "text": payload.get("text", ""),
                            "metadata": {k: v for k, v in payload.items() if k != "text"},
                        }
                        dense_map[doc_id] = info
                    else:
                        continue  # Doc not found in Qdrant
                except Exception:
                    continue  # Skip on error

            source = "hybrid" if len(rankings) > 1 else ("bm25" if embeddings_failed else "dense")
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
            bm25_count=len(bm25_ranking),
            fused_count=len(fused),
            rbac_filtered_count=len(results),
            degraded=embeddings_failed,
            user_id=user_context.user_id,
        )
        return results

    async def dense_only_search(
        self,
        query: str,
        user_context: UserContext,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Perform dense-only search (no BM25) with RBAC filtering.

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


if TYPE_CHECKING:
    from retrieval.embeddings import EmbeddingService
    from retrieval.qdrant_client import QdrantManager
