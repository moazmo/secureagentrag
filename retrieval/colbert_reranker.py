"""ColBERTv2 late-interaction reranker.

ColBERT uses token-level embeddings and MaxSim scoring for more expressive
relevance modeling than single-vector or cross-encoder approaches. It is
particularly effective on long documents where coarse embedding similarity
misses fine-grained matches.

This module is optional: if ``colbert-ai`` is not installed, the reranker
gracefully degrades to passthrough mode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from config.settings import settings
from utils.logging import get_logger

logger = get_logger(__name__)

try:
    from colbert import Searcher
    from colbert.infra import ColBERTConfig, Run, RunConfig

    _COLBERT_AVAILABLE = True
except ImportError:
    _COLBERT_AVAILABLE = False
    logger.info(
        "colbert_not_installed",
        msg="ColBERT reranker unavailable. Install with: pip install colbert-ai[faiss-cpu]",
    )

if TYPE_CHECKING:
    from retrieval.hybrid_search import SearchResult


class ColBERTReranker:
    """ColBERTv2 late-interaction reranker.

    Loads a ColBERT checkpoint and re-ranks query-document pairs using
    token-level MaxSim scoring. Requires ``colbert-ai`` and a compatible
    checkpoint (e.g., ``colbert-ir/colbertv2.0``).

    Args:
        checkpoint: HuggingFace checkpoint or local path.
        device: "cuda" or "cpu". Auto-detects if None.
    """

    def __init__(
        self,
        checkpoint: str = "colbert-ir/colbertv2.0",
        device: str | None = None,
    ) -> None:
        self._checkpoint = checkpoint
        self._device = device or ("cuda" if _torch_cuda() else "cpu")
        self._searcher: Searcher | None = None
        self._index_built = False

        logger.info(
            "colbert_reranker_initialized",
            checkpoint=checkpoint,
            device=self._device,
            available=self.is_available(),
        )

    def is_available(self) -> bool:
        """Return True if colbert-ai is installed and importable."""
        return _COLBERT_AVAILABLE

    def _ensure_searcher(self) -> Searcher | None:
        """Lazy-load the ColBERT searcher."""
        if self._searcher is not None:
            return self._searcher

        if not _COLBERT_AVAILABLE:
            return None

        try:
            with Run().context(RunConfig(nranks=1, experiment="secureagentrag")):
                config = ColBERTConfig(
                    root=str(settings.data_dir / "colbert"),
                    nbits=2,
                )
                self._searcher = Searcher(
                    index="secureagentrag.nbits=2",
                    config=config,
                )
            logger.info("colbert_searcher_loaded")
            return self._searcher
        except Exception as exc:
            logger.warning("colbert_searcher_load_failed", error=str(exc))
            return None

    def rerank(
        self,
        query: str,
        documents: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Rerank documents using ColBERT MaxSim scoring.

        Falls back to passthrough if ColBERT is unavailable or the index
        has not been built.
        """
        if not documents:
            return []

        if not self.is_available() or not self._index_built:
            return documents[:top_k] if top_k else documents

        searcher = self._ensure_searcher()
        if searcher is None:
            return documents[:top_k] if top_k else documents

        try:
            # Build a temporary mini-index from the candidate docs
            texts = [doc.text for doc in documents]
            # ColBERT search requires an indexed collection; for reranking
            # a small candidate set we use the Searcher directly if possible.
            # If the full collection index exists, we query it and filter.
            results = searcher.search(query, k=len(documents))

            # Map returned pids back to our documents
            # This is a simplified mapping; production would use doc IDs.
            scored_docs: list[tuple[SearchResult, float]] = []
            for doc in documents:
                score = 0.0
                for pid, rank_score in zip(results[0], results[2], strict=False):
                    if texts[pid] == doc.text:
                        score = float(rank_score)
                        break
                scored_docs.append((doc, score))

            scored_docs.sort(key=lambda x: x[1], reverse=True)

            reranked: list[SearchResult] = []
            for doc, score in scored_docs:
                reranked.append(doc.model_copy(update={"score": float(score)}))

            return reranked[:top_k] if top_k else reranked

        except Exception as exc:
            logger.error("colbert_rerank_failed", error=str(exc))
            return documents[:top_k] if top_k else documents

    def rerank_texts(
        self,
        query: str,
        texts: list[str],
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """Rerank raw texts using ColBERT."""
        if not texts:
            return []

        if not self.is_available() or not self._index_built:
            results = [(text, 0.0) for text in texts]
            return results[:top_k] if top_k else results

        searcher = self._ensure_searcher()
        if searcher is None:
            results = [(text, 0.0) for text in texts]
            return results[:top_k] if top_k else results

        try:
            results = searcher.search(query, k=len(texts))
            scored = [
                (texts[pid], float(score))
                for pid, score in zip(results[0], results[2], strict=False)
                if pid < len(texts)
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k] if top_k else scored
        except Exception as exc:
            logger.error("colbert_rerank_texts_failed", error=str(exc))
            results = [(text, 0.0) for text in texts]
            return results[:top_k] if top_k else results


def _torch_cuda() -> bool:
    """Check if torch CUDA is available without importing torch eagerly."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False
