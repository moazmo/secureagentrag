"""Reranker using cross-encoder models for improved retrieval precision."""

from __future__ import annotations

from typing import TYPE_CHECKING

from utils.logging import get_logger

logger = get_logger(__name__)

try:
    from sentence_transformers import CrossEncoder

    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.info(
        "sentence_transformers_not_installed",
        detail="Reranker will operate in passthrough mode",
    )

if TYPE_CHECKING:
    from retrieval.hybrid_search import SearchResult


class Reranker:
    """Cross-encoder reranker for improving retrieval precision.

    Lazily loads a cross-encoder model and uses it to re-score query-document
    pairs for more accurate relevance ranking. Falls back to passthrough mode
    if sentence-transformers is not installed.

    Args:
        model_name: HuggingFace model identifier for the cross-encoder.
        device: Target device ("cuda", "cpu", or None for auto-detection).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
    ) -> None:
        """Initialize the reranker with lazy model loading.

        Args:
            model_name: Cross-encoder model name from HuggingFace Hub.
            device: Computation device. Auto-detects CUDA if available when None.
        """
        self._model_name = model_name
        self._device = device
        self._model: CrossEncoder | None = None

        logger.info(
            "reranker_initialized",
            model_name=model_name,
            device=device or "auto",
            available=self.is_available(),
        )

    def _load_model(self) -> None:
        """Load the cross-encoder model on first use.

        Detects CUDA availability automatically if device is not specified.
        """
        if not _SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning(
                "cannot_load_reranker_model", reason="sentence-transformers not installed"
            )
            return

        try:
            import torch

            device = self._device
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            self._model = CrossEncoder(self._model_name, device=device)
            logger.info(
                "reranker_model_loaded",
                model_name=self._model_name,
                device=device,
            )
        except Exception as exc:
            logger.error(
                "reranker_model_load_failed",
                model_name=self._model_name,
                error=str(exc),
            )
            self._model = None

    def is_available(self) -> bool:
        """Check if the sentence-transformers library is installed.

        Returns:
            True if reranking is possible, False otherwise.
        """
        return _SENTENCE_TRANSFORMERS_AVAILABLE

    def rerank(
        self,
        query: str,
        documents: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Rerank search results using the cross-encoder model.

        If the model is not available, returns documents unchanged (passthrough).

        Args:
            query: The user query.
            documents: List of SearchResult objects to rerank.
            top_k: Maximum number of results to return. Returns all if None.

        Returns:
            Reranked list of SearchResult objects with updated scores.
        """
        if not documents:
            return []

        if not self.is_available():
            logger.info("reranker_passthrough", reason="model not available")
            return documents[:top_k] if top_k else documents

        if self._model is None:
            self._load_model()

        if self._model is None:
            # Model failed to load — passthrough
            logger.warning("reranker_passthrough_after_load_failure")
            return documents[:top_k] if top_k else documents

        try:
            # Create (query, document_text) pairs
            pairs = [(query, doc.text) for doc in documents]

            # Score with cross-encoder
            scores = self._model.predict(pairs)

            # Pair documents with their reranker scores
            scored_docs = list(zip(documents, scores, strict=False))
            scored_docs.sort(key=lambda x: float(x[1]), reverse=True)

            # Update scores and return
            results: list[SearchResult] = []
            for doc, score in scored_docs:
                reranked = doc.model_copy(update={"score": float(score)})
                results.append(reranked)

            if top_k:
                results = results[:top_k]

            logger.info(
                "rerank_completed",
                input_count=len(documents),
                output_count=len(results),
            )
            return results

        except Exception as exc:
            logger.error("rerank_failed", error=str(exc))
            return documents[:top_k] if top_k else documents

    def rerank_texts(
        self,
        query: str,
        texts: list[str],
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """Rerank raw texts using the cross-encoder model.

        A simpler interface that accepts raw text strings instead of SearchResult objects.

        Args:
            query: The user query.
            texts: List of text strings to rerank.
            top_k: Maximum number of results to return. Returns all if None.

        Returns:
            List of (text, score) tuples sorted by reranker score descending.
        """
        if not texts:
            return []

        if not self.is_available():
            # Return with zero scores in original order
            results = [(text, 0.0) for text in texts]
            return results[:top_k] if top_k else results

        if self._model is None:
            self._load_model()

        if self._model is None:
            results = [(text, 0.0) for text in texts]
            return results[:top_k] if top_k else results

        try:
            pairs = [(query, text) for text in texts]
            scores = self._model.predict(pairs)

            scored_texts = [(text, float(score)) for text, score in zip(texts, scores, strict=False)]
            scored_texts.sort(key=lambda x: x[1], reverse=True)

            return scored_texts[:top_k] if top_k else scored_texts

        except Exception as exc:
            logger.error("rerank_texts_failed", error=str(exc))
            results = [(text, 0.0) for text in texts]
            return results[:top_k] if top_k else results
