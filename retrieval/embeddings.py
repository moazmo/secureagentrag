"""Embedding service with Ollama primary and sentence-transformers fallback."""

from __future__ import annotations

import asyncio
import hashlib
import threading

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from utils.logging import get_logger

logger = get_logger(__name__)

# Lazy singleton for local embedding model
_local_embedder = None
_local_embedder_lock = threading.Lock()


def _get_local_embedder():
    """Lazily initialize and return a sentence-transformers embedder.

    Thread-safe singleton pattern. Falls back to None if the library
    is not installed.
    """
    global _local_embedder
    if _local_embedder is None:
        with _local_embedder_lock:
            if _local_embedder is None:
                try:
                    from sentence_transformers import SentenceTransformer

                    _local_embedder = SentenceTransformer(settings.local_embedding_model)
                    logger.info(
                        "local_embedder_loaded",
                        model=settings.local_embedding_model,
                    )
                except ImportError:
                    logger.error(
                        "sentence_transformers_not_installed",
                        hint="pip install sentence-transformers",
                    )
                    raise RuntimeError(
                        "sentence-transformers is not installed. "
                        "Install it with: pip install sentence-transformers"
                    ) from None
    return _local_embedder


class EmbeddingService:
    """Generates text embeddings using Ollama or local sentence-transformers.

    Tries Ollama first (better quality, GPU-accelerated). If Ollama is
    unreachable and settings.embedding_backend is "local" or auto-fallback
    is enabled, falls back to sentence-transformers.

    Provides both single-text and batch embedding capabilities with
    automatic retry logic for transient failures.

    Args:
        model: Embedding model name. Defaults to settings.embedding_model.
        ollama_url: Ollama API base URL. Defaults to settings.ollama_url.
    """

    def __init__(
        self,
        model: str | None = None,
        ollama_url: str | None = None,
        max_cache_size: int = 1000,
    ) -> None:
        """Initialize the embedding service.

        Args:
            model: Model identifier for embeddings. Uses settings default if None.
            ollama_url: Base URL for Ollama API. Uses settings default if None.
            max_cache_size: Maximum number of embeddings to cache in memory.
        """
        self._model = model if model is not None else settings.embedding_model
        self._ollama_url = ollama_url if ollama_url is not None else settings.ollama_url
        self._embedding_dim = settings.embedding_dim
        self._cache: dict[str, list[float]] = {}
        self._max_cache_size = max_cache_size
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._use_local = settings.embedding_backend == "local"
        self._ollama_available: bool | None = None

        logger.info(
            "embedding_service_initialized",
            model=self._model,
            ollama_url=self._ollama_url,
            embedding_dim=self._embedding_dim,
            max_cache_size=self._max_cache_size,
            backend=settings.embedding_backend,
        )

    def get_embedding_dim(self) -> int:
        """Return the configured embedding dimension.

        Returns:
            Integer dimension of embedding vectors.
        """
        return self._embedding_dim

    @staticmethod
    def _cache_key(text: str) -> str:
        """Generate a cache key for the given text using MD5 hash.

        Args:
            text: Input text to generate key for.

        Returns:
            Hex digest string suitable as a dictionary key.
        """
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def clear_cache(self) -> None:
        """Clear the embedding cache and reset statistics."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("embedding_cache_cleared")

    def cache_stats(self) -> dict:
        """Return cache statistics.

        Returns:
            Dictionary with hits, misses, and current size.
        """
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._cache),
            "max_size": self._max_cache_size,
        }

    def _store_in_cache(self, key: str, embedding: list[float]) -> None:
        """Store an embedding in the cache, evicting oldest if at capacity.

        Args:
            key: Cache key (MD5 hash of input text).
            embedding: Embedding vector to store.
        """
        if len(self._cache) >= self._max_cache_size:
            # Evict the oldest entry (first inserted)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = embedding

    async def embed_text(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text with caching.

        Checks the in-memory cache first. On miss, calls Ollama API.
        If Ollama is unreachable, falls back to sentence-transformers.

        Args:
            text: Input text to embed.

        Returns:
            List of floats representing the embedding vector.

        Raises:
            httpx.HTTPStatusError: If the Ollama API returns an error status.
            httpx.ConnectError: If Ollama is unreachable and no fallback is available.
        """
        key = self._cache_key(text)

        # Check cache
        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]

        self._cache_misses += 1

        # If explicitly configured for local, use it directly
        if self._use_local:
            return await self._embed_local(text, key)

        # Try Ollama first
        try:
            embedding = await self._embed_ollama(text)
            self._store_in_cache(key, embedding)
            self._ollama_available = True
            return embedding
        except httpx.ConnectError:
            logger.warning("ollama_unavailable_falling_back_to_local")
            self._ollama_available = False
            return await self._embed_local(text, key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _embed_ollama(self, text: str) -> list[float]:
        """Call Ollama embedding API.

        Args:
            text: Input text to embed.

        Returns:
            Embedding vector from Ollama.
        """
        url = f"{self._ollama_url}/api/embed"
        payload = {
            "model": self._model,
            "input": text,
            "keep_alive": settings.ollama_keep_alive,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        embeddings = data.get("embeddings", [])
        if embeddings and len(embeddings) > 0:
            return embeddings[0]

        embedding = data.get("embedding", [])
        if embedding:
            return embedding

        logger.error("embedding_empty_response", model=self._model, text_len=len(text))
        raise ValueError("Ollama returned empty embedding response")

    async def _embed_local(self, text: str, key: str | None = None) -> list[float]:
        """Generate embedding using local sentence-transformers model.

        Args:
            text: Input text to embed.
            key: Optional cache key to store result.

        Returns:
            Embedding vector from local model.
        """
        embedder = _get_local_embedder()
        # sentence-transformers is synchronous; offload to default executor.
        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(None, embedder.encode, text)
        result = embedding.tolist()
        if key:
            self._store_in_cache(key, result)
        return result

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts in batches.

        Processes texts in groups to avoid memory issues and API timeouts.
        Respects ``settings.embedding_batch_size`` and
        ``settings.embedding_max_concurrent_batches`` for safe defaults.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts per batch. Uses settings default if None.

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            httpx.HTTPStatusError: If the Ollama API returns an error status.
            ValueError: If any batch returns invalid results.
        """
        if not texts:
            return []

        batch_size = batch_size or settings.embedding_batch_size
        max_concurrent = settings.embedding_max_concurrent_batches
        total = len(texts)

        if total > batch_size * max_concurrent * 10:
            logger.warning(
                "embedding_large_batch",
                total=total,
                batch_size=batch_size,
                max_concurrent=max_concurrent,
                estimated_batches=(total + batch_size - 1) // batch_size,
            )

        all_embeddings: list[list[float]] = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _embed_with_limit(batch: list[str], start_idx: int) -> list[list[float]]:
            async with semaphore:
                logger.info(
                    "embedding_batch_processing",
                    batch_start=start_idx,
                    batch_size=len(batch),
                    total=total,
                )
                return await self._embed_batch_request(batch)

        # Process batches with concurrency limit
        tasks = []
        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            tasks.append(_embed_with_limit(batch, i))

        results = await asyncio.gather(*tasks)
        for batch_embeddings in results:
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def _embed_batch_request(self, texts: list[str]) -> list[list[float]]:
        """Send a batch embedding request.

        Uses Ollama if available, otherwise falls back to local model.

        Args:
            texts: Batch of texts to embed.

        Returns:
            List of embedding vectors for the batch.
        """
        if self._use_local or self._ollama_available is False:
            return await self._embed_batch_local(texts)

        try:
            return await self._embed_batch_ollama(texts)
        except httpx.ConnectError:
            logger.warning("ollama_batch_unavailable_falling_back_to_local")
            self._ollama_available = False
            return await self._embed_batch_local(texts)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _embed_batch_ollama(self, texts: list[str]) -> list[list[float]]:
        """Send a batch embedding request to Ollama.

        Args:
            texts: Batch of texts to embed.

        Returns:
            List of embedding vectors for the batch.
        """
        url = f"{self._ollama_url}/api/embed"
        payload = {
            "model": self._model,
            "input": texts,
            "keep_alive": settings.ollama_keep_alive,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        embeddings = data.get("embeddings", [])
        if embeddings and len(embeddings) == len(texts):
            return embeddings

        # Fallback: embed one by one if batch format not supported
        logger.warning(
            "batch_embedding_fallback",
            expected=len(texts),
            received=len(embeddings) if embeddings else 0,
        )
        results: list[list[float]] = []
        for text in texts:
            single_payload = {
                "model": self._model,
                "input": text,
                "keep_alive": settings.ollama_keep_alive,
            }
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, json=single_payload)
                resp.raise_for_status()
                single_data = resp.json()

            emb = single_data.get("embeddings", [[]])[0]
            if not emb:
                emb = single_data.get("embedding", [])
            results.append(emb)

        return results

    async def _embed_batch_local(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch using local sentence-transformers.

        Args:
            texts: Batch of texts to embed.

        Returns:
            List of embedding vectors for the batch.
        """
        embedder = _get_local_embedder()
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(None, embedder.encode, texts)
        return [emb.tolist() for emb in embeddings]
