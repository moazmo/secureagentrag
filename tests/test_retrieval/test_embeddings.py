"""Tests for the embedding service caching functionality."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture()
def embedding_service():
    """Create an EmbeddingService with mocked settings."""
    with patch("retrieval.embeddings.settings") as mock_settings:
        mock_settings.embedding_model = "bge-m3"
        mock_settings.ollama_url = "http://localhost:11434"
        mock_settings.embedding_dim = 1024
        from retrieval.embeddings import EmbeddingService

        service = EmbeddingService(
            model="bge-m3",
            ollama_url="http://localhost:11434",
            max_cache_size=5,
        )
        return service


class TestCacheKey:
    """Tests for cache key generation."""

    def test_cache_key_deterministic(self, embedding_service):
        """Same text produces same cache key."""
        key1 = embedding_service._cache_key("hello world")
        key2 = embedding_service._cache_key("hello world")

        assert key1 == key2

    def test_cache_key_different_texts(self, embedding_service):
        """Different texts produce different cache keys."""
        key1 = embedding_service._cache_key("hello")
        key2 = embedding_service._cache_key("world")

        assert key1 != key2

    def test_cache_key_is_md5_hex(self, embedding_service):
        """Cache key is a valid MD5 hex digest (32 chars)."""
        key = embedding_service._cache_key("test")

        assert len(key) == 32
        assert all(c in "0123456789abcdef" for c in key)

    def test_cache_key_unicode(self, embedding_service):
        """Cache key handles unicode text."""
        key = embedding_service._cache_key("مرحبا بالعالم")

        assert len(key) == 32


class TestCacheHitMiss:
    """Tests for cache hit/miss behavior."""

    @pytest.mark.asyncio
    async def test_cache_miss_calls_api(self, embedding_service):
        """Cache miss triggers API call."""
        mock_embedding = [0.1] * 1024

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_response = MagicMock()
            mock_response.json.return_value = {"embeddings": [mock_embedding]}
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await embedding_service.embed_text("test text")

            assert result == mock_embedding
            mock_client.post.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cache_hit_skips_api(self, embedding_service):
        """Cache hit returns cached result without API call."""
        mock_embedding = [0.2] * 1024

        # Pre-populate cache
        key = embedding_service._cache_key("cached text")
        embedding_service._cache[key] = mock_embedding

        result = await embedding_service.embed_text("cached text")

        assert result == mock_embedding

    @pytest.mark.asyncio
    async def test_cache_hit_increments_counter(self, embedding_service):
        """Cache hit increments the hit counter."""
        key = embedding_service._cache_key("cached")
        embedding_service._cache[key] = [0.1] * 10

        await embedding_service.embed_text("cached")

        assert embedding_service._cache_hits == 1
        assert embedding_service._cache_misses == 0

    @pytest.mark.asyncio
    async def test_cache_miss_increments_counter(self, embedding_service):
        """Cache miss increments the miss counter."""
        mock_embedding = [0.1] * 1024

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_response = MagicMock()
            mock_response.json.return_value = {"embeddings": [mock_embedding]}
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            await embedding_service.embed_text("new text")

        assert embedding_service._cache_misses == 1
        assert embedding_service._cache_hits == 0


class TestCacheStats:
    """Tests for cache_stats()."""

    def test_initial_stats(self, embedding_service):
        """Initial stats show zeros."""
        stats = embedding_service.cache_stats()

        assert stats == {
            "hits": 0,
            "misses": 0,
            "size": 0,
            "max_size": 5,
        }

    def test_stats_after_population(self, embedding_service):
        """Stats reflect cache state."""
        embedding_service._cache["key1"] = [0.1]
        embedding_service._cache["key2"] = [0.2]
        embedding_service._cache_hits = 3
        embedding_service._cache_misses = 7

        stats = embedding_service.cache_stats()

        assert stats["hits"] == 3
        assert stats["misses"] == 7
        assert stats["size"] == 2
        assert stats["max_size"] == 5


class TestClearCache:
    """Tests for clear_cache()."""

    def test_clear_removes_entries(self, embedding_service):
        """clear_cache removes all cached embeddings."""
        embedding_service._cache["k1"] = [0.1]
        embedding_service._cache["k2"] = [0.2]

        embedding_service.clear_cache()

        assert len(embedding_service._cache) == 0

    def test_clear_resets_counters(self, embedding_service):
        """clear_cache resets hit/miss counters."""
        embedding_service._cache_hits = 10
        embedding_service._cache_misses = 20

        embedding_service.clear_cache()

        assert embedding_service._cache_hits == 0
        assert embedding_service._cache_misses == 0


class TestCacheEviction:
    """Tests for cache size limit and eviction."""

    def test_eviction_at_max_size(self, embedding_service):
        """Cache evicts oldest entry when at max capacity."""
        # Fill cache to max (5)
        for i in range(5):
            embedding_service._cache[f"key{i}"] = [float(i)]

        assert len(embedding_service._cache) == 5

        # Store one more — should evict oldest (key0)
        embedding_service._store_in_cache("key_new", [9.9])

        assert len(embedding_service._cache) == 5
        assert "key0" not in embedding_service._cache
        assert "key_new" in embedding_service._cache


class TestEmbedBatch:
    """Tests for embed_batch with mocked responses."""

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self, embedding_service):
        """embed_batch with empty list returns empty."""
        result = await embedding_service.embed_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_batch_returns_embeddings(self, embedding_service):
        """embed_batch returns all embeddings."""
        mock_embeddings = [[0.1] * 1024, [0.2] * 1024]

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_response = MagicMock()
            mock_response.json.return_value = {"embeddings": mock_embeddings}
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await embedding_service.embed_batch(["text1", "text2"])

        assert len(result) == 2
        assert result[0] == mock_embeddings[0]
        assert result[1] == mock_embeddings[1]
