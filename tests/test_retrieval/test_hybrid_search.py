"""Tests for hybrid search module — BM25, RRF, and HybridSearcher."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ingestion.metadata import UserContext
from retrieval.hybrid_search import (
    BM25Index,
    HybridSearcher,
    SearchResult,
    reciprocal_rank_fusion,
)


class TestSearchResult:
    """Tests for the SearchResult model."""

    def test_create_search_result(self):
        """SearchResult can be created with all fields."""
        result = SearchResult(
            id="point-1",
            text="Hello world",
            score=0.95,
            metadata={"org_id": "org-1"},
            source="dense",
        )
        assert result.id == "point-1"
        assert result.text == "Hello world"
        assert result.score == 0.95
        assert result.metadata == {"org_id": "org-1"}
        assert result.source == "dense"

    def test_default_values(self):
        """SearchResult uses correct defaults."""
        result = SearchResult(id="p1", text="test")
        assert result.score == 0.0
        assert result.metadata == {}
        assert result.source == "hybrid"


class TestReciprocalRankFusion:
    """Tests for the reciprocal_rank_fusion function."""

    def test_single_ranking_preserves_order(self):
        """Single ranking list should preserve original order."""
        ranking = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        fused = reciprocal_rank_fusion([ranking], k=60)

        # Order should be preserved
        assert fused[0][0] == "doc1"
        assert fused[1][0] == "doc2"
        assert fused[2][0] == "doc3"

    def test_single_ranking_scores(self):
        """Scores in single ranking follow RRF formula."""
        ranking = [("doc1", 0.9), ("doc2", 0.8)]
        fused = reciprocal_rank_fusion([ranking], k=60)

        # RRF score for rank 1: 1/(60+1) = ~0.01639
        assert abs(fused[0][1] - 1.0 / 61) < 1e-6
        # RRF score for rank 2: 1/(60+2) = ~0.01613
        assert abs(fused[1][1] - 1.0 / 62) < 1e-6

    def test_two_rankings_interleave(self):
        """Two rankings should fuse documents appearing in both higher."""
        ranking1 = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        ranking2 = [("doc2", 0.95), ("doc3", 0.85), ("doc4", 0.7)]

        fused = reciprocal_rank_fusion([ranking1, ranking2], k=60)

        # doc2 appears at rank 2 in list1 and rank 1 in list2
        # doc2 RRF = 1/(60+2) + 1/(60+1) = 1/62 + 1/61
        # doc1 RRF = 1/(60+1) = 1/61 (only in list1)
        # doc3 RRF = 1/(60+3) + 1/(60+2) = 1/63 + 1/62

        fused_ids = [doc_id for doc_id, _ in fused]
        # doc2 should rank highest (appears in both lists at high positions)
        assert fused_ids[0] == "doc2"

    def test_empty_rankings(self):
        """Empty ranking list returns empty."""
        fused = reciprocal_rank_fusion([])
        assert fused == []

    def test_empty_sublists(self):
        """Empty sublists produce empty result."""
        fused = reciprocal_rank_fusion([[]])
        assert fused == []

    def test_custom_k_value(self):
        """Custom k value changes scores appropriately."""
        ranking = [("doc1", 0.9)]
        fused_k1 = reciprocal_rank_fusion([ranking], k=1)
        fused_k60 = reciprocal_rank_fusion([ranking], k=60)

        # k=1: score = 1/(1+1) = 0.5
        # k=60: score = 1/(60+1) ≈ 0.0164
        assert fused_k1[0][1] > fused_k60[0][1]
        assert abs(fused_k1[0][1] - 0.5) < 1e-6

    def test_duplicate_docs_across_rankings(self):
        """Documents in multiple rankings accumulate scores."""
        ranking1 = [("doc1", 1.0)]
        ranking2 = [("doc1", 1.0)]
        ranking3 = [("doc1", 1.0)]

        fused = reciprocal_rank_fusion([ranking1, ranking2, ranking3], k=60)

        # Score should be 3 * 1/(60+1) = 3/61
        assert abs(fused[0][1] - 3.0 / 61) < 1e-6


class TestBM25Index:
    """Tests for the BM25Index class."""

    def test_initial_state(self):
        """Newly created index with nonexistent path is not built."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            index = BM25Index(index_path=f"{tmpdir}/nonexistent.pkl")
            assert not index.is_built()

    def test_build_index(self):
        """Building index marks it as built."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            index = BM25Index(index_path=f"{tmpdir}/test_bm25.pkl")
            docs = ["hello world", "foo bar baz", "hello foo"]
            ids = ["d1", "d2", "d3"]

            index.build_index(docs, ids)

            assert index.is_built()

    def test_build_index_length_mismatch(self):
        """build_index raises ValueError on length mismatch."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            index = BM25Index(index_path=f"{tmpdir}/test_bm25.pkl")
            with pytest.raises(ValueError, match="Length mismatch"):
                index.build_index(["doc1", "doc2"], ["id1"])

    def test_search_returns_relevant(self):
        """Search returns documents matching the query."""
        index = BM25Index()
        docs = [
            "machine learning algorithms",
            "deep learning neural networks",
            "cooking recipes for dinner",
        ]
        ids = ["d1", "d2", "d3"]
        index.build_index(docs, ids)

        results = index.search("learning", top_k=2)

        # Should return d1 and d2 (both contain "learning")
        result_ids = [doc_id for doc_id, _ in results]
        assert "d1" in result_ids
        assert "d2" in result_ids
        assert "d3" not in result_ids

    def test_search_respects_top_k(self):
        """Search returns at most top_k results."""
        index = BM25Index()
        docs = ["word " * 10 for _ in range(20)]
        ids = [f"d{i}" for i in range(20)]
        index.build_index(docs, ids)

        results = index.search("word", top_k=5)
        assert len(results) <= 5

    def test_search_empty_index(self):
        """Searching un-built index returns empty list."""
        index = BM25Index()
        results = index.search("hello")
        assert results == []

    def test_search_no_match(self):
        """Search with no matching terms returns empty."""
        index = BM25Index()
        docs = ["alpha beta gamma", "delta epsilon"]
        ids = ["d1", "d2"]
        index.build_index(docs, ids)

        results = index.search("zzz_nonexistent_xyz")
        assert results == []


class TestHybridSearcher:
    """Tests for the HybridSearcher class."""

    @pytest.fixture()
    def mock_qdrant(self):
        """Create a mock QdrantManager."""
        mock = MagicMock()
        mock.search_with_rbac.return_value = []
        return mock

    @pytest.fixture()
    def mock_embedder(self):
        """Create a mock EmbeddingService."""
        mock = MagicMock()
        mock.embed_text = AsyncMock(return_value=[0.1] * 1024)
        return mock

    @pytest.fixture()
    def user_context(self) -> UserContext:
        """Standard test user context."""
        return UserContext(
            user_id="user-1",
            org_id="org-1",
            roles=["viewer"],
            clearance_level=2,
        )

    @pytest.mark.asyncio
    async def test_search_with_no_results(self, mock_qdrant, mock_embedder, user_context):
        """Search returns empty list when no results found."""
        searcher = HybridSearcher(mock_qdrant, mock_embedder)

        results = await searcher.search("test query", user_context)

        assert results == []
        mock_embedder.embed_text.assert_awaited_once_with("test query")

    @pytest.mark.asyncio
    async def test_search_calls_qdrant_with_rbac(self, mock_qdrant, mock_embedder, user_context):
        """Search calls qdrant search_with_rbac."""
        searcher = HybridSearcher(mock_qdrant, mock_embedder)

        await searcher.search("test", user_context, top_k=5)

        mock_qdrant.search_with_rbac.assert_called_once()
        call_kwargs = mock_qdrant.search_with_rbac.call_args.kwargs
        assert call_kwargs["user_context"] == user_context
        assert call_kwargs["top_k"] == 10  # 5 * 2 over-fetch

    @pytest.mark.asyncio
    async def test_search_with_dense_results(self, mock_qdrant, mock_embedder, user_context):
        """Search returns results when dense search finds documents."""
        mock_point = MagicMock()
        mock_point.id = "point-1"
        mock_point.score = 0.9
        mock_point.payload = {"text": "Hello world", "org_id": "org-1"}

        mock_qdrant.search_with_rbac.return_value = [mock_point]
        searcher = HybridSearcher(mock_qdrant, mock_embedder)

        results = await searcher.search("hello", user_context, top_k=5)

        assert len(results) == 1
        assert results[0].id == "point-1"
        assert results[0].text == "Hello world"

    @pytest.mark.asyncio
    async def test_dense_only_search(self, mock_qdrant, mock_embedder, user_context):
        """dense_only_search skips BM25."""
        mock_point = MagicMock()
        mock_point.id = "p1"
        mock_point.score = 0.85
        mock_point.payload = {"text": "Test doc", "org_id": "org-1"}

        mock_qdrant.search_with_rbac.return_value = [mock_point]
        searcher = HybridSearcher(mock_qdrant, mock_embedder)

        results = await searcher.dense_only_search("test", user_context, top_k=5)

        assert len(results) == 1
        assert results[0].source == "dense"
        assert results[0].score == 0.85

    @pytest.mark.asyncio
    async def test_search_handles_embedding_error(self, mock_qdrant, mock_embedder, user_context):
        """Search returns empty on embedding failure."""
        mock_embedder.embed_text = AsyncMock(side_effect=Exception("API error"))
        searcher = HybridSearcher(mock_qdrant, mock_embedder)

        results = await searcher.search("test", user_context)

        assert results == []

    @pytest.mark.asyncio
    async def test_search_with_bm25(self, mock_qdrant, mock_embedder, user_context):
        """Search uses BM25 when index is available."""
        # Set up BM25 index
        bm25 = BM25Index()
        bm25.build_index(
            ["machine learning models", "deep learning networks"],
            ["p1", "p2"],
        )

        # Set up dense results
        mock_point = MagicMock()
        mock_point.id = "p1"
        mock_point.score = 0.8
        mock_point.payload = {"text": "machine learning models", "org_id": "org-1"}
        mock_qdrant.search_with_rbac.return_value = [mock_point]

        searcher = HybridSearcher(mock_qdrant, mock_embedder, bm25_index=bm25)
        results = await searcher.search("machine learning", user_context, top_k=5)

        # Should have results from fusion
        assert len(results) >= 1
