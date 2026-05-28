"""Tests for hybrid search module — sparse vectors, RRF, and HybridSearcher."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ingestion.metadata import UserContext
from retrieval.hybrid_search import (
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


class TestHybridSearcher:
    """Tests for the HybridSearcher class."""

    @pytest.fixture()
    def mock_qdrant(self):
        """Create a mock QdrantManager."""
        mock = MagicMock()
        mock.search_with_rbac.return_value = []
        mock.search_sparse_with_rbac.return_value = []
        mock.for_org.return_value = mock
        mock.collection_name = "test_collection"
        return mock

    @pytest.fixture()
    def mock_embedder(self):
        """Create a mock EmbeddingService."""
        mock = MagicMock()
        mock.embed_text = AsyncMock(return_value=[0.1] * 1024)
        return mock

    @pytest.fixture()
    def mock_sparse(self):
        """Create a mock SparseEmbeddingService."""
        mock = MagicMock()
        mock.embed_text.return_value = MagicMock()
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
    async def test_search_dense_only(self, mock_qdrant, mock_embedder, user_context):
        """search_dense_only skips sparse."""
        mock_point = MagicMock()
        mock_point.id = "p1"
        mock_point.score = 0.85
        mock_point.payload = {"text": "Test doc", "org_id": "org-1"}

        mock_qdrant.search_with_rbac.return_value = [mock_point]
        searcher = HybridSearcher(mock_qdrant, mock_embedder)

        results = await searcher.search_dense_only("test", user_context, top_k=5)

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
    async def test_search_with_sparse(self, mock_qdrant, mock_embedder, mock_sparse, user_context):
        """Search uses sparse vectors when service is available."""
        # Dense results
        mock_point = MagicMock()
        mock_point.id = "p1"
        mock_point.score = 0.8
        mock_point.payload = {"text": "machine learning models", "org_id": "org-1"}
        mock_qdrant.search_with_rbac.return_value = [mock_point]

        # Sparse results from Qdrant
        sparse_point = MagicMock()
        sparse_point.id = "p2"
        sparse_point.score = 0.7
        sparse_point.payload = {"text": "deep learning networks", "org_id": "org-1"}
        mock_qdrant.search_sparse_with_rbac.return_value = [sparse_point]

        searcher = HybridSearcher(mock_qdrant, mock_embedder, sparse_service=mock_sparse)
        results = await searcher.search("machine learning", user_context, top_k=5)

        # Should have results from fusion
        assert len(results) >= 1
        mock_qdrant.search_sparse_with_rbac.assert_called_once()

    @pytest.mark.asyncio
    async def test_sparse_drops_unauthorised_when_dense_returns_zero(
        self, mock_qdrant, mock_embedder, mock_sparse, user_context
    ):
        """REGRESSION: when a user has zero access, sparse search with RBAC
        filter must also return nothing. Qdrant applies the same payload
        filter on sparse vectors natively, but we verify the contract."""
        # Dense returns nothing (Qdrant filter rejects everything for this user).
        mock_qdrant.search_with_rbac.return_value = []

        # Sparse search also returns nothing when RBAC-filtered
        mock_qdrant.search_sparse_with_rbac.return_value = []

        searcher = HybridSearcher(mock_qdrant, mock_embedder, sparse_service=mock_sparse)
        results = await searcher.search("compensation", user_context, top_k=5)

        assert results == [], (
            "Sparse search with RBAC filter must return zero docs for "
            "unauthorised users (cross-org / over-clearance / role-mismatch)."
        )
        # Both dense and sparse paths were queried with RBAC
        mock_qdrant.search_with_rbac.assert_called_once()
        mock_qdrant.search_sparse_with_rbac.assert_called_once()

    @pytest.mark.asyncio
    async def test_sparse_admits_only_authorised_when_dense_returns_some(
        self, mock_qdrant, mock_embedder, mock_sparse, user_context
    ):
        """When dense returns one doc and sparse surfaces an extra doc,
        both are already RBAC-authorised by Qdrant."""
        # Dense returns one authorised doc.
        mock_point = MagicMock()
        mock_point.id = "ok-1"
        mock_point.score = 0.8
        mock_point.payload = {"text": "ok doc compensation", "org_id": "org-1"}
        mock_qdrant.search_with_rbac.return_value = [mock_point]

        # Sparse returns the same doc + another authorised doc
        sparse_point = MagicMock()
        sparse_point.id = "ok-2"
        sparse_point.score = 0.75
        sparse_point.payload = {"text": "another ok doc", "org_id": "org-1"}
        mock_qdrant.search_sparse_with_rbac.return_value = [mock_point, sparse_point]
        mock_qdrant.client.retrieve.return_value = [sparse_point]

        searcher = HybridSearcher(mock_qdrant, mock_embedder, sparse_service=mock_sparse)
        results = await searcher.search("compensation", user_context, top_k=5)

        ids = {r.id for r in results}
        assert "ok-1" in ids
        assert "ok-2" in ids


class TestSessionScopedSearch:
    """Tests for the dual-collection (base + session) BYOK upload search path."""

    @pytest.fixture
    def user_context(self) -> UserContext:
        return UserContext(
            user_id="demo-sess-1",
            org_id="demo",
            clearance_level=2,
            roles=["engineering"],
        )

    @pytest.fixture
    def mock_qdrant(self) -> MagicMock:
        """Base manager. ``for_session`` returns the session manager below."""
        return MagicMock()

    @pytest.fixture
    def mock_embedder(self) -> MagicMock:
        m = MagicMock()
        m.embed_text = AsyncMock(return_value=[0.1] * 1024)
        return m

    @pytest.mark.asyncio
    async def test_session_id_invokes_for_session(self, mock_qdrant, mock_embedder, user_context):
        """When session_id is passed, hybrid search must call for_session(sid)."""
        # Base + session managers
        base_mgr = MagicMock()
        sess_mgr = MagicMock()
        # Distinct collections so the "session_qdrant is tenant_qdrant" guard
        # does NOT short-circuit the session path.
        base_mgr.collection_name = "documents"
        sess_mgr.collection_name = "documents_sess_xyz"
        mock_qdrant.for_org.return_value = base_mgr
        mock_qdrant.for_session.return_value = sess_mgr

        base_point = MagicMock()
        base_point.id = "base-1"
        base_point.score = 0.9
        base_point.payload = {"text": "demo corpus chunk", "org_id": "demo"}

        sess_point = MagicMock()
        sess_point.id = "sess-1"
        sess_point.score = 0.85
        sess_point.payload = {"text": "visitor upload chunk", "org_id": "demo"}

        base_mgr.search_with_rbac.return_value = [base_point]
        sess_mgr.search_with_rbac.return_value = [sess_point]

        searcher = HybridSearcher(mock_qdrant, mock_embedder, sparse_service=None)
        results = await searcher.search("test", user_context, top_k=10, session_id="xyz")

        mock_qdrant.for_session.assert_called_once_with("xyz")
        ids = {r.id for r in results}
        # Both collections must surface their chunks in the fused ranking.
        assert "base-1" in ids
        assert "sess-1" in ids

    @pytest.mark.asyncio
    async def test_session_id_omitted_skips_for_session(
        self, mock_qdrant, mock_embedder, user_context
    ):
        """No session_id -> never call for_session (production /query path)."""
        base_mgr = MagicMock()
        base_mgr.search_with_rbac.return_value = []
        mock_qdrant.for_org.return_value = base_mgr

        searcher = HybridSearcher(mock_qdrant, mock_embedder, sparse_service=None)
        await searcher.search("test", user_context, top_k=5)
        mock_qdrant.for_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_session_search_failure_does_not_break_base(
        self, mock_qdrant, mock_embedder, user_context
    ):
        """Session-collection search exception must NOT take out the base result."""
        base_mgr = MagicMock()
        sess_mgr = MagicMock()
        base_mgr.collection_name = "documents"
        sess_mgr.collection_name = "documents_sess_broken"
        mock_qdrant.for_org.return_value = base_mgr
        mock_qdrant.for_session.return_value = sess_mgr

        base_point = MagicMock()
        base_point.id = "base-1"
        base_point.score = 0.9
        base_point.payload = {"text": "ok", "org_id": "demo"}
        base_mgr.search_with_rbac.return_value = [base_point]
        # Simulate the session collection blowing up under us.
        sess_mgr.search_with_rbac.side_effect = RuntimeError("session-down")

        searcher = HybridSearcher(mock_qdrant, mock_embedder, sparse_service=None)
        results = await searcher.search("test", user_context, top_k=5, session_id="broken")
        # The base result must survive even when the session path raises.
        assert any(r.id == "base-1" for r in results)
