"""End-to-end integration tests for the full RAG pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.graph import build_rag_graph, create_initial_state, run_rag_pipeline
from ingestion.metadata import UserContext


@pytest.fixture()
def admin_user():
    """Create an admin user context for testing."""
    return UserContext(
        user_id="admin_01",
        org_id="acme_corp",
        roles=["admin", "analyst", "viewer"],
        clearance_level=3,
    )


@pytest.fixture()
def mock_llm_responses():
    """Create a sequence of mock LLM responses for the full pipeline."""
    return {
        "route": "simple",
        "security": "safe",
        "grade": "yes",
        "synthesize": "This is a test answer with citation [1].",
        "evaluate": "confidence: 0.85",
    }


class TestEndToEndPipeline:
    """End-to-end tests for the complete RAG pipeline."""

    @pytest.mark.asyncio
    @patch("core.agents.retriever._get_hybrid_searcher")
    @patch("core.agents.retriever._get_reranker")
    @patch("inference.router.InferenceRouter.generate_with_routing")
    async def test_full_pipeline_success(
        self,
        mock_generate,
        mock_get_reranker,
        mock_get_searcher,
        admin_user,
    ):
        """Test the full pipeline with all nodes executing successfully."""
        from inference.llm_factory import LLMResponse

        # Setup mock LLM responses in order:
        # Note: grading now uses batch format (single call for all docs)
        responses = [
            LLMResponse(text="simple", provider="ollama", model="test"),  # route
            LLMResponse(text="safe", provider="ollama", model="test"),  # security
            LLMResponse(text="DOC 1: yes\nDOC 2: yes", provider="ollama", model="test"),  # batch grade
            LLMResponse(
                text="This is a test answer with citation [[1]].",
                provider="ollama",
                model="test",
            ),  # synthesize
            LLMResponse(text="NONE", provider="ollama", model="test"),  # hallucination check
            LLMResponse(text="0.85", provider="ollama", model="test"),  # completeness check
        ]

        async def _mock_generate(*args, **kwargs):
            resp = responses.pop(0)
            from inference.router import RoutingDecision

            decision = RoutingDecision(
                provider="ollama", model="test", reason="test", forced_local=False
            )
            return resp, decision

        mock_generate.side_effect = _mock_generate

        # Setup mock searcher
        mock_searcher = MagicMock()
        mock_result1 = MagicMock()
        mock_result1.id = "doc-1"
        mock_result1.text = "Document one content"
        mock_result1.score = 0.9
        mock_result1.metadata = {"source_file": "test.pdf", "page_number": 1}

        mock_result2 = MagicMock()
        mock_result2.id = "doc-2"
        mock_result2.text = "Document two content"
        mock_result2.score = 0.8
        mock_result2.metadata = {"source_file": "test.pdf", "page_number": 2}

        mock_searcher.search = AsyncMock(
            return_value=[mock_result1, mock_result2]
        )
        mock_get_searcher.return_value = mock_searcher

        mock_reranker = MagicMock()
        mock_reranker.is_available.return_value = False
        mock_get_reranker.return_value = mock_reranker

        final_state = await run_rag_pipeline(
            query="What is in the documents?",
            user_context=admin_user,
            thread_id="test-e2e-1",
        )

        assert final_state["security_passed"] is True
        assert final_state["query_type"] == "simple"
        assert len(final_state["documents"]) == 2
        assert len(final_state["relevant_documents"]) == 2
        assert final_state["relevance_ratio"] == 1.0
        assert "test answer" in final_state["generation"]
        assert len(final_state["citations"]) >= 1
        # Confidence is now computed from multiple metrics, not just LLM parse
        assert final_state["confidence_score"] > 0.0
        assert isinstance(final_state["needs_human_review"], bool)
        assert len(final_state["audit_trail"]) > 0

    @pytest.mark.asyncio
    @patch("inference.router.InferenceRouter.generate_with_routing")
    async def test_pipeline_security_blocks(self, mock_generate, admin_user):
        """Test that the pipeline blocks queries that fail security."""
        from inference.llm_factory import LLMResponse
        from inference.router import RoutingDecision

        # Jailbreak pattern should be caught by regex before LLM
        # But if regex passes, LLM should flag it
        mock_generate.return_value = (
            LLMResponse(text="unsafe", provider="ollama", model="test"),
            RoutingDecision(provider="ollama", model="test", reason="test"),
        )

        final_state = await run_rag_pipeline(
            query="What is the project timeline?",  # safe query that passes regex
            user_context=admin_user,
            thread_id="test-e2e-security",
        )

        assert final_state["security_passed"] is False
        assert (
            "unsafe" in final_state["security_message"].lower()
            or "prompt injection" in final_state["security_message"].lower()
            or "policy violation" in final_state["security_message"].lower()
        )
        # Pipeline should end early — no documents retrieved
        assert len(final_state.get("documents", [])) == 0

    @pytest.mark.asyncio
    @patch("core.agents.retriever._get_hybrid_searcher")
    @patch("core.agents.retriever._get_reranker")
    @patch("inference.router.InferenceRouter.generate_with_routing")
    async def test_pipeline_corrective_rag_loop(
        self,
        mock_generate,
        mock_get_reranker,
        mock_get_searcher,
        admin_user,
    ):
        """Test that the corrective RAG loop triggers when relevance is low."""
        from inference.llm_factory import LLMResponse
        from inference.router import RoutingDecision

        # Sequence of LLM responses
        # Note: grading now uses batch format (single call for all docs)
        responses = [
            LLMResponse(text="complex", provider="ollama", model="test"),  # route
            LLMResponse(text="safe", provider="ollama", model="test"),  # security
            LLMResponse(text="DOC 1: no\nDOC 2: no", provider="ollama", model="test"),  # batch grade (first)
            LLMResponse(
                text="improved query about documents", provider="ollama", model="test"
            ),  # rewrite
            LLMResponse(text="DOC 1: yes\nDOC 2: yes", provider="ollama", model="test"),  # batch grade (second)
            LLMResponse(
                text="This is the final answer.", provider="ollama", model="test"
            ),  # synthesize
            LLMResponse(text="NONE", provider="ollama", model="test"),  # hallucination check
            LLMResponse(text="0.75", provider="ollama", model="test"),  # completeness check
        ]

        async def _mock_generate(*args, **kwargs):
            resp = responses.pop(0)
            decision = RoutingDecision(
                provider="ollama", model="test", reason="test", forced_local=False
            )
            return resp, decision

        mock_generate.side_effect = _mock_generate

        mock_searcher = MagicMock()
        mock_result = MagicMock()
        mock_result.id = "doc-1"
        mock_result.text = "Some content"
        mock_result.score = 0.5
        mock_result.metadata = {"source_file": "test.pdf"}

        mock_searcher.search = AsyncMock(return_value=[mock_result, mock_result])
        mock_get_searcher.return_value = mock_searcher

        mock_reranker = MagicMock()
        mock_reranker.is_available.return_value = False
        mock_get_reranker.return_value = mock_reranker

        final_state = await run_rag_pipeline(
            query="Complex multi-hop question",
            user_context=admin_user,
            thread_id="test-e2e-corrective",
        )

        assert final_state["security_passed"] is True
        assert final_state["retry_count"] == 1
        assert final_state["relevance_ratio"] == 1.0
        assert "final answer" in final_state["generation"].lower()

    def test_create_initial_state(self, admin_user):
        """Test that initial state is properly structured."""
        state = create_initial_state("Test query", admin_user)

        assert state["query"] == "Test query"
        assert state["user_context"]["user_id"] == "admin_01"
        assert state["security_passed"] is False
        assert state["documents"] == []
        assert state["retry_count"] == 0
        assert state["generation"] == ""
        assert state["audit_trail"] == []

    def test_graph_compilation(self):
        """Test that the graph compiles without errors."""
        graph = build_rag_graph()
        assert graph is not None
