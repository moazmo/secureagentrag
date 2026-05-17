"""Tests for the retriever and document grading agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.agents.retriever import (
    grade_documents,
    retrieve_documents,
    should_retry,
)
from utils.async_helpers import run_async


async def _call_retrieve_documents(state):
    """Helper to call async retrieve_documents in sync tests."""
    return await retrieve_documents(state)


@pytest.fixture()
def retriever_state():
    """Create a base state for retriever tests."""
    return {
        "query": "What is the deployment process?",
        "user_context": {
            "user_id": "user1",
            "org_id": "org1",
            "roles": ["engineer"],
            "clearance_level": 2,
        },
        "query_type": "simple",
        "rewritten_query": "What is the deployment process?",
        "security_passed": True,
        "security_message": "",
        "documents": [],
        "relevant_documents": [],
        "relevance_ratio": 0.0,
        "retry_count": 0,
        "max_retries": 2,
        "generation": "",
        "citations": [],
        "confidence_score": 0.0,
        "needs_human_review": False,
        "evaluation_notes": "",
        "audit_trail": [],
    }


@pytest.fixture()
def mock_search_results():
    """Create mock SearchResult objects."""
    result1 = MagicMock()
    result1.id = "doc-1"
    result1.text = "The deployment process involves CI/CD pipelines."
    result1.score = 0.9
    result1.metadata = {"source_file": "deploy.pdf", "page_number": 1}

    result2 = MagicMock()
    result2.id = "doc-2"
    result2.text = "Testing should happen before deployment."
    result2.score = 0.75
    result2.metadata = {"source_file": "testing.pdf", "page_number": 3}

    return [result1, result2]


class TestRetrieveDocuments:
    """Tests for the retrieve_documents function."""

    @patch("core.agents.retriever._get_reranker")
    @patch("core.agents.retriever._get_hybrid_searcher")
    def test_retrieve_documents_success(
        self,
        mock_get_searcher,
        mock_get_reranker,
        retriever_state,
        mock_search_results,
    ):
        """Test successful document retrieval."""
        mock_searcher = MagicMock()
        mock_searcher.search = AsyncMock(return_value=mock_search_results)
        mock_get_searcher.return_value = mock_searcher

        mock_reranker = MagicMock()
        mock_reranker.is_available.return_value = False
        mock_get_reranker.return_value = mock_reranker

        result = run_async(_call_retrieve_documents(retriever_state))

        assert "documents" in result
        assert len(result["documents"]) == 2
        assert result["documents"][0]["doc_id"] == "doc-1"
        assert result["documents"][0]["text"] == "The deployment process involves CI/CD pipelines."
        assert result["documents"][0]["relevant"] is False  # Not graded yet

    @patch("core.agents.retriever._get_reranker")
    @patch("core.agents.retriever._get_hybrid_searcher")
    def test_retrieve_documents_empty_results(
        self, mock_get_searcher, mock_get_reranker, retriever_state
    ):
        """Test retrieval with no results."""
        mock_searcher = MagicMock()
        mock_searcher.search = AsyncMock(return_value=[])
        mock_get_searcher.return_value = mock_searcher

        mock_reranker = MagicMock()
        mock_reranker.is_available.return_value = False
        mock_get_reranker.return_value = mock_reranker

        result = run_async(_call_retrieve_documents(retriever_state))

        assert result["documents"] == []
        assert len(result["audit_trail"]) == 1
        assert result["audit_trail"][0]["documents_count"] == 0

    @patch("core.agents.retriever._get_reranker")
    @patch("core.agents.retriever._get_hybrid_searcher")
    def test_retrieve_documents_appends_audit(
        self,
        mock_get_searcher,
        mock_get_reranker,
        retriever_state,
        mock_search_results,
    ):
        """Test that retrieval appends to audit trail."""
        mock_searcher = MagicMock()
        mock_searcher.search = AsyncMock(return_value=mock_search_results)
        mock_get_searcher.return_value = mock_searcher

        mock_reranker = MagicMock()
        mock_reranker.is_available.return_value = False
        mock_get_reranker.return_value = mock_reranker

        result = run_async(_call_retrieve_documents(retriever_state))

        assert len(result["audit_trail"]) == 1
        assert result["audit_trail"][0]["node"] == "retriever"
        assert result["audit_trail"][0]["action"] == "retrieve_documents"


class TestGradeDocuments:
    """Tests for the grade_documents function."""

    @patch("core.agents.retriever.call_llm_async")
    def test_grade_documents_all_relevant(self, mock_llm, retriever_state):
        """Test grading where all documents are relevant."""
        # Batch format: DOC N: yes for each document
        mock_llm.return_value = "DOC 1: yes\nDOC 2: yes"

        retriever_state["documents"] = [
            {
                "doc_id": "d1",
                "text": "Relevant content",
                "score": 0.9,
                "relevant": False,
                "metadata": {},
            },
            {
                "doc_id": "d2",
                "text": "Also relevant",
                "score": 0.8,
                "relevant": False,
                "metadata": {},
            },
        ]

        result = run_async(grade_documents(retriever_state))

        assert result["relevance_ratio"] == 1.0
        assert len(result["relevant_documents"]) == 2
        assert all(doc["relevant"] for doc in result["documents"])

    @patch("core.agents.retriever.call_llm_async")
    def test_grade_documents_none_relevant(self, mock_llm, retriever_state):
        """Test grading where no documents are relevant."""
        # Batch format: DOC N: no for each document
        mock_llm.return_value = "DOC 1: no\nDOC 2: no"

        retriever_state["documents"] = [
            {"doc_id": "d1", "text": "Irrelevant", "score": 0.3, "relevant": False, "metadata": {}},
            {"doc_id": "d2", "text": "Not useful", "score": 0.2, "relevant": False, "metadata": {}},
        ]

        result = run_async(grade_documents(retriever_state))

        assert result["relevance_ratio"] == 0.0
        assert len(result["relevant_documents"]) == 0

    @patch("core.agents.retriever.call_llm_async")
    def test_grade_documents_mixed_relevance(self, mock_llm, retriever_state):
        """Test grading with mixed relevance results."""
        # Batch format: mixed yes/no responses
        mock_llm.return_value = "DOC 1: yes\nDOC 2: no\nDOC 3: yes\nDOC 4: no"

        retriever_state["documents"] = [
            {"doc_id": "d1", "text": "Text 1", "score": 0.9, "relevant": False, "metadata": {}},
            {"doc_id": "d2", "text": "Text 2", "score": 0.7, "relevant": False, "metadata": {}},
            {"doc_id": "d3", "text": "Text 3", "score": 0.6, "relevant": False, "metadata": {}},
            {"doc_id": "d4", "text": "Text 4", "score": 0.5, "relevant": False, "metadata": {}},
        ]

        result = run_async(grade_documents(retriever_state))

        assert result["relevance_ratio"] == 0.5
        assert len(result["relevant_documents"]) == 2

    @patch("core.agents.retriever.call_llm_async")
    def test_grade_documents_empty_list(self, mock_llm, retriever_state):
        """Test grading with empty documents list."""
        retriever_state["documents"] = []

        result = run_async(grade_documents(retriever_state))

        assert result["relevance_ratio"] == 0.0
        assert result["relevant_documents"] == []
        mock_llm.assert_not_called()


class TestShouldRetry:
    """Tests for the should_retry conditional edge function."""

    def test_returns_rewrite_low_relevance_retries_available(self):
        """Test that low relevance with retries available returns 'rewrite'."""
        state = {
            "relevance_ratio": 0.3,
            "retry_count": 0,
            "max_retries": 2,
        }
        assert should_retry(state) == "rewrite"

    def test_returns_generate_high_relevance(self):
        """Test that high relevance returns 'generate'."""
        state = {
            "relevance_ratio": 0.8,
            "retry_count": 0,
            "max_retries": 2,
        }
        assert should_retry(state) == "generate"

    def test_returns_generate_retries_exhausted(self):
        """Test that exhausted retries returns 'generate' even with low relevance."""
        state = {
            "relevance_ratio": 0.2,
            "retry_count": 2,
            "max_retries": 2,
        }
        assert should_retry(state) == "generate"

    def test_returns_rewrite_at_threshold(self):
        """Test behavior at the 0.5 threshold boundary (below returns rewrite)."""
        state = {
            "relevance_ratio": 0.49,
            "retry_count": 0,
            "max_retries": 2,
        }
        assert should_retry(state) == "rewrite"

    def test_returns_generate_at_threshold(self):
        """Test behavior at exactly 0.5 relevance (returns generate)."""
        state = {
            "relevance_ratio": 0.5,
            "retry_count": 0,
            "max_retries": 2,
        }
        assert should_retry(state) == "generate"

    def test_defaults_when_fields_missing(self):
        """Test default behavior when fields are missing from state."""
        state = {}
        # relevance_ratio defaults to 0.0, retry_count to 0, max_retries to 2
        # 0.0 < 0.5 and 0 < 2 -> rewrite
        assert should_retry(state) == "rewrite"
