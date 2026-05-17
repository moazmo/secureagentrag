"""Tests for the query router and rewriter agent."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from core.agents.router import (
    _get_rewrite_prompt,
    _get_routing_prompt,
    rewrite_query,
    route_query,
)
from utils.async_helpers import run_async


@pytest.fixture()
def base_state():
    """Create a minimal GraphState for testing router functions."""
    return {
        "query": "What is the company's vacation policy?",
        "user_context": {
            "user_id": "user1",
            "org_id": "org1",
            "roles": ["employee"],
            "clearance_level": 2,
        },
        "query_type": "",
        "rewritten_query": "",
        "security_passed": False,
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


class TestRouteQuery:
    """Tests for the route_query function."""

    @patch("core.agents.router.call_llm_async")
    def test_route_query_simple(self, mock_llm, base_state):
        """Test routing a simple factual query."""
        mock_llm.return_value = "simple"

        result = run_async(route_query(base_state))

        assert result["query_type"] == "simple"
        assert result["rewritten_query"] == base_state["query"]
        assert len(result["audit_trail"]) == 1
        assert result["audit_trail"][0]["node"] == "router"
        assert result["audit_trail"][0]["action"] == "route_query"

    @patch("core.agents.router.call_llm_async")
    def test_route_query_complex(self, mock_llm, base_state):
        """Test routing a complex multi-hop query."""
        mock_llm.return_value = "complex"

        result = run_async(route_query(base_state))

        assert result["query_type"] == "complex"

    @patch("core.agents.router.call_llm_async")
    def test_route_query_out_of_scope(self, mock_llm, base_state):
        """Test routing an out-of-scope query."""
        mock_llm.return_value = "out_of_scope"

        result = run_async(route_query(base_state))

        assert result["query_type"] == "out_of_scope"

    @patch("core.agents.router.call_llm_async")
    def test_route_query_invalid_response_defaults_to_complex(self, mock_llm, base_state):
        """Test that invalid LLM response defaults to 'complex'."""
        mock_llm.return_value = "I think this is a simple query about..."

        result = run_async(route_query(base_state))

        assert result["query_type"] == "complex"

    @patch("core.agents.router.call_llm_async")
    def test_route_query_empty_response_defaults_to_complex(self, mock_llm, base_state):
        """Test that empty LLM response defaults to 'complex'."""
        mock_llm.return_value = ""

        result = run_async(route_query(base_state))

        assert result["query_type"] == "complex"


class TestRewriteQuery:
    """Tests for the rewrite_query function."""

    @patch("core.agents.router.call_llm_async")
    def test_rewrite_query_produces_different_query(self, mock_llm, base_state):
        """Test that rewrite_query generates a different query."""
        mock_llm.return_value = "What are the details of the employee vacation policy?"

        base_state["rewritten_query"] = base_state["query"]
        result = run_async(rewrite_query(base_state))

        assert result["rewritten_query"] != base_state["query"]
        assert "vacation" in result["rewritten_query"]
        assert result["retry_count"] == 1

    @patch("core.agents.router.call_llm_async")
    def test_rewrite_query_increments_retry_count(self, mock_llm, base_state):
        """Test that retry_count is incremented."""
        mock_llm.return_value = "improved query"
        base_state["retry_count"] = 1

        result = run_async(rewrite_query(base_state))

        assert result["retry_count"] == 2

    @patch("core.agents.router.call_llm_async")
    def test_rewrite_query_falls_back_on_empty_response(self, mock_llm, base_state):
        """Test that empty LLM response falls back to current query."""
        mock_llm.return_value = ""
        base_state["rewritten_query"] = "original query"

        result = run_async(rewrite_query(base_state))

        assert result["rewritten_query"] == "original query"

    @patch("core.agents.router.call_llm_async")
    def test_rewrite_query_appends_to_audit_trail(self, mock_llm, base_state):
        """Test that rewrite appends to audit trail."""
        mock_llm.return_value = "new query"

        result = run_async(rewrite_query(base_state))

        assert len(result["audit_trail"]) == 1
        assert result["audit_trail"][0]["action"] == "rewrite_query"
        assert "timestamp" in result["audit_trail"][0]


class TestRoutingPrompt:
    """Tests for the _get_routing_prompt helper."""

    def test_prompt_contains_query(self):
        """Test that the prompt includes the user query."""
        prompt = _get_routing_prompt("What is machine learning?")
        assert "What is machine learning?" in prompt

    def test_prompt_contains_all_categories(self):
        """Test that the prompt mentions all classification categories."""
        prompt = _get_routing_prompt("test")
        assert "simple" in prompt
        assert "complex" in prompt
        assert "out_of_scope" in prompt


class TestRewritePrompt:
    """Tests for the _get_rewrite_prompt helper."""

    def test_rewrite_prompt_includes_query_and_summary(self):
        """Test that rewrite prompt contains both query and failed docs summary."""
        prompt = _get_rewrite_prompt("my query", "doc about cats; doc about dogs")
        assert "my query" in prompt
        assert "doc about cats" in prompt
