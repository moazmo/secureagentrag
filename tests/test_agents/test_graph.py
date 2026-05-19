"""Tests for the graph compilation and execution module."""

from __future__ import annotations

import pytest

from core.graph import build_rag_graph, create_initial_state
from ingestion.metadata import UserContext


@pytest.fixture()
def sample_user_context():
    """Create a sample UserContext for testing."""
    return UserContext(
        user_id="test-user",
        org_id="test-org",
        roles=["admin", "engineer"],
        clearance_level=3,
    )


class TestBuildRagGraph:
    """Tests for the build_rag_graph function."""

    def test_returns_compiled_graph(self):
        """Test that build_rag_graph returns a compiled graph object."""
        graph = build_rag_graph()
        # Compiled graph should be invocable (has invoke/ainvoke methods)
        assert hasattr(graph, "invoke") or hasattr(graph, "ainvoke")

    def test_graph_has_correct_nodes(self):
        """Test that the graph contains all expected node names."""
        graph = build_rag_graph()

        # Access the underlying graph's nodes via get_graph()
        graph_repr = graph.get_graph()
        # nodes may be strings or objects with .id attribute depending on version
        node_ids = set()
        for node in graph_repr.nodes:
            node_ids.add(node if isinstance(node, str) else node.id)

        expected_nodes = {
            "router",
            "security",
            "retriever",
            "grader",
            "rewriter",
            "synthesizer",
            "evaluator",
        }
        for expected in expected_nodes:
            assert expected in node_ids, f"Missing node: {expected}"

    def test_graph_has_start_and_end(self):
        """Test that the graph has __start__ and __end__ nodes."""
        graph = build_rag_graph()
        graph_repr = graph.get_graph()
        node_ids = set()
        for node in graph_repr.nodes:
            node_ids.add(node if isinstance(node, str) else node.id)

        assert "__start__" in node_ids
        assert "__end__" in node_ids

    def test_graph_edges_exist(self):
        """Test that key edges exist in the graph."""
        graph = build_rag_graph()
        graph_repr = graph.get_graph()

        # Collect edges as (source, target) pairs
        edge_pairs = set()
        for edge in graph_repr.edges:
            if isinstance(edge, tuple):
                edge_pairs.add((edge[0], edge[1]))
            else:
                edge_pairs.add((edge.source, edge.target))

        # Check critical direct edges
        assert ("__start__", "router") in edge_pairs
        # Guardrails sits between router and security to catch prompt-
        # injection before any retrieval / LLM budget is spent.
        assert ("router", "guardrails") in edge_pairs
        assert ("guardrails", "security") in edge_pairs
        assert ("retriever", "grader") in edge_pairs
        assert ("rewriter", "retriever") in edge_pairs
        assert ("synthesizer", "evaluator") in edge_pairs


class TestCreateInitialState:
    """Tests for the create_initial_state function."""

    def test_creates_proper_state(self, sample_user_context):
        """Test that initial state is properly structured."""
        state = create_initial_state(
            query="What is machine learning?",
            user_context=sample_user_context,
        )

        assert state["query"] == "What is machine learning?"
        assert state["user_context"]["user_id"] == "test-user"
        assert state["user_context"]["org_id"] == "test-org"
        assert state["user_context"]["roles"] == ["admin", "engineer"]
        assert state["user_context"]["clearance_level"] == 3

    def test_initial_state_defaults(self, sample_user_context):
        """Test that initial state has proper default values."""
        state = create_initial_state("test query", sample_user_context)

        assert state["query_type"] == ""
        assert state["rewritten_query"] == ""
        assert state["security_passed"] is False
        assert state["documents"] == []
        assert state["relevant_documents"] == []
        assert state["relevance_ratio"] == 0.0
        assert state["retry_count"] == 0
        assert state["max_retries"] == 2
        assert state["generation"] == ""
        assert state["citations"] == []
        assert state["confidence_score"] == 0.0
        assert state["needs_human_review"] is False
        assert state["evaluation_notes"] == ""
        assert state["audit_trail"] == []

    def test_user_context_serialized_as_dict(self, sample_user_context):
        """Test that UserContext is properly serialized to dict."""
        state = create_initial_state("query", sample_user_context)

        uc = state["user_context"]
        assert isinstance(uc, dict)
        assert "user_id" in uc
        assert "org_id" in uc
        assert "roles" in uc
        assert "clearance_level" in uc
