"""Tests for the graph compilation and execution module."""

from __future__ import annotations

import asyncio
from typing import ClassVar
from unittest.mock import AsyncMock, patch

import pytest

import core.graph as graph_mod
from config.settings import settings
from core.graph import build_rag_graph, create_initial_state, run_rag_pipeline
from ingestion.metadata import UserContext


class _DummySearcher:
    """Minimal HybridSearcher stub for the streaming contract test."""

    async def search(self, *_args, **_kwargs):
        from retrieval.hybrid_search import SearchResult

        return [
            SearchResult(
                id="d1",
                text="Document X explains the answer.",
                score=0.9,
                metadata={"source_file": "x.txt", "page_number": 1},
                source="dense",
            )
        ]


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
        # Faithfulness gate sits between synth and evaluator so the
        # evaluator can read faithfulness_ratio when it's enabled, and the
        # node is a no-op when disabled.
        assert ("synthesizer", "faithfulness") in edge_pairs
        assert ("faithfulness", "evaluator") in edge_pairs


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


class TestRequestDeadline:
    """The pipeline must enforce settings.request_timeout_s and refuse
    gracefully — never block the caller forever."""

    def test_run_rag_pipeline_returns_timeout_state(self, sample_user_context):
        """A graph.ainvoke that exceeds the budget produces a timeout final state."""

        class _SlowGraph:
            async def ainvoke(self, *_args, **_kwargs):
                # Sleep well beyond the test budget so the deadline must trigger.
                await asyncio.sleep(2.0)
                raise AssertionError("graph should have been cancelled by deadline")

        async def _fake_build():
            return _SlowGraph()

        with (
            patch.object(settings, "request_timeout_s", 0.05),
            patch.object(graph_mod, "build_rag_graph_async", _fake_build),
        ):
            final = asyncio.run(
                run_rag_pipeline(
                    query="anything",
                    user_context=sample_user_context,
                    thread_id="t-timeout",
                )
            )

        assert "exceeded" in final["generation"].lower()
        assert final["needs_human_review"] is True
        assert final["evaluation_notes"] == "request_timeout"
        assert final["confidence_score"] == 0.0
        assert final["audit_trail"], "timeout must leave an audit entry"
        assert final["audit_trail"][0]["node"] == "deadline"
        assert final["audit_trail"][0]["action"] == "timeout"

    def test_streaming_emits_token_events_via_writer(self, sample_user_context):
        """Contract: streaming flows through graph.astream(stream_mode=['updates',
        'custom']) and the synthesizer pushes token events through the
        LangGraph writer. No hand-walked graph anywhere."""
        from core.graph import run_rag_pipeline_stream

        async def _fake_stream(prompt, **_kwargs):
            for chunk in ("Hello ", "world", " [1]."):
                yield chunk

        # Make every other LLM call a no-op so the graph runs locally.
        async def _fake_decision(prompt, **_kwargs):
            class _R:
                provider = "ollama"
                model = "test"
                reason = "test"
                forced_local = False

            class _Resp:
                usage: ClassVar[dict] = {}
                latency_ms = 0.0

            return ("yes", _R(), _Resp())

        async def _drive():
            tokens: list[str] = []
            saw_phase = False
            with (
                patch("core.agents.synthesizer.call_llm_stream", _fake_stream),
                patch("core.agents.router.call_llm_with_decision", _fake_decision),
                patch("core.agents.retriever.call_llm_async", AsyncMock(return_value="yes")),
                patch(
                    "core.agents.retriever._get_hybrid_searcher",
                    return_value=_DummySearcher(),
                ),
                patch("core.agents.security.call_llm_async", AsyncMock(return_value="SAFE")),
                patch("core.agents.evaluator.call_llm_async", AsyncMock(return_value="0.8")),
            ):
                async for event in run_rag_pipeline_stream(
                    query="What is X?",
                    user_context=sample_user_context,
                    thread_id="t-stream-contract",
                ):
                    if event.get("type") == "token":
                        tokens.append(event["text"])
                    elif event.get("type") == "phase":
                        saw_phase = True
            return tokens, saw_phase

        tokens, saw_phase = asyncio.run(_drive())
        assert saw_phase, "phase events must surface from astream(updates)"
        # At least the three streamed chunks landed via the writer.
        joined = "".join(tokens)
        assert "Hello " in joined and "world" in joined and "[1]." in joined

    def test_run_rag_pipeline_disabled_when_budget_zero(self, sample_user_context):
        """SAR_REQUEST_TIMEOUT_S=0 disables the deadline."""

        called = {"n": 0}

        class _FastGraph:
            async def ainvoke(self, initial_state, **_kwargs):
                called["n"] += 1
                # Return a minimal final state the pipeline can post-process.
                initial_state["generation"] = "ok"
                initial_state["audit_trail"] = [{"node": "router"}]
                return initial_state

        async def _fake_build():
            return _FastGraph()

        with (
            patch.object(settings, "request_timeout_s", 0.0),
            patch.object(graph_mod, "build_rag_graph_async", _fake_build),
        ):
            final = asyncio.run(
                run_rag_pipeline(
                    query="hi",
                    user_context=sample_user_context,
                    thread_id="t-nobudget",
                )
            )
        assert called["n"] == 1
        assert final["generation"] == "ok"
