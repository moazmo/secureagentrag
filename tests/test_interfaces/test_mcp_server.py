"""Tests for the MCP server interface."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ingestion.metadata import UserContext
from interfaces.mcp_server import (
    _MCP_AVAILABLE,
    _build_user_context,
    _query_impl,
    _retrieve_impl,
    build_server,
)


class TestBuildUserContext:
    """Tests for _build_user_context helper."""

    def test_defaults_viewer_role(self):
        """When no roles are provided, default to ['viewer']."""
        ctx = _build_user_context("u1", "o1", [], 2)
        assert ctx.user_id == "u1"
        assert ctx.org_id == "o1"
        assert ctx.roles == ["viewer"]
        assert ctx.clearance_level == 2

    def test_preserves_provided_roles(self):
        """Custom roles are preserved."""
        ctx = _build_user_context("u1", "o1", ["admin", "engineer"], 3)
        assert ctx.roles == ["admin", "engineer"]


class TestRetrieveImpl:
    """Tests for _retrieve_impl (RBAC-filtered hybrid search)."""

    @patch("core.agents.retriever._get_hybrid_searcher")
    async def test_returns_formatted_results(self, mock_get_searcher):
        """Results are transformed into dicts with doc_id, text, score, metadata."""
        mock_result = MagicMock()
        mock_result.id = "pt-1"
        mock_result.text = "chunk text"
        mock_result.score = 0.95
        mock_result.metadata = {"source": "file.pdf"}
        searcher = MagicMock()
        searcher.search = AsyncMock(return_value=[mock_result])
        mock_get_searcher.return_value = searcher

        result = await _retrieve_impl(
            query="test",
            user_id="u1",
            org_id="o1",
            roles=["viewer"],
            clearance_level=1,
            top_k=5,
        )

        assert len(result) == 1
        assert result[0]["doc_id"] == "pt-1"
        assert result[0]["text"] == "chunk text"
        assert result[0]["score"] == 0.95
        assert result[0]["metadata"]["source"] == "file.pdf"
        searcher.search.assert_awaited_once()

    @patch("core.agents.retriever._get_hybrid_searcher")
    async def test_empty_results(self, mock_get_searcher):
        """No matches returns an empty list."""
        searcher = MagicMock()
        searcher.search = AsyncMock(return_value=[])
        mock_get_searcher.return_value = searcher

        result = await _retrieve_impl("q", "u1")

        assert result == []

    @patch("core.agents.retriever._get_hybrid_searcher")
    async def test_rbac_params_passed(self, mock_get_searcher):
        """User context is built and forwarded to the searcher."""
        searcher = MagicMock()
        searcher.search = AsyncMock(return_value=[])
        mock_get_searcher.return_value = searcher

        await _retrieve_impl(
            query="q",
            user_id="u1",
            org_id="acme",
            roles=["admin"],
            clearance_level=3,
            top_k=10,
        )

        call_kwargs = searcher.search.call_args.kwargs
        assert call_kwargs["top_k"] == 10
        user_ctx = call_kwargs["user_context"]
        assert isinstance(user_ctx, UserContext)
        assert user_ctx.user_id == "u1"
        assert user_ctx.org_id == "acme"
        assert user_ctx.roles == ["admin"]
        assert user_ctx.clearance_level == 3


class TestQueryImpl:
    """Tests for _query_impl (full RAG pipeline)."""

    @patch("interfaces.mcp_server.run_rag_pipeline", new_callable=AsyncMock)
    async def test_returns_query_response_dict(self, mock_pipeline):
        """The state is serialised through QueryResponse.from_state."""
        mock_pipeline.return_value = {
            "generation": "The answer.",
            "citations": [],
            "confidence_score": 0.9,
            "needs_human_review": False,
            "query_type": "simple",
            "retry_count": 0,
            "security_passed": True,
            "guardrails_passed": True,
            "synth_provider": "ollama",
            "synth_model": "qwen3:8b",
            "synth_latency_ms": 1200.0,
            "synth_usage": {},
        }

        result = await _query_impl("What is RAG?", "u1")

        assert result["answer"] == "The answer."
        assert result["confidence_score"] == 0.9
        assert result["provenance"]["provider"] == "ollama"
        mock_pipeline.assert_awaited_once()
        kwargs = mock_pipeline.call_args.kwargs
        assert kwargs["query"] == "What is RAG?"
        assert kwargs["thread_id"] == "mcp-u1"
        assert kwargs["prefer_cloud"] is False

    @patch("interfaces.mcp_server.run_rag_pipeline", new_callable=AsyncMock)
    async def test_prefer_cloud_passed_through(self, mock_pipeline):
        """The prefer_cloud flag is forwarded to the pipeline."""
        mock_pipeline.return_value = {
            "generation": "",
            "citations": [],
            "confidence_score": 0.0,
            "needs_human_review": False,
            "query_type": "",
            "retry_count": 0,
            "security_passed": True,
            "guardrails_passed": True,
            "synth_provider": "",
            "synth_model": "",
            "synth_latency_ms": 0.0,
            "synth_usage": {},
        }

        await _query_impl("q", "u1", prefer_cloud=True)

        assert mock_pipeline.call_args.kwargs["prefer_cloud"] is True


class TestBuildServer:
    """Tests for build_server."""

    @pytest.mark.skipif(not _MCP_AVAILABLE, reason="mcp package not installed")
    def test_returns_fastmcp_instance(self):
        """build_server returns a FastMCP object with tools registered."""
        server = build_server()
        assert server is not None
        # FastMCP exposes tools via _tool_manager._tools
        tools = getattr(server._tool_manager, "_tools", {})
        assert "retrieve" in tools
        assert "query" in tools

    def test_raises_when_mcp_unavailable(self):
        """If the mcp package is not installed, build_server raises RuntimeError."""
        with (
            patch("interfaces.mcp_server._MCP_AVAILABLE", False),
            pytest.raises(RuntimeError, match="mcp package not installed"),
        ):
            build_server()

    @pytest.mark.skipif(not _MCP_AVAILABLE, reason="mcp package not installed")
    @patch("interfaces.mcp_server._retrieve_impl", new_callable=AsyncMock)
    async def test_retrieve_tool_serialises_json(self, mock_retrieve):
        """The retrieve tool returns a JSON string."""
        mock_retrieve.return_value = [{"doc_id": "1", "text": "t", "score": 0.5, "metadata": {}}]
        server = build_server()
        result, _ = await server.call_tool("retrieve", {"query": "q", "user_id": "u1"})
        parsed = json.loads(result[0].text)
        assert isinstance(parsed, list)
        assert parsed[0]["doc_id"] == "1"

    @pytest.mark.skipif(not _MCP_AVAILABLE, reason="mcp package not installed")
    @patch("interfaces.mcp_server._query_impl", new_callable=AsyncMock)
    async def test_query_tool_serialises_json(self, mock_query):
        """The query tool returns a JSON string."""
        mock_query.return_value = {"answer": "yes", "confidence_score": 0.99}
        server = build_server()
        result, _ = await server.call_tool("query", {"query": "q", "user_id": "u1"})
        parsed = json.loads(result[0].text)
        assert parsed["answer"] == "yes"
