"""MCP server exposing SecureAgentRAG retrieval + query as tools.

Run with ``uv run python -m interfaces.mcp_server`` (stdio transport). Add
to your Claude Desktop / Claude Code / Cursor config under ``mcpServers``:

    {
      "secureagentrag": {
        "command": "uv",
        "args": ["run", "python", "-m", "interfaces.mcp_server"],
        "cwd": "F:/CV_project/secureagentrag"
      }
    }

Two tools are exposed:

- ``retrieve(query, user_id, org_id, roles, clearance_level, top_k)`` —
  RBAC-filtered hybrid search; returns ranked chunks with metadata.
- ``query(query, user_id, org_id, roles, clearance_level, prefer_cloud)`` —
  full multi-agent RAG pipeline; returns answer + citations + provenance.

The server is intentionally thin — it serialises ``QueryResponse`` (defined
in ``core/schemas.py``) so clients get the same shape FastAPI returns.
"""

from __future__ import annotations

import json
from typing import Any

from core.graph import run_rag_pipeline
from core.schemas import QueryResponse
from ingestion.metadata import UserContext
from utils.logging import get_logger

logger = get_logger(__name__)

try:
    from mcp.server.fastmcp import FastMCP  # type: ignore[import-not-found]

    _MCP_AVAILABLE = True
except ImportError:
    FastMCP = None  # type: ignore[assignment,misc]
    _MCP_AVAILABLE = False


def _build_user_context(
    user_id: str, org_id: str, roles: list[str], clearance_level: int
) -> UserContext:
    return UserContext(
        user_id=user_id,
        org_id=org_id,
        roles=roles or ["viewer"],
        clearance_level=clearance_level,
    )


async def _retrieve_impl(
    query: str,
    user_id: str,
    org_id: str = "",
    roles: list[str] | None = None,
    clearance_level: int = 1,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Run RBAC-filtered hybrid search and return raw chunks (no synthesis)."""
    from core.agents.retriever import _get_hybrid_searcher

    user_ctx = _build_user_context(user_id, org_id, roles or ["viewer"], clearance_level)
    searcher = _get_hybrid_searcher()
    results = await searcher.search(query=query, user_context=user_ctx, top_k=top_k)
    return [
        {
            "doc_id": r.id,
            "text": r.text,
            "score": r.score,
            "metadata": r.metadata,
        }
        for r in results
    ]


async def _query_impl(
    query: str,
    user_id: str,
    org_id: str = "",
    roles: list[str] | None = None,
    clearance_level: int = 1,
    prefer_cloud: bool = False,
) -> dict[str, Any]:
    """Run the full multi-agent RAG pipeline and return a ``QueryResponse``."""
    user_ctx = _build_user_context(user_id, org_id, roles or ["viewer"], clearance_level)
    state = await run_rag_pipeline(
        query=query,
        user_context=user_ctx,
        thread_id=f"mcp-{user_id}",
        prefer_cloud=prefer_cloud,
    )
    return QueryResponse.from_state(state).model_dump()


def build_server() -> Any:
    """Build the FastMCP server with the two SecureAgentRAG tools registered."""
    if not _MCP_AVAILABLE:
        raise RuntimeError("mcp package not installed. Run: uv sync --extra mcp")

    mcp = FastMCP("secureagentrag")

    @mcp.tool()
    async def retrieve(
        query: str,
        user_id: str,
        org_id: str = "",
        roles: list[str] | None = None,
        clearance_level: int = 1,
        top_k: int = 5,
    ) -> str:
        """Search the SecureAgentRAG corpus with RBAC filters and return ranked chunks.

        Use this when you want the raw evidence rather than a synthesised
        answer. RBAC is enforced at the Qdrant payload level — only chunks
        the user's roles grant access to are returned.
        """
        results = await _retrieve_impl(
            query=query,
            user_id=user_id,
            org_id=org_id,
            roles=roles,
            clearance_level=clearance_level,
            top_k=top_k,
        )
        return json.dumps(results, ensure_ascii=False)

    @mcp.tool()
    async def query(
        query: str,
        user_id: str,
        org_id: str = "",
        roles: list[str] | None = None,
        clearance_level: int = 1,
        prefer_cloud: bool = False,
    ) -> str:
        """Run the full multi-agent RAG pipeline. Returns answer + citations + provenance.

        Routes through guardrails -> security -> retrieve -> grade -> synth ->
        eval. HIGH-sensitivity data is forced local regardless of
        ``prefer_cloud``.
        """
        response = await _query_impl(
            query=query,
            user_id=user_id,
            org_id=org_id,
            roles=roles,
            clearance_level=clearance_level,
            prefer_cloud=prefer_cloud,
        )
        return json.dumps(response, ensure_ascii=False)

    return mcp


def main() -> None:
    """Stdio entrypoint — invoked by Claude Desktop / Code via ``mcpServers``."""
    if not _MCP_AVAILABLE:
        raise SystemExit("mcp package not installed. Run: uv sync --extra mcp")
    server = build_server()
    server.run()


if __name__ == "__main__":
    main()
