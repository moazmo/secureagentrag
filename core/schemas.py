"""Public Pydantic response/request models shared across the API surface.

These wrap the internal ``GraphState`` (a TypedDict) into stable, validated
shapes that the FastAPI layer, the MCP server, and any future client SDK
can rely on. The internal pipeline keeps using the TypedDict for cheap
mutation; serialisation happens at the edges.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class CitationModel(BaseModel):
    """A citation pointing to a chunk in a source document."""

    source_file: str
    page_number: int
    chunk_text: str
    relevance_score: float


class ProvenanceModel(BaseModel):
    """Where and how the synthesizer ran for a given response."""

    provider: str = ""  # "ollama" | "groq" | "openai" | "anthropic"
    model: str = ""
    forced_local: bool = False
    latency_ms: float = 0.0
    usage: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    """Request payload for ``POST /query`` and MCP ``query`` tool."""

    query: str = Field(min_length=1, max_length=4000)
    user_id: str = Field(min_length=1)
    org_id: str = ""
    roles: list[str] = Field(default_factory=lambda: ["viewer"])
    # Clearance is a 1..3 scale (low/medium/high). Bound it so a caller can't
    # pass an arbitrarily high integer to over-clear themselves past the RBAC
    # sensitivity filter.
    clearance_level: int = Field(default=1, ge=1, le=3)
    prefer_cloud: bool = False
    override_provider: str = ""


class QueryResponse(BaseModel):
    """Structured RAG response.

    The shape downstream clients (FastAPI, MCP, SDKs) bind to. Decouples the
    internal mutable ``GraphState`` from the public contract so we can refactor
    pipeline state without breaking consumers.
    """

    answer: str
    citations: list[CitationModel] = Field(default_factory=list)
    confidence_score: float = 0.0
    needs_human_review: bool = False
    query_type: str = ""
    retry_count: int = 0
    provenance: ProvenanceModel = Field(default_factory=ProvenanceModel)
    blocked: bool = False
    blocked_reason: str = ""

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> QueryResponse:
        """Build the response model from a final ``GraphState`` dict."""
        blocked = not state.get("security_passed", True) or not state.get("guardrails_passed", True)
        blocked_reason = ""
        if not state.get("guardrails_passed", True):
            blocked_reason = f"guardrails:{state.get('guardrails_reason', '')}"
        elif not state.get("security_passed", True):
            blocked_reason = state.get("security_message", "rbac_blocked")
        return cls(
            answer=state.get("generation", ""),
            citations=[CitationModel(**c) for c in state.get("citations", [])],
            confidence_score=state.get("confidence_score", 0.0),
            needs_human_review=state.get("needs_human_review", False),
            query_type=state.get("query_type", ""),
            retry_count=state.get("retry_count", 0),
            provenance=ProvenanceModel(
                provider=state.get("synth_provider", ""),
                model=state.get("synth_model", ""),
                # Report a real value instead of a hardcoded False: if the
                # synthesizer ran on local Ollama, the answer was produced
                # locally. (HIGH-sensitivity content is forced here on
                # self-hosted deploys.)
                forced_local=(state.get("synth_provider", "") == "ollama"),
                latency_ms=state.get("synth_latency_ms", 0.0),
                usage=state.get("synth_usage", {}),
            ),
            blocked=blocked,
            blocked_reason=blocked_reason,
        )


class IngestRequestModel(BaseModel):
    """Request payload for ``POST /ingest`` and MCP ``ingest`` tool."""

    file_path: str
    user_id: str
    org_id: str = ""
    roles: list[str] = Field(default_factory=lambda: ["viewer"])
    sensitivity_level: str = "low"

    @field_validator("file_path")
    @classmethod
    def _no_path_traversal(cls, v: str) -> str:
        """Reject directory-traversal sequences and null bytes."""
        if not v or not v.strip():
            raise ValueError("file_path must not be empty")
        if ".." in v or "\x00" in v:
            raise ValueError("file_path must not contain '..' or null bytes")
        return v


class IngestResponseModel(BaseModel):
    """Structured ingestion result."""

    file_path: str
    status: str
    num_chunks: int
    point_ids: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    processing_time_seconds: float = 0.0
