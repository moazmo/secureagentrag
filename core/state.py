"""LangGraph state schema for the multi-agent RAG workflow."""

from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict


class DocumentGrade(TypedDict):
    """Grade for a retrieved document.

    Attributes:
        doc_id: Unique identifier for the document chunk.
        text: The text content of the document chunk.
        score: Relevance score from retrieval.
        relevant: Whether the document was judged relevant by the grader.
        metadata: Associated metadata (source, page, sensitivity, etc.).
    """

    doc_id: str
    text: str
    score: float
    relevant: bool
    metadata: dict


class Citation(TypedDict):
    """Citation for a source document.

    Attributes:
        source_file: Original file name or path.
        page_number: Page number in the source document.
        chunk_text: Excerpt of the cited text.
        relevance_score: Score indicating relevance to the answer.
    """

    source_file: str
    page_number: int
    chunk_text: str
    relevance_score: float


class GraphState(TypedDict):
    """State for the multi-agent RAG graph.

    This TypedDict defines all fields flowing through the LangGraph workflow.
    Each node reads from and writes to subsets of this state.
    """

    # Input
    query: str
    user_context: dict  # UserContext serialized as dict

    # Inference routing preferences (set by UI / API caller)
    prefer_cloud: bool  # True when caller opts into cloud providers for LOW/MEDIUM
    override_provider: str  # "" or one of "ollama" / "groq" / "openai" / "anthropic"

    # Optional tone hint injected into the synthesizer's system prompt.
    # Empty string = use the default research-assistant voice. The BYOK
    # demo endpoint populates this from the X-Demo-Persona header so the
    # three personas produce visibly distinct answers.
    persona_style: str

    # Streaming dispatch flag — set by run_rag_pipeline_stream so the
    # synthesizer chooses call_llm_stream over call_llm_with_decision and
    # pushes tokens through the LangGraph stream writer. Not part of the
    # public API; leading underscore signals "internal pipeline plumbing".
    _stream: bool

    # Router
    query_type: str  # "simple", "complex", "out_of_scope"
    rewritten_query: str
    query_sensitivity: str  # "low" | "medium" | "high" — inferred from the query itself

    # Guardrails (prompt-injection / jailbreak detection)
    guardrails_passed: bool
    guardrails_reason: str

    # Security
    security_passed: bool
    security_message: str

    # Retrieval
    documents: list[DocumentGrade]

    # Grading
    relevant_documents: list[DocumentGrade]
    relevance_ratio: float

    # Corrective RAG
    retry_count: int
    max_retries: int

    # Generation
    generation: str
    citations: list[Citation]
    confidence_score: float
    # Provenance of the synthesizer LLM call (set by synthesize_answer/_stream).
    synth_provider: str  # "ollama" | "groq" | "openai" | "anthropic"
    synth_model: str
    synth_usage: dict  # {prompt_tokens, completion_tokens, total_tokens}
    synth_latency_ms: float

    # Faithfulness (NLI-gated)
    faithfulness_ratio: float  # entailed sentences / total cited sentences
    faithfulness_unsupported: list[dict]  # [{"sentence": str, "cited": [int], "verdict": str}]

    # Evaluation
    needs_human_review: bool
    evaluation_notes: str

    # Audit
    audit_trail: Annotated[list[dict], add]  # Append-only via reducer
