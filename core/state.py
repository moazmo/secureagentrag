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

    # Router
    query_type: str  # "simple", "complex", "out_of_scope"
    rewritten_query: str

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

    # Evaluation
    needs_human_review: bool
    evaluation_notes: str

    # Audit
    audit_trail: Annotated[list[dict], add]  # Append-only via reducer
