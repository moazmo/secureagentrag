"""Answer synthesis agent with mandatory citations."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from core.agents.router import call_llm_async, call_llm_stream
from core.state import Citation, GraphState  # noqa: TC001
from utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from core.state import DocumentGrade

logger = get_logger(__name__)


def _max_sensitivity(docs_to_use: list[DocumentGrade]) -> str:
    """Determine highest sensitivity level among the documents used.

    Args:
        docs_to_use: Documents that will be fed as synthesis context.

    Returns:
        "high" | "medium" | "low".
    """
    levels = [doc.get("metadata", {}).get("sensitivity_level", "low") for doc in docs_to_use]
    if "high" in levels:
        return "high"
    if "medium" in levels:
        return "medium"
    return "low"


def _build_synthesis_prompt(query: str, documents: list[DocumentGrade], sensitivity: str) -> str:
    """Build the synthesis prompt with source markers for citation tracking.

    Args:
        query: The user's query.
        documents: List of relevant documents to use as context.
        sensitivity: Sensitivity level string for disclaimer handling.

    Returns:
        Formatted prompt string for the LLM.
    """
    context_parts: list[str] = []
    for i, doc in enumerate(documents, start=1):
        source = doc.get("metadata", {}).get("source_file", "unknown")
        page = doc.get("metadata", {}).get("page_number", 0)
        context_parts.append(f"[{i}] (Source: {source}, Page: {page})\n{doc['text'][:600]}")

    context_str = "\n\n".join(context_parts)

    sensitivity_instruction = ""
    if sensitivity in ("high", "medium"):
        sensitivity_instruction = (
            "\n\nIMPORTANT: This involves sensitive information. "
            "Include appropriate disclaimers about data sensitivity and "
            "note that verification may be required."
        )

    return (
        "You are an expert assistant. Answer the user's question based ONLY on the "
        "provided context. You MUST cite your sources using [[1]], [[2]], etc. markers "
        "corresponding to the numbered sources below. Use double brackets for citations.\n\n"
        "If the context does not contain enough information to answer, say so clearly.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {query}\n"
        f"{sensitivity_instruction}\n\n"
        "Provide a comprehensive answer with citations:"
    )


def _extract_citations(response: str, documents: list[DocumentGrade]) -> list[Citation]:
    """Extract citation references from the LLM response.

    Parses citation markers of the form [[N]] from the response text and maps
    them back to source documents. Uses double-bracket format [[N]] to avoid
    conflicts with markdown links [text](url) and numbered lists.

    Args:
        response: The generated response text with [[N]] citation markers.
        documents: The list of documents used as context.

    Returns:
        List of Citation TypedDicts with source information.
    """
    # Match [[N]] format — distinct from markdown links [text](url)
    citation_refs = re.findall(r"\[\[(\d+)\]\]", response)

    # Fallback: also check for legacy [N] format (but be more strict)
    if not citation_refs:
        # Only match [N] when not followed by '(' (excluding markdown links)
        citation_refs = re.findall(r"\[(\d+)\](?!\s*\()", response)

    seen_indices: set[int] = set()
    citations: list[Citation] = []

    for ref in citation_refs:
        idx = int(ref) - 1  # Convert to 0-based index
        if idx < 0 or idx >= len(documents) or idx in seen_indices:
            continue
        seen_indices.add(idx)

        doc = documents[idx]
        metadata = doc.get("metadata", {})
        citation: Citation = {
            "source_file": metadata.get("source_file", "unknown"),
            "page_number": metadata.get("page_number", 0),
            "chunk_text": doc["text"][:200],
            "relevance_score": doc.get("score", 0.0),
        }
        citations.append(citation)

    return citations


def _compute_synthesis_confidence(
    documents: list[DocumentGrade],
    citations: list[Citation],
    generation: str,
) -> float:
    """Compute a preliminary confidence score for the synthesized answer.

    This is a fast heuristic-based score that the evaluator later refines
    with LLM-based assessment. It considers:
    - Average relevance score of retrieved documents
    - Citation density (citations per sentence)
    - Document coverage (fraction of retrieved docs that were cited)

    Args:
        documents: Retrieved documents used for synthesis.
        citations: Extracted citations from the generated answer.
        generation: The generated response text.

    Returns:
        Preliminary confidence score between 0.0 and 1.0.
    """
    if not documents or not generation:
        return 0.0

    # Factor 1: Average retrieval relevance score (normalized)
    scores = [doc.get("score", 0.0) for doc in documents if doc.get("score")]
    avg_relevance = sum(scores) / len(scores) if scores else 0.0
    relevance_component = min(1.0, max(0.0, (avg_relevance - 0.3) / 0.5))

    # Factor 2: Citation density
    sentences = re.split(r"[.!?]+\s+", generation)
    sentences = [s.strip() for s in sentences if s.strip()]
    citation_density = len(citations) / max(len(sentences), 1)
    density_component = min(1.0, citation_density * 2.0)  # 1 cite per 2 sentences = full

    # Factor 3: Document coverage (cited docs / total docs)
    coverage_component = len(citations) / max(len(documents), 1)

    # Weighted combination
    confidence = relevance_component * 0.40 + density_component * 0.30 + coverage_component * 0.30
    return round(max(0.0, min(1.0, confidence)), 3)


def _add_disclaimers(response: str, sensitivity_level: str) -> str:
    """Add disclaimers to the response based on sensitivity level.

    Args:
        response: The generated response text.
        sensitivity_level: The sensitivity level of the documents used.

    Returns:
        Response text with appropriate disclaimers appended.
    """
    if sensitivity_level == "high":
        disclaimer = (
            "\n\n---\n"
            "**DISCLAIMER**: This response contains information derived from "
            "highly sensitive documents. Please verify with authorized personnel "
            "before acting on this information. Do not share externally."
        )
        return response + disclaimer
    elif sensitivity_level == "medium":
        disclaimer = (
            "\n\n---\n"
            "**Note**: This response references documents with moderate sensitivity. "
            "Please handle according to your organization's data policies."
        )
        return response + disclaimer

    return response


async def synthesize_answer(state: GraphState) -> dict:
    """Synthesize a comprehensive answer from relevant documents with citations.

    Builds context from relevant (or all available) documents, prompts the LLM
    to generate a cited answer, and extracts structured citation metadata.

    Args:
        state: Current graph state with relevant_documents and query.

    Returns:
        Partial state update with generation, citations, and audit_trail entry.
    """
    query = state.get("rewritten_query") or state["query"]
    relevant_documents = state.get("relevant_documents", [])
    all_documents = state.get("documents", [])

    # Use relevant docs preferably, fall back to all if retries exhausted
    docs_to_use = relevant_documents if relevant_documents else all_documents

    logger.info("synthesizing_answer", doc_count=len(docs_to_use))

    if not docs_to_use:
        generation = (
            "I was unable to find relevant documents to answer your question. "
            "Please try rephrasing your query or check that the relevant "
            "documents have been ingested."
        )
        return {
            "generation": generation,
            "citations": [],
            "audit_trail": [
                {
                    "node": "synthesizer",
                    "action": "synthesize_answer",
                    "doc_count": 0,
                    "generation_len": len(generation),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            ],
        }

    max_sensitivity = _max_sensitivity(docs_to_use)

    # Build prompt and call LLM with inference routing
    prompt = _build_synthesis_prompt(query, docs_to_use, max_sensitivity)
    response = await call_llm_async(
        prompt,
        system_prompt="You are an expert research assistant that always cites sources.",
        sensitivity_level=max_sensitivity,
    )

    if not response.strip():
        response = "Unable to generate a response. Please try again."

    # Extract citations
    citations = _extract_citations(response, docs_to_use)

    # Add disclaimers based on sensitivity
    generation = _add_disclaimers(response, max_sensitivity)

    # Compute preliminary confidence score for the evaluator to refine
    confidence_score = _compute_synthesis_confidence(docs_to_use, citations, generation)

    logger.info(
        "answer_synthesized",
        generation_len=len(generation),
        citation_count=len(citations),
        sensitivity=max_sensitivity,
        preliminary_confidence=confidence_score,
    )

    return {
        "generation": generation,
        "citations": citations,
        "confidence_score": confidence_score,
        "audit_trail": [
            {
                "node": "synthesizer",
                "action": "synthesize_answer",
                "doc_count": len(docs_to_use),
                "citation_count": len(citations),
                "sensitivity": max_sensitivity,
                "generation_len": len(generation),
                "preliminary_confidence": confidence_score,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ],
    }


async def synthesize_answer_stream(state: GraphState) -> AsyncGenerator[dict, None]:
    """Streaming variant of synthesize_answer.

    Yields events as the LLM generates tokens, then a final event with
    parsed citations, disclaimers and preliminary confidence.

    Event shapes:
        {"type": "token", "text": str}
        {"type": "final", "generation": str, "citations": [...],
         "confidence_score": float, "audit_entry": {...}}

    Args:
        state: Current graph state with relevant_documents (or documents fallback)
            and query / rewritten_query.

    Yields:
        Event dicts as described above.
    """
    query = state.get("rewritten_query") or state["query"]
    relevant_documents = state.get("relevant_documents", [])
    all_documents = state.get("documents", [])
    docs_to_use = relevant_documents if relevant_documents else all_documents

    logger.info("synthesizing_answer_stream", doc_count=len(docs_to_use))

    if not docs_to_use:
        generation = (
            "I was unable to find relevant documents to answer your question. "
            "Please try rephrasing your query or check that the relevant "
            "documents have been ingested."
        )
        yield {"type": "token", "text": generation}
        yield {
            "type": "final",
            "generation": generation,
            "citations": [],
            "confidence_score": 0.0,
            "audit_entry": {
                "node": "synthesizer",
                "action": "synthesize_answer_stream",
                "doc_count": 0,
                "generation_len": len(generation),
                "timestamp": datetime.now(UTC).isoformat(),
            },
        }
        return

    max_sensitivity = _max_sensitivity(docs_to_use)
    prompt = _build_synthesis_prompt(query, docs_to_use, max_sensitivity)

    collected: list[str] = []
    async for token in call_llm_stream(
        prompt,
        system_prompt="You are an expert research assistant that always cites sources.",
        sensitivity_level=max_sensitivity,
    ):
        collected.append(token)
        yield {"type": "token", "text": token}

    raw_response = "".join(collected).strip()
    if not raw_response:
        raw_response = "Unable to generate a response. Please try again."
        yield {"type": "token", "text": raw_response}

    citations = _extract_citations(raw_response, docs_to_use)
    generation = _add_disclaimers(raw_response, max_sensitivity)

    # Emit disclaimer suffix as a final token so UI sees full text
    disclaimer_suffix = generation[len(raw_response) :]
    if disclaimer_suffix:
        yield {"type": "token", "text": disclaimer_suffix}

    confidence_score = _compute_synthesis_confidence(docs_to_use, citations, generation)

    logger.info(
        "answer_synthesized_stream",
        generation_len=len(generation),
        citation_count=len(citations),
        sensitivity=max_sensitivity,
        preliminary_confidence=confidence_score,
    )

    yield {
        "type": "final",
        "generation": generation,
        "citations": citations,
        "confidence_score": confidence_score,
        "audit_entry": {
            "node": "synthesizer",
            "action": "synthesize_answer_stream",
            "doc_count": len(docs_to_use),
            "citation_count": len(citations),
            "sensitivity": max_sensitivity,
            "generation_len": len(generation),
            "preliminary_confidence": confidence_score,
            "timestamp": datetime.now(UTC).isoformat(),
        },
    }
