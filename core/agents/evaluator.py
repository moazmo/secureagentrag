"""Response evaluation and confidence scoring agent.

Performs multi-dimensional quality assessment:
1. Citation coverage — what fraction of claims are backed by sources
2. Hallucination detection — claims not supported by retrieved documents
3. Answer completeness — whether all parts of the query were addressed
4. Confidence calibration — statistical confidence based on evidence strength
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

from config.settings import settings
from core.agents.router import call_llm_async
from core.state import Citation, DocumentGrade, GraphState  # noqa: TC001
from utils.logging import get_logger

logger = get_logger(__name__)


def _compute_citation_coverage(generation: str, citations: list[Citation]) -> float:
    """Compute what fraction of the response is backed by citations.

    Heuristic: count sentences in the generation and check which ones
    have citation markers. A well-cited answer should have most factual
    claims backed by [N] markers.

    Args:
        generation: The generated response text.
        citations: List of extracted citations.

    Returns:
        Coverage ratio between 0.0 and 1.0.
    """
    if not generation or not citations:
        return 0.0

    # Split into sentences (simple heuristic)
    sentences = re.split(r"[.!?]+\s+", generation)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return 0.0

    # Count sentences that have at least one citation marker
    cited_sentences = 0
    for sentence in sentences:
        if re.search(r"\[\d+\]", sentence):
            cited_sentences += 1

    # Also factor in: do we have citations for all document sources used?
    # A perfect score requires both sentence-level and source-level coverage
    sentence_coverage = cited_sentences / len(sentences)

    # Penalize if very few citations relative to document count
    # (extracted citations vs documents that could have been cited)
    return min(1.0, sentence_coverage)


def _compute_evidence_strength(
    citations: list[Citation], documents: list[DocumentGrade]
) -> float:
    """Compute average relevance score of cited documents.

    Higher relevance scores in the retrieved documents suggest stronger
    evidence backing the answer.

    Args:
        citations: Extracted citations with relevance scores.
        documents: All retrieved documents with scores.

    Returns:
        Evidence strength score between 0.0 and 1.0.
    """
    if not citations:
        return 0.0

    # Average relevance score of cited documents
    scores = [c.get("relevance_score", 0.0) for c in citations if c.get("relevance_score")]
    if not scores:
        return 0.0

    avg_score = sum(scores) / len(scores)
    # Normalize: retrieval scores are typically 0.5-0.9, scale to 0-1
    return min(1.0, max(0.0, (avg_score - 0.3) / 0.6))


def _get_hallucination_check_prompt(
    query: str, answer: str, context: str
) -> str:
    """Build prompt for hallucination detection.

    Args:
        query: User query.
        answer: Generated answer.
        context: Retrieved document excerpts.

    Returns:
        Formatted prompt string.
    """
    return (
        "You are a fact-checking assistant. Your task is to identify claims in the "
        "generated answer that are NOT supported by the provided context.\n\n"
        "Instructions:\n"
        "1. Read the context carefully.\n"
        "2. Read the generated answer.\n"
        "3. List any claims in the answer that cannot be verified from the context.\n"
        "4. If the answer contains no unsupported claims, respond with 'NONE'.\n\n"
        f"Context:\n{context[:1500]}\n\n"
        f"Generated Answer:\n{answer[:800]}\n\n"
        "Unsupported claims (one per line, or 'NONE' if all claims are supported):"
    )


def _get_completeness_prompt(query: str, answer: str) -> str:
    """Build prompt for answer completeness check.

    Args:
        query: User query.
        answer: Generated answer.

    Returns:
        Formatted prompt string.
    """
    return (
        "You are evaluating whether an answer fully addresses a user's question.\n\n"
        "Rate the completeness on a scale of 0.0 to 1.0:\n"
        "- 1.0: Answer addresses ALL parts of the question completely.\n"
        "- 0.7-0.9: Answer addresses most parts but may miss minor aspects.\n"
        "- 0.4-0.6: Answer addresses some parts but misses significant aspects.\n"
        "- 0.0-0.3: Answer fails to address the question or is off-topic.\n\n"
        f"Question: {query}\n\n"
        f"Answer: {answer[:800]}\n\n"
        "Respond with ONLY a decimal number between 0.0 and 1.0."
    )


def _parse_score(response: str) -> float:
    """Parse a numeric score from LLM response.

    Args:
        response: Raw LLM response text.

    Returns:
        Float score clamped between 0.0 and 1.0.
    """
    try:
        cleaned = response.strip()
        match = re.search(r"(\d+\.?\d*)", cleaned)
        if match:
            score = float(match.group(1))
            if score > 1.0:
                score = score / 100.0
            return max(0.0, min(1.0, score))
    except (ValueError, AttributeError):
        pass
    return 0.5


def _count_hallucinations(response: str) -> int:
    """Count number of hallucinated claims from LLM response.

    Args:
        response: LLM response listing unsupported claims.

    Returns:
        Number of unsupported claims (0 if response is 'NONE').
    """
    cleaned = response.strip().upper()
    if cleaned == "NONE" or cleaned.startswith("NONE"):
        return 0
    # Count non-empty lines
    lines = [line.strip() for line in response.split("\n") if line.strip()]
    return len(lines)


async def evaluate_response(state: GraphState) -> dict:
    """Evaluate the generated response with multi-dimensional quality assessment.

    Computes:
    - Citation coverage: fraction of claims backed by sources
    - Evidence strength: average relevance of cited documents
    - Hallucination count: claims not supported by context
    - Completeness: whether all parts of the query were addressed
    - Calibrated confidence: weighted combination of above metrics

    Args:
        state: Current graph state with generation and relevant_documents.

    Returns:
        Partial state update with confidence_score, needs_human_review,
        evaluation_notes, and audit_trail entry.
    """
    query = state.get("rewritten_query") or state["query"]
    generation = state.get("generation", "")
    citations = state.get("citations", [])
    relevant_documents = state.get("relevant_documents", [])
    all_documents = state.get("documents", [])
    docs_to_use = relevant_documents if relevant_documents else all_documents

    logger.info(
        "evaluating_response",
        generation_len=len(generation),
        doc_count=len(docs_to_use),
        citation_count=len(citations),
    )

    # ── Metric 1: Citation Coverage (heuristic, no LLM call) ────────────────
    citation_coverage = _compute_citation_coverage(generation, citations)

    # ── Metric 2: Evidence Strength (heuristic, no LLM call) ────────────────
    evidence_strength = _compute_evidence_strength(citations, docs_to_use)

    # ── Metric 3 & 4: Hallucination Check + Completeness (batched LLM) ──────
    context_str = "\n---\n".join(
        doc.get("text", "")[:300] for doc in docs_to_use[:5]
    )

    # Run hallucination and completeness checks in parallel
    import asyncio

    hallucination_prompt = _get_hallucination_check_prompt(
        query, generation, context_str
    )
    completeness_prompt = _get_completeness_prompt(query, generation)

    hallucination_task = call_llm_async(
        hallucination_prompt,
        system_prompt="You are a strict fact-checking assistant.",
        sensitivity_level="high",  # Always local for evaluation
    )
    completeness_task = call_llm_async(
        completeness_prompt,
        system_prompt="You are an answer quality evaluator.",
        sensitivity_level="high",
    )

    hallucination_response, completeness_response = await asyncio.gather(
        hallucination_task, completeness_task
    )

    hallucination_count = _count_hallucinations(hallucination_response)
    completeness_score = _parse_score(completeness_response)

    # ── Calibrated Confidence Score ─────────────────────────────────────────
    # Weighted combination of all metrics
    # Citation coverage: 25% | Evidence strength: 20% | Completeness: 30% | Hallucination penalty: 25%
    hallucination_penalty = max(0.0, 1.0 - (hallucination_count * 0.3))

    confidence_score = (
        citation_coverage * 0.25
        + evidence_strength * 0.20
        + completeness_score * 0.30
        + hallucination_penalty * 0.25
    )
    confidence_score = round(max(0.0, min(1.0, confidence_score)), 3)

    # Determine if human review is needed
    needs_human_review = (
        confidence_score < settings.confidence_threshold
        or hallucination_count > 0
        or citation_coverage < 0.3
    )

    # Build detailed evaluation notes
    notes_parts: list[str] = []
    if hallucination_count > 0:
        notes_parts.append(
            f"⚠️ {hallucination_count} potentially unsupported claim(s) detected. "
            "Verify against source documents."
        )
    if citation_coverage < 0.5:
        notes_parts.append(
            f"📎 Low citation coverage ({citation_coverage:.0%}). "
            "Many claims lack source backing."
        )
    if completeness_score < 0.5:
        notes_parts.append(
            f"❓ Answer may be incomplete ({completeness_score:.0%}). "
            "Some aspects of the query may not be addressed."
        )

    if confidence_score >= 0.8 and not notes_parts:
        evaluation_notes = (
            f"✅ High confidence ({confidence_score:.0%}). Well-cited, complete, "
            f"and supported by strong evidence."
        )
    elif confidence_score >= 0.6:
        evaluation_notes = (
            f"Info: Moderate confidence ({confidence_score:.0%}). "
            + " ".join(notes_parts)
            if notes_parts
            else "Answer appears reasonable with adequate support."
        )
    else:
        base_note = (
            f"⚠️ Low confidence ({confidence_score:.0%}). Human review recommended."
        )
        evaluation_notes = base_note + " " + " ".join(notes_parts) if notes_parts else base_note

    logger.info(
        "response_evaluated",
        confidence_score=confidence_score,
        citation_coverage=round(citation_coverage, 3),
        evidence_strength=round(evidence_strength, 3),
        completeness=round(completeness_score, 3),
        hallucinations=hallucination_count,
        needs_human_review=needs_human_review,
    )

    return {
        "confidence_score": confidence_score,
        "needs_human_review": needs_human_review,
        "evaluation_notes": evaluation_notes,
        "audit_trail": [
            {
                "node": "evaluator",
                "action": "evaluate_response",
                "confidence_score": confidence_score,
                "citation_coverage": round(citation_coverage, 3),
                "evidence_strength": round(evidence_strength, 3),
                "completeness": round(completeness_score, 3),
                "hallucinations": hallucination_count,
                "needs_human_review": needs_human_review,
                "evaluation_notes": evaluation_notes,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ],
    }



