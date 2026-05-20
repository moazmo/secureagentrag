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


_CITATION_MARKER_RE = re.compile(r"\[\[?\d+\]?\]")
"""Match both `[N]` and `[[N]]` citation markers used by the synthesizer."""


def _compute_citation_coverage(generation: str, citations: list[Citation]) -> float:
    """Compute what fraction of the response is backed by citation markers.

    A response is considered well-cited when most non-trivial sentences carry
    a `[N]` or `[[N]]` marker linking back to a source. Very short sentences
    (transition phrases, list intros) are excluded from the denominator so a
    well-cited answer with a few connective sentences is not penalised.

    Args:
        generation: The generated response text.
        citations: List of extracted citations.

    Returns:
        Coverage ratio between 0.0 and 1.0.
    """
    if not generation or not citations:
        return 0.0

    # Split on both sentence terminators and bullet/line breaks so each
    # bullet in a markdown answer is one "claim".
    units = re.split(r"[.!?]+\s+|\n[-*]\s+|\n\d+\.\s+", generation)
    # Substantive = unit has >=5 words. Drops bullet labels and transitions.
    substantive = [u.strip() for u in units if len(u.strip().split()) >= 5]
    if not substantive:
        return 0.0

    cited = sum(1 for u in substantive if _CITATION_MARKER_RE.search(u))
    raw_density = cited / len(substantive)

    # Scoring curve: full credit at 50% density. A well-grounded answer
    # with citations on half of its substantive claims (plus the rest
    # being recap/structure) earns a 1.0 here.
    return min(1.0, raw_density / 0.5)


def _compute_evidence_strength(citations: list[Citation], documents: list[DocumentGrade]) -> float:
    """Compute how thoroughly the answer draws on the retrieved corpus.

    Old implementation averaged the `relevance_score` field on citations, but
    that field holds the Reciprocal Rank Fusion score (typically 0.01-0.05),
    which after normalisation collapsed to ~0 every time. Replaced with a
    source-coverage signal: ratio of cited documents to documents available
    to cite, capped at 1.0. Encourages the synthesizer to use multiple
    sources rather than recycling one chunk.

    Args:
        citations: Extracted citations.
        documents: All retrieved documents the synthesizer had access to.

    Returns:
        Evidence strength score between 0.0 and 1.0.
    """
    if not citations:
        return 0.0
    if not documents:
        # No documents available means nothing to credit; treat citations as
        # presence-only evidence.
        return min(1.0, len(citations) / 3.0)

    # De-duplicate by chunk (source_file + page + first 60 chars of chunk text)
    # so 3 cites of the same chunk don't inflate the score, but cites of
    # different chunks within the same file still count as breadth.
    # Target = 3 unique chunks for full credit; smaller corpora are not
    # penalised for having fewer total docs.
    unique_chunks = {
        (
            c.get("source_file"),
            c.get("page_number"),
            (c.get("chunk_text") or "")[:60],
        )
        for c in citations
    }
    target = max(1, min(len(documents), 3))
    return min(1.0, len(unique_chunks) / target)


def _get_hallucination_check_prompt(query: str, answer: str, context: str) -> str:
    """Build prompt for hallucination detection.

    Uses a strict structured output (CLAIM markers) so the parser does not
    have to guess between preamble and actual unsupported claims.

    Args:
        query: User query.
        answer: Generated answer.
        context: Retrieved document excerpts.

    Returns:
        Formatted prompt string.
    """
    return (
        "You are a conservative fact-checking assistant. Only flag claims that "
        "directly contradict the context or introduce specific facts (names, "
        "numbers, dates, quotes) that are not present in the context. Do NOT "
        "flag general statements, summaries, paraphrases, or commonly-known "
        "background information — those are acceptable.\n\n"
        "STRICT OUTPUT FORMAT (no preamble, no reasoning, no `<think>` blocks):\n"
        "- If every specific factual claim is supported by the context, output "
        "exactly:\n"
        "    NONE\n"
        "- Otherwise output one line per unsupported claim, each prefixed with "
        "the marker `CLAIM:` and nothing else:\n"
        "    CLAIM: <short description of the unsupported claim>\n\n"
        "EXAMPLES:\n"
        "- Context says 'revenue grew 12%'. Answer says 'revenue grew 12%'. "
        "Output: NONE\n"
        "- Context says 'revenue grew 12%'. Answer says 'revenue grew 18%'. "
        "Output: CLAIM: Revenue figure 18% contradicts context (12%).\n"
        "- Context describes data classes. Answer adds general framing like "
        "'Access control is important'. Output: NONE\n\n"
        f"Context:\n{context[:1500]}\n\n"
        f"Generated Answer:\n{answer[:800]}\n\n"
        "Output:"
    )


def _get_completeness_prompt(query: str, answer: str) -> str:
    """Build prompt for answer completeness check.

    Calibrated for retrieval-grounded answers: a focused, factually correct
    answer that addresses the question with citations earns a high score even
    when it is short. Stylistic perfection is not the bar — coverage of the
    question's intent is.

    Args:
        query: User query.
        answer: Generated answer.

    Returns:
        Formatted prompt string.
    """
    return (
        "You are evaluating whether an answer addresses a user's question, "
        "given that the answer must be grounded in retrieved documents.\n\n"
        "Score the answer on a 0.0-1.0 scale based ONLY on whether it covers "
        "what the question asks. Do NOT penalise for brevity, formatting, or "
        "style — only for missing or incorrect coverage of the asked topics.\n\n"
        "- 1.0: Every part of the question is addressed.\n"
        "- 0.8: Main question fully addressed; minor sub-aspects missing.\n"
        "- 0.6: Question is addressed but with meaningful gaps.\n"
        "- 0.4: Partial answer — some aspects covered, some missing.\n"
        "- 0.2: Answer is off-topic or barely addresses the question.\n\n"
        f"Question: {query}\n\n"
        f"Answer: {answer[:1200]}\n\n"
        "Respond with ONLY a single decimal number (e.g. `0.8`), no explanation."
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

    Parser is strict: only lines starting with ``CLAIM:`` are counted.
    Free-text preamble, reasoning, and reasoning-mode ``<think>`` blocks
    are ignored so chatty models do not produce false-positive hallucination
    counts. ``NONE`` (case-insensitive, anywhere on its own line) shortcuts
    to zero.

    Args:
        response: LLM response (structured per ``_get_hallucination_check_prompt``).

    Returns:
        Number of unsupported claims (0 if no CLAIM lines found).
    """
    if not response or not response.strip():
        return 0

    # Strip reasoning-model think blocks (e.g., Qwen3 thinking mode).
    no_think = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE)

    # Explicit NONE shortcut.
    for line in no_think.splitlines():
        stripped = line.strip().rstrip(".").upper()
        if stripped == "NONE":
            return 0

    # Count CLAIM: lines (the strict format requested in the prompt).
    claim_lines = [
        line for line in no_think.splitlines() if re.match(r"^\s*CLAIM\s*:", line, re.IGNORECASE)
    ]
    return len(claim_lines)


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
    context_str = "\n---\n".join(doc.get("text", "")[:300] for doc in docs_to_use[:5])

    # Run hallucination and completeness checks in parallel
    import asyncio

    hallucination_prompt = _get_hallucination_check_prompt(query, generation, context_str)
    completeness_prompt = _get_completeness_prompt(query, generation)

    # Evaluator routing: respects user's prefer_cloud flag like every other
    # agent. The default sensitivity is "medium" (the answer + retrieved
    # context have already been seen by the synthesizer, which itself
    # routed based on sensitivity), so when the user opts into cloud, eval
    # follows. HIGH-sensitivity content still pins local via the router's
    # internal gate.
    prefer_cloud = state.get("prefer_cloud", False)
    doc_sens = state.get("query_sensitivity", "low")
    if any((d.get("metadata", {}) or {}).get("sensitivity_level") == "high" for d in docs_to_use):
        doc_sens = "high"
    eval_sensitivity = doc_sens

    hallucination_task = call_llm_async(
        hallucination_prompt,
        system_prompt="You are a strict fact-checking assistant.",
        sensitivity_level=eval_sensitivity,
        prefer_cloud=prefer_cloud,
    )
    completeness_task = call_llm_async(
        completeness_prompt,
        system_prompt="You are an answer quality evaluator.",
        sensitivity_level=eval_sensitivity,
        prefer_cloud=prefer_cloud,
    )

    hallucination_response, completeness_response = await asyncio.gather(
        hallucination_task, completeness_task
    )

    hallucination_count = _count_hallucinations(hallucination_response)
    completeness_score = _parse_score(completeness_response)

    # ── Calibrated Confidence Score ─────────────────────────────────────────
    # Weights reward what local 8B-class models actually do well: citing
    # sources, producing complete answers, and (when the NLI gate is on)
    # producing sentences the cited chunks actually entail.
    #
    # When SAR_FAITHFULNESS_GATE_ENABLED=true the NLI ratio replaces the
    # weaker self-fact-check signal because faithfulness has been measured
    # against the actual source, not the LLM's recollection of it.
    #
    # Citation coverage:   30%  (strongest grounding signal)
    # Evidence strength:   15%  (source-coverage breadth)
    # Completeness:        30%  (LLM-graded against the query)
    # Faithfulness:        25%  (NLI gate or hallucination penalty)
    hallucination_penalty = max(0.0, 1.0 - (hallucination_count * 0.15))
    faithfulness_ratio = float(state.get("faithfulness_ratio", 1.0))
    if settings.faithfulness_gate_enabled:
        faithfulness_signal = faithfulness_ratio
    else:
        faithfulness_signal = hallucination_penalty

    confidence_score = (
        citation_coverage * 0.30
        + evidence_strength * 0.15
        + completeness_score * 0.30
        + faithfulness_signal * 0.25
    )
    confidence_score = round(max(0.0, min(1.0, confidence_score)), 3)

    # Human review triggers on low overall confidence OR (when the gate is
    # on) faithfulness ratio below threshold. The NLI gate is a deterministic
    # source-grounded signal, so a failure there is reliable enough to flag
    # by itself.
    faithfulness_below_threshold = (
        settings.faithfulness_gate_enabled and faithfulness_ratio < settings.faithfulness_threshold
    )
    needs_human_review = (
        confidence_score < settings.confidence_threshold or faithfulness_below_threshold
    )

    # Build detailed evaluation notes
    notes_parts: list[str] = []
    if faithfulness_below_threshold:
        unsupported_count = len(state.get("faithfulness_unsupported", []) or [])
        notes_parts.append(
            f"🛡️ Faithfulness {faithfulness_ratio:.0%} < threshold "
            f"{settings.faithfulness_threshold:.0%} "
            f"({unsupported_count} unsupported claim(s))."
        )
    if hallucination_count > 0:
        notes_parts.append(
            f"⚠️ {hallucination_count} potentially unsupported claim(s) detected. "
            "Verify against source documents."
        )
    if citation_coverage < 0.5:
        notes_parts.append(
            f"📎 Low citation coverage ({citation_coverage:.0%}). Many claims lack source backing."
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
            f"Info: Moderate confidence ({confidence_score:.0%}). " + " ".join(notes_parts)
            if notes_parts
            else "Answer appears reasonable with adequate support."
        )
    else:
        base_note = f"⚠️ Low confidence ({confidence_score:.0%}). Human review recommended."
        evaluation_notes = base_note + " " + " ".join(notes_parts) if notes_parts else base_note

    logger.info(
        "response_evaluated",
        confidence_score=confidence_score,
        citation_coverage=round(citation_coverage, 3),
        evidence_strength=round(evidence_strength, 3),
        completeness=round(completeness_score, 3),
        hallucinations=hallucination_count,
        faithfulness_ratio=round(faithfulness_ratio, 3),
        faithfulness_gated=settings.faithfulness_gate_enabled,
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
                "faithfulness_ratio": round(faithfulness_ratio, 3),
                "faithfulness_gated": settings.faithfulness_gate_enabled,
                "faithfulness_below_threshold": faithfulness_below_threshold,
                "needs_human_review": needs_human_review,
                "evaluation_notes": evaluation_notes,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ],
    }
