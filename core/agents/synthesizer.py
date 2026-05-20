"""Answer synthesis agent with mandatory citations."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, ClassVar

from config.settings import settings
from core.agents.router import call_llm_stream, call_llm_with_decision
from core.state import Citation, GraphState  # noqa: TC001
from utils.logging import get_logger

if TYPE_CHECKING:
    from core.state import DocumentGrade

logger = get_logger(__name__)


_SENSITIVITY_RANK = {"low": 1, "medium": 2, "high": 3}


def _max_label(*labels: str) -> str:
    """Return the highest sensitivity label across the inputs."""
    rank = max((_SENSITIVITY_RANK.get(lbl, 1) for lbl in labels), default=1)
    for label, value in _SENSITIVITY_RANK.items():
        if value == rank:
            return label
    return "low"


def _max_sensitivity(docs_to_use: list[DocumentGrade]) -> str:
    """Determine highest sensitivity level among the documents used.

    Args:
        docs_to_use: Documents that will be fed as synthesis context.

    Returns:
        "high" | "medium" | "low".
    """
    levels = [doc.get("metadata", {}).get("sensitivity_level", "low") for doc in docs_to_use]
    return _max_label(*levels) if levels else "low"


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
        "You are an expert research assistant. Answer the user's question using "
        "ONLY the provided context. Follow these citation rules strictly:\n\n"
        "CITATION RULES:\n"
        "1. Every factual statement MUST end with a citation marker `[N]` where "
        "N is the source number from the Context list below.\n"
        "2. If two sources support a claim, cite both: `... [1][3]`.\n"
        "3. Do NOT use double brackets, footnotes, or any other format. Just `[N]`.\n"
        "4. Do NOT write a 'Sources:' or 'References:' section at the end — the "
        "system extracts citations automatically from inline markers.\n"
        "5. If the context lacks information to answer fully, say so explicitly "
        "rather than inventing details.\n\n"
        "STYLE:\n"
        "- Be concise but complete. Cover every part of the question.\n"
        "- Use short paragraphs or bullet points for readability.\n"
        "- Do not preface the answer with phrases like 'Based on the context'.\n"
        "- Do not include `<think>` or reasoning trace blocks in the output.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {query}\n"
        f"{sensitivity_instruction}\n\n"
        "Answer (with inline `[N]` citations on every factual claim):"
    )


def _build_json_synthesis_prompt(
    query: str, documents: list[DocumentGrade], sensitivity: str
) -> str:
    """Build a JSON-mode synthesis prompt requesting structured output.

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
        "You are an expert research assistant. Answer the user's question using "
        "ONLY the provided context. You MUST respond with a single valid JSON object "
        "and nothing else. Do not wrap the JSON in markdown code blocks.\n\n"
        "The JSON object must have exactly these two fields:\n"
        '- "answer": a string with the full answer text. Every factual statement '
        "must end with an inline citation marker `[N]` where N is the source number.\n"
        '- "citations": a list of integers (source numbers) that were cited, '
        "in the order they first appear in the answer. Each integer must be >= 1.\n\n"
        "CITATION RULES:\n"
        "1. Every factual statement MUST end with a citation marker `[N]`.\n"
        "2. If two sources support a claim, cite both: `... [1][3]`.\n"
        "3. Do NOT use double brackets, footnotes, or any other format.\n"
        "4. Do NOT write a 'Sources:' or 'References:' section.\n"
        "5. If the context lacks information to answer fully, say so explicitly.\n\n"
        "STYLE:\n"
        "- Be concise but complete.\n"
        "- Use short paragraphs or bullet points.\n"
        "- Do not preface the answer with phrases like 'Based on the context'.\n"
        "- Do not include `<think>` or reasoning trace blocks.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {query}\n"
        f"{sensitivity_instruction}\n\n"
        "Respond with ONLY valid JSON in this exact format: "
        '{"answer": "...", "citations": [1, 3]}'
    )


def _extract_citations(response: str, documents: list[DocumentGrade]) -> list[Citation]:
    """Extract citation references from the LLM response.

    Parses `[N]` citation markers (the format the synthesizer is prompted to
    produce) and the legacy `[[N]]` form. Skips markdown link syntax `[text](url)`
    by requiring the bracket to NOT be followed by `(`. Strips reasoning-mode
    `<think>...</think>` blocks before extraction so think-stream citations
    do not leak into the output.

    Args:
        response: The generated response text.
        documents: The list of documents used as context.

    Returns:
        List of Citation TypedDicts with source information, in citation order.
    """
    # Drop reasoning blocks before extraction.
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE)

    # Match `[[N]]` (legacy) first, then `[N]` (current canonical form).
    # `(?!\s*\()` excludes markdown link syntax `[text](url)`.
    citation_refs = re.findall(r"\[\[(\d+)\]\]|\[(\d+)\](?!\s*\()", cleaned)
    # Each tuple has one populated group; take whichever is non-empty.
    citation_refs = [a or b for a, b in citation_refs]

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


def _extract_json_citations(
    response: str, documents: list[DocumentGrade]
) -> tuple[str, list[Citation]]:
    """Parse JSON-mode response and extract answer text plus citations.

    Falls back to regex extraction if the response is not valid JSON or
    lacks the expected fields.

    Args:
        response: The generated response text (expected to be JSON).
        documents: The list of documents used as context.

    Returns:
        Tuple of (answer_text, citations). If JSON parsing fails, answer_text
        is empty and citations come from regex fallback.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("\n", 1)[0]

    cleaned = cleaned.strip()
    if not cleaned:
        return "", _extract_citations(response, documents)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return "", _extract_citations(response, documents)

    if not isinstance(data, dict):
        return "", _extract_citations(response, documents)

    answer = data.get("answer", "")
    if not isinstance(answer, str):
        answer = str(answer)

    citations: list[Citation] = []
    seen_indices: set[int] = set()
    raw_citations = data.get("citations", [])
    if not isinstance(raw_citations, list):
        raw_citations = []

    for ref in raw_citations:
        if not isinstance(ref, int):
            try:
                ref = int(ref)
            except (ValueError, TypeError):
                continue
        idx = ref - 1
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

    if not citations:
        fallback_citations = _extract_citations(answer, documents)
        if fallback_citations:
            citations = fallback_citations

    return answer, citations


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


def _maybe_get_stream_writer(state: GraphState):
    """Return a LangGraph stream writer iff the caller opted into streaming.

    LangGraph 1.x binds a writer in every node context (no-op when no
    consumer is listening), so writer-presence alone is not a reliable
    signal. Instead we look at the caller-set ``_stream`` flag — only
    ``run_rag_pipeline_stream`` flips it to True before invocation. This
    keeps ``synthesize_answer`` deterministic from a single dispatch
    signal we control.
    """
    if not state.get("_stream"):
        return None
    try:
        from langgraph.config import get_stream_writer  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        return get_stream_writer()
    except Exception:
        return None


async def synthesize_answer(state: GraphState) -> dict:
    """Synthesize a comprehensive answer from relevant documents with citations.

    Two execution modes share this single node so the streaming and
    non-streaming pipelines stay byte-identical in behaviour:

    * **Streaming** — when invoked via ``graph.astream(stream_mode="custom")``
      a LangGraph stream writer is available; we call the underlying
      ``call_llm_stream`` and push each token through the writer as
      ``{"type": "token", "text": ...}``.
    * **Single-shot** — when invoked via ``graph.ainvoke`` or direct unit
      tests, no writer is bound, so we issue one ``call_llm_with_decision``
      and return the full text.

    Both branches converge on the same return dict (generation, citations,
    confidence_score, synth_provider/model/usage/latency_ms, audit_trail)
    so downstream nodes never need to know which path ran.

    Args:
        state: Current graph state with relevant_documents and query.

    Returns:
        Partial state update with generation, citations, and audit_trail entry.
    """
    query = state.get("rewritten_query") or state["query"]
    relevant_documents = state.get("relevant_documents", [])
    all_documents = state.get("documents", [])
    retry_count = state.get("retry_count", 0)

    # Corrective RAG: only synthesize from documents the grader judged relevant.
    # Falling back to all_documents when relevant_documents is empty defeats the
    # whole point of the grader + rewrite loop — we would synthesize from text
    # we already decided was off-topic. Refuse instead.
    docs_to_use = relevant_documents

    logger.info(
        "synthesizing_answer",
        doc_count=len(docs_to_use),
        retrieved_total=len(all_documents),
        retries=retry_count,
    )

    if not docs_to_use:
        # Distinguish "nothing retrieved at all" from "retrieved but all
        # judged irrelevant after retries". The user-facing message is the
        # same — but the audit trail records the real reason.
        if not all_documents:
            refuse_reason = "no_documents_retrieved"
            generation = (
                "I was unable to find any documents matching your question. "
                "Please check that the relevant documents have been ingested "
                "and that you have permission to access them."
            )
        else:
            refuse_reason = "all_documents_off_topic"
            generation = (
                "I retrieved documents but none were judged relevant to your "
                "question after corrective retries. Please try rephrasing the "
                "query with more specific terms, or confirm that the indexed "
                "corpus actually covers this topic."
            )
        return {
            "generation": generation,
            "citations": [],
            "confidence_score": 0.0,
            "audit_trail": [
                {
                    "node": "synthesizer",
                    "action": "refuse",
                    "reason": refuse_reason,
                    "doc_count": 0,
                    "retrieved_total": len(all_documents),
                    "retries": retry_count,
                    "generation_len": len(generation),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            ],
        }

    doc_sensitivity = _max_sensitivity(docs_to_use)
    query_sensitivity = state.get("query_sensitivity", "low")
    max_sensitivity = _max_label(doc_sensitivity, query_sensitivity)
    prefer_cloud = state.get("prefer_cloud", False)

    # Build prompt and call LLM with inference routing. prefer_cloud only
    # takes effect for LOW/MEDIUM sensitivity — HIGH always routes local.
    # max_sensitivity is the higher of (doc sensitivity, query sensitivity)
    # so a sensitive QUERY against low-tagged docs still routes local.
    json_mode = settings.json_citations_enabled
    if json_mode:
        prompt = _build_json_synthesis_prompt(query, docs_to_use, max_sensitivity)
    else:
        prompt = _build_synthesis_prompt(query, docs_to_use, max_sensitivity)

    writer = _maybe_get_stream_writer(state)
    if writer is not None:
        # Streaming path — same node, just pushes tokens through the
        # LangGraph writer as they arrive. Provenance is resolved up-front
        # from the InferenceRouter (it's pure / cheap) so the audit_trail
        # carries the provider/model even though we never see the
        # underlying LLMResponse object.
        from inference.router import InferenceRouter

        stream_decision = InferenceRouter().route(
            sensitivity_level=max_sensitivity, prefer_cloud=prefer_cloud
        )
        import time as _time

        t0 = _time.perf_counter()
        collected: list[str] = []
        async for token in call_llm_stream(
            prompt,
            system_prompt="You are an expert research assistant that always cites sources.",
            sensitivity_level=max_sensitivity,
            prefer_cloud=prefer_cloud,
        ):
            collected.append(token)
            writer({"type": "token", "text": token})
        stream_latency_ms = (_time.perf_counter() - t0) * 1000

        response = "".join(collected).strip() or "Unable to generate a response. Please try again."
        decision = stream_decision
        # Synthesise an LLMResponse-shape stub so the downstream code can
        # read .latency_ms and .usage uniformly.

        class _StubResp:
            usage: ClassVar[dict] = {}
            latency_ms: float = stream_latency_ms

        llm_response = _StubResp()
    else:
        response_text, decision, llm_response = await call_llm_with_decision(
            prompt,
            system_prompt="You are an expert research assistant that always cites sources.",
            sensitivity_level=max_sensitivity,
            prefer_cloud=prefer_cloud,
            json_mode=json_mode,
        )
        response = response_text
        if not response.strip():
            response = "Unable to generate a response. Please try again."

    # Extract citations
    if json_mode:
        answer_text, citations = _extract_json_citations(response, docs_to_use)
        if not answer_text.strip():
            answer_text = response
        generation = _add_disclaimers(answer_text, max_sensitivity)
    else:
        citations = _extract_citations(response, docs_to_use)
        generation = _add_disclaimers(response, max_sensitivity)

    # On the streaming path, push the disclaimer suffix through so the UI
    # sees the final, complete text.
    if writer is not None:
        disclaimer_suffix = generation[len(response) :]
        if disclaimer_suffix:
            writer({"type": "token", "text": disclaimer_suffix})

    # Compute preliminary confidence score for the evaluator to refine
    confidence_score = _compute_synthesis_confidence(docs_to_use, citations, generation)

    logger.info(
        "answer_synthesized",
        generation_len=len(generation),
        citation_count=len(citations),
        sensitivity=max_sensitivity,
        preliminary_confidence=confidence_score,
        streamed=writer is not None,
    )

    return {
        "generation": generation,
        "citations": citations,
        "confidence_score": confidence_score,
        "synth_provider": decision.provider if decision else "unknown",
        "synth_model": decision.model if decision else "unknown",
        "synth_usage": dict(llm_response.usage) if llm_response else {},
        "synth_latency_ms": (llm_response.latency_ms if llm_response else 0.0),
        "audit_trail": [
            {
                "node": "synthesizer",
                "action": "synthesize_answer",
                "doc_count": len(docs_to_use),
                "citation_count": len(citations),
                "sensitivity": max_sensitivity,
                "generation_len": len(generation),
                "preliminary_confidence": confidence_score,
                "provider": decision.provider if decision else "unknown",
                "model": decision.model if decision else "unknown",
                "forced_local": decision.forced_local if decision else False,
                "routing_reason": decision.reason if decision else "",
                "tokens": dict(llm_response.usage) if llm_response else {},
                "latency_ms": (llm_response.latency_ms if llm_response else 0.0),
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ],
    }


# synthesize_answer_stream was removed: the streaming + non-streaming
# pipelines now share the same `synthesize_answer` node, which dispatches
# based on whether a LangGraph stream writer is bound (see
# `_maybe_get_stream_writer`). One source of truth = no drift.
