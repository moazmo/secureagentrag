"""Query routing and rewriting agent."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from config.settings import settings
from core.state import GraphState  # noqa: TC001
from utils.logging import get_logger
from utils.observability import trace_llm_call

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = get_logger(__name__)

# Keyword groups for fast-path query sensitivity classification.
# These are the kinds of queries that should NEVER leave local infrastructure
# regardless of `prefer_cloud`. The synthesizer takes max(query_sensitivity,
# doc_sensitivity) so a sensitive query on low-classified docs still locks
# inference to local.
_HIGH_SENSITIVITY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(ssn|social\s*security|passport|driver'?s?\s*licen[cs]e|tax\s*id)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(salary|compensation|payroll|bonus|stock\s*grant|equity\s*grant)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(password|api[\s_-]?key|secret|token|credential|private[\s_-]?key)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(medical|health|diagnosis|prescription|hipaa|patient|phi\b)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(credit\s*card|bank\s*account|routing\s*number|iban|swift)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(trade\s*secret|m&a|acquisition|merger|insider|earnings\s*call)\b",
        re.IGNORECASE,
    ),
]
_MEDIUM_SENSITIVITY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(confidential|internal\s*only|restricted|proprietary)\b", re.IGNORECASE),
    re.compile(r"\b(employee|hr|hiring|firing|performance\s*review)\b", re.IGNORECASE),
    re.compile(r"\b(customer\s*data|user\s*data|pii|personal\s*data)\b", re.IGNORECASE),
]


def classify_query_sensitivity(query: str) -> str:
    """Classify a query's data-sensitivity tier from its text alone.

    Pure-regex (no LLM call) for predictable latency. Used to force local
    inference for queries that touch sensitive topics even when the
    retrieved documents are tagged low-sensitivity. Returns one of
    "high" / "medium" / "low".

    Args:
        query: User's raw query text.

    Returns:
        Sensitivity label string.
    """
    if not query:
        return "low"
    for pat in _HIGH_SENSITIVITY_PATTERNS:
        if pat.search(query):
            return "high"
    for pat in _MEDIUM_SENSITIVITY_PATTERNS:
        if pat.search(query):
            return "medium"
    return "low"


async def call_llm_async(
    prompt: str,
    system_prompt: str = "",
    sensitivity_level: str = "low",
    prefer_cloud: bool = False,
    json_mode: bool = False,
) -> str:
    """Call LLM asynchronously with inference routing.

    Backwards-compatible wrapper returning just the text. Most call sites
    don't need the routing decision and use this variant. Synth uses
    ``call_llm_with_decision`` instead so it can record provider/model
    in the audit trail.

    Args:
        prompt: The user/instruction prompt.
        system_prompt: Optional system prompt for context.
        sensitivity_level: Data sensitivity for routing (high/medium/low).
        prefer_cloud: Whether to prefer cloud providers for low-sensitivity.
        json_mode: Whether to request JSON-formatted output.

    Returns:
        The generated text response, or empty string on failure.
    """
    text, _decision, _response = await call_llm_with_decision(
        prompt=prompt,
        system_prompt=system_prompt,
        sensitivity_level=sensitivity_level,
        prefer_cloud=prefer_cloud,
        json_mode=json_mode,
    )
    return text


async def call_llm_with_decision(
    prompt: str,
    system_prompt: str = "",
    sensitivity_level: str = "low",
    prefer_cloud: bool = False,
    json_mode: bool = False,
):
    """Like ``call_llm_async`` but returns (text, RoutingDecision, LLMResponse).

    Useful when the caller needs to surface which provider/model was actually
    used (e.g. to write provenance into the audit trail).
    """
    from inference.router import InferenceRouter

    router = InferenceRouter()
    try:
        response, decision = await router.generate_with_routing(
            prompt=prompt,
            system_prompt=system_prompt,
            sensitivity_level=sensitivity_level,
            prefer_cloud=prefer_cloud,
            json_mode=json_mode,
        )
        logger.info(
            "call_llm_async_routed",
            provider=decision.provider,
            model=decision.model,
            latency_ms=response.latency_ms,
        )
        trace_llm_call(
            provider=decision.provider,
            model=decision.model,
            prompt=prompt,
            response=response.text,
            latency_ms=response.latency_ms,
            tokens=response.usage,
        )
        return response.text, decision, response
    except Exception as exc:
        logger.error("call_llm_async_failed", error=str(exc))
        return "", None, None


async def call_llm_stream(
    prompt: str,
    system_prompt: str = "",
    sensitivity_level: str = "low",
    prefer_cloud: bool = False,
) -> AsyncGenerator[str, None]:
    """Stream LLM response asynchronously with inference routing.

    Args:
        prompt: The user/instruction prompt.
        system_prompt: Optional system prompt for context.
        sensitivity_level: Data sensitivity for routing (high/medium/low).
        prefer_cloud: Whether to prefer cloud providers for low-sensitivity.

    Yields:
        Token strings as they are generated.
    """
    from inference.router import InferenceRouter

    router = InferenceRouter()
    try:
        async for token in router.generate_stream_with_routing(
            prompt=prompt,
            system_prompt=system_prompt,
            sensitivity_level=sensitivity_level,
            prefer_cloud=prefer_cloud,
        ):
            yield token
    except Exception as exc:
        logger.error("call_llm_stream_failed", error=str(exc))
        # Map common failure modes to actionable user-facing copy. Bare
        # "[Error generating response]" left visitors guessing whether
        # the bug was in the corpus, the prompt, or the LLM provider.
        err_text = str(exc).lower()
        if "429" in err_text or "rate" in err_text or "quota" in err_text:
            yield (
                "Sorry — the demo's shared Groq quota is exhausted for "
                "this hour. Paste your own Groq / OpenAI / Anthropic key "
                "via the 🔑 button at the top of the page for an "
                "unthrottled bucket."
            )
        elif "timeout" in err_text or "connect" in err_text:
            yield (
                "The LLM provider didn't respond in time. Try again in a "
                "few seconds, or paste your own key via the 🔑 button."
            )
        else:
            yield (
                "I couldn't reach the LLM provider this turn. Try again, "
                "or paste your own key via the 🔑 button at the top."
            )


def _get_routing_prompt(query: str) -> str:
    """Build the classification prompt for query routing.

    Args:
        query: The user's query to classify.

    Returns:
        Formatted prompt string for the LLM.
    """
    return (
        "Classify the following user query into exactly one category.\n\n"
        "Categories:\n"
        '- "simple": Direct factual question answerable from a single document chunk.\n'
        '- "complex": Requires reasoning, multi-hop retrieval, or synthesis across documents.\n'
        '- "out_of_scope": Not answerable from the document corpus (personal opinions, '
        "unrelated topics, etc.).\n\n"
        f"Query: {query}\n\n"
        "Respond with ONLY the category name (simple, complex, or out_of_scope), "
        "nothing else."
    )


def _get_rewrite_prompt(query: str, failed_docs_summary: str) -> str:
    """Build the rewrite prompt for corrective RAG.

    Args:
        query: The original or previously rewritten query.
        failed_docs_summary: Summary of documents that were deemed irrelevant.

    Returns:
        Formatted prompt string for the LLM.
    """
    return (
        "The following query did not retrieve sufficiently relevant documents.\n"
        "Rewrite it to improve retrieval quality. Make it more specific, add context, "
        "or rephrase to better match potential document content.\n\n"
        f"Original query: {query}\n\n"
        f"Summary of irrelevant results retrieved: {failed_docs_summary}\n\n"
        "Respond with ONLY the rewritten query, nothing else."
    )


async def route_query(state: GraphState) -> dict:
    """Route the user query by classifying its type and setting routing metadata.

    Classifies the query as simple, complex, or out_of_scope and sets
    routing parameters that downstream nodes use to adjust behavior:
    - simple: fewer retries, smaller top_k, skip grader if docs look good
    - complex: full corrective RAG with all retries
    - out_of_scope: early termination with polite refusal

    Args:
        state: Current graph state.

    Returns:
        Partial state update with query_type, rewritten_query, max_retries,
        top_k, and audit_trail entry.
    """
    query = state["query"]
    prefer_cloud = state.get("prefer_cloud", False)
    logger.info("routing_query", query_len=len(query), prefer_cloud=prefer_cloud)

    # Short-query heuristic: bypass the LLM classifier entirely when the
    # query is short enough that the "simple" vs "complex" distinction
    # rarely matters. Saves one Groq call per chat in BYOK demo mode
    # (where the 30 RPM cap is the binding constraint). Threshold of
    # 80 chars covers most demo prompts; longer queries still pay the
    # classifier cost so the corrective-RAG retry budget tunes correctly.
    if settings.byok_mode and len(query) <= 80:
        query_type = "simple"
        logger.info("route_query_shortcut", reason="byok_short_query")
    else:
        prompt = _get_routing_prompt(query)
        response = await call_llm_async(
            prompt,
            system_prompt="You are a query classification assistant.",
            prefer_cloud=prefer_cloud,
        )

        # Parse the response — normalize to expected categories
        response_clean = response.strip().lower().replace('"', "").replace("'", "")
        valid_types = {"simple", "complex", "out_of_scope"}

        if response_clean in valid_types:
            query_type = response_clean
        else:
            # Default to complex if LLM response is unparseable
            query_type = "complex"
            logger.warning("route_query_fallback", raw_response=response_clean)

    # Set routing parameters based on query type
    routing_config = _get_routing_config(query_type)

    # Query-level sensitivity classification — independent of doc tagging.
    # Synthesizer will take max() of this and document sensitivity so a
    # sensitive query never escapes to cloud even on low-tagged docs.
    query_sensitivity = classify_query_sensitivity(query)

    logger.info(
        "query_routed",
        query_type=query_type,
        max_retries=routing_config["max_retries"],
        top_k=routing_config["top_k"],
        query_sensitivity=query_sensitivity,
    )

    return {
        "query_type": query_type,
        "query_sensitivity": query_sensitivity,
        "rewritten_query": query,  # First pass: no rewrite
        "max_retries": routing_config["max_retries"],
        "audit_trail": [
            {
                "node": "router",
                "action": "route_query",
                "query_type": query_type,
                "max_retries": routing_config["max_retries"],
                "top_k": routing_config["top_k"],
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ],
    }


def _get_routing_config(query_type: str) -> dict:
    """Get routing configuration for a given query type.

    Args:
        query_type: The classified query type.

    Returns:
        Dict with routing parameters:
            - max_retries: Number of corrective retries allowed
            - top_k: Number of documents to retrieve initially
            - skip_grader: Whether to skip grading for speed (simple queries)
    """
    configs: dict[str, dict] = {
        "simple": {
            "max_retries": 1,  # Simple queries need fewer retries
            "top_k": 5,  # Fewer docs needed for simple factual questions
            "skip_grader": False,  # Still grade, but be lenient
        },
        "complex": {
            "max_retries": 2,  # Full corrective RAG
            "top_k": 10,  # More docs for synthesis
            "skip_grader": False,
        },
        "out_of_scope": {
            "max_retries": 0,  # No retries for out-of-scope
            "top_k": 3,  # Minimal retrieval attempt
            "skip_grader": True,  # Skip grading, will fail fast
        },
    }
    return configs.get(query_type, configs["complex"])


async def rewrite_query(state: GraphState) -> dict:
    """Rewrite the query for better retrieval during corrective RAG loop.

    Called when initial retrieval did not produce enough relevant documents.
    Uses the LLM to produce an improved query variant.

    Args:
        state: Current graph state with documents and relevance info.

    Returns:
        Partial state update with rewritten_query, incremented retry_count,
        and audit_trail entry.
    """
    current_query = state.get("rewritten_query") or state["query"]
    documents = state.get("documents", [])
    prefer_cloud = state.get("prefer_cloud", False)

    # Build summary of irrelevant docs for context
    irrelevant_docs = [d for d in documents if not d.get("relevant", False)]
    failed_summary = "; ".join(doc.get("text", "")[:100] for doc in irrelevant_docs[:3])

    logger.info("rewriting_query", current_query_len=len(current_query), prefer_cloud=prefer_cloud)

    prompt = _get_rewrite_prompt(current_query, failed_summary)
    response = await call_llm_async(
        prompt,
        system_prompt="You are a query rewriting assistant.",
        prefer_cloud=prefer_cloud,
    )

    rewritten = response.strip() if response.strip() else current_query
    retry_count = state.get("retry_count", 0) + 1

    logger.info("query_rewritten", retry_count=retry_count, new_query_len=len(rewritten))

    return {
        "rewritten_query": rewritten,
        "retry_count": retry_count,
        "audit_trail": [
            {
                "node": "router",
                "action": "rewrite_query",
                "original_query": current_query,
                "rewritten_query": rewritten,
                "retry_count": retry_count,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ],
    }
