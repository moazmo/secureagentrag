"""Query routing and rewriting agent."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from core.state import GraphState  # noqa: TC001
from utils.logging import get_logger
from utils.observability import trace_llm_call

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = get_logger(__name__)


async def call_llm_async(
    prompt: str,
    system_prompt: str = "",
    sensitivity_level: str = "low",
    prefer_cloud: bool = False,
) -> str:
    """Call LLM asynchronously with inference routing.

    Routes through InferenceRouter so sensitivity-based provider selection
    and cloud fallback actually work in the pipeline.

    Args:
        prompt: The user/instruction prompt.
        system_prompt: Optional system prompt for context.
        sensitivity_level: Data sensitivity for routing (high/medium/low).
        prefer_cloud: Whether to prefer cloud providers for low-sensitivity.

    Returns:
        The generated text response, or empty string on failure.
    """
    from inference.router import InferenceRouter

    router = InferenceRouter()
    try:
        response, decision = await router.generate_with_routing(
            prompt=prompt,
            system_prompt=system_prompt,
            sensitivity_level=sensitivity_level,
            prefer_cloud=prefer_cloud,
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
        return response.text
    except Exception as exc:
        logger.error("call_llm_async_failed", error=str(exc))
        return ""


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
        yield "[Error generating response]"


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
    logger.info("routing_query", query_len=len(query))

    prompt = _get_routing_prompt(query)
    response = await call_llm_async(
        prompt, system_prompt="You are a query classification assistant."
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

    logger.info(
        "query_routed",
        query_type=query_type,
        max_retries=routing_config["max_retries"],
        top_k=routing_config["top_k"],
    )

    return {
        "query_type": query_type,
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

    # Build summary of irrelevant docs for context
    irrelevant_docs = [d for d in documents if not d.get("relevant", False)]
    failed_summary = "; ".join(doc.get("text", "")[:100] for doc in irrelevant_docs[:3])

    logger.info("rewriting_query", current_query_len=len(current_query))

    prompt = _get_rewrite_prompt(current_query, failed_summary)
    response = await call_llm_async(
        prompt, system_prompt="You are a query rewriting assistant."
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
