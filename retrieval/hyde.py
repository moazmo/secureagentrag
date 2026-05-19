"""Hypothetical Document Embeddings (HyDE) — Gao et al., 2022.

Before searching, ask the LLM to write the *kind of document* that would
answer the query, then embed that hypothetical answer instead of (or in
addition to) the raw query. The hypothesis sits in document-space rather
than question-space, so the dense vector lines up better with real docs.

Cost: one LLM call per query (mitigated by routing — for benign queries we
let it ride on cloud when ``prefer_cloud`` is True).
"""

from __future__ import annotations

from core.agents.router import call_llm_async
from utils.logging import get_logger

logger = get_logger(__name__)

_HYDE_PROMPT = (
    "Write a short, factual passage (3-5 sentences) that would directly "
    "answer the following question, as if quoting a relevant document. "
    "Do not hedge, do not add caveats, do not say 'I think' — just write "
    "the passage as the document itself would phrase it.\n\n"
    "Question: {query}\n\n"
    "Passage:"
)


async def generate_hyde_passage(
    query: str,
    *,
    sensitivity_level: str = "low",
    prefer_cloud: bool = False,
) -> str:
    """Return a hypothetical answer passage for ``query``.

    Falls back to the raw query on any failure so retrieval still runs.

    Args:
        query: User's natural language query.
        sensitivity_level: Passed to the inference router (HIGH stays local).
        prefer_cloud: User routing preference.

    Returns:
        A short passage suitable for use as the embedding input.
    """
    try:
        passage = await call_llm_async(
            _HYDE_PROMPT.format(query=query),
            system_prompt="You generate concise factual passages for retrieval.",
            sensitivity_level=sensitivity_level,
            prefer_cloud=prefer_cloud,
        )
        passage = passage.strip()
        if not passage:
            return query
        logger.info("hyde_passage_generated", chars=len(passage))
        # Concatenate with original query so BM25 still benefits from the
        # original keywords (dense + sparse balance).
        return f"{query}\n\n{passage}"
    except Exception as exc:
        logger.warning("hyde_passage_failed", error=str(exc))
        return query
