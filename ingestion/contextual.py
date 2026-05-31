"""Anthropic-style Contextual Retrieval.

Before embedding, each chunk is prefixed with a short LLM-written ``context``
that grounds it inside its source document ("This section describes the
GOVERN function of the NIST AI RMF, specifically the role of risk
tolerance..."). Anthropic reported a 35-49% reduction in retrieval failures
on their internal benchmark.

The chunk text shown to the user remains the original — only the *embedding
input* (and BM25 tokenisation) carries the prepended context. So display
quality is unchanged while retrieval recall improves.

Trade-off: one LLM call per chunk at ingestion time. We parallelise with a
bounded asyncio.Semaphore and route via ``call_llm_async`` so the call obeys
the same sensitivity rules as the rest of the system (HIGH stays local).
"""

from __future__ import annotations

import asyncio

from core.agents.router import call_llm_async
from utils.logging import get_logger

logger = get_logger(__name__)

_PROMPT_TEMPLATE = (
    "<document>\n{document}\n</document>\n\n"
    "Here is the chunk we want to situate within the whole document:\n"
    "<chunk>\n{chunk}\n</chunk>\n\n"
    "Please give a short succinct context to situate this chunk within "
    "the overall document for the purposes of improving search retrieval "
    "of the chunk. Answer only with the succinct context (1-3 sentences, "
    "under 100 tokens) and nothing else."
)


async def _generate_one(
    document_text: str,
    chunk_text: str,
    semaphore: asyncio.Semaphore,
    prefer_cloud: bool,
    max_doc_chars: int,
    sensitivity_level: str = "low",
) -> str:
    """Generate a single chunk's context summary.

    Args:
        document_text: Full source document text (truncated to ``max_doc_chars``).
        chunk_text: The chunk to situate.
        semaphore: Bound on concurrent LLM calls.
        prefer_cloud: Honour user routing preference (HIGH still stays local).
        max_doc_chars: Cap document text included in the prompt.

    Returns:
        Short context string, or empty string on failure.
    """
    async with semaphore:
        prompt = _PROMPT_TEMPLATE.format(
            document=document_text[:max_doc_chars],
            chunk=chunk_text,
        )
        try:
            ctx = await call_llm_async(
                prompt,
                system_prompt="You generate short retrieval context summaries.",
                # Use the document's real sensitivity so HIGH content is summarised
                # on local inference and never sent to a cloud provider at ingest.
                sensitivity_level=sensitivity_level,
                prefer_cloud=prefer_cloud,
            )
            return ctx.strip()
        except Exception as exc:
            logger.debug("contextual_chunk_failed", error=str(exc))
            return ""


async def generate_chunk_contexts(
    document_text: str,
    chunks: list[str],
    *,
    prefer_cloud: bool = False,
    max_concurrent: int = 8,
    max_doc_chars: int = 50_000,
    sensitivity_level: str = "low",
) -> list[str]:
    """Generate contexts for every chunk concurrently.

    Args:
        document_text: Full source document text.
        chunks: List of chunk texts in order.
        prefer_cloud: Pass through to the routing layer.
        max_concurrent: Maximum simultaneous LLM calls.
        max_doc_chars: Truncate document text to this many chars in each
            prompt (long docs balloon prompt cost without proportional benefit).

    Returns:
        List of context strings, one per chunk (same length & order).
    """
    if not chunks:
        return []
    sem = asyncio.Semaphore(max_concurrent)
    tasks = [
        _generate_one(document_text, c, sem, prefer_cloud, max_doc_chars, sensitivity_level)
        for c in chunks
    ]
    contexts = await asyncio.gather(*tasks, return_exceptions=False)
    logger.info(
        "contextual_retrieval_generated",
        chunks=len(chunks),
        successful=sum(1 for c in contexts if c),
    )
    return list(contexts)


def merge_chunks(chunks: list[str], contexts: list[str]) -> list[str]:
    """Return ``[context + "\\n\\n" + chunk]`` for embedding input.

    Args:
        chunks: Original chunk texts.
        contexts: Per-chunk contexts (same length, may have empty entries).

    Returns:
        Augmented texts. Where a context is empty the original chunk is
        returned unmodified.
    """
    out: list[str] = []
    for chunk, ctx in zip(chunks, contexts, strict=False):
        if ctx:
            out.append(f"Context: {ctx}\n\n{chunk}")
        else:
            out.append(chunk)
    return out
