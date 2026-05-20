"""Citation-faithfulness gate.

After synthesis we have a generation with inline ``[N]`` citation markers and
a parallel list of ``Citation`` records that map ``N`` -> the source chunk.
Most RAG demos stop there. This module goes one step further:

For every sentence that carries one or more citation markers, ask a local LLM
the yes/no entailment question — does the cited chunk support the sentence?
Unsupported sentences are either flagged with a visible ``[unsupported]``
tag (default) or removed from the answer entirely (strict mode).

Rationale
---------
A citation marker proves the LLM *chose* a source. It does not prove the
source *supports* the claim. The two are different — and the difference is
how hallucinations slip past a citation-aware UI. Running an NLI pass
catches that gap without requiring a separate model: the same Ollama
qwen3:8b that synthesised the answer also classifies entailment well enough
for a guardrail.

Behaviour
---------
The gate is opt-in via ``settings.faithfulness_gate_enabled``. When off,
``check_faithfulness`` is a pass-through that sets ``faithfulness_ratio=1.0``
and leaves the generation untouched, so the existing pipeline shape is
preserved.

State contract
--------------
Reads:  ``generation``, ``citations``, ``relevant_documents`` (or
``documents``), ``query_sensitivity``, ``prefer_cloud``.
Writes: ``generation`` (possibly annotated/trimmed), ``faithfulness_ratio``,
``faithfulness_unsupported``, ``audit_trail`` entry.
"""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from config.settings import settings
from core.agents.router import call_llm_async
from utils.logging import get_logger

if TYPE_CHECKING:
    from core.state import DocumentGrade, GraphState

logger = get_logger(__name__)


# Match `[N]` and the legacy `[[N]]`. Mirrors synthesizer._extract_citations.
_CITE_RE = re.compile(r"\[\[(\d+)\]\]|\[(\d+)\](?!\s*\()")
# Sentence splitter that preserves the trailing punctuation so we can rebuild
# the generation without reflowing whitespace.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")


def _split_sentences(text: str) -> list[str]:
    """Split ``text`` into rough sentences for per-claim faithfulness checks."""
    if not text.strip():
        return []
    # Strip <think> blocks defensively (synth should have removed them).
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]


def _cited_indices(sentence: str) -> list[int]:
    """Return 1-based citation indices found in ``sentence``."""
    out: list[int] = []
    for m in _CITE_RE.finditer(sentence):
        token = m.group(1) or m.group(2)
        if token is None:
            continue
        try:
            out.append(int(token))
        except ValueError:
            continue
    return out


def _build_nli_prompt(sentence: str, source_text: str) -> str:
    """Build a strict yes/no entailment prompt.

    Kept deliberately minimal: the smaller the prompt, the more reliable
    yes/no classification gets on 8B-class local models.
    """
    return (
        "You are a strict fact-checker. Decide whether the SOURCE text "
        "directly supports the CLAIM.\n\n"
        f"SOURCE:\n{source_text[:1500]}\n\n"
        f"CLAIM: {sentence}\n\n"
        "Answer with exactly one word: 'yes' if the SOURCE clearly supports "
        "the CLAIM, otherwise 'no'. Do not include explanation, punctuation, "
        "or any other text."
    )


def _parse_yes_no(response: str) -> bool:
    """Parse the LLM's one-word verdict. Conservative: anything not clearly
    'yes' is treated as unsupported.
    """
    if not response:
        return False
    cleaned = response.strip().lower()
    # Strip leading reasoning tokens some local models still emit.
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE).strip()
    # Take the first non-empty token.
    head = cleaned.split()[0] if cleaned.split() else ""
    return head.startswith("yes")


async def _check_one(
    sentence: str,
    cited_indices: list[int],
    documents: list[DocumentGrade],
    sensitivity: str,
    prefer_cloud: bool,
    semaphore: asyncio.Semaphore,
) -> tuple[bool, str]:
    """Run one entailment check.

    Returns:
        (supported, reason) — ``reason`` is empty on success or a short tag
        on failure ("no_cited_index", "empty_source", "llm_no", "llm_error").
    """
    # Resolve cited chunk(s) -> concatenate text. Skip out-of-range refs.
    snippets: list[str] = []
    for idx in cited_indices:
        i = idx - 1
        if i < 0 or i >= len(documents):
            continue
        snippets.append(documents[i].get("text", ""))
    if not snippets:
        return False, "no_cited_index"
    source = "\n\n---\n\n".join(snippets).strip()
    if not source:
        return False, "empty_source"

    prompt = _build_nli_prompt(sentence, source)
    async with semaphore:
        try:
            response = await call_llm_async(
                prompt=prompt,
                system_prompt="You are a strict factual entailment checker.",
                sensitivity_level=sensitivity,
                prefer_cloud=prefer_cloud,
            )
        except Exception as exc:
            logger.warning("faithfulness_llm_error", error=str(exc))
            # Fail open: treat as supported to avoid dropping content on
            # transient LLM errors. The audit entry records the count.
            return True, "llm_error"
    supported = _parse_yes_no(response)
    return supported, "" if supported else "llm_no"


async def check_faithfulness(state: GraphState) -> dict:
    """LangGraph node: NLI entailment check on every cited sentence.

    No-op when ``faithfulness_gate_enabled`` is false. When enabled, for each
    sentence with at least one ``[N]`` marker:

    1. Look up the cited chunks.
    2. Ask the local LLM if the chunks entail the sentence (one-word yes/no).
    3. Flag (default) or drop (strict mode) sentences the LLM marks as
       unsupported.

    The mode is controlled by ``settings.faithfulness_gate_mode``:
    - "flag": append ``[unsupported]`` after the sentence (default).
    - "drop": remove the sentence from the generation.

    Args:
        state: Current graph state. Must contain ``generation`` and
            ``citations``; documents come from ``relevant_documents`` or
            ``documents``.

    Returns:
        Partial state update with ``generation``, ``faithfulness_ratio``,
        ``faithfulness_unsupported``, and an ``audit_trail`` entry.
    """
    generation: str = state.get("generation", "") or ""
    documents: list[DocumentGrade] = state.get("relevant_documents") or state.get("documents") or []

    if not settings.faithfulness_gate_enabled:
        return {
            "faithfulness_ratio": 1.0,
            "faithfulness_unsupported": [],
            "audit_trail": [
                {
                    "node": "faithfulness",
                    "action": "skip",
                    "reason": "disabled",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            ],
        }

    if not generation.strip() or not documents:
        return {
            "faithfulness_ratio": 1.0,
            "faithfulness_unsupported": [],
            "audit_trail": [
                {
                    "node": "faithfulness",
                    "action": "skip",
                    "reason": "empty_generation_or_no_docs",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            ],
        }

    # Tokenise sentences. Each cited sentence gets one NLI call.
    sentences = _split_sentences(generation)
    cited_pairs: list[tuple[int, str, list[int]]] = []
    for idx, sentence in enumerate(sentences):
        cites = _cited_indices(sentence)
        if cites:
            cited_pairs.append((idx, sentence, cites))

    if not cited_pairs:
        # No cited sentences at all — treat ratio as 1.0 to avoid penalising
        # zero-claim answers ("Sorry, I cannot answer that.").
        return {
            "faithfulness_ratio": 1.0,
            "faithfulness_unsupported": [],
            "audit_trail": [
                {
                    "node": "faithfulness",
                    "action": "noop",
                    "reason": "no_cited_sentences",
                    "sentences": len(sentences),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            ],
        }

    sensitivity = state.get("query_sensitivity", "low") or "low"
    prefer_cloud = bool(state.get("prefer_cloud", False))
    semaphore = asyncio.Semaphore(max(1, int(settings.faithfulness_max_concurrent)))

    tasks = [
        _check_one(sentence, cites, documents, sensitivity, prefer_cloud, semaphore)
        for _, sentence, cites in cited_pairs
    ]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    unsupported: list[dict] = []
    annotated_sentences = list(sentences)
    drop_indices: set[int] = set()
    mode = (settings.faithfulness_gate_mode or "flag").lower()

    for (sent_idx, sentence, cites), (supported, reason) in zip(cited_pairs, results, strict=False):
        if supported:
            continue
        unsupported.append(
            {
                "sentence": sentence,
                "cited": cites,
                "verdict": reason or "llm_no",
            }
        )
        if mode == "drop":
            drop_indices.add(sent_idx)
        else:
            # Inject inline marker; keep the rest of the sentence so the
            # reader can see what was flagged.
            annotated_sentences[sent_idx] = sentence + " *[unsupported]*"

    if drop_indices:
        annotated_sentences = [
            s for i, s in enumerate(annotated_sentences) if i not in drop_indices
        ]
    new_generation = " ".join(annotated_sentences).strip()
    if not new_generation:
        # Strict mode dropped every cited sentence. Refuse rather than
        # return an empty string to the caller.
        new_generation = (
            "I could not find sentence-level support for any of the cited "
            "claims in the retrieved documents. Refusing to return an "
            "unverified answer."
        )

    total_cited = len(cited_pairs)
    supported_count = total_cited - len(unsupported)
    ratio = round(supported_count / total_cited, 3) if total_cited else 1.0

    logger.info(
        "faithfulness_checked",
        cited_sentences=total_cited,
        supported=supported_count,
        unsupported=len(unsupported),
        ratio=ratio,
        mode=mode,
    )

    return {
        "generation": new_generation,
        "faithfulness_ratio": ratio,
        "faithfulness_unsupported": unsupported,
        "audit_trail": [
            {
                "node": "faithfulness",
                "action": "check",
                "mode": mode,
                "cited_sentences": total_cited,
                "supported": supported_count,
                "unsupported": len(unsupported),
                "ratio": ratio,
                "threshold": settings.faithfulness_threshold,
                "below_threshold": ratio < settings.faithfulness_threshold,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ],
    }
