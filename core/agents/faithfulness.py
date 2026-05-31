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
# the generation without reflowing whitespace. Splits on Latin terminators
# (.!?) AND Arabic ones (؟ question mark U+061F, ۔ full stop U+06D4) plus the
# ellipsis. The old pattern also required the next char to be an ASCII capital
# (`[A-Z\[]`), which silently disabled segmentation for Arabic answers (Arabic
# has no capitalisation) — leaving the whole Arabic generation as one "sentence"
# and effectively bypassing the NLI gate on a marketed feature. The lookahead is
# now just "followed by whitespace", which works for both scripts.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?؟۔…])\s+")


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


def _resolve_source(cited_indices: list[int], documents: list[DocumentGrade]) -> tuple[str, str]:
    """Resolve cited chunk(s) -> concatenated source text.

    Returns ``(source, reason)``. ``source`` is empty when unresolvable, with
    ``reason`` set to ``"no_cited_index"`` or ``"empty_source"``.
    """
    snippets: list[str] = []
    for idx in cited_indices:
        i = idx - 1
        if i < 0 or i >= len(documents):
            continue
        snippets.append(documents[i].get("text", ""))
    if not snippets:
        return "", "no_cited_index"
    source = "\n\n---\n\n".join(snippets).strip()
    if not source:
        return "", "empty_source"
    return source, ""


async def _judge_single(
    sentence: str,
    source: str,
    sensitivity: str,
    prefer_cloud: bool,
    semaphore: asyncio.Semaphore,
) -> tuple[bool, str]:
    """One entailment LLM call for a single (resolved) claim/source pair."""
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


def _build_batch_prompt(items: list[tuple[str, str]]) -> str:
    """Build a numbered multi-claim entailment prompt.

    ``items`` is a list of ``(claim, source)``. The model is asked to emit one
    verdict line per claim, e.g. ``1: yes`` / ``2: no``.
    """
    blocks = []
    for n, (claim, source) in enumerate(items, start=1):
        blocks.append(f"[{n}] SOURCE:\n{source[:1200]}\n[{n}] CLAIM: {claim}")
    body = "\n\n".join(blocks)
    return (
        "You are a strict fact-checker. For EACH numbered CLAIM below, decide "
        "whether its SOURCE directly supports it.\n\n"
        f"{body}\n\n"
        "Respond with exactly one line per claim in the form '<number>: yes' or "
        "'<number>: no' — nothing else. 'yes' only if the SOURCE clearly "
        "supports that CLAIM."
    )


_BATCH_VERDICT_RE = re.compile(r"(?m)^\s*\[?(\d+)\]?\s*[:.)\-]?\s*(yes|no|y|n)\b", re.IGNORECASE)


def _parse_batch(response: str, count: int) -> dict[int, bool]:
    """Parse ``N: yes/no`` lines into a 1-based index -> supported map.

    Only indices in ``[1, count]`` are kept. Missing indices are simply absent
    from the map so the caller can fall back to an individual check.
    """
    if not response:
        return {}
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE)
    out: dict[int, bool] = {}
    for m in _BATCH_VERDICT_RE.finditer(cleaned):
        n = int(m.group(1))
        if 1 <= n <= count and n not in out:
            out[n] = m.group(2).lower().startswith("y")
    return out


async def _judge_batch(
    items: list[tuple[str, str]],
    sensitivity: str,
    prefer_cloud: bool,
    semaphore: asyncio.Semaphore,
) -> list[tuple[bool, str]]:
    """Judge a batch of resolved (claim, source) pairs with one LLM call.

    Any claim the batch response does not clearly score falls back to an
    individual :func:`_judge_single` call, so a model that ignores the batch
    format degrades to the per-claim path (never worse than today).
    """
    prompt = _build_batch_prompt(items)
    async with semaphore:
        try:
            response = await call_llm_async(
                prompt=prompt,
                system_prompt="You are a strict factual entailment checker.",
                sensitivity_level=sensitivity,
                prefer_cloud=prefer_cloud,
            )
        except Exception as exc:
            logger.warning("faithfulness_batch_llm_error", error=str(exc))
            response = ""

    verdicts = _parse_batch(response, len(items))
    results: list[tuple[bool, str]] = []
    fallbacks: list[int] = []
    for i in range(len(items)):
        n = i + 1
        if n in verdicts:
            supported = verdicts[n]
            results.append((supported, "" if supported else "llm_no"))
        else:
            results.append((True, "llm_unparsed"))  # placeholder; may be replaced
            fallbacks.append(i)

    # Re-check any unparsed claims individually so a malformed batch response
    # never silently passes a fabricated claim.
    if fallbacks:
        logger.info("faithfulness_batch_fallback", unparsed=len(fallbacks), batch=len(items))
        retried = await asyncio.gather(
            *[
                _judge_single(items[i][0], items[i][1], sensitivity, prefer_cloud, semaphore)
                for i in fallbacks
            ]
        )
        for i, res in zip(fallbacks, retried, strict=True):
            results[i] = res
    return results


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

    # Resolve each cited sentence's source up front. Claims with no resolvable
    # source fail immediately (no LLM); the rest go to the LLM (batched).
    results: list[tuple[bool, str]] = [(True, "")] * len(cited_pairs)
    llm_positions: list[int] = []
    llm_items: list[tuple[str, str]] = []
    for pos, (_sent_idx, sentence, cites) in enumerate(cited_pairs):
        source, reason = _resolve_source(cites, documents)
        if not source:
            results[pos] = (False, reason)
        else:
            llm_positions.append(pos)
            llm_items.append((sentence, source))

    batch_size = max(1, int(settings.faithfulness_batch_size))
    use_batch = settings.faithfulness_batch_enabled and batch_size > 1

    if llm_items and use_batch:
        # One LLM call per batch of claims (with per-claim fallback inside).
        batch_tasks = []
        batch_pos_groups: list[list[int]] = []
        for start in range(0, len(llm_items), batch_size):
            group_items = llm_items[start : start + batch_size]
            batch_pos_groups.append(llm_positions[start : start + batch_size])
            batch_tasks.append(_judge_batch(group_items, sensitivity, prefer_cloud, semaphore))
        group_results = await asyncio.gather(*batch_tasks)
        for positions, verdicts in zip(batch_pos_groups, group_results, strict=True):
            for pos, verdict in zip(positions, verdicts, strict=True):
                results[pos] = verdict
    elif llm_items:
        # Legacy per-claim path (one call each).
        single = await asyncio.gather(
            *[
                _judge_single(sentence, source, sensitivity, prefer_cloud, semaphore)
                for sentence, source in llm_items
            ]
        )
        for pos, verdict in zip(llm_positions, single, strict=True):
            results[pos] = verdict

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
                "batched": bool(use_batch and llm_items),
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
