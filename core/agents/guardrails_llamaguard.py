"""LlamaGuard 3 classifier as a drop-in guardrails escalation backend.

Why a separate module?
----------------------
The legacy escalation in :mod:`core.agents.guardrails_llm` calls the
synth-grade LLM (``qwen3:8b``) and asks for a free-form SAFE/UNSAFE token.
That works but is loose: any prompt the model rephrases ends up scored
SAFE. LlamaGuard 3 is a 8B model fine-tuned by Meta specifically for
content-policy classification with a fixed taxonomy (``S1-S14``).

Selecting between backends
--------------------------
``settings.guardrails_backend``:

* ``"regex"`` — only the fast regex gate runs (default for cheap workloads).
* ``"llm"`` — the legacy ``guardrails_llm.llm_guardrails_check`` escalation.
* ``"llamaguard"`` — this module. Calls ``settings.llamaguard_model`` via
  Ollama using the official chat template Meta ships with the model card.

The graph node in :mod:`core.agents.guardrails` always runs the regex gate
first, then escalates ambiguous + passing queries to the configured
backend. Backend errors fail open (return SAFE) so a transient Ollama
outage does not silently drop user content.

Output contract
---------------
``check`` returns ``(passed: bool, reason: str)``. The reason on failure is
the LlamaGuard category if we could parse it (e.g. ``S5_defamation``,
``S2_non_violent_crimes``), or ``llamaguard_unsafe`` if the model just
said unsafe without a category.
"""

from __future__ import annotations

import re

from config.settings import settings
from utils.logging import get_logger

logger = get_logger(__name__)

# Mapping from LlamaGuard 3 S1-S14 codes to human-readable reasons that
# slot into our `guardrails_reason` enum. Sourced from the model card:
# https://huggingface.co/meta-llama/Llama-Guard-3-8B
_CATEGORY_MAP: dict[str, str] = {
    "S1": "violent_crimes",
    "S2": "non_violent_crimes",
    "S3": "sex_crimes",
    "S4": "child_exploitation",
    "S5": "defamation",
    "S6": "specialized_advice",
    "S7": "privacy_violation",
    "S8": "intellectual_property",
    "S9": "indiscriminate_weapons",
    "S10": "hate_speech",
    "S11": "self_harm",
    "S12": "sexual_content",
    "S13": "elections_misinformation",
    "S14": "code_interpreter_abuse",
}

# LlamaGuard outputs either:
#   "safe"
# or:
#   "unsafe\nS2,S5"
# We match the category line in a separate regex so we can record which
# policy fired in the audit trail.
_UNSAFE_RE = re.compile(r"\bunsafe\b", re.IGNORECASE)
_CAT_RE = re.compile(r"\bS(?P<n>\d{1,2})\b")


def _prompt(query: str) -> str:
    """Build the LlamaGuard prompt using Meta's official chat template.

    The model expects a tightly-formatted instruction block. We omit the
    full system policy text (the model has it baked in) and pass only the
    user turn to keep the request small.
    """
    return (
        "Task: Check if there is unsafe content in the user message according "
        "to the safety policy.\n\n"
        "<BEGIN UNSAFE CONTENT CATEGORIES>\n"
        "S1: Violent Crimes\nS2: Non-Violent Crimes\nS3: Sex Crimes\n"
        "S4: Child Exploitation\nS5: Defamation\nS6: Specialized Advice\n"
        "S7: Privacy Violation\nS8: Intellectual Property\n"
        "S9: Indiscriminate Weapons\nS10: Hate Speech\nS11: Self-Harm\n"
        "S12: Sexual Content\nS13: Elections Misinformation\n"
        "S14: Code Interpreter Abuse\n"
        "<END UNSAFE CONTENT CATEGORIES>\n\n"
        "<BEGIN CONVERSATION>\n"
        f"User: {query}\n"
        "<END CONVERSATION>\n\n"
        "Provide your safety assessment:\n"
        "- First line: 'safe' or 'unsafe'\n"
        "- If unsafe, second line lists comma-separated category codes."
    )


def _parse(response: str) -> tuple[bool, str]:
    """Return ``(passed, reason)``. ``passed=True`` means SAFE."""
    if not response:
        # No response — fail open (transport-level failures handled by caller).
        return True, ""
    if not _UNSAFE_RE.search(response):
        return True, ""

    # Find the first SN code; fall back to a generic reason if none parsed.
    match = _CAT_RE.search(response)
    if match:
        code = f"S{int(match.group('n'))}"
        reason = _CATEGORY_MAP.get(code, f"llamaguard_{code.lower()}")
        return False, reason
    return False, "llamaguard_unsafe"


async def check(query: str) -> tuple[bool, str]:
    """LlamaGuard 3 classification call.

    Args:
        query: The user's query text.

    Returns:
        ``(passed, reason)``. ``passed=False`` blocks the request and the
        reason maps to one of ``_CATEGORY_MAP`` values (or
        ``"llamaguard_unsafe"`` if the code did not parse).
    """
    # Late import keeps the dependency footprint of importing this module
    # to zero — the actual Ollama client is only resolved at call time.
    from inference.llm_factory import get_llm

    model = settings.llamaguard_model
    try:
        client = get_llm("ollama", model=model)
        response = await client.generate(
            prompt=_prompt(query),
            system_prompt="You are LlamaGuard 3, a content classifier.",
            temperature=0.0,
            max_tokens=64,
        )
        text = response.text if response else ""
        passed, reason = _parse(text)
        if not passed:
            logger.warning(
                "llamaguard_blocked",
                reason=reason,
                model=model,
                query_preview=query[:100],
            )
        return passed, reason
    except Exception as exc:
        logger.warning(
            "llamaguard_check_failed",
            error=str(exc),
            model=model,
            query_preview=query[:100],
        )
        # Fail-open on transport-level errors (model not pulled, Ollama
        # down). The regex gate already ran ahead of us; the principle is
        # never to drop user content on infrastructure flakes.
        return True, "llamaguard_check_failed"
