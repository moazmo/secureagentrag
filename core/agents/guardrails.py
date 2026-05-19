"""Prompt-injection / jailbreak guardrails agent.

Runs *before* the security/RBAC node so injection attempts are blocked before
the request consumes embedding/LLM budget. The check is a layered regex
heuristic — fast (≤1ms) and dependency-free. The output of the synthesizer
is similarly scanned for system-prompt leakage.

Why not just an LLM classifier?
- Latency: adding an LLM call on every query doubles end-to-end time for
  the common (benign) case.
- Defense-in-depth: a deterministic gate complements the RBAC + sensitivity
  gates already in place.
- Optional escalation: when ``guardrails_strict`` is enabled in settings the
  caller can chain a model-based classifier on top by inspecting the
  ``state["guardrails_reason"]`` field.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

from config.settings import settings
from core.state import GraphState  # noqa: TC001
from utils.audit import audit_logger
from utils.logging import get_logger

logger = get_logger(__name__)

# Patterns that signal an attempt to override the system prompt / RBAC.
_INJECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Most specific / highest signal first so they beat broader matches.
    (re.compile(r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>"), "chat_template_injection"),
    (re.compile(r"</?system\b", re.IGNORECASE), "system_tag_injection"),
    (
        re.compile(
            r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instruction|prompt)",
            re.IGNORECASE,
        ),
        "ignore_instructions",
    ),
    (
        re.compile(
            r"\bdisregard\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instruction|prompt)",
            re.IGNORECASE,
        ),
        "disregard_instructions",
    ),
    (
        re.compile(
            r"\b(?:reveal|show|print|dump|leak)\s+(?:the\s+)?(?:system\s+)?(?:prompt|instructions?)\b",
            re.IGNORECASE,
        ),
        "prompt_extraction",
    ),
    (re.compile(r"\bDAN\s+mode\b|\bdeveloper\s+mode\b", re.IGNORECASE), "jailbreak_persona"),
    (
        re.compile(
            r"\b(?:you\s+are\s+now|you'?re\s+now|act\s+as)\s+(?:a|an)?\s*(?:dan|jailbreak|developer\s*mode|sudo|root|admin)\b",
            re.IGNORECASE,
        ),
        "role_override",
    ),
    (
        re.compile(
            r"\bbypass\s+(?:the\s+)?(?:rbac|security|filter|guardrail|safety)", re.IGNORECASE
        ),
        "explicit_bypass",
    ),
    (
        re.compile(
            r"\bgrant\s+me\s+(?:admin|root|elevated)\s+(?:access|role|permission)", re.IGNORECASE
        ),
        "privilege_escalation",
    ),
]

# Patterns that signal the model leaked its system prompt back into the answer.
_LEAK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\byou are a helpful (?:assistant|RAG)\b", re.IGNORECASE),
    re.compile(r"\bsystem prompt[:\s]", re.IGNORECASE),
    re.compile(r"\b(?:RBAC|sensitivity_level_int|org_id|user_context)\b"),
]


def check_query(query: str) -> tuple[bool, str]:
    """Return ``(passed, reason)`` for the given query.

    Args:
        query: Raw user query text.

    Returns:
        Tuple of (passed, reason). ``passed=False`` indicates a likely
        injection attempt; ``reason`` names the matched pattern.
    """
    if not query or not query.strip():
        return False, "empty_query"
    if len(query) > 4000:
        return False, "query_too_long"
    for pattern, name in _INJECTION_PATTERNS:
        if pattern.search(query):
            return False, name
    return True, ""


def check_output(text: str) -> tuple[bool, str]:
    """Return ``(safe, reason)`` for synthesized output.

    Args:
        text: Generated answer text.

    Returns:
        Tuple of (safe, reason). ``safe=False`` if the answer appears to
        leak the system prompt or internal config fields.
    """
    if not text:
        return True, ""
    for pat in _LEAK_PATTERNS:
        if pat.search(text):
            return False, "system_prompt_leak"
    return True, ""


async def guardrails_check(state: GraphState) -> dict:
    """LangGraph node — gate the query before retrieval.

    Args:
        state: Current graph state.

    Returns:
        Partial state update with ``guardrails_passed``,
        ``guardrails_reason``, and an audit-trail entry.
    """
    if not settings.guardrails_enabled:
        return {
            "guardrails_passed": True,
            "guardrails_reason": "disabled",
            "audit_trail": [
                {
                    "node": "guardrails",
                    "action": "skipped",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            ],
        }

    passed, reason = check_query(state["query"])

    # Strict mode: escalate to an LLM-based classifier for a second opinion.
    # Regex-blocked queries are blocked immediately; regex-passed queries get
    # the LLM check when strict mode is on.
    if passed and settings.guardrails_strict:
        from core.agents.guardrails_llm import llm_guardrails_check

        passed, reason = await llm_guardrails_check(state["query"])

    if not passed:
        user = state.get("user_context", {}) or {}
        audit_logger.log_security_event(
            user_id=user.get("user_id", "unknown"),
            org_id=user.get("org_id", ""),
            event_type="prompt_injection_attempt",
            details={"reason": reason, "query_preview": state["query"][:200]},
        )
        logger.warning("guardrails_blocked", reason=reason, user_id=user.get("user_id"))

    return {
        "guardrails_passed": passed,
        "guardrails_reason": reason,
        "audit_trail": [
            {
                "node": "guardrails",
                "action": "guardrails_check",
                "passed": passed,
                "reason": reason,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ],
    }


def guardrails_gate(state: GraphState) -> str:
    """Conditional-edge function. ``"proceed"`` or ``"blocked"``."""
    return "proceed" if state.get("guardrails_passed", True) else "blocked"
