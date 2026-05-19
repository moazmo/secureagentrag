"""PII redaction for secondary stores (audit log, query cache, conversation history).

Two strategies:
- Regex-based (always available) — covers email, phone, SSN, credit card,
  IBAN, IPv4, URL with credentials.
- Microsoft Presidio (optional dependency) — invoked when installed; higher
  recall and language-aware NER for names, locations, organizations.

This module never sees plaintext from the LLM — it operates only on text that
is about to be persisted to disk. Live prompts and retrieved contexts remain
unmodified so model quality is not affected.
"""

from __future__ import annotations

import re
from typing import Any

from config.settings import settings
from utils.logging import get_logger

logger = get_logger(__name__)

# Order matters — most specific patterns first so they win against the
# broader phone regex.
_REGEX_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"https?://[^\s/]+:[^\s/]+@[^\s]+"), "[URL_WITH_CREDS]"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[EMAIL]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "[CC]"),  # Luhn-validated below
    (re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"), "[IBAN]"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[IP]"),
    (re.compile(r"\b(?:sk|pk|api|key)[-_][A-Za-z0-9]{16,}\b", re.IGNORECASE), "[API_KEY]"),
    (re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"), "[PHONE]"),
]

# Try Presidio for richer detection (names, locations, etc.)
try:
    from presidio_analyzer import AnalyzerEngine  # type: ignore[import-not-found]
    from presidio_anonymizer import AnonymizerEngine  # type: ignore[import-not-found]

    _PRESIDIO_AVAILABLE = True
    _analyzer: Any = AnalyzerEngine()
    _anonymizer: Any = AnonymizerEngine()
except Exception:
    _PRESIDIO_AVAILABLE = False
    _analyzer = None
    _anonymizer = None


def _luhn_valid(num: str) -> bool:
    """Luhn checksum to filter false-positive credit-card matches."""
    digits = [int(c) for c in num if c.isdigit()]
    if not (13 <= len(digits) <= 19):
        return False
    s = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        s += d
    return s % 10 == 0


def redact(text: str) -> str:
    """Return ``text`` with PII tokens masked.

    Args:
        text: Arbitrary string that may contain PII.

    Returns:
        Redacted copy of the text. If redaction is disabled via settings
        the original string is returned unchanged.
    """
    if not settings.pii_redaction_enabled or not text:
        return text

    out = text
    for pattern, token in _REGEX_PATTERNS:
        if token == "[CC]":
            # Apply Luhn to avoid over-masking phone numbers / arbitrary digits.
            out = pattern.sub(lambda m: "[CC]" if _luhn_valid(m.group(0)) else m.group(0), out)
        else:
            out = pattern.sub(token, out)

    if _PRESIDIO_AVAILABLE and _analyzer is not None and _anonymizer is not None:
        try:
            results = _analyzer.analyze(text=out, language="en")
            if results:
                out = _anonymizer.anonymize(text=out, analyzer_results=results).text
        except Exception as exc:
            logger.debug("presidio_redact_failed", error=str(exc))

    return out


def redact_dict(data: dict[str, Any], fields: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Recursively redact string values in a dict.

    Args:
        data: Dict (possibly nested) to redact.
        fields: If given, only redact these top-level keys. Otherwise redact
            every string in the structure.

    Returns:
        Deep-redacted copy.
    """
    if not settings.pii_redaction_enabled:
        return data

    def _walk(value: Any, *, force: bool) -> Any:
        if isinstance(value, str):
            return redact(value) if force else value
        if isinstance(value, dict):
            return {
                k: _walk(v, force=force or (fields is not None and k in fields))
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [_walk(v, force=force) for v in value]
        return value

    return _walk(data, force=fields is None)
