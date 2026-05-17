"""Input validation and sanitization utilities.

Provides query validation, length limits, character filtering, and
injection pattern detection as a defense-in-depth layer beyond the
security agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from utils.logging import get_logger

logger = get_logger(__name__)

# Maximum query length to prevent abuse
MAX_QUERY_LENGTH: int = 4000
MAX_QUERY_WORDS: int = 500

# Dangerous characters/patterns to flag
_DANGEROUS_PATTERNS: list[re.Pattern] = [
    # SQL injection attempts
    re.compile(r"(\b(union|select|insert|update|delete|drop|create|alter)\b.*\b(from|into|table|database)\b)", re.IGNORECASE),
    # Command injection
    re.compile(r"[;&|`$]\s*\w+", re.IGNORECASE),
    # Path traversal
    re.compile(r"\.\.[\/\\]"),
    # Null bytes
    re.compile(r"\x00"),
    # Excessive repetition (DoS pattern)
    re.compile(r"(.)\1{50,}"),
]

# Allowed characters for basic sanitization
_SAFE_QUERY_PATTERN = re.compile(r"^[\w\s\-.,;:!?'\"()[\]{}@#$%&*+/=<>|~^`\u0600-\u06FF]+$", re.UNICODE)


@dataclass
class ValidationResult:
    """Result of input validation."""

    valid: bool
    sanitized_query: str
    message: str = ""
    violations: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.violations is None:
            self.violations = []


def validate_query(
    query: str,
    max_length: int | None = None,
    max_words: int | None = None,
    allow_dangerous: bool = False,
) -> ValidationResult:
    """Validate and sanitize a user query.

    Checks length limits, word count, dangerous patterns, and optionally
    sanitizes the query by removing/replacing problematic characters.

    Args:
        query: The raw user query.
        max_length: Maximum character length. Defaults to MAX_QUERY_LENGTH.
        max_words: Maximum word count. Defaults to MAX_QUERY_WORDS.
        allow_dangerous: If True, only warns about dangerous patterns
            instead of rejecting. Defaults to False.

    Returns:
        ValidationResult with valid flag, sanitized query, and any violations.
    """
    violations: list[str] = []

    # Check for None or empty
    if not query or not query.strip():
        return ValidationResult(
            valid=False,
            sanitized_query="",
            message="Query cannot be empty.",
            violations=["empty_query"],
        )

    max_len = max_length or MAX_QUERY_LENGTH
    max_w = max_words or MAX_QUERY_WORDS

    # Length check
    if len(query) > max_len:
        violations.append(f"query_too_long: {len(query)} > {max_len}")

    # Word count check
    word_count = len(query.split())
    if word_count > max_w:
        violations.append(f"query_too_many_words: {word_count} > {max_w}")

    # Dangerous pattern checks
    for pattern in _DANGEROUS_PATTERNS:
        if pattern.search(query):
            violations.append(f"dangerous_pattern: {pattern.pattern[:40]}...")

    # Sanitize: trim whitespace, normalize newlines
    sanitized = query.strip()
    sanitized = re.sub(r"\r\n|\r", "\n", sanitized)
    # Collapse multiple spaces
    sanitized = re.sub(r" {2,}", " ", sanitized)

    if violations:
        if allow_dangerous:
            logger.warning(
                "query_validation_warnings",
                violations=violations,
                query_len=len(query),
            )
            return ValidationResult(
                valid=True,
                sanitized_query=sanitized,
                message="Query passed with warnings.",
                violations=violations,
            )
        logger.warning(
            "query_validation_failed",
            violations=violations,
            query_len=len(query),
        )
        return ValidationResult(
            valid=False,
            sanitized_query=sanitized,
            message=f"Query validation failed: {'; '.join(violations)}",
            violations=violations,
        )

    return ValidationResult(
        valid=True,
        sanitized_query=sanitized,
        message="Query validation passed.",
    )


def sanitize_metadata(metadata: dict) -> dict:
    """Sanitize metadata values to prevent injection in stored payloads.

    Args:
        metadata: Raw metadata dict.

    Returns:
        Sanitized metadata with string values cleaned.
    """
    sanitized: dict = {}
    for key, value in metadata.items():
        # Sanitize keys
        safe_key = re.sub(r"[^\w\-_.]", "_", str(key))[:128]

        if isinstance(value, str):
            # Trim and sanitize strings
            safe_value = value.strip()[:2000]
            safe_value = safe_value.replace("\x00", "")
            sanitized[safe_key] = safe_value
        elif isinstance(value, (int, float, bool)):
            sanitized[safe_key] = value
        elif isinstance(value, list):
            sanitized[safe_key] = [
                v.strip()[:500].replace("\x00", "") if isinstance(v, str) else v
                for v in value[:50]  # Limit list size
            ]
        elif isinstance(value, dict):
            sanitized[safe_key] = sanitize_metadata(value)
        else:
            sanitized[safe_key] = str(value)[:500]

    return sanitized


def validate_file_path(file_path: str) -> ValidationResult:
    """Validate a file path for upload safety.

    Args:
        file_path: The file path to validate.

    Returns:
        ValidationResult indicating if the path is safe.
    """
    violations: list[str] = []

    if not file_path or not file_path.strip():
        return ValidationResult(
            valid=False,
            sanitized_query="",
            message="File path cannot be empty.",
            violations=["empty_path"],
        )

    # Path traversal check
    if ".." in file_path or "~" in file_path:
        violations.append("path_traversal_attempt")

    # Null byte injection
    if "\x00" in file_path:
        violations.append("null_byte_injection")

    # Length check
    if len(file_path) > 512:
        violations.append("path_too_long")

    sanitized = file_path.strip()

    if violations:
        logger.warning(
            "file_path_validation_failed",
            violations=violations,
            path=file_path[:100],
        )
        return ValidationResult(
            valid=False,
            sanitized_query=sanitized,
            message=f"File path validation failed: {'; '.join(violations)}",
            violations=violations,
        )

    return ValidationResult(
        valid=True,
        sanitized_query=sanitized,
        message="File path validation passed.",
    )
