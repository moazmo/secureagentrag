"""Tests for BYOK API-key shape redaction in audit / cache / persistence paths.

This is the security regression guard for the launch.
``utils.pii.redact`` must mask every key shape the platform might receive,
even one the developer forgot existed. New provider added? Add a row here.

Synthetic key shapes are used throughout — these are NOT real credentials.

See ``launch-plan/11-security-checklist.md`` § Key never lands on disk.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from config.settings import settings
from utils.pii import redact

# ── Provider key shapes ─────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _enable_redaction():
    """Tests must run with PII redaction enabled even when global setting is off."""
    with patch.object(settings, "pii_redaction_enabled", True):
        yield


@pytest.mark.parametrize(
    "key, label",
    [
        # Groq — gsk_ + 52 alnum
        ("gsk_" + "A" * 52, "Groq"),
        # OpenAI legacy short — sk- + 32 alnum
        ("sk-" + "A" * 32, "OpenAI-legacy"),
        # OpenAI new project-scoped — sk-proj-...
        ("sk-proj-" + "a" * 48, "OpenAI-proj"),
        # OpenAI service account — sk-svcacct-...
        ("sk-svcacct-" + "b" * 40, "OpenAI-svc"),
        # Anthropic — sk-ant-api03-...
        ("sk-ant-api03-" + "a" * 40, "Anthropic"),
        # Hugging Face write token — hf_ + 30+ alnum
        ("hf_" + "X" * 34, "Hugging-Face"),
        # Vercel — vcp_ + alnum
        ("vcp_" + "Z" * 52, "Vercel"),
    ],
)
def test_key_shape_redacted(key: str, label: str) -> None:
    body = f"User pasted key: {key} for inference"
    redacted = redact(body)
    assert key not in redacted, (
        f"{label} key survived redaction:\n  raw:      {key!r}\n  redacted: {redacted!r}"
    )
    assert "[API_KEY]" in redacted


def test_qdrant_jwt_cluster_key_redacted() -> None:
    """Qdrant Cloud database-api-keys v2 returns JWT-shaped keys."""
    # synthetic three-part base64url (real shape, dummy payload)
    jwt = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        ".eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6dGVzdCJ9"
        ".testsignature1234567890abcdef"
    )
    body = f"qdrant key = {jwt} for cluster X"
    out = redact(body)
    assert jwt not in out
    assert "[API_KEY]" in out


def test_qdrant_cloud_management_key_redacted() -> None:
    """Qdrant Cloud management keys: ``<uuid>|<random>``."""
    mgmt = "12345678-1234-1234-1234-123456789abc|ZZ" + "Z" * 40
    body = f"Header: apikey {mgmt}"
    out = redact(body)
    assert mgmt not in out
    assert "[API_KEY]" in out


def test_multiple_keys_in_same_string_all_redacted() -> None:
    """Compound payload: every shape gets masked in one pass."""
    body = (
        "groq="
        + "gsk_"
        + "A" * 52
        + " openai="
        + "sk-proj-"
        + "a" * 48
        + " hf="
        + "hf_"
        + "X" * 34
        + " vercel="
        + "vcp_"
        + "Z" * 52
    )
    out = redact(body)
    # Count: 4 [API_KEY] tokens (one per provider) at minimum
    assert out.count("[API_KEY]") >= 4
    assert "gsk_" not in out
    assert "sk-proj-" not in out
    assert "hf_" not in out
    assert "vcp_" not in out


# ── False-positive guards ───────────────────────────────────────────────────


def test_non_key_words_left_alone() -> None:
    """Plain prose must not be redacted as if it were a key."""
    body = "The skydiver wore a parkour outfit while keyboarding at a hackathon."
    out = redact(body)
    assert "skydiver" in out
    assert "parkour" in out
    assert "keyboarding" in out


def test_short_strings_not_caught() -> None:
    """Three-character random strings are not keys — must not be redacted."""
    body = "Use the abc and xyz tokens for the route."
    out = redact(body)
    assert "abc" in out
    assert "xyz" in out


def test_redaction_disabled_returns_original() -> None:
    """When ``pii_redaction_enabled=False`` redact is a no-op (dev-only setting)."""
    body = "key=gsk_" + "A" * 52
    with patch.object(settings, "pii_redaction_enabled", False):
        assert redact(body) == body


# ── Audit-log round-trip regression ─────────────────────────────────────────


def test_byok_key_does_not_survive_audit_dict_redaction() -> None:
    """End-to-end: keys nested in audit-style dicts get redacted too.

    ``redact_dict`` is what the audit logger calls before writing JSONL.
    A key buried in a nested ``request_body.headers.authorization`` field
    must still get masked.
    """
    from utils.pii import redact_dict

    key = "gsk_" + "A" * 52
    payload = {
        "user_id": "visitor-A",
        "request_body": {
            "headers": {
                "authorization": f"Bearer {key}",
            },
            "metadata": {"jti": "abc"},
        },
        "response_body": "ok",
    }
    out = redact_dict(payload)
    flat = str(out)
    assert key not in flat
