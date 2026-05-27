"""Tests for the BYOK (Bring Your Own Key) request-extraction layer.

The dependency itself is pure data extraction — see
``launch-plan/03-backend-byok.md``. Throttling and key redaction live in
separate test modules so each invariant is held by exactly one place.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from interfaces.byok import (
    SUPPORTED_PROVIDERS,
    ByokCreds,
    _derive_session_id,
    build_creds,
    client_ip_from_request,
)

# ── Client IP extraction (X-Forwarded-For aware) ───────────────────────────


def _fake_request(headers: dict[str, str], peer: str | None = None) -> object:
    """Minimal FastAPI Request stub for client_ip_from_request tests."""

    class _Client:
        def __init__(self, host: str) -> None:
            self.host = host

    obj = MagicMock()
    obj.headers = headers
    obj.client = _Client(peer) if peer else None
    return obj


def test_client_ip_prefers_x_forwarded_for_leftmost() -> None:
    """XFF chain ``client, proxy1, proxy2`` -> the leftmost token is returned.

    HF Spaces sits behind multiple proxies; the rightmost entries are the
    HF infrastructure IPs and would collapse every visitor into one
    throttle bucket. The original client IP is always leftmost.
    """
    req = _fake_request(
        {"x-forwarded-for": "203.0.113.42, 10.0.0.1, 10.0.0.2"},
        peer="172.16.0.1",
    )
    assert client_ip_from_request(req) == "203.0.113.42"


def test_client_ip_falls_back_to_x_real_ip() -> None:
    """X-Real-IP wins over the socket peer when XFF is absent."""
    req = _fake_request({"x-real-ip": "198.51.100.7"}, peer="172.16.0.1")
    assert client_ip_from_request(req) == "198.51.100.7"


def test_client_ip_falls_back_to_socket_peer() -> None:
    """No proxy headers -> use request.client.host."""
    req = _fake_request({}, peer="172.16.0.50")
    assert client_ip_from_request(req) == "172.16.0.50"


def test_client_ip_anonymous_when_everything_missing() -> None:
    """Returns the literal ``anon`` string -- the throttle never crashes on None."""
    req = _fake_request({}, peer=None)
    assert client_ip_from_request(req) == "anon"

# ── Pure factory tests ──────────────────────────────────────────────────────


def test_build_creds_full_payload() -> None:
    """All five headers populated, client host known."""
    creds = build_creds(
        user_key="sk-test",
        provider="groq",
        ollama_url=None,
        session_id="abc-123",
        demo_persona="engineer",
        client_host="1.2.3.4",
    )
    assert isinstance(creds, ByokCreds)
    assert creds.user_key == "sk-test"
    assert creds.provider == "groq"
    assert creds.session_id == "abc-123"
    assert creds.demo_persona == "engineer"
    assert creds.has_user_key() is True
    assert creds.safe_provider() == "groq"


def test_build_creds_generates_session_id_when_missing() -> None:
    """No client-supplied session → server-generates a 17-char ID."""
    creds = build_creds(
        user_key=None,
        provider=None,
        ollama_url=None,
        session_id=None,
        demo_persona=None,
        client_host="1.2.3.4",
    )
    # format: 8-char-host-hash + "-" + 8-char-random-uuid
    assert len(creds.session_id) == 17
    assert creds.session_id[8] == "-"
    assert creds.has_user_key() is False


def test_build_creds_generates_session_id_when_blank() -> None:
    """Whitespace-only session header treated as missing."""
    creds = build_creds(
        user_key=None,
        provider=None,
        ollama_url=None,
        session_id="   ",
        demo_persona=None,
        client_host="1.2.3.4",
    )
    assert len(creds.session_id) == 17


def test_build_creds_handles_anonymous_client() -> None:
    """No client host → falls back to ``anon`` digest. Same shape, same length."""
    creds = build_creds(
        user_key=None,
        provider=None,
        ollama_url=None,
        session_id=None,
        demo_persona=None,
        client_host=None,
    )
    assert len(creds.session_id) == 17


def test_derive_session_id_sticky_for_same_host() -> None:
    """Host hash prefix is deterministic — sticky to a single client.

    Important for keeping a returning visitor on the same Qdrant collection
    within a session. The random UUID suffix prevents indefinite stickiness.
    """
    a = _derive_session_id("1.2.3.4")
    b = _derive_session_id("1.2.3.4")
    assert a[:8] == b[:8]  # same host prefix
    assert a[9:] != b[9:]  # different random tail


def test_derive_session_id_unique_for_different_hosts() -> None:
    """Different client hosts → different prefixes."""
    assert _derive_session_id("1.2.3.4")[:8] != _derive_session_id("5.6.7.8")[:8]


# ── BYOK semantics ──────────────────────────────────────────────────────────


def test_has_user_key_false_for_empty_string() -> None:
    """Empty string is the same as missing — owner-key fallback path."""
    creds = build_creds(
        user_key="",
        provider=None,
        ollama_url=None,
        session_id="sess",
        demo_persona=None,
        client_host="1.2.3.4",
    )
    assert creds.has_user_key() is False


def test_has_user_key_false_for_whitespace_only() -> None:
    """Pydantic str_strip_whitespace coerces ``'   '`` to ``''`` → not a key."""
    creds = build_creds(
        user_key="   ",
        provider=None,
        ollama_url=None,
        session_id="sess",
        demo_persona=None,
        client_host="1.2.3.4",
    )
    assert creds.has_user_key() is False


def test_safe_provider_rejects_unknown() -> None:
    """Bogus provider headers do NOT propagate downstream."""
    creds = build_creds(
        user_key="sk-x",
        provider="evil-provider-please-route-here",
        ollama_url=None,
        session_id="sess",
        demo_persona=None,
        client_host="1.2.3.4",
    )
    assert creds.safe_provider() is None


def test_safe_provider_normalises_case() -> None:
    """Provider header is case-insensitive."""
    creds = build_creds(
        user_key="sk-x",
        provider="GROQ",
        ollama_url=None,
        session_id="sess",
        demo_persona=None,
        client_host="1.2.3.4",
    )
    assert creds.safe_provider() == "groq"


def test_supported_providers_complete() -> None:
    """Documenting the allowlist so any future provider addition is intentional."""
    assert {"groq", "openai", "anthropic", "ollama"} == SUPPORTED_PROVIDERS


# ── Immutability ────────────────────────────────────────────────────────────


def test_creds_are_frozen() -> None:
    """``ByokCreds`` must be immutable so downstream code cannot mutate the key.

    Without this guard, a deeper function could silently substitute the
    owner key in for ``user_key`` and bypass the throttle.
    """
    creds = build_creds(
        user_key="sk-x",
        provider="groq",
        ollama_url=None,
        session_id="sess",
        demo_persona=None,
        client_host="1.2.3.4",
    )
    # Frozen Pydantic models raise pydantic.ValidationError on attribute set.
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        creds.user_key = "different-key"  # type: ignore[misc]


# ── FastAPI dependency (lightly mocked) ─────────────────────────────────────


def test_extract_byok_handles_no_client() -> None:
    """``request.client`` can be None (test clients, certain proxies)."""
    from interfaces.byok import extract_byok

    req = MagicMock()
    req.client = None
    creds = extract_byok(
        request=req,
        x_user_llm_key=None,
        x_user_provider=None,
        x_user_ollama_url=None,
        x_session_id=None,
        x_demo_persona=None,
    )
    assert len(creds.session_id) == 17


def test_extract_byok_forwards_all_headers() -> None:
    """The dependency wires every supported header into the model."""
    from interfaces.byok import extract_byok

    req = MagicMock()
    req.client.host = "10.0.0.1"
    creds = extract_byok(
        request=req,
        x_user_llm_key="my-key",
        x_user_provider="openai",
        x_user_ollama_url="http://localhost:11434",
        x_session_id="client-sess",
        x_demo_persona="compliance",
    )
    assert creds.user_key == "my-key"
    assert creds.safe_provider() == "openai"
    assert creds.ollama_url == "http://localhost:11434"
    assert creds.session_id == "client-sess"
    assert creds.demo_persona == "compliance"
