"""Regression tests: the visitor's BYOK key actually powers their request.

Before this wiring, ``X-User-LLM-Key`` / ``X-User-Provider`` were parsed into
``ByokCreds`` and then ignored — the pipeline always used the owner's cached
client, and a bare key still bypassed the per-IP owner-key throttle. These tests
pin the contract so that never regresses:

1. ``ByokRuntime.is_active`` / ``ByokCreds.byok_active`` gate correctly.
2. ``InferenceRouter._client_for`` builds a *fresh per-request* client bound to
   the visitor key when (and only when) an active BYOK runtime matches the
   chosen provider; otherwise it returns the owner's cached client.
3. ``InferenceRouter.route`` honours the BYOK provider like an override — but
   the HIGH-sensitivity local guard still wins on a self-hosted deploy.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from config.settings import settings
from inference.byok_context import ByokRuntime, reset_byok_runtime, set_byok_runtime
from inference.cloud_clients import GroqClient
from inference.router import InferenceRouter
from interfaces.byok import ByokCreds

# ── ByokRuntime.is_active ────────────────────────────────────────────────────


def test_runtime_is_active_key_providers() -> None:
    assert ByokRuntime(provider="groq", user_key="sk-x").is_active()
    assert not ByokRuntime(provider="groq", user_key="").is_active()
    assert not ByokRuntime(provider="groq", user_key=None).is_active()
    assert not ByokRuntime(provider="groq", user_key="   ").is_active()


def test_runtime_is_active_ollama_needs_url() -> None:
    assert ByokRuntime(provider="ollama", ollama_url="http://x.example.com:11434").is_active()
    assert not ByokRuntime(provider="ollama").is_active()
    assert not ByokRuntime(provider=None).is_active()


# ── ByokCreds.byok_active ────────────────────────────────────────────────────


def test_creds_byok_active() -> None:
    assert ByokCreds(session_id="s", user_key="sk", provider="groq").byok_active()
    # key without a usable provider must NOT count (closes the throttle bypass)
    assert not ByokCreds(session_id="s", user_key="sk", provider="evil").byok_active()
    assert not ByokCreds(session_id="s", user_key="sk").byok_active()
    # provider without a key must NOT count
    assert not ByokCreds(session_id="s", provider="groq").byok_active()
    assert ByokCreds(
        session_id="s", provider="ollama", ollama_url="http://x.example.com:11434"
    ).byok_active()
    assert not ByokCreds(session_id="s", provider="ollama").byok_active()


# ── _client_for ──────────────────────────────────────────────────────────────


def test_client_for_builds_visitor_client_when_active() -> None:
    tok = set_byok_runtime(ByokRuntime(provider="groq", user_key="sk-visitor"))
    try:
        client, ephemeral = InferenceRouter._client_for("groq", "llama-3.1-8b-instant")
    finally:
        reset_byok_runtime(tok)
    assert isinstance(client, GroqClient)
    assert client.api_key == "sk-visitor"  # the VISITOR key, not the owner's
    assert ephemeral is True
    # Close the per-request httpx client we just built so it does not leak an
    # open socket into a later test's teardown (Windows selector-loop artifact).
    asyncio.run(client.close())


def test_client_for_uses_owner_client_when_no_runtime() -> None:
    sentinel = MagicMock()
    with patch("inference.router.get_llm", return_value=sentinel) as g:
        client, ephemeral = InferenceRouter._client_for("groq", "m")
    assert client is sentinel
    assert ephemeral is False
    g.assert_called_once()


def test_client_for_uses_owner_client_on_provider_mismatch() -> None:
    """Visitor brought a groq key but the router decided ollama (HIGH→local):
    the owner client is used, never the visitor key on the wrong provider."""
    sentinel = MagicMock()
    tok = set_byok_runtime(ByokRuntime(provider="groq", user_key="sk-visitor"))
    try:
        with patch("inference.router.get_llm", return_value=sentinel):
            client, ephemeral = InferenceRouter._client_for("ollama", "qwen3:8b")
    finally:
        reset_byok_runtime(tok)
    assert client is sentinel
    assert ephemeral is False


# ── route() honours BYOK provider, but HIGH guard still wins ─────────────────


def test_route_honours_byok_provider_for_low() -> None:
    r = InferenceRouter()
    tok = set_byok_runtime(ByokRuntime(provider="groq", user_key="sk-x"))
    try:
        decision = r.route(sensitivity_level="low")
    finally:
        reset_byok_runtime(tok)
    assert decision.provider == "groq"


def test_route_byok_cannot_move_high_off_local_when_enforced() -> None:
    r = InferenceRouter(force_local_for_sensitive=True)
    tok = set_byok_runtime(ByokRuntime(provider="groq", user_key="sk-x"))
    try:
        with patch.object(settings, "allow_cloud_for_high", False):
            decision = r.route(sensitivity_level="high")
    finally:
        reset_byok_runtime(tok)
    assert decision.provider == "ollama"
    assert decision.forced_local is True


def test_route_byok_high_allowed_when_cloud_unlocked() -> None:
    """Hosted demo (SAR_ALLOW_CLOUD_FOR_HIGH=true): visitor key powers HIGH too."""
    r = InferenceRouter(force_local_for_sensitive=True)
    tok = set_byok_runtime(ByokRuntime(provider="groq", user_key="sk-x"))
    try:
        with patch.object(settings, "allow_cloud_for_high", True):
            decision = r.route(sensitivity_level="high")
    finally:
        reset_byok_runtime(tok)
    assert decision.provider == "groq"
