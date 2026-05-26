"""Tests for the BYOK per-request LLM client factories.

The factories must:
- return a fresh client instance each call (no shared mutable state)
- bind the visitor's key/url to *that* instance only — never to a module
  singleton that could leak to subsequent owner-key calls
- reject empty / unsupported inputs loudly

See ``launch-plan/03-backend-byok.md`` § Cloud + Ollama per-request key
override and ``launch-plan/11-security-checklist.md`` § Per-request key.
"""

from __future__ import annotations

import pytest

from inference.cloud_clients import (
    AnthropicClient,
    GroqClient,
    OpenAIClient,
    make_byok_cloud_client,
)
from inference.ollama_client import OllamaClient, make_byok_ollama_client

# ── Cloud factory ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "provider, expected_cls, expected_default_model",
    [
        ("groq", GroqClient, "llama-3.1-8b-instant"),
        ("openai", OpenAIClient, "gpt-4o-mini"),
        ("anthropic", AnthropicClient, "claude-sonnet-4-20250514"),
    ],
)
def test_make_byok_cloud_client_routes_by_provider(
    provider: str, expected_cls: type, expected_default_model: str
) -> None:
    c = make_byok_cloud_client(provider=provider, user_key="sk-test-key")
    assert isinstance(c, expected_cls)
    assert c.api_key == "sk-test-key"
    assert c.model == expected_default_model


def test_make_byok_cloud_client_provider_case_insensitive() -> None:
    c = make_byok_cloud_client(provider="GROQ", user_key="sk-x")
    assert isinstance(c, GroqClient)


def test_make_byok_cloud_client_model_override() -> None:
    c = make_byok_cloud_client(provider="groq", user_key="sk-x", model="mixtral-8x7b-32768")
    assert c.model == "mixtral-8x7b-32768"


def test_make_byok_cloud_client_rejects_blank_key() -> None:
    with pytest.raises(ValueError, match="without a user key"):
        make_byok_cloud_client(provider="groq", user_key="")
    with pytest.raises(ValueError, match="without a user key"):
        make_byok_cloud_client(provider="groq", user_key="   ")


def test_make_byok_cloud_client_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="not supported"):
        make_byok_cloud_client(provider="evil-llm", user_key="sk-x")


def test_make_byok_cloud_client_strips_whitespace_from_key() -> None:
    """Trailing newlines from clipboard pastes must not break the auth header."""
    c = make_byok_cloud_client(provider="groq", user_key="  sk-x  \n")
    assert c.api_key == "sk-x"


def test_make_byok_cloud_client_returns_fresh_instance_each_call() -> None:
    """Two BYOK calls with different keys must NOT share state.

    Regression guard for the class of bug where a memoised singleton would
    silently keep visitor-A's key and re-use it for visitor-B's request.
    """
    a = make_byok_cloud_client(provider="groq", user_key="sk-visitor-A")
    b = make_byok_cloud_client(provider="groq", user_key="sk-visitor-B")
    assert a is not b
    assert a.api_key == "sk-visitor-A"
    assert b.api_key == "sk-visitor-B"


# ── Ollama factory ──────────────────────────────────────────────────────────


def test_make_byok_ollama_client_binds_url() -> None:
    c = make_byok_ollama_client(base_url="http://my-ollama.example.com:11434")
    assert isinstance(c, OllamaClient)
    assert c.base_url == "http://my-ollama.example.com:11434"


def test_make_byok_ollama_client_strips_whitespace_and_trailing_slash() -> None:
    """OllamaClient.__init__ already strips trailing slash — we add whitespace strip on top."""
    c = make_byok_ollama_client(base_url="  http://my-ollama.example.com/  ")
    assert c.base_url == "http://my-ollama.example.com"


def test_make_byok_ollama_client_rejects_blank_url() -> None:
    with pytest.raises(ValueError, match="without a base_url"):
        make_byok_ollama_client(base_url="")
    with pytest.raises(ValueError, match="without a base_url"):
        make_byok_ollama_client(base_url="   ")


def test_make_byok_ollama_client_returns_fresh_instance_each_call() -> None:
    """No shared state across visitors."""
    a = make_byok_ollama_client(base_url="http://a.example.com:11434")
    b = make_byok_ollama_client(base_url="http://b.example.com:11434")
    assert a is not b
    assert a.base_url != b.base_url


def test_make_byok_ollama_client_model_override() -> None:
    c = make_byok_ollama_client(base_url="http://a:11434", model="qwen3:8b")
    assert c.model == "qwen3:8b"


# ── State-leakage regression ────────────────────────────────────────────────


def test_owner_default_client_unaffected_by_byok_calls() -> None:
    """Building a BYOK client must NOT mutate any module-level state.

    We don't have a module-level default cloud client today, but if one is
    ever added (e.g. ``inference.llm_factory.get_default_cloud_client``),
    this test should be extended to assert it stays bound to the owner key
    after BYOK calls.
    """
    from inference.cloud_clients import settings as cc_settings

    owner_key_before = cc_settings.groq_api_key
    make_byok_cloud_client(provider="groq", user_key="sk-visitor")
    owner_key_after = cc_settings.groq_api_key
    assert owner_key_before == owner_key_after
