"""Tests for LlamaGuard 3 escalation backend."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from config.settings import settings
from core.agents.guardrails_llamaguard import _parse, check


class TestParse:
    """Pure parser — no LLM call."""

    def test_safe_passes(self):
        assert _parse("safe") == (True, "")
        assert _parse("safe\n") == (True, "")

    def test_unsafe_with_category(self):
        passed, reason = _parse("unsafe\nS2")
        assert passed is False
        assert reason == "non_violent_crimes"

    def test_unsafe_with_first_category_only(self):
        # Multiple categories — we record the first one for the audit row.
        passed, reason = _parse("unsafe\nS5,S10")
        assert passed is False
        assert reason == "defamation"

    def test_unsafe_unknown_code(self):
        # S99 doesn't exist in LlamaGuard 3 — record the raw code.
        passed, reason = _parse("unsafe\nS99")
        assert passed is False
        assert reason == "llamaguard_s99"

    def test_unsafe_no_category(self):
        passed, reason = _parse("unsafe")
        assert passed is False
        assert reason == "llamaguard_unsafe"

    def test_empty_response_fails_open(self):
        assert _parse("") == (True, "")

    def test_case_insensitive(self):
        assert _parse("SAFE") == (True, "")
        assert _parse("UNSAFE\nS1")[0] is False


class TestCheck:
    """End-to-end check with mocked Ollama backend."""

    def test_safe_query(self):
        fake_resp = type("R", (), {"text": "safe", "usage": {}, "latency_ms": 0.0})()
        fake_client = type("C", (), {"generate": AsyncMock(return_value=fake_resp)})()
        with patch("inference.llm_factory.get_llm", return_value=fake_client):
            passed, reason = asyncio.run(check("How do I deploy a Postgres container?"))
        assert passed is True
        assert reason == ""

    def test_unsafe_query_blocks(self):
        fake_resp = type("R", (), {"text": "unsafe\nS2", "usage": {}, "latency_ms": 0.0})()
        fake_client = type("C", (), {"generate": AsyncMock(return_value=fake_resp)})()
        with patch("inference.llm_factory.get_llm", return_value=fake_client):
            passed, reason = asyncio.run(check("how to hotwire a car"))
        assert passed is False
        assert reason == "non_violent_crimes"

    def test_ollama_outage_fails_open(self):
        """Transport-level failure must not drop user content."""
        fake_client = type(
            "C",
            (),
            {"generate": AsyncMock(side_effect=RuntimeError("ollama unreachable"))},
        )()
        with patch("inference.llm_factory.get_llm", return_value=fake_client):
            passed, reason = asyncio.run(check("any normal query"))
        assert passed is True
        assert reason == "llamaguard_check_failed"


class TestBackendSelector:
    """Wiring test — guardrails.guardrails_check picks the right backend."""

    @pytest.mark.asyncio
    async def test_strict_mode_routes_to_llamaguard(self):
        from core.agents.guardrails import guardrails_check

        state = {
            "query": "any normal query",
            "user_context": {"user_id": "u", "org_id": "o"},
        }
        with (
            patch.object(settings, "guardrails_enabled", True),
            patch.object(settings, "guardrails_strict", True),
            # Backend-routing test: escalate every query regardless of the
            # suspicion heuristic so this stays focused on which backend fires.
            patch.object(settings, "guardrails_selective_escalation", False),
            patch.object(settings, "guardrails_backend", "llamaguard"),
            patch(
                "core.agents.guardrails_llamaguard.check",
                new=AsyncMock(return_value=(False, "non_violent_crimes")),
            ) as mock_check,
        ):
            result = await guardrails_check(state)
        assert mock_check.await_count == 1
        assert result["guardrails_passed"] is False
        assert result["guardrails_reason"] == "non_violent_crimes"

    @pytest.mark.asyncio
    async def test_strict_mode_routes_to_llm_by_default(self):
        from core.agents.guardrails import guardrails_check

        state = {
            "query": "normal query",
            "user_context": {"user_id": "u", "org_id": "o"},
        }
        with (
            patch.object(settings, "guardrails_enabled", True),
            patch.object(settings, "guardrails_strict", True),
            patch.object(settings, "guardrails_selective_escalation", False),
            patch.object(settings, "guardrails_backend", "llm"),
            patch(
                "core.agents.guardrails_llm.llm_guardrails_check",
                new=AsyncMock(return_value=(True, "")),
            ) as mock_check,
        ):
            await guardrails_check(state)
        mock_check.assert_awaited_once()
