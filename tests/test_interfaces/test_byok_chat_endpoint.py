"""Integration tests for the /byok/chat FastAPI endpoint.

The endpoint exists only when ``settings.byok_mode=True``, so each test
patches the env, reloads ``interfaces.api``, and exercises the route via
TestClient. The RAG pipeline is mocked so the suite runs without Ollama
or Qdrant.
"""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

from config.settings import settings
from utils.rate_limiter import reset_owner_key_throttle


@pytest.fixture
def byok_app() -> Iterator:
    """Reload ``interfaces.api`` with byok_mode=True and yield a TestClient."""
    from fastapi.testclient import TestClient

    reset_owner_key_throttle()
    with (
        patch.object(settings, "byok_mode", True),
        patch.object(settings, "byok_owner_key_quota_per_hour", 3),
        patch.object(settings, "cors_allow_origins", ["https://app.eilm.live"]),
    ):
        # Force a fresh import so the conditional route registration runs.
        if "interfaces.api" in sys.modules:
            del sys.modules["interfaces.api"]
        api_mod = importlib.import_module("interfaces.api")
        client = TestClient(api_mod.app)
        try:
            yield client, api_mod
        finally:
            client.close()
            reset_owner_key_throttle()


def _make_state(text: str = "ok"):
    """Build a minimal QueryResponse-like state for the mocked pipeline."""

    state = {
        "answer": text,
        "user_id": "demo-user",
        "thread_id": "byok-test",
        "citations": [],
        "confidence": 0.9,
        "needs_human_review": False,
        "audit_trail": [],
        "graph_path": [],
    }
    # QueryResponse.from_state may need a dict-shaped state; instead build the
    # response directly and substitute it via the mock's return_value.
    return state


def _make_response():
    from core.schemas import QueryResponse

    return QueryResponse(
        answer="hello world",
        user_id="demo-user",
        thread_id="byok-test",
        citations=[],
        confidence=0.9,
        needs_human_review=False,
        audit_trail=[],
        graph_path=[],
    )


def test_byok_route_present_only_in_byok_mode() -> None:
    """Sanity baseline — without byok_mode the route is NOT registered."""
    from fastapi.testclient import TestClient

    with patch.object(settings, "byok_mode", False):
        if "interfaces.api" in sys.modules:
            del sys.modules["interfaces.api"]
        api_mod = importlib.import_module("interfaces.api")
        client = TestClient(api_mod.app)
        try:
            paths = {route.path for route in api_mod.app.routes}
            assert "/byok/chat" not in paths
        finally:
            client.close()


def test_byok_chat_uses_owner_key_when_no_user_key(byok_app) -> None:
    """No ``X-User-LLM-Key`` header → owner-key path, throttle consumes a slot."""
    client, _ = byok_app
    fake_resp = _make_response()

    # ``QueryResponse.from_state`` expects a state; patch it alongside the
    # pipeline so the endpoint flows regardless of the real schema shape.
    with (
        patch("interfaces.api.run_rag_pipeline", new=AsyncMock(return_value=fake_resp)),
        patch("interfaces.api.QueryResponse.from_state", return_value=fake_resp),
    ):
        r = client.post("/byok/chat", json={"query": "ping"})
    assert r.status_code == 200
    body = r.json()
    assert body["byok_used"] is False
    assert body["persona"] == "anonymous"


def test_byok_chat_owner_key_quota_returns_429_after_three(byok_app) -> None:
    """Same IP → fourth owner-key call returns 429 with BYOK hint."""
    client, _ = byok_app
    fake_resp = _make_response()
    with (
        patch("interfaces.api.run_rag_pipeline", new=AsyncMock(return_value=fake_resp)),
        patch("interfaces.api.QueryResponse.from_state", return_value=fake_resp),
    ):
        for _ in range(3):
            r = client.post("/byok/chat", json={"query": "ping"})
            assert r.status_code == 200
        r4 = client.post("/byok/chat", json={"query": "ping"})
    assert r4.status_code == 429
    detail = r4.json()["detail"]
    assert detail["reason"] == "owner_key_hourly_quota_exhausted"
    assert "your own LLM key" in detail["hint"]


def test_byok_chat_with_user_key_bypasses_throttle(byok_app) -> None:
    """Visitor BYOK never consumes owner-key quota."""
    client, _ = byok_app
    fake_resp = _make_response()
    with (
        patch("interfaces.api.run_rag_pipeline", new=AsyncMock(return_value=fake_resp)),
        patch("interfaces.api.QueryResponse.from_state", return_value=fake_resp),
    ):
        for _ in range(10):  # 10 > quota_per_hour=3
            r = client.post(
                "/byok/chat",
                json={"query": "ping"},
                headers={
                    "X-User-LLM-Key": "sk-visitor",
                    "X-User-Provider": "groq",
                },
            )
            assert r.status_code == 200
            assert r.json()["byok_used"] is True


def test_byok_chat_translates_persona_to_user_ctx(byok_app) -> None:
    """Persona header reaches the run_rag_pipeline call as the right UserContext."""
    client, _ = byok_app
    fake_resp = _make_response()
    seen_ctx = {}

    async def _capture(**kwargs):
        seen_ctx["user_context"] = kwargs["user_context"]
        return fake_resp

    with (
        patch("interfaces.api.run_rag_pipeline", new=AsyncMock(side_effect=_capture)),
        patch("interfaces.api.QueryResponse.from_state", return_value=fake_resp),
    ):
        r = client.post(
            "/byok/chat",
            json={"query": "ping"},
            headers={"X-Demo-Persona": "compliance"},
        )
    assert r.status_code == 200
    ctx = seen_ctx["user_context"]
    assert ctx.org_id == "demo-compliance"
    assert ctx.clearance_level == 4
    assert "compliance" in ctx.roles


def test_byok_chat_unknown_persona_falls_back_to_anon_clearance_1(byok_app) -> None:
    """Bogus persona must NOT leak into a higher clearance bucket."""
    client, _ = byok_app
    fake_resp = _make_response()
    seen_ctx = {}

    async def _capture(**kwargs):
        seen_ctx["user_context"] = kwargs["user_context"]
        return fake_resp

    with (
        patch("interfaces.api.run_rag_pipeline", new=AsyncMock(side_effect=_capture)),
        patch("interfaces.api.QueryResponse.from_state", return_value=fake_resp),
    ):
        r = client.post(
            "/byok/chat",
            json={"query": "ping"},
            headers={"X-Demo-Persona": "ceo-of-everything"},
        )
    assert r.status_code == 200
    assert seen_ctx["user_context"].clearance_level == 1
    assert seen_ctx["user_context"].org_id == "demo-anon"


def test_cors_middleware_allowlist_enforced(byok_app) -> None:
    """Origin in allowlist gets CORS headers; foreign origin does NOT."""
    client, _ = byok_app
    r_ok = client.options(
        "/byok/chat",
        headers={
            "Origin": "https://app.eilm.live",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert r_ok.headers.get("access-control-allow-origin") == "https://app.eilm.live"

    r_bad = client.options(
        "/byok/chat",
        headers={
            "Origin": "https://evil.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    # foreign origin → no allow-origin header echoed back
    assert "evil.example.com" not in (r_bad.headers.get("access-control-allow-origin") or "")
