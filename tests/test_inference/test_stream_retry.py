"""Tests for streaming 429 / connection retry in the cloud client.

The live BYOK demo streams (SSE). A transient Groq per-minute 429 used to kill
the whole answer because the streaming path never retried. These tests pin the
new behaviour: retry-before-first-token, Retry-After honoured, give up cleanly
after the cap.
"""

from __future__ import annotations

import httpx
import pytest

from inference import cloud_clients as cc


class _FakeResp:
    def __init__(self, status: int, lines: list[str] | None = None, headers: dict | None = None):
        self.status_code = status
        self._lines = lines or []
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            req = httpx.Request("POST", "http://test/stream")
            raise httpx.HTTPStatusError(
                f"status {self.status_code}",
                request=req,
                response=httpx.Response(self.status_code, request=req),
            )

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeStreamCtx:
    def __init__(self, resp: _FakeResp):
        self._resp = resp

    async def __aenter__(self) -> _FakeResp:
        return self._resp

    async def __aexit__(self, *_a) -> bool:
        return False


class _FakeClient:
    """Returns a queued response per .stream() call; records call count."""

    def __init__(self, responses: list[_FakeResp]):
        self._responses = responses
        self.calls = 0

    def stream(self, _method, _url, headers=None, json=None):
        resp = self._responses[self.calls]
        self.calls += 1
        return _FakeStreamCtx(resp)


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Record sleeps instead of waiting."""
    sleeps: list[float] = []

    async def _fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(cc.asyncio, "sleep", _fake_sleep)
    return sleeps


def test_retry_after_seconds_parsing():
    assert cc._retry_after_seconds("3") == 3.0
    assert cc._retry_after_seconds("7.5") == 7.5
    assert cc._retry_after_seconds(None) is None
    assert cc._retry_after_seconds("Wed, 21 Oct 2099 07:28:00 GMT") is None


@pytest.mark.asyncio
async def test_stream_retries_429_then_succeeds(_no_real_sleep):
    client = _FakeClient(
        [
            _FakeResp(429, headers={"Retry-After": "3"}),
            _FakeResp(200, lines=["data: hello", "data: [DONE]"]),
        ]
    )
    lines = [
        line
        async for line in cc._stream_lines_with_retry(
            client, "http://test/stream", {}, {}, provider="groq"
        )
    ]
    assert client.calls == 2  # one retry
    assert _no_real_sleep == [3.0]  # honoured Retry-After
    assert "data: hello" in lines


@pytest.mark.asyncio
async def test_stream_gives_up_after_max_attempts(_no_real_sleep):
    client = _FakeClient([_FakeResp(429) for _ in range(cc._STREAM_MAX_ATTEMPTS)])
    with pytest.raises(httpx.HTTPStatusError):
        _ = [
            line
            async for line in cc._stream_lines_with_retry(
                client, "http://test/stream", {}, {}, provider="groq"
            )
        ]
    assert client.calls == cc._STREAM_MAX_ATTEMPTS
    # Slept between the first (MAX-1) attempts, not after the final failure.
    assert len(_no_real_sleep) == cc._STREAM_MAX_ATTEMPTS - 1


@pytest.mark.asyncio
async def test_stream_no_retry_on_clean_first_response(_no_real_sleep):
    client = _FakeClient([_FakeResp(200, lines=["data: ok", "data: [DONE]"])])
    lines = [
        line
        async for line in cc._stream_lines_with_retry(
            client, "http://test/stream", {}, {}, provider="groq"
        )
    ]
    assert client.calls == 1
    assert _no_real_sleep == []
    assert "data: ok" in lines


@pytest.mark.asyncio
async def test_call_llm_stream_rate_limit_copy(monkeypatch):
    """On an exhausted rate limit, the user-facing copy says 'per-minute / try
    again', not 'exhausted for the hour'."""
    from core.agents import router

    async def _boom(*_a, **_k):
        raise RuntimeError("HTTP 429 rate limit exceeded")
        yield  # pragma: no cover - makes this an async generator

    class _FakeRouter:
        def generate_stream_with_routing(self, *_a, **_k):
            return _boom()

    monkeypatch.setattr("inference.router.InferenceRouter", lambda: _FakeRouter())

    out = "".join([tok async for tok in router.call_llm_stream("q", sensitivity_level="low")])
    assert "per-minute" in out
    assert "exhausted for" not in out.lower()
