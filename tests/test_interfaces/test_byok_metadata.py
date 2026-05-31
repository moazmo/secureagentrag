"""Tests for the public metadata endpoints /byok/personas + /byok/corpus.

These endpoints power the frontend's ``/corpus`` and ``/personas`` pages.
They take no auth, no BYOK key, and they expose only metadata that is
already implied by the demo (filename, roles, sensitivity, chunk
counts) -- never raw chunk text. The tests pin that contract.
"""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

from config.settings import settings


@pytest.fixture
def byok_app() -> Iterator:
    from fastapi.testclient import TestClient

    with (
        patch.object(settings, "byok_mode", True),
        patch.object(settings, "cors_allow_origins", ["https://example.test"]),
    ):
        if "interfaces.api" in sys.modules:
            del sys.modules["interfaces.api"]
        api_mod = importlib.import_module("interfaces.api")
        client = TestClient(api_mod.app)
        try:
            yield client, api_mod
        finally:
            client.close()


def test_personas_endpoint_returns_three_presets(byok_app):
    client, _ = byok_app
    resp = client.get("/byok/personas")
    assert resp.status_code == 200
    body = resp.json()
    assert body["default"] == "engineer"
    assert body["org_id"] == "demo"
    keys = {p["key"] for p in body["items"]}
    assert keys == {"engineer", "compliance", "executive"}
    # Each preset must carry the four fields the frontend reads.
    for p in body["items"]:
        assert isinstance(p["clearance_level"], int)
        assert isinstance(p["roles"], list) and len(p["roles"]) >= 1
        assert isinstance(p["style"], str) and len(p["style"]) > 0
        assert p["label"] == p["key"].capitalize()


def test_personas_endpoint_clearance_matches_legacy_dispatch(byok_app):
    """Engineer < Compliance == Executive on the clearance ladder."""
    client, _ = byok_app
    items = {p["key"]: p for p in client.get("/byok/personas").json()["items"]}
    assert items["engineer"]["clearance_level"] == 2
    assert items["compliance"]["clearance_level"] == 3
    assert items["executive"]["clearance_level"] == 3
    # Executive's role bag must be a superset of engineering + compliance.
    assert "engineering" in items["executive"]["roles"]
    assert "compliance" in items["executive"]["roles"]


def _mock_searcher_with_corpus(monkeypatch, points):
    """Stub `_get_hybrid_searcher` so it returns a manager that scrolls `points`."""

    class _FakeClient:
        def scroll(self, **_kw):
            return points, None

    class _FakeQdrant:
        client = _FakeClient()
        collection_name = "documents_demo"

        def for_org(self, _org):
            return self

    class _FakeSearcher:
        _qdrant = _FakeQdrant()

    from core.agents import retriever as retriever_mod

    monkeypatch.setattr(retriever_mod, "_get_hybrid_searcher", lambda: _FakeSearcher())


def test_corpus_endpoint_groups_points_by_source_file(byok_app, monkeypatch):
    client, _api = byok_app
    points = [
        SimpleNamespace(
            payload={
                "source_file": "/data/policy.txt",
                "roles": ["engineering", "viewer"],
                "sensitivity_level": "medium",
            }
        ),
        SimpleNamespace(
            payload={
                "source_file": "/data/policy.txt",
                "roles": ["engineering", "viewer"],
                "sensitivity_level": "medium",
            }
        ),
        SimpleNamespace(
            payload={
                "source_file": "/data/finance_q3.txt",
                "roles": ["finance_manager", "executive"],
                "sensitivity_level": "high",
            }
        ),
    ]
    _mock_searcher_with_corpus(monkeypatch, points)

    resp = client.get("/byok/corpus")
    assert resp.status_code == 200
    body = resp.json()
    # The endpoint now reports the real collection name the searcher resolved
    # (the mock's QdrantManager.for_org yields ``documents_demo``) rather than a
    # hardcoded literal — ADR-040 F2.
    assert body["collection"] == "documents_demo"
    assert body["count"] == 2
    assert body["total_chunks"] == 3
    by_name = {f["source_file"]: f for f in body["items"]}
    assert by_name["policy.txt"]["chunks"] == 2
    assert by_name["policy.txt"]["sensitivity_level"] == "medium"
    assert by_name["finance_q3.txt"]["chunks"] == 1
    assert by_name["finance_q3.txt"]["sensitivity_level"] == "high"


def test_corpus_endpoint_never_returns_chunk_text(byok_app, monkeypatch):
    """Defense-in-depth: corpus browser must NEVER leak raw chunk content."""
    client, _api = byok_app
    points = [
        SimpleNamespace(
            payload={
                "source_file": "secret.txt",
                "roles": ["admin"],
                "sensitivity_level": "high",
                # If a future refactor pulled the text into the row, this
                # test should fail loudly.
                "text": "PRIVILEGED LITERAL THAT MUST NOT LEAK",
            }
        )
    ]
    _mock_searcher_with_corpus(monkeypatch, points)

    body = client.get("/byok/corpus").json()
    raw = repr(body)
    assert "PRIVILEGED LITERAL" not in raw
    assert "text" not in body["items"][0]


def test_corpus_endpoint_returns_empty_when_qdrant_unreachable(byok_app, monkeypatch):
    """Fail-open on transport: a Qdrant outage returns an empty corpus, not a 5xx."""
    client, _api = byok_app

    class _BrokenClient:
        def scroll(self, **_kw):
            raise RuntimeError("qdrant connection refused")

    class _BrokenQdrant:
        client = _BrokenClient()
        collection_name = "documents_demo"

        def for_org(self, _org):
            return self

    class _BrokenSearcher:
        _qdrant = _BrokenQdrant()

    from core.agents import retriever as retriever_mod

    monkeypatch.setattr(retriever_mod, "_get_hybrid_searcher", lambda: _BrokenSearcher())
    resp = client.get("/byok/corpus")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 0
    assert body["items"] == []


def test_metadata_endpoints_have_no_auth(byok_app):
    """Both endpoints must be reachable with zero headers."""
    client, _ = byok_app
    assert client.get("/byok/personas").status_code == 200
    # corpus may return empty when no searcher is configured, but must be 200.
    assert client.get("/byok/corpus").status_code == 200
