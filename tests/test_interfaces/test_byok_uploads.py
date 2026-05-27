"""Integration tests for the BYOK upload endpoints.

The endpoint group exists only when ``settings.byok_mode=True``. We patch
the runtime hooks (``_get_hybrid_searcher`` and the ingestion pipeline)
so the suite never touches Qdrant Cloud or sentence-transformers, but
still exercises the public validation surface end-to-end through the
FastAPI dependency stack.
"""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

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
        patch.object(settings, "cors_allow_origins", ["https://example.com"]),
        patch.object(settings, "byok_upload_max_bytes", 1024),
        patch.object(settings, "byok_upload_max_files", 2),
        patch.object(settings, "byok_upload_allowed_extensions", [".txt", ".md"]),
    ):
        if "interfaces.api" in sys.modules:
            del sys.modules["interfaces.api"]
        api_mod = importlib.import_module("interfaces.api")
        client = TestClient(api_mod.app)
        try:
            yield client, api_mod
        finally:
            client.close()
            reset_owner_key_throttle()


def _stub_searcher(initial_uploads: list[dict] | None = None) -> MagicMock:
    """Return a fake hybrid-searcher whose `_qdrant.for_session` returns a manager
    with a `client.scroll` / `client.set_payload` / `client.delete` interface."""
    fake_client = MagicMock()
    # Initial scroll: return points carrying the source_file payload so the
    # list endpoint can group + count them.
    points = []
    for entry in initial_uploads or []:
        pt = MagicMock()
        pt.payload = {
            "source_file": entry["filename"],
            "source_file_id": entry.get("file_id", entry["filename"]),
            "ingested_at": "2026-05-27T00:00:00Z",
        }
        points.append(pt)
    fake_client.scroll.return_value = (points, None)
    fake_client.set_payload.return_value = None
    fake_client.delete.return_value = None
    fake_client.count.return_value = MagicMock(count=len(points))

    sess_mgr = MagicMock()
    sess_mgr.client = fake_client
    sess_mgr.collection_name = "documents_sess_test"

    searcher = MagicMock()
    searcher._qdrant.for_session.return_value = sess_mgr
    searcher._embeddings = MagicMock()
    searcher._sparse = MagicMock()
    return searcher


def test_uploads_list_empty_session(byok_app) -> None:
    client, _ = byok_app
    with patch("interfaces.api._get_hybrid_searcher", return_value=_stub_searcher()):
        r = client.get("/byok/uploads", headers={"X-Session-ID": "sess-1"})
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == "sess-1"
    assert body["count"] == 0
    assert body["max_files"] == 2
    assert body["max_bytes"] == 1024
    assert ".txt" in body["allowed_extensions"]


def test_uploads_rejects_unsupported_extension(byok_app) -> None:
    client, _ = byok_app
    with patch("interfaces.api._get_hybrid_searcher", return_value=_stub_searcher()):
        r = client.post(
            "/byok/uploads",
            headers={"X-Session-ID": "sess-x"},
            files={"file": ("hello.exe", b"binary", "application/octet-stream")},
        )
    assert r.status_code == 400
    assert r.json()["detail"]["reason"] == "unsupported_extension"


def test_uploads_rejects_oversize(byok_app) -> None:
    """File > byok_upload_max_bytes (1024 in fixture) -> 413."""
    client, _ = byok_app
    big = b"x" * 2048
    with patch("interfaces.api._get_hybrid_searcher", return_value=_stub_searcher()):
        r = client.post(
            "/byok/uploads",
            headers={"X-Session-ID": "sess-x"},
            files={"file": ("hello.txt", big, "text/plain")},
        )
    assert r.status_code == 413
    assert r.json()["detail"]["reason"] == "file_too_large"


def test_uploads_rejects_empty(byok_app) -> None:
    client, _ = byok_app
    with patch("interfaces.api._get_hybrid_searcher", return_value=_stub_searcher()):
        r = client.post(
            "/byok/uploads",
            headers={"X-Session-ID": "sess-x"},
            files={"file": ("hello.txt", b"", "text/plain")},
        )
    assert r.status_code == 400
    assert r.json()["detail"]["reason"] == "empty_file"


def test_uploads_enforces_file_count_cap(byok_app) -> None:
    """When max_files (2) is reached, a 3rd upload returns 409."""
    client, _ = byok_app
    existing = [
        {"filename": "a.txt", "file_id": "id-a"},
        {"filename": "b.txt", "file_id": "id-b"},
    ]
    with patch(
        "interfaces.api._get_hybrid_searcher",
        return_value=_stub_searcher(initial_uploads=existing),
    ):
        r = client.post(
            "/byok/uploads",
            headers={"X-Session-ID": "sess-x"},
            files={"file": ("c.txt", b"content", "text/plain")},
        )
    assert r.status_code == 409
    assert r.json()["detail"]["reason"] == "upload_quota_exceeded"


def test_uploads_happy_path(byok_app) -> None:
    """Valid .txt upload -> 200 with ingest result + payload tag applied."""
    client, _ = byok_app
    searcher = _stub_searcher()

    # Mock the actual ingestion pipeline so we don't touch sentence-transformers.
    result = MagicMock()
    result.point_ids = ["p1", "p2"]
    result.num_chunks = 2
    result.status = "success"
    result.errors = []
    result.processing_time_seconds = 0.5

    fake_pipeline = MagicMock()
    fake_pipeline.ingest_document = AsyncMock(return_value=result)

    with (
        patch("interfaces.api._get_hybrid_searcher", return_value=searcher),
        patch("interfaces.api.IngestionPipeline", return_value=fake_pipeline),
    ):
        r = client.post(
            "/byok/uploads",
            headers={"X-Session-ID": "sess-x"},
            files={"file": ("notes.txt", b"hello world content", "text/plain")},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == "sess-x"
    assert body["status"] == "success"
    assert body["chunks"] == 2
    assert body["filename"] == "notes.txt"
    # file_id is a uuid hex string (32 chars).
    assert isinstance(body["file_id"], str) and len(body["file_id"]) == 32

    # Tag write must include source_file_id + ingested_at on the new point ids.
    sess_mgr = searcher._qdrant.for_session.return_value
    sess_mgr.client.set_payload.assert_called_once()
    payload_kwargs = sess_mgr.client.set_payload.call_args.kwargs
    assert "source_file_id" in payload_kwargs["payload"]
    assert "ingested_at" in payload_kwargs["payload"]
    assert payload_kwargs["points"] == ["p1", "p2"]


def test_uploads_delete_drops_points(byok_app) -> None:
    client, _ = byok_app
    searcher = _stub_searcher()
    with patch("interfaces.api._get_hybrid_searcher", return_value=searcher):
        r = client.delete(
            "/byok/uploads/file-123",
            headers={"X-Session-ID": "sess-x"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["file_id"] == "file-123"
    # Defaults to count=0 in our stub, but client.delete must still be called.
    sess_mgr = searcher._qdrant.for_session.return_value
    sess_mgr.client.delete.assert_called_once()


def test_uploads_listing_groups_by_source_file(byok_app) -> None:
    client, _ = byok_app
    initial = [
        {"filename": "a.txt", "file_id": "id-a"},
        {"filename": "a.txt", "file_id": "id-a"},
        {"filename": "b.txt", "file_id": "id-b"},
    ]
    with patch(
        "interfaces.api._get_hybrid_searcher",
        return_value=_stub_searcher(initial_uploads=initial),
    ):
        r = client.get("/byok/uploads", headers={"X-Session-ID": "sess-x"})
    body = r.json()
    assert body["count"] == 2
    by_filename = {item["filename"]: item for item in body["items"]}
    assert by_filename["a.txt"]["chunks"] == 2
    assert by_filename["b.txt"]["chunks"] == 1
