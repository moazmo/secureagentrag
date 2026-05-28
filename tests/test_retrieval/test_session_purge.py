"""Tests for the BYOK per-session Qdrant collection purge.

The purge sweep must:
- only touch collections matching the session prefix (never org collections,
  never the base collection)
- skip undated collections (we can't time-stamp = we don't delete)
- delete only collections older than the TTL
- survive a Qdrant SDK error mid-sweep (one bad collection != abort)

See ``launch-plan/03-backend-byok.md`` § Session purge cron.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from config.settings import settings
from retrieval.session_purge import (
    _is_session_collection,
    _parse_created_at,
    purge_expired_sessions,
    schedule_session_purge,
)


def _fake_collection(name: str, created_at: str | None) -> SimpleNamespace:
    """Build a mock Qdrant CollectionInfo with optional ``created_at`` metadata."""
    params = SimpleNamespace(metadata=({"created_at": created_at} if created_at else None))
    config = SimpleNamespace(params=params)
    return SimpleNamespace(name=name, config=config)


def _client_returning(collections: list, info_for: dict | None = None):
    """Build a client mock with deterministic list + get behaviour."""
    info_for = info_for or {}
    cli = MagicMock()
    cli.get_collections.return_value = SimpleNamespace(collections=collections)
    cli.get_collection.side_effect = lambda name: info_for.get(name, _fake_collection(name, None))
    return cli


# ── Naming predicates ───────────────────────────────────────────────────────


def test_is_session_collection_recognises_prefix() -> None:
    base = settings.qdrant_collection
    assert _is_session_collection(f"{base}_sess_abc") is True
    assert _is_session_collection(f"{base}_sess_") is True  # empty session_id still has prefix


def test_is_session_collection_rejects_org_collection() -> None:
    """Org collections must NEVER be deleted by this sweep."""
    base = settings.qdrant_collection
    assert _is_session_collection(f"{base}_acme_corp") is False


def test_is_session_collection_rejects_base() -> None:
    """Base collection itself must NEVER be deleted by this sweep."""
    assert _is_session_collection(settings.qdrant_collection) is False


# ── Timestamp parsing ───────────────────────────────────────────────────────


def test_parse_created_at_handles_iso_utc() -> None:
    out = _parse_created_at({"created_at": "2026-05-26T13:00:00+00:00"})
    assert out == datetime(2026, 5, 26, 13, 0, tzinfo=UTC)


def test_parse_created_at_handles_trailing_z() -> None:
    out = _parse_created_at({"created_at": "2026-05-26T13:00:00Z"})
    assert out == datetime(2026, 5, 26, 13, 0, tzinfo=UTC)


def test_parse_created_at_returns_none_for_missing_meta() -> None:
    assert _parse_created_at(None) is None
    assert _parse_created_at({}) is None


def test_parse_created_at_returns_none_for_bad_value() -> None:
    assert _parse_created_at({"created_at": "not-a-date"}) is None


# ── Purge sweep ─────────────────────────────────────────────────────────────


def test_purge_deletes_only_expired_session_collections() -> None:
    base = settings.qdrant_collection
    now = datetime(2026, 5, 26, 13, 0, tzinfo=UTC)
    horizon = now - timedelta(hours=24)
    collections = [
        _fake_collection(f"{base}_sess_old", (horizon - timedelta(hours=1)).isoformat()),
        _fake_collection(f"{base}_sess_fresh", (horizon + timedelta(hours=2)).isoformat()),
        _fake_collection(f"{base}_acme_corp", (horizon - timedelta(hours=48)).isoformat()),
        _fake_collection(base, (horizon - timedelta(hours=72)).isoformat()),
    ]
    info_for = {c.name: c for c in collections}
    cli = _client_returning(collections, info_for)

    summary = purge_expired_sessions(cli, ttl_hours=24, now=now)

    cli.delete_collection.assert_called_once_with(f"{base}_sess_old")
    assert summary["deleted"] == 1
    assert summary["inspected"] == 2  # only sess_old + sess_fresh
    assert f"{base}_sess_old" in summary["deleted_names"]


def test_purge_skips_undated_session_collections() -> None:
    """Defensive: collections without ``created_at`` are skipped, not deleted."""
    base = settings.qdrant_collection
    collections = [_fake_collection(f"{base}_sess_legacy", None)]
    cli = _client_returning(collections, {c.name: c for c in collections})

    summary = purge_expired_sessions(cli, ttl_hours=24, now=datetime.now(UTC))

    cli.delete_collection.assert_not_called()
    assert summary["deleted"] == 0
    assert summary["skipped"] == 1


def test_purge_continues_after_collection_error() -> None:
    """A single bad collection does NOT stop the rest of the sweep."""
    base = settings.qdrant_collection
    now = datetime(2026, 5, 26, 13, 0, tzinfo=UTC)
    old_iso = (now - timedelta(hours=48)).isoformat()
    collections = [
        _fake_collection(f"{base}_sess_explode", old_iso),
        _fake_collection(f"{base}_sess_ok", old_iso),
    ]
    info_for = {c.name: c for c in collections}
    cli = _client_returning(collections, info_for)

    def _delete(name: str) -> None:
        if name == f"{base}_sess_explode":
            raise RuntimeError("qdrant 500")

    cli.delete_collection.side_effect = _delete
    summary = purge_expired_sessions(cli, ttl_hours=24, now=now)

    assert summary["deleted"] == 1
    assert summary["errors"] == 1
    assert f"{base}_sess_ok" in summary["deleted_names"]


def test_purge_handles_list_collections_failure() -> None:
    """If even the initial list call fails, return a clean error summary."""
    cli = MagicMock()
    cli.get_collections.side_effect = RuntimeError("qdrant unreachable")
    summary = purge_expired_sessions(cli, ttl_hours=24, now=datetime.now(UTC))
    assert summary["errors"] == 1
    assert summary["deleted"] == 0


def test_purge_respects_ttl_from_settings_by_default() -> None:
    """When ``ttl_hours`` is None, the function reads from settings."""
    base = settings.qdrant_collection
    now = datetime(2026, 5, 26, 13, 0, tzinfo=UTC)
    # Collection 12h old; default TTL is 24h, so it should NOT be deleted.
    collections = [_fake_collection(f"{base}_sess_x", (now - timedelta(hours=12)).isoformat())]
    cli = _client_returning(collections, {c.name: c for c in collections})

    summary = purge_expired_sessions(cli, now=now)

    cli.delete_collection.assert_not_called()
    assert summary["deleted"] == 0
    assert summary["ttl_hours"] == settings.session_collection_ttl_hours


# ── Scheduler hook ──────────────────────────────────────────────────────────


def test_schedule_skips_when_byok_mode_off() -> None:
    """Sweep is wasted work when BYOK isn't on — no session collections exist."""
    cli = MagicMock()
    with patch.object(settings, "byok_mode", False):
        out = schedule_session_purge(cli)
    assert out is None
    cli.get_collections.assert_not_called()


def test_schedule_single_shot_when_apscheduler_missing() -> None:
    """Without APScheduler the function still sweeps once at startup."""
    base = settings.qdrant_collection
    collections = [
        _fake_collection(
            f"{base}_sess_old",
            (datetime.now(UTC) - timedelta(hours=48)).isoformat(),
        )
    ]
    cli = _client_returning(collections, {c.name: c for c in collections})

    with (
        patch.object(settings, "byok_mode", True),
        patch.dict(
            "sys.modules",
            {
                "apscheduler": None,
                "apscheduler.schedulers": None,
                "apscheduler.schedulers.asyncio": None,
            },
        ),
    ):
        out = schedule_session_purge(cli)

    assert out is None
    cli.delete_collection.assert_called_once_with(f"{base}_sess_old")
