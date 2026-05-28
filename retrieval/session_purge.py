"""Per-session Qdrant collection purge for BYOK mode.

In BYOK mode each visitor's uploads land in a collection named
``documents_sess_<sanitized_session_id>``. Without a cleanup pass these
collections accumulate until the 1 GB Qdrant Cloud free tier fills up.

This module provides:

- :func:`purge_expired_sessions` — synchronous, idempotent sweep that
  deletes collections whose creation timestamp is older than
  ``settings.session_collection_ttl_hours``.
- :func:`schedule_session_purge` — APScheduler hook the FastAPI lifespan
  calls so the sweep runs every 6 hours inside the same process. No
  separate cron container required.

The creation timestamp is read from Qdrant's
``CollectionInfo.config.params.metadata`` (set at create-time by the
ingestion pipeline). Collections without a creation timestamp are treated
as legacy and **skipped** — we never delete data we can't date.

See ``launch-plan/03-backend-byok.md`` § Session purge cron.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from config.settings import settings
from utils.logging import get_logger

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

logger = get_logger(__name__)


SESSION_COLLECTION_PREFIX = "_sess_"
"""Suffix introduced into the collection name by ``get_collection_name`` when
``byok_mode`` is on and a ``session_id`` is supplied. Used here to filter the
purge sweep to BYOK collections only — multi-tenant org collections are NOT
touched."""


def _session_collection_prefix() -> str:
    """Concrete prefix for the current base collection (e.g. ``documents_sess_``)."""
    return f"{settings.qdrant_collection}{SESSION_COLLECTION_PREFIX}"


def _is_session_collection(name: str) -> bool:
    """True iff ``name`` was emitted by ``get_collection_name`` with a session_id."""
    return name.startswith(_session_collection_prefix())


def _parse_created_at(meta: dict[str, Any] | None) -> datetime | None:
    """Return the collection's recorded creation datetime, or None if missing.

    The ingestion pipeline writes ``created_at`` as an ISO-8601 UTC string into
    the collection's metadata payload when first creating a session
    collection. Older collections lack the field — those are intentionally
    skipped to avoid deleting data we cannot date.
    """
    if not meta:
        return None
    raw = meta.get("created_at")
    if not raw:
        return None
    try:
        # Accept both ``2026-05-26T13:00:00+00:00`` and trailing ``Z`` forms.
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        logger.warning("session_purge_bad_timestamp", value=str(raw))
        return None


def purge_expired_sessions(
    client: QdrantClient,
    *,
    ttl_hours: int | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Delete BYOK session collections older than the TTL.

    Args:
        client: Live ``QdrantClient`` (cloud or local).
        ttl_hours: Override ``settings.session_collection_ttl_hours``. Tests
            pass small values; production uses the default 24.
        now: Override the clock for deterministic tests.

    Returns:
        Summary dict with counts (``inspected``, ``deleted``, ``skipped``,
        ``errors``) suitable for emission to the audit log.
    """
    ttl = ttl_hours if ttl_hours is not None else settings.session_collection_ttl_hours
    horizon = (now or datetime.now(UTC)) - timedelta(hours=ttl)
    inspected = deleted = skipped = errors = 0
    deleted_names: list[str] = []

    try:
        collections = client.get_collections().collections
    except Exception as exc:
        logger.error("session_purge_list_failed", error=str(exc))
        return {"inspected": 0, "deleted": 0, "skipped": 0, "errors": 1}

    for col in collections:
        name = col.name
        if not _is_session_collection(name):
            continue
        inspected += 1
        try:
            info = client.get_collection(name)
            meta = getattr(info.config.params, "metadata", None) or {}
            created = _parse_created_at(meta)
            if created is None:
                # Undated -> skip; we don't delete what we can't time-stamp.
                skipped += 1
                continue
            if created < horizon:
                client.delete_collection(name)
                deleted += 1
                deleted_names.append(name)
                logger.info(
                    "session_purge_deleted",
                    collection=name,
                    created_at=created.isoformat(),
                    age_hours=round((horizon - created).total_seconds() / 3600.0 + ttl, 1),
                )
            else:
                skipped += 1
        except Exception as exc:
            errors += 1
            logger.warning("session_purge_collection_failed", collection=name, error=str(exc))

    summary = {
        "inspected": inspected,
        "deleted": deleted,
        "skipped": skipped,
        "errors": errors,
        "deleted_names": deleted_names,
        "ttl_hours": ttl,
    }
    logger.info(
        "session_purge_summary", **{k: v for k, v in summary.items() if k != "deleted_names"}
    )
    return summary


# ── FastAPI lifespan hook ────────────────────────────────────────────────────


def schedule_session_purge(client: QdrantClient, *, interval_hours: int = 6) -> Any | None:
    """Start an APScheduler job that runs :func:`purge_expired_sessions` periodically.

    Called from the FastAPI ``lifespan`` context manager. Returns the
    ``AsyncIOScheduler`` instance (or None when APScheduler is not
    installed — we then run as a single-shot at startup so at least one
    sweep happens per restart).
    """
    if not settings.byok_mode:
        logger.debug("session_purge_not_scheduled", reason="byok_mode is off")
        return None

    try:
        from apscheduler.schedulers.asyncio import (
            AsyncIOScheduler,  # type: ignore[import-not-found]
        )
    except ImportError:
        # Optional dep absent: at least sweep once so the Space does not
        # accumulate indefinitely on long uptimes.
        logger.warning("apscheduler_missing", action="single-shot purge instead")
        purge_expired_sessions(client)
        return None

    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        purge_expired_sessions,
        "interval",
        hours=interval_hours,
        args=[client],
        id="byok-session-purge",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("session_purge_scheduled", every_hours=interval_hours)
    return scheduler
