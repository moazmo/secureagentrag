"""Multi-tenancy utilities for Qdrant collection naming."""

from __future__ import annotations

from config.settings import settings


def _sanitize(s: str) -> str:
    """Coerce ``s`` to a Qdrant-safe identifier (alnum + underscore only)."""
    return "".join(c if c.isalnum() else "_" for c in s)


def get_collection_name(
    org_id: str | None = None,
    *,
    session_id: str | None = None,
) -> str:
    """Return the Qdrant collection name for a given org or BYOK session.

    Resolution order:

    1. **BYOK mode** (``settings.byok_mode=True``) with ``session_id`` →
       returns ``"{base}_sess_{sanitized_session}"``. Session-scoped
       collections isolate each visitor's uploads.
    2. **Multi-tenant** (``settings.multi_tenant_collections=True``) with
       ``org_id`` → returns ``"{base}_{sanitized_org}"``.
    3. **Single-tenant** (default) → returns ``settings.qdrant_collection``.

    Args:
        org_id: Organisation identifier (multi-tenant mode).
        session_id: Per-visitor session UUID (BYOK mode). Takes priority over
            ``org_id`` when both are set and BYOK is on, because BYOK is the
            stricter isolation boundary.

    Returns:
        Collection name string suitable for QdrantManager.
    """
    base = settings.qdrant_collection
    if settings.byok_mode and session_id:
        return f"{base}_sess_{_sanitize(session_id)}"
    if not settings.multi_tenant_collections or not org_id:
        return base
    return f"{base}_{_sanitize(org_id)}"
