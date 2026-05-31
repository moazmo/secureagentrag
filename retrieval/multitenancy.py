"""Multi-tenancy utilities for Qdrant collection naming."""

from __future__ import annotations

import hashlib

from config.settings import settings


def _sanitize(s: str) -> str:
    """Coerce ``s`` to a Qdrant-safe identifier (alnum + underscore only).

    Collapsing every non-alphanumeric to ``_`` is lossy: ``acme-corp`` and
    ``acme_corp`` would both map to ``acme_corp`` and share a collection. When
    the mapping IS lossy we append a short hash of the **original** string so
    distinct inputs never collide onto the same tenant/session collection. Pure
    alphanumeric inputs are returned unchanged (stable, readable names). The
    result is bounded to a Qdrant-safe length.
    """
    safe = "".join(c if c.isalnum() else "_" for c in s)
    if safe != s:
        digest = hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]
        safe = f"{safe[:200]}_{digest}"
    return safe[:255]


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
