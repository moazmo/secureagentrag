"""Multi-tenancy utilities for Qdrant collection naming."""

from __future__ import annotations

from config.settings import settings


def get_collection_name(org_id: str | None = None) -> str:
    """Return the Qdrant collection name for a given organization.

    When ``settings.multi_tenant_collections`` is True, each organization
    gets its own collection (``documents_{org_id}``). When False, all
    organizations share the default collection.

    Args:
        org_id: Organization identifier. If None, returns the default
            collection name.

    Returns:
        Collection name string suitable for QdrantManager.
    """
    base = settings.qdrant_collection
    if not settings.multi_tenant_collections or not org_id:
        return base
    # Sanitize org_id for use in a collection name
    safe_org = "".join(c if c.isalnum() else "_" for c in org_id)
    return f"{base}_{safe_org}"
