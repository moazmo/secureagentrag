"""Multi-tenancy collection-routing tests."""

from __future__ import annotations

from unittest.mock import patch

from config.settings import settings
from retrieval.multitenancy import get_collection_name


def test_collection_name_default_single_tenant() -> None:
    name = get_collection_name("acme_corp")
    assert name == settings.qdrant_collection


def test_collection_name_multi_tenant_branches_per_org() -> None:
    with patch.object(settings, "multi_tenant_collections", True):
        assert get_collection_name("acme_corp") == f"{settings.qdrant_collection}_acme_corp"
        assert get_collection_name("partner_inc") == f"{settings.qdrant_collection}_partner_inc"


def test_collection_name_sanitises_org_id() -> None:
    with patch.object(settings, "multi_tenant_collections", True):
        # Punctuation / spaces get replaced with underscores so the collection
        # name is always a valid Qdrant identifier.
        assert get_collection_name("acme.corp") == f"{settings.qdrant_collection}_acme_corp"
        assert get_collection_name("acme corp") == f"{settings.qdrant_collection}_acme_corp"
        assert get_collection_name("acme/corp") == f"{settings.qdrant_collection}_acme_corp"


def test_collection_name_empty_org_returns_default() -> None:
    with patch.object(settings, "multi_tenant_collections", True):
        assert get_collection_name(None) == settings.qdrant_collection
        assert get_collection_name("") == settings.qdrant_collection


def test_qdrant_manager_for_org_returns_self_in_single_tenant() -> None:
    from retrieval.qdrant_client import QdrantManager

    mgr = QdrantManager.__new__(QdrantManager)
    mgr._collection_name = settings.qdrant_collection
    with patch.object(settings, "multi_tenant_collections", False):
        assert mgr.for_org("any_org") is mgr


def test_qdrant_manager_for_org_returns_new_in_multi_tenant() -> None:
    from retrieval.qdrant_client import QdrantManager

    mgr = QdrantManager.__new__(QdrantManager)
    mgr._collection_name = settings.qdrant_collection
    mgr._url = "http://localhost:6333"
    mgr._api_key = None
    with (
        patch.object(settings, "multi_tenant_collections", True),
        patch.object(QdrantManager, "__init__", return_value=None) as init_mock,
        patch.object(QdrantManager, "ensure_collection", return_value=None),
    ):
        scoped = mgr.for_org("acme_corp")
        assert scoped is not mgr
        init_mock.assert_called_once()
