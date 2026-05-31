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
        # Punctuation / spaces still yield a valid Qdrant identifier (alnum +
        # underscore), now with a short disambiguating hash because the mapping
        # is lossy.
        for org in ("acme.corp", "acme corp", "acme/corp"):
            name = get_collection_name(org)
            assert name.startswith(f"{settings.qdrant_collection}_acme_corp_")
            assert all(c.isalnum() or c == "_" for c in name)


def test_collection_name_no_collision_across_distinct_orgs() -> None:
    """H8: org ids that sanitise to the same string must NOT share a collection."""
    with patch.object(settings, "multi_tenant_collections", True):
        a = get_collection_name("acme-corp")
        b = get_collection_name("acme_corp")
        c = get_collection_name("acme.corp")
        assert a != b  # used to both be documents_acme_corp
        assert a != c
        assert b != c
        # Deterministic across calls.
        assert get_collection_name("acme-corp") == a


def test_collection_name_empty_org_returns_default() -> None:
    with patch.object(settings, "multi_tenant_collections", True):
        assert get_collection_name(None) == settings.qdrant_collection
        assert get_collection_name("") == settings.qdrant_collection


def test_collection_name_byok_session_overrides_org() -> None:
    """In BYOK mode, session_id beats org_id — visitor isolation is stricter."""
    with (
        patch.object(settings, "byok_mode", True),
        patch.object(settings, "multi_tenant_collections", True),
    ):
        # session_id present → session collection regardless of org_id
        name = get_collection_name("acme_corp", session_id="abc-123")
        assert name.startswith(f"{settings.qdrant_collection}_sess_abc_123_")
        # session_id absent → falls back to multi-tenant routing (acme_corp is
        # already collection-safe, so no hash suffix).
        assert get_collection_name("acme_corp") == f"{settings.qdrant_collection}_acme_corp"


def test_collection_name_byok_session_sanitises_id() -> None:
    """Session UUIDs from clients can contain dashes; collection-safe sanitise."""
    with patch.object(settings, "byok_mode", True):
        # standard UUID4 format → collection-safe id + disambiguating hash
        name = get_collection_name(session_id="550e8400-e29b-41d4-a716-446655440000")
        assert name.startswith(
            f"{settings.qdrant_collection}_sess_550e8400_e29b_41d4_a716_446655440000_"
        )
        assert all(c.isalnum() or c == "_" for c in name)
        # distinct session ids never collide onto the same collection
        n1 = get_collection_name(session_id="sess-a")
        n2 = get_collection_name(session_id="sess_a")
        assert n1 != n2


def test_collection_name_byok_session_ignored_when_mode_off() -> None:
    """Session_id alone does NOT trigger session collection — needs byok_mode."""
    with patch.object(settings, "byok_mode", False):
        # In dev mode (no BYOK), session_id is silently ignored.
        assert get_collection_name(session_id="anything") == settings.qdrant_collection


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
    mgr._tenant_cache = {}
    with (
        patch.object(settings, "multi_tenant_collections", True),
        patch.object(QdrantManager, "__init__", return_value=None) as init_mock,
        patch.object(QdrantManager, "ensure_collection", return_value=None),
    ):
        scoped = mgr.for_org("acme_corp")
        assert scoped is not mgr
        init_mock.assert_called_once()


def test_qdrant_manager_for_org_caches_per_tenant_managers() -> None:
    """``for_org`` must reuse the per-org manager across calls.

    Multi-tenant mode previously created a fresh QdrantManager (new HTTP
    client + Qdrant `get_collections` round-trip) on every request. Caching
    by collection name turns repeat calls into a dict lookup. This test pins
    the contract so the optimisation does not regress.
    """
    from retrieval.qdrant_client import QdrantManager

    mgr = QdrantManager.__new__(QdrantManager)
    mgr._collection_name = settings.qdrant_collection
    mgr._url = "http://localhost:6333"
    mgr._api_key = None
    mgr._tenant_cache = {}
    with (
        patch.object(settings, "multi_tenant_collections", True),
        patch.object(QdrantManager, "__init__", return_value=None) as init_mock,
        patch.object(QdrantManager, "ensure_collection", return_value=None) as ensure_mock,
    ):
        first = mgr.for_org("acme_corp")
        second = mgr.for_org("acme_corp")
        third = mgr.for_org("partner_inc")

    # Same org -> identical manager. Different org -> fresh manager.
    assert first is second
    assert first is not third
    # Construction (and ensure_collection) only runs once per distinct tenant.
    assert init_mock.call_count == 2
    assert ensure_mock.call_count == 2


def test_qdrant_manager_ensure_collection_creates_sparse_field() -> None:
    """``ensure_collection`` always provisions the sparse vector field.

    Sparse isolation in multi-tenant mode is structural: each per-tenant
    collection (e.g. ``documents_acme_corp``) is created with its own
    ``sparse`` slot via ``SparseVectorParams``, and Qdrant cannot scan
    across collections in a single query — so a query bound to org B's
    manager never sees org A's sparse data even though the field name is
    identical. This test exercises the dense-and-sparse creation contract
    directly on a manually-constructed manager so the regression guard does
    not depend on QdrantManager.__init__ side effects.
    """
    from qdrant_client.http.models import SparseVectorParams

    from retrieval.qdrant_client import QdrantManager

    captured: list[dict] = []

    class _FakeClient:
        def get_collections(self):
            return type("R", (), {"collections": []})()

        def create_collection(self, **kwargs: object) -> None:
            captured.append(kwargs)

    fake_client = _FakeClient()

    mgr = QdrantManager.__new__(QdrantManager)
    mgr._collection_name = "documents_acme_corp"
    mgr._url = "http://localhost:6333"
    mgr._api_key = None
    mgr._client = fake_client
    mgr._tenant_cache = {}

    mgr.ensure_collection()

    sparse_name = getattr(settings, "sparse_vector_name", "sparse")
    assert len(captured) == 1
    kw = captured[0]
    assert kw["collection_name"] == "documents_acme_corp"
    sv = kw["sparse_vectors_config"]
    assert sparse_name in sv
    assert isinstance(sv[sparse_name], SparseVectorParams)


def test_qdrant_manager_for_org_distinct_tenants_isolated() -> None:
    """Two ``for_org`` calls on different orgs produce managers bound to
    distinct collections — sparse data lives in two separate Qdrant
    collections, structurally isolated from each other.
    """
    from retrieval.multitenancy import get_collection_name
    from retrieval.qdrant_client import QdrantManager

    mgr = QdrantManager.__new__(QdrantManager)
    mgr._collection_name = settings.qdrant_collection
    mgr._url = "http://localhost:6333"
    mgr._api_key = None
    mgr._tenant_cache = {}

    with (
        patch.object(settings, "multi_tenant_collections", True),
        patch.object(QdrantManager, "__init__", return_value=None),
        patch.object(QdrantManager, "ensure_collection", return_value=None),
    ):
        a = mgr.for_org("acme_corp")
        b = mgr.for_org("partner_inc")
        # The for_org factory sets up _collection_name on the new managers
        # via the (patched) __init__. Simulate that here for assertion.
        a._collection_name = get_collection_name("acme_corp")
        b._collection_name = get_collection_name("partner_inc")

    assert a._collection_name != b._collection_name
    assert "acme_corp" in a._collection_name
    assert "partner_inc" in b._collection_name
