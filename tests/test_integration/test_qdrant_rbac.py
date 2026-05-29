"""Real-Qdrant integration test for the RBAC payload filter.

This is the hero invariant — "RBAC at the vector-DB layer" — exercised against
a *live* Qdrant rather than a mock. It upserts documents across two orgs,
sensitivity levels, and role sets, then asserts ``search_with_rbac`` returns
only what each ``UserContext`` is entitled to. Cross-tenant, over-clearance,
and role-mismatch documents must all be filtered out by Qdrant itself.

Marked ``integration``: skipped unless ``SAR_QDRANT_URL`` points at a reachable
Qdrant. CI runs it in a dedicated job with a Qdrant service container; the
default unit job excludes it via ``-m "not integration"``.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

pytestmark = pytest.mark.integration

_QDRANT_URL = os.getenv("SAR_QDRANT_URL")
_QDRANT_API_KEY = os.getenv("SAR_QDRANT_API_KEY")
_DIM = 8


def _vec(seed: float) -> list[float]:
    return [seed] * _DIM


@pytest.fixture()
def manager():
    """A QdrantManager bound to a throwaway collection on a live Qdrant."""
    if not _QDRANT_URL:
        pytest.skip("SAR_QDRANT_URL not set — live Qdrant required")

    from qdrant_client import QdrantClient

    from retrieval.qdrant_client import QdrantManager

    # Fail fast (and skip) if the server is unreachable.
    try:
        QdrantClient(url=_QDRANT_URL, api_key=_QDRANT_API_KEY, timeout=5).get_collections()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Qdrant unreachable at {_QDRANT_URL}: {exc}")

    collection = f"it_rbac_{uuid.uuid4().hex[:8]}"
    mgr = QdrantManager(url=_QDRANT_URL, collection_name=collection, api_key=_QDRANT_API_KEY)
    mgr.ensure_collection(vector_size=_DIM)
    try:
        yield mgr
    finally:
        mgr.delete_collection()


def _seed(mgr) -> None:
    chunks = [
        "acme low eng",  # 1
        "acme high admin",  # 2
        "acme low admin",  # 3
        "globex low eng",  # 4
    ]
    embeddings = [_vec(0.1), _vec(0.2), _vec(0.3), _vec(0.4)]
    metadatas = [
        {"org_id": "acme", "sensitivity_level_int": 1, "roles": ["engineer"]},
        {"org_id": "acme", "sensitivity_level_int": 3, "roles": ["admin"]},
        {"org_id": "acme", "sensitivity_level_int": 1, "roles": ["admin"]},
        {"org_id": "globex", "sensitivity_level_int": 1, "roles": ["engineer"]},
    ]
    asyncio.run(mgr.upsert_documents(chunks=chunks, embeddings=embeddings, metadatas=metadatas))


def _texts(points) -> set[str]:
    return {p.payload.get("text") for p in points}


def test_rbac_filter_enforces_org_clearance_and_roles(manager):
    from ingestion.metadata import UserContext

    _seed(manager)

    # Engineer at acme, clearance 1: only the acme/low/engineer doc.
    eng = UserContext(user_id="u1", org_id="acme", roles=["engineer"], clearance_level=1)
    got = _texts(manager.search_with_rbac(_vec(0.1), eng, top_k=10))
    assert got == {"acme low eng"}, got

    # Admin at acme, clearance 3: both acme/admin docs (low + high), no globex,
    # and NOT the engineer-only doc (role mismatch).
    admin = UserContext(user_id="u2", org_id="acme", roles=["admin"], clearance_level=3)
    got = _texts(manager.search_with_rbac(_vec(0.2), admin, top_k=10))
    assert got == {"acme high admin", "acme low admin"}, got

    # Cross-tenant isolation: globex engineer sees only the globex doc.
    other = UserContext(user_id="u3", org_id="globex", roles=["engineer"], clearance_level=3)
    got = _texts(manager.search_with_rbac(_vec(0.4), other, top_k=10))
    assert got == {"globex low eng"}, got


def test_over_clearance_doc_is_filtered(manager):
    from ingestion.metadata import UserContext

    _seed(manager)
    # acme admin but clearance 1 must not see the sensitivity-3 doc.
    low_admin = UserContext(user_id="u4", org_id="acme", roles=["admin"], clearance_level=1)
    got = _texts(manager.search_with_rbac(_vec(0.3), low_admin, top_k=10))
    assert "acme high admin" not in got
    assert got == {"acme low admin"}, got
