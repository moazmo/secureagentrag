"""Tests for Qdrant client RBAC filtering."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ingestion.metadata import UserContext


@pytest.fixture()
def user_context_admin() -> UserContext:
    """Admin user with high clearance."""
    return UserContext(
        user_id="admin-001",
        org_id="org-alpha",
        roles=["admin", "engineer"],
        clearance_level=3,
    )


@pytest.fixture()
def user_context_viewer() -> UserContext:
    """Basic viewer with low clearance."""
    return UserContext(
        user_id="viewer-001",
        org_id="org-alpha",
        roles=["viewer"],
        clearance_level=1,
    )


@pytest.fixture()
def user_context_engineer() -> UserContext:
    """Engineer with medium clearance."""
    return UserContext(
        user_id="eng-001",
        org_id="org-beta",
        roles=["engineer", "viewer"],
        clearance_level=2,
    )


@pytest.fixture()
def qdrant_manager():
    """Create a QdrantManager with mocked Qdrant client."""
    with patch("retrieval.qdrant_client.QdrantClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        from retrieval.qdrant_client import QdrantManager

        manager = QdrantManager(
            url="http://localhost:6333",
            collection_name="test_collection",
        )
        return manager


class TestBuildRbacFilter:
    """Tests for QdrantManager.build_rbac_filter()."""

    def test_filter_has_must_conditions(self, qdrant_manager, user_context_admin):
        """Filter must contain org_id, sensitivity_level_int, and roles conditions."""
        filter_ = qdrant_manager.build_rbac_filter(user_context_admin)

        assert filter_.must is not None
        assert len(filter_.must) == 3

    def test_filter_org_id_match(self, qdrant_manager, user_context_admin):
        """First must condition checks org_id."""
        filter_ = qdrant_manager.build_rbac_filter(user_context_admin)

        org_condition = filter_.must[0]
        assert org_condition.key == "org_id"
        assert org_condition.match.value == "org-alpha"

    def test_filter_sensitivity_range(self, qdrant_manager, user_context_admin):
        """Second must condition checks sensitivity_level_int with range lte."""
        filter_ = qdrant_manager.build_rbac_filter(user_context_admin)

        sensitivity_condition = filter_.must[1]
        assert sensitivity_condition.key == "sensitivity_level_int"
        assert sensitivity_condition.range.lte == 3

    def test_filter_roles_in_must(self, qdrant_manager, user_context_admin):
        """Roles condition must be in must (not should) for security."""
        filter_ = qdrant_manager.build_rbac_filter(user_context_admin)

        assert filter_.should is None
        roles_condition = filter_.must[2]
        assert roles_condition.key == "roles"
        assert set(roles_condition.match.any) == {"admin", "engineer"}

    def test_filter_low_clearance(self, qdrant_manager, user_context_viewer):
        """Low clearance user gets restrictive sensitivity filter."""
        filter_ = qdrant_manager.build_rbac_filter(user_context_viewer)

        sensitivity_condition = filter_.must[1]
        assert sensitivity_condition.range.lte == 1

    def test_filter_medium_clearance(self, qdrant_manager, user_context_engineer):
        """Medium clearance user gets appropriate sensitivity filter."""
        filter_ = qdrant_manager.build_rbac_filter(user_context_engineer)

        sensitivity_condition = filter_.must[1]
        assert sensitivity_condition.range.lte == 2

    def test_filter_different_org(self, qdrant_manager, user_context_engineer):
        """Different org produces different org_id filter."""
        filter_ = qdrant_manager.build_rbac_filter(user_context_engineer)

        org_condition = filter_.must[0]
        assert org_condition.match.value == "org-beta"

    def test_filter_single_role(self, qdrant_manager, user_context_viewer):
        """Single role user produces MatchAny with one element."""
        filter_ = qdrant_manager.build_rbac_filter(user_context_viewer)

        roles_condition = filter_.must[2]
        assert roles_condition.match.any == ["viewer"]

    def test_filter_multiple_roles(self, qdrant_manager, user_context_engineer):
        """Multiple roles produce MatchAny with all roles."""
        filter_ = qdrant_manager.build_rbac_filter(user_context_engineer)

        roles_condition = filter_.must[2]
        assert set(roles_condition.match.any) == {"engineer", "viewer"}


class TestSearchWithRbac:
    """Tests for QdrantManager.search_with_rbac()."""

    def test_search_calls_client_with_filter(self, qdrant_manager, user_context_admin):
        """search_with_rbac calls client.search with RBAC filter."""
        qdrant_manager.client.search.return_value = []

        results = qdrant_manager.search_with_rbac(
            query_embedding=[0.1] * 1024,
            user_context=user_context_admin,
            top_k=5,
        )

        qdrant_manager.client.search.assert_called_once()
        call_kwargs = qdrant_manager.client.search.call_args
        assert call_kwargs.kwargs["limit"] == 5
        assert call_kwargs.kwargs["query_filter"] is not None
        assert results == []

    def test_search_returns_empty_on_error(self, qdrant_manager, user_context_admin):
        """search_with_rbac returns empty list on exception."""
        qdrant_manager.client.search.side_effect = Exception("Connection failed")

        results = qdrant_manager.search_with_rbac(
            query_embedding=[0.1] * 1024,
            user_context=user_context_admin,
        )

        assert results == []


class TestSearchWithoutRbac:
    """Tests for QdrantManager.search_without_rbac()."""

    def test_search_calls_client_without_filter(self, qdrant_manager, user_context_admin):
        """search_without_rbac calls client.search without filter when admin."""
        qdrant_manager.client.search.return_value = []

        results = qdrant_manager.search_without_rbac(
            query_embedding=[0.1] * 1024,
            top_k=10,
            admin_context=user_context_admin,
        )

        call_kwargs = qdrant_manager.client.search.call_args
        assert call_kwargs.kwargs["limit"] == 10
        assert "query_filter" not in call_kwargs.kwargs
        assert results == []

    def test_search_raises_without_admin_context(self, qdrant_manager):
        """search_without_rbac raises PermissionError without admin context."""
        import pytest

        with pytest.raises(PermissionError, match="Admin role required"):
            qdrant_manager.search_without_rbac(query_embedding=[0.1] * 1024)

    def test_search_raises_for_non_admin_role(self, qdrant_manager, user_context_viewer):
        """search_without_rbac raises PermissionError for non-admin user."""
        import pytest

        with pytest.raises(PermissionError, match="Admin role required"):
            qdrant_manager.search_without_rbac(
                query_embedding=[0.1] * 1024,
                admin_context=user_context_viewer,
            )

    def test_search_returns_empty_on_error(self, qdrant_manager, user_context_admin):
        """search_without_rbac returns empty list on exception."""
        qdrant_manager.client.search.side_effect = Exception("Timeout")

        results = qdrant_manager.search_without_rbac(
            query_embedding=[0.1] * 1024,
            admin_context=user_context_admin,
        )

        assert results == []


class TestGetDocumentCount:
    """Tests for QdrantManager.get_document_count()."""

    def test_returns_count(self, qdrant_manager):
        """get_document_count returns correct count."""
        mock_info = MagicMock()
        mock_info.points_count = 42
        qdrant_manager.client.get_collection.return_value = mock_info

        count = qdrant_manager.get_document_count()

        assert count == 42

    def test_returns_zero_on_error(self, qdrant_manager):
        """get_document_count returns 0 on failure."""
        qdrant_manager.client.get_collection.side_effect = Exception("Not found")

        count = qdrant_manager.get_document_count()

        assert count == 0


class TestScrollDocuments:
    """Tests for QdrantManager.scroll_documents()."""

    def test_scroll_returns_results(self, qdrant_manager):
        """scroll_documents returns list of records."""
        mock_records = [MagicMock(), MagicMock()]
        qdrant_manager.client.scroll.return_value = (mock_records, None)

        results = qdrant_manager.scroll_documents(limit=50)

        assert len(results) == 2
        qdrant_manager.client.scroll.assert_called_once()

    def test_scroll_returns_empty_on_error(self, qdrant_manager):
        """scroll_documents returns empty list on failure."""
        qdrant_manager.client.scroll.side_effect = Exception("Error")

        results = qdrant_manager.scroll_documents()

        assert results == []
