"""Tests for document metadata models and RBAC helpers."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ingestion.metadata import (
    DocumentMetadata,
    IngestRequest,
    SensitivityLevel,
    UserContext,
    sensitivity_to_int,
)


class TestSensitivityLevel:
    """Tests for SensitivityLevel enum."""

    def test_enum_values(self) -> None:
        """Enum should have correct string values."""
        assert SensitivityLevel.LOW == "low"
        assert SensitivityLevel.MEDIUM == "medium"
        assert SensitivityLevel.HIGH == "high"

    def test_enum_is_string(self) -> None:
        """Enum values should be usable as strings."""
        assert isinstance(SensitivityLevel.LOW, str)
        assert f"level: {SensitivityLevel.HIGH}" == "level: high"

    def test_enum_from_value(self) -> None:
        """Enum should be constructible from string values."""
        assert SensitivityLevel("low") == SensitivityLevel.LOW
        assert SensitivityLevel("medium") == SensitivityLevel.MEDIUM
        assert SensitivityLevel("high") == SensitivityLevel.HIGH


class TestSensitivityToInt:
    """Tests for the sensitivity_to_int helper function."""

    def test_low_maps_to_1(self) -> None:
        assert sensitivity_to_int(SensitivityLevel.LOW) == 1

    def test_medium_maps_to_2(self) -> None:
        assert sensitivity_to_int(SensitivityLevel.MEDIUM) == 2

    def test_high_maps_to_3(self) -> None:
        assert sensitivity_to_int(SensitivityLevel.HIGH) == 3


class TestDocumentMetadata:
    """Tests for DocumentMetadata model."""

    def test_creation_with_required_fields(self) -> None:
        """Should create metadata with only required fields."""
        meta = DocumentMetadata(
            user_id="user1",
            org_id="org1",
            source_file="test.pdf",
        )
        assert meta.user_id == "user1"
        assert meta.org_id == "org1"
        assert meta.source_file == "test.pdf"
        assert meta.sensitivity_level == SensitivityLevel.LOW
        assert meta.roles == ["viewer"]
        assert meta.page_number == 0
        assert meta.chunk_index == 0
        assert meta.file_type == ""
        assert meta.language is None

    def test_creation_with_all_fields(self) -> None:
        """Should create metadata with all fields specified."""
        now = datetime(2024, 1, 1, 12, 0, 0)
        meta = DocumentMetadata(
            user_id="admin",
            org_id="acme",
            sensitivity_level=SensitivityLevel.HIGH,
            roles=["admin", "manager"],
            source_file="report.pdf",
            page_number=5,
            chunk_index=3,
            ingested_at=now,
            file_type="pdf",
            language="en",
        )
        assert meta.sensitivity_level == SensitivityLevel.HIGH
        assert meta.roles == ["admin", "manager"]
        assert meta.page_number == 5
        assert meta.chunk_index == 3
        assert meta.ingested_at == now
        assert meta.file_type == "pdf"
        assert meta.language == "en"

    def test_to_qdrant_payload(self) -> None:
        """to_qdrant_payload should return a flat dict with serialized values."""
        now = datetime(2024, 6, 15, 10, 30, 0)
        meta = DocumentMetadata(
            user_id="user1",
            org_id="org1",
            sensitivity_level=SensitivityLevel.MEDIUM,
            roles=["editor", "viewer"],
            source_file="doc.pdf",
            page_number=2,
            chunk_index=7,
            ingested_at=now,
            file_type="pdf",
            language="ar",
        )
        payload = meta.to_qdrant_payload()

        assert payload["user_id"] == "user1"
        assert payload["org_id"] == "org1"
        assert payload["sensitivity_level"] == "medium"
        assert payload["sensitivity_int"] == 2
        assert payload["roles"] == ["editor", "viewer"]
        assert payload["source_file"] == "doc.pdf"
        assert payload["page_number"] == 2
        assert payload["chunk_index"] == 7
        assert payload["ingested_at"] == "2024-06-15T10:30:00"
        assert payload["file_type"] == "pdf"
        assert payload["language"] == "ar"

    def test_to_qdrant_payload_types(self) -> None:
        """Payload values should be the correct primitive types."""
        meta = DocumentMetadata(
            user_id="u1",
            org_id="o1",
            source_file="f.pdf",
        )
        payload = meta.to_qdrant_payload()

        assert isinstance(payload["sensitivity_level"], str)
        assert isinstance(payload["sensitivity_int"], int)
        assert isinstance(payload["roles"], list)
        assert isinstance(payload["ingested_at"], str)

    def test_default_ingested_at(self) -> None:
        """ingested_at should default to approximately now."""
        before = datetime.now(UTC).replace(tzinfo=None)
        meta = DocumentMetadata(
            user_id="u1",
            org_id="o1",
            source_file="f.pdf",
        )
        after = datetime.now(UTC).replace(tzinfo=None)

        assert before <= meta.ingested_at <= after


class TestUserContext:
    """Tests for UserContext model."""

    def test_creation(self) -> None:
        """Should create a valid user context."""
        ctx = UserContext(
            user_id="user123",
            org_id="org456",
            roles=["admin", "analyst"],
            clearance_level=3,
        )
        assert ctx.user_id == "user123"
        assert ctx.org_id == "org456"
        assert ctx.roles == ["admin", "analyst"]
        assert ctx.clearance_level == 3

    def test_clearance_level_values(self) -> None:
        """Clearance level should accept valid integer values."""
        for level in [1, 2, 3]:
            ctx = UserContext(
                user_id="u",
                org_id="o",
                roles=["viewer"],
                clearance_level=level,
            )
            assert ctx.clearance_level == level


class TestIngestRequest:
    """Tests for IngestRequest model."""

    def test_creation_with_defaults(self) -> None:
        """Should create request with default sensitivity and roles."""
        req = IngestRequest(
            file_path="/docs/report.pdf",
            user_id="user1",
            org_id="org1",
        )
        assert req.file_path == "/docs/report.pdf"
        assert req.user_id == "user1"
        assert req.org_id == "org1"
        assert req.sensitivity_level == SensitivityLevel.LOW
        assert req.roles == ["viewer"]

    def test_creation_with_custom_values(self) -> None:
        """Should accept custom sensitivity level and roles."""
        req = IngestRequest(
            file_path="/secret/data.pdf",
            user_id="admin",
            org_id="acme",
            sensitivity_level=SensitivityLevel.HIGH,
            roles=["admin", "ciso"],
        )
        assert req.sensitivity_level == SensitivityLevel.HIGH
        assert req.roles == ["admin", "ciso"]

    def test_validation_missing_required(self) -> None:
        """Should raise validation error when required fields are missing."""
        with pytest.raises(ValueError):  # Pydantic ValidationError
            IngestRequest(file_path="/test.pdf")  # type: ignore[call-arg]
