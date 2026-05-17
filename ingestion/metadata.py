"""Document metadata models for RBAC-aware ingestion."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class SensitivityLevel(StrEnum):
    """Classification levels controlling document access."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def sensitivity_to_int(level: SensitivityLevel) -> int:
    """Convert a SensitivityLevel to its numeric equivalent for Qdrant range filters.

    Args:
        level: The sensitivity level enum value.

    Returns:
        Integer mapping: low=1, medium=2, high=3.
    """
    mapping: dict[SensitivityLevel, int] = {
        SensitivityLevel.LOW: 1,
        SensitivityLevel.MEDIUM: 2,
        SensitivityLevel.HIGH: 3,
    }
    return mapping[level]


class DocumentMetadata(BaseModel):
    """Metadata attached to each document chunk stored in the vector database.

    Attributes:
        user_id: Owner who uploaded the document.
        org_id: Organization the document belongs to.
        sensitivity_level: Access classification level.
        roles: Roles that can access this document.
        source_file: Original file path or name.
        page_number: Page number in the source document (0-indexed).
        chunk_index: Sequential chunk index within the document.
        ingested_at: Timestamp of ingestion.
        file_type: Document type (pdf/docx/image).
        language: Detected language if available.
    """

    user_id: str
    org_id: str
    sensitivity_level: SensitivityLevel = SensitivityLevel.LOW
    roles: list[str] = Field(default_factory=lambda: ["viewer"])
    source_file: str
    page_number: int = 0
    chunk_index: int = 0
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC).replace(tzinfo=None))
    file_type: str = ""
    language: str | None = None

    def to_qdrant_payload(self) -> dict:
        """Convert metadata to a flat dictionary suitable for Qdrant payload storage.

        Enums are converted to their string values, datetimes to ISO format strings,
        and None values are preserved as-is for optional fields.

        Returns:
            Flat dictionary with serialized values.
        """
        return {
            "user_id": self.user_id,
            "org_id": self.org_id,
            "sensitivity_level": self.sensitivity_level.value,
            "sensitivity_level_int": sensitivity_to_int(self.sensitivity_level),
            "roles": self.roles,
            "source_file": self.source_file,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "ingested_at": self.ingested_at.isoformat(),
            "file_type": self.file_type,
            "language": self.language,
        }


class UserContext(BaseModel):
    """Represents the authenticated user context for RBAC filtering during retrieval.

    Attributes:
        user_id: Identifier of the querying user.
        org_id: Organization the user belongs to.
        roles: Roles assigned to the user.
        clearance_level: Numeric clearance (1=low, 2=medium, 3=high) for Qdrant range filters.
    """

    user_id: str
    org_id: str
    roles: list[str]
    clearance_level: int


class IngestRequest(BaseModel):
    """Request model for document ingestion.

    Attributes:
        file_path: Path to the file to ingest.
        user_id: Identifier of the user triggering ingestion.
        org_id: Organization context for the document.
        sensitivity_level: Classification level for the document.
        roles: Roles that should have access.
    """

    file_path: str
    user_id: str
    org_id: str
    sensitivity_level: SensitivityLevel = SensitivityLevel.LOW
    roles: list[str] = Field(default_factory=lambda: ["viewer"])
