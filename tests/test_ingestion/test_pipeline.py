"""Tests for the ingestion pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ingestion.metadata import IngestRequest, SensitivityLevel
from ingestion.pipeline import IngestionPipeline, IngestionResult


class TestIngestionResult:
    """Tests for the IngestionResult model."""

    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        result = IngestionResult(file_path="test.pdf")
        assert result.file_path == "test.pdf"
        assert result.num_chunks == 0
        assert result.point_ids == []
        assert result.status == "success"
        assert result.errors == []
        assert result.processing_time_seconds == 0.0

    def test_full_creation(self) -> None:
        """Should accept all fields."""
        result = IngestionResult(
            file_path="/docs/report.pdf",
            num_chunks=15,
            point_ids=["id1", "id2", "id3"],
            status="partial",
            errors=["OCR failed for page 3"],
            processing_time_seconds=2.5,
        )
        assert result.num_chunks == 15
        assert len(result.point_ids) == 3
        assert result.status == "partial"
        assert len(result.errors) == 1
        assert result.processing_time_seconds == 2.5

    def test_failed_status(self) -> None:
        """Should support failed status."""
        result = IngestionResult(
            file_path="broken.pdf",
            status="failed",
            errors=["File could not be parsed"],
        )
        assert result.status == "failed"


class TestIngestionPipeline:
    """Tests for IngestionPipeline initialization and basic behavior."""

    def test_initialization(self) -> None:
        """Pipeline should initialize with mocked dependencies."""
        mock_qdrant = MagicMock()
        mock_embeddings = MagicMock()

        pipeline = IngestionPipeline(
            qdrant_manager=mock_qdrant,
            embedding_service=mock_embeddings,
        )
        assert pipeline._qdrant is mock_qdrant
        assert pipeline._embeddings is mock_embeddings
        assert pipeline._chunker is not None
        assert pipeline._ocr is not None

    def test_initialization_with_custom_deps(self) -> None:
        """Pipeline should accept custom chunker and OCR."""
        mock_qdrant = MagicMock()
        mock_embeddings = MagicMock()
        mock_chunker = MagicMock()
        mock_ocr = MagicMock()

        pipeline = IngestionPipeline(
            qdrant_manager=mock_qdrant,
            embedding_service=mock_embeddings,
            chunker=mock_chunker,
            ocr_processor=mock_ocr,
        )
        assert pipeline._chunker is mock_chunker
        assert pipeline._ocr is mock_ocr

    @pytest.mark.asyncio
    async def test_ingest_document_file_not_found(self) -> None:
        """Should return failed result for non-existent file."""
        mock_qdrant = MagicMock()
        mock_embeddings = MagicMock()

        pipeline = IngestionPipeline(
            qdrant_manager=mock_qdrant,
            embedding_service=mock_embeddings,
        )

        request = IngestRequest(
            file_path="/nonexistent/file.pdf",
            user_id="user1",
            org_id="org1",
        )

        result = await pipeline.ingest_document(request)
        assert result.status == "failed"
        assert len(result.errors) > 0
        assert "Load failed" in result.errors[0]

    @pytest.mark.asyncio
    async def test_ingest_document_unsupported_format(self) -> None:
        """Should return failed result for unsupported file types."""
        mock_qdrant = MagicMock()
        mock_embeddings = MagicMock()

        pipeline = IngestionPipeline(
            qdrant_manager=mock_qdrant,
            embedding_service=mock_embeddings,
        )

        request = IngestRequest(
            file_path="/docs/file.xyz",
            user_id="user1",
            org_id="org1",
        )

        result = await pipeline.ingest_document(request)
        assert result.status == "failed"

    @pytest.mark.asyncio
    async def test_ingest_document_success_flow(self, tmp_path) -> None:
        """Should successfully ingest a document with mocked services."""
        # Create a test image file (simplest to test since no parsing needed)
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"\x89PNG\r\n\x1a\n")

        # Mock dependencies
        mock_qdrant = MagicMock()
        mock_qdrant.ensure_collection = MagicMock()
        mock_qdrant.upsert_documents = AsyncMock(return_value=["point-1"])
        # Pipeline calls qdrant.for_org(org_id); single-tenant mode returns
        # self, so the mock must mirror that behaviour.
        mock_qdrant.for_org.return_value = mock_qdrant
        mock_qdrant.get_documents_by_source.return_value = []

        mock_embeddings = MagicMock()
        mock_embeddings.embed_batch = AsyncMock(return_value=[[0.1] * 1024])

        # Mock OCR to return some text
        mock_ocr = MagicMock()
        mock_ocr.is_available.return_value = True
        mock_ocr.extract_text_from_image.return_value = "OCR extracted text content here."

        pipeline = IngestionPipeline(
            qdrant_manager=mock_qdrant,
            embedding_service=mock_embeddings,
            ocr_processor=mock_ocr,
        )

        request = IngestRequest(
            file_path=str(test_file),
            user_id="user1",
            org_id="org1",
            sensitivity_level=SensitivityLevel.MEDIUM,
            roles=["analyst"],
        )

        result = await pipeline.ingest_document(request)
        assert result.status == "success"
        assert result.num_chunks == 1
        assert result.point_ids == ["point-1"]
        assert result.processing_time_seconds >= 0

    @pytest.mark.asyncio
    async def test_ingest_batch(self) -> None:
        """Should process multiple requests and return results list."""
        mock_qdrant = MagicMock()
        mock_embeddings = MagicMock()

        pipeline = IngestionPipeline(
            qdrant_manager=mock_qdrant,
            embedding_service=mock_embeddings,
        )

        requests = [
            IngestRequest(
                file_path="/nonexistent/a.pdf",
                user_id="u1",
                org_id="o1",
            ),
            IngestRequest(
                file_path="/nonexistent/b.pdf",
                user_id="u2",
                org_id="o1",
            ),
        ]

        results = await pipeline.ingest_batch(requests)
        assert len(results) == 2
        # Both should fail since files don't exist
        assert all(r.status == "failed" for r in results)
