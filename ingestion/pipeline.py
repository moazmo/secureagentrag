"""End-to-end document ingestion pipeline with deduplication."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from config.settings import settings
from ingestion.chunker import TextChunker
from ingestion.contextual import generate_chunk_contexts, merge_chunks
from ingestion.loaders import LoadedDocument, load_document
from ingestion.metadata import DocumentMetadata, IngestRequest
from ingestion.ocr import OCRProcessor
from utils.audit import audit_logger
from utils.logging import get_logger

if TYPE_CHECKING:
    from retrieval.embeddings import EmbeddingService
    from retrieval.hybrid_search import BM25Index
    from retrieval.qdrant_client import QdrantManager

logger = get_logger(__name__)


class IngestionResult(BaseModel):
    """Result of a document ingestion operation.

    Attributes:
        file_path: Path to the ingested file.
        num_chunks: Total number of chunks created.
        point_ids: List of Qdrant point IDs for stored vectors.
        status: Ingestion status — "success", "partial", or "failed".
        errors: List of error messages encountered during processing.
        processing_time_seconds: Total time taken for ingestion.
    """

    file_path: str
    num_chunks: int = 0
    point_ids: list[str] = Field(default_factory=list)
    status: str = "success"
    errors: list[str] = Field(default_factory=list)
    processing_time_seconds: float = 0.0


class IngestionPipeline:
    """Orchestrates the end-to-end document ingestion workflow.

    Coordinates document loading, OCR processing, text chunking,
    embedding generation, vector storage with RBAC metadata, and BM25
    sparse index maintenance.

    Args:
        qdrant_manager: Qdrant vector store manager instance.
        embedding_service: Embedding generation service instance.
        chunker: Optional text chunker. Creates default if not provided.
        ocr_processor: Optional OCR processor. Creates default if not provided.
        bm25_index: Optional shared BM25Index to populate during ingestion.
    """

    def __init__(
        self,
        qdrant_manager: QdrantManager,
        embedding_service: EmbeddingService,
        chunker: TextChunker | None = None,
        ocr_processor: OCRProcessor | None = None,
        bm25_index: BM25Index | None = None,
    ) -> None:
        """Initialize the ingestion pipeline with its dependencies.

        Args:
            qdrant_manager: Manager for Qdrant vector store operations.
            embedding_service: Service for generating text embeddings.
            chunker: Text chunker instance. Uses default settings if None.
            ocr_processor: OCR processor instance. Creates new one if None.
            bm25_index: Shared BM25Index to populate with chunk texts.
        """
        self._qdrant = qdrant_manager
        self._embeddings = embedding_service
        self._chunker = chunker or TextChunker()
        self._ocr = ocr_processor or OCRProcessor()
        self._bm25 = bm25_index

        logger.info("ingestion_pipeline_initialized")

    def _compute_content_hash(self, text: str) -> str:
        """Compute a hash for deduplication of document chunks.

        Args:
            text: Chunk text content.

        Returns:
            MD5 hash string of the normalized text.
        """
        normalized = " ".join(text.lower().split())
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()

    async def ingest_document(
        self,
        request: IngestRequest,
        force_reingest: bool = False,
    ) -> IngestionResult:
        """Ingest a single document through the full pipeline.

        Steps:
            1. Load document using appropriate loader
            2. For pages with insufficient text, attempt OCR
            3. Chunk all extracted text
            4. Deduplicate against existing chunks (unless force_reingest)
            5. Create RBAC-aware metadata for each chunk
            6. Generate embeddings in batch
            7. Upsert to Qdrant vector store
            8. Return ingestion result

        Args:
            request: Ingestion request containing file path and RBAC context.
            force_reingest: If True, skip deduplication and re-ingest all chunks.

        Returns:
            IngestionResult with status, chunk count, and point IDs.
        """
        start_time = time.time()
        errors: list[str] = []
        file_path = request.file_path

        logger.info("ingestion_started", file=file_path, user=request.user_id)

        # Step 1: Load document
        try:
            documents = load_document(file_path)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            logger.error("ingestion_load_failed", file=file_path, error=str(exc))
            return IngestionResult(
                file_path=file_path,
                status="failed",
                errors=[f"Load failed: {exc}"],
                processing_time_seconds=time.time() - start_time,
            )

        # Step 2: OCR for pages with little/no text
        if self._ocr.is_available():
            documents = self._apply_ocr_fallback(documents, file_path)

        # Step 3: Chunk text
        chunked = self._chunker.chunk_documents(documents, source_file=file_path)

        if not chunked:
            logger.warning("ingestion_no_chunks", file=file_path)
            return IngestionResult(
                file_path=file_path,
                num_chunks=0,
                status="partial",
                errors=["No text content could be extracted from document"],
                processing_time_seconds=time.time() - start_time,
            )

        # Step 4: Deduplication — check for existing chunks by source+hash
        if not force_reingest:
            existing_docs = self._qdrant.get_documents_by_source(
                source_file=file_path,
                org_id=request.org_id,
            )
            existing_hashes = set()
            for doc in existing_docs:
                text = doc.payload.get("text", "") if doc.payload else ""
                existing_hashes.add(self._compute_content_hash(text))

            new_chunked = []
            duplicates = 0
            for chunk_text, chunk_meta in chunked:
                chunk_hash = self._compute_content_hash(chunk_text)
                if chunk_hash in existing_hashes:
                    duplicates += 1
                    continue
                new_chunked.append((chunk_text, chunk_meta))

            if duplicates > 0:
                logger.info(
                    "ingestion_deduplicated",
                    file=file_path,
                    duplicates=duplicates,
                    new_chunks=len(new_chunked),
                )
                if not new_chunked:
                    return IngestionResult(
                        file_path=file_path,
                        num_chunks=0,
                        status="success",
                        errors=[f"All {duplicates} chunks already exist. Skipping."],
                        processing_time_seconds=time.time() - start_time,
                    )
            chunked = new_chunked

        # Step 5: Create metadata for each chunk
        chunk_texts: list[str] = []
        metadatas: list[dict] = []
        file_ext = Path(file_path).suffix.lower().lstrip(".")

        for chunk_text, chunk_meta in chunked:
            chunk_texts.append(chunk_text)

            doc_metadata = DocumentMetadata(
                user_id=request.user_id,
                org_id=request.org_id,
                sensitivity_level=request.sensitivity_level,
                roles=request.roles,
                source_file=file_path,
                page_number=chunk_meta.get("page_number", 0),
                chunk_index=chunk_meta.get("chunk_index", 0),
                file_type=file_ext,
            )
            metadatas.append(doc_metadata.to_qdrant_payload())

        # Step 5b: (optional) Anthropic-style Contextual Retrieval — prepend
        # an LLM-generated context summary to each chunk *for embedding only*.
        # The chunk text shown to users (and stored in payload) is unchanged.
        embed_inputs = chunk_texts
        if settings.contextual_retrieval_enabled and chunk_texts:
            try:
                full_doc = "\n".join(d.text for d in documents)
                contexts = await generate_chunk_contexts(
                    full_doc,
                    chunk_texts,
                    prefer_cloud=False,
                )
                embed_inputs = merge_chunks(chunk_texts, contexts)
                logger.info(
                    "contextual_retrieval_applied",
                    file=file_path,
                    augmented=sum(1 for c in contexts if c),
                )
            except Exception as exc:
                logger.warning("contextual_retrieval_failed", error=str(exc))
                embed_inputs = chunk_texts

        # Step 6: Generate embeddings
        try:
            embeddings = await self._embeddings.embed_batch(embed_inputs)
        except Exception as exc:
            logger.error("ingestion_embedding_failed", file=file_path, error=str(exc))
            return IngestionResult(
                file_path=file_path,
                num_chunks=len(chunk_texts),
                status="failed",
                errors=[f"Embedding generation failed: {exc}"],
                processing_time_seconds=time.time() - start_time,
            )

        # Step 7: Upsert to Qdrant
        try:
            self._qdrant.ensure_collection()
            point_ids = await self._qdrant.upsert_documents(
                chunks=chunk_texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except Exception as exc:
            logger.error("ingestion_upsert_failed", file=file_path, error=str(exc))
            return IngestionResult(
                file_path=file_path,
                num_chunks=len(chunk_texts),
                status="failed",
                errors=[f"Vector store upsert failed: {exc}"],
                processing_time_seconds=time.time() - start_time,
            )

        # Step 7: Append to BM25 index. Previously this called build_index
        # which WIPED the existing corpus on every upload — so only the last
        # ingested document was searchable via BM25. add_documents extends
        # in place.
        if self._bm25 is not None and point_ids:
            try:
                # Index the same augmented text we embedded so BM25 + dense
                # see the same surface form (Anthropic recommends parity for
                # contextual retrieval — context tokens help BM25 too).
                self._bm25.add_documents(
                    documents=embed_inputs,
                    doc_ids=point_ids,
                )
                logger.info("bm25_index_updated", chunks=len(point_ids))
            except Exception as exc:
                logger.warning("bm25_index_update_failed", error=str(exc))
                errors.append(f"BM25 index update failed: {exc}")

        # Step 8: Record audit event and return result
        processing_time = time.time() - start_time

        audit_logger.log_ingestion(
            user_id=request.user_id,
            document_name=file_path,
            chunk_count=len(point_ids),
            metadata={
                "org_id": request.org_id,
                "sensitivity_level": request.sensitivity_level.value,
                "processing_time_seconds": processing_time,
            },
        )

        status = "success" if not errors else "partial"
        logger.info(
            "ingestion_completed",
            file=file_path,
            chunks=len(point_ids),
            time_seconds=processing_time,
            status=status,
        )

        return IngestionResult(
            file_path=file_path,
            num_chunks=len(point_ids),
            point_ids=point_ids,
            status=status,
            errors=errors,
            processing_time_seconds=processing_time,
        )

    async def ingest_batch(self, requests: list[IngestRequest]) -> list[IngestionResult]:
        """Ingest multiple documents sequentially.

        Args:
            requests: List of ingestion requests to process.

        Returns:
            List of IngestionResult, one per request.
        """
        results: list[IngestionResult] = []

        logger.info("batch_ingestion_started", count=len(requests))

        for request in requests:
            result = await self.ingest_document(request)
            results.append(result)

        successful = sum(1 for r in results if r.status == "success")
        failed = sum(1 for r in results if r.status == "failed")

        logger.info(
            "batch_ingestion_completed",
            total=len(results),
            successful=successful,
            failed=failed,
        )

        return results

    def _apply_ocr_fallback(
        self,
        documents: list[LoadedDocument],
        file_path: str,
    ) -> list[LoadedDocument]:
        """Apply OCR to documents with insufficient text content.

        Args:
            documents: List of loaded documents to process.
            file_path: Original file path for OCR processing.

        Returns:
            Updated list of documents with OCR-enhanced text where applicable.
        """
        enhanced: list[LoadedDocument] = []

        for doc in documents:
            if len(doc.text.strip()) < 50:
                # Try OCR for this page
                if doc.file_type == "image" or doc.metadata.get("ocr_needed"):
                    ocr_text = self._ocr.extract_text_from_image(file_path)
                    if ocr_text:
                        enhanced.append(
                            LoadedDocument(
                                text=ocr_text,
                                page_number=doc.page_number,
                                source_file=doc.source_file,
                                file_type=doc.file_type,
                                metadata={**doc.metadata, "ocr_applied": True},
                            )
                        )
                        continue
                elif doc.file_type == "pdf":
                    ocr_text = self._ocr.extract_text_from_pdf_page(file_path, doc.page_number)
                    if ocr_text:
                        enhanced.append(
                            LoadedDocument(
                                text=ocr_text,
                                page_number=doc.page_number,
                                source_file=doc.source_file,
                                file_type=doc.file_type,
                                metadata={**doc.metadata, "ocr_applied": True},
                            )
                        )
                        continue

            enhanced.append(doc)

        return enhanced
