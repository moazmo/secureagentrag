"""Upload page — document ingestion with metadata tagging."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import streamlit as st

from ingestion.metadata import IngestRequest, SensitivityLevel
from ingestion.pipeline import IngestionPipeline
from retrieval.embeddings import EmbeddingService
from retrieval.hybrid_search import BM25Index
from retrieval.qdrant_client import QdrantManager
from utils.async_helpers import run_async
from utils.logging import get_logger
from utils.rate_limiter import check_upload_rate_limit

logger = get_logger(__name__)

ACCEPTED_TYPES = ["pdf", "docx", "txt", "png", "jpg", "jpeg", "tiff"]


@st.cache_resource
def _get_qdrant_manager() -> QdrantManager:
    """Get a cached QdrantManager instance.

    Returns:
        Singleton QdrantManager.
    """
    return QdrantManager()


@st.cache_resource
def _get_embedding_service() -> EmbeddingService:
    """Get a cached EmbeddingService instance.

    Returns:
        Singleton EmbeddingService.
    """
    return EmbeddingService()


def render_upload_page() -> None:
    """Render the document upload page with metadata form and ingestion pipeline."""
    st.title("📤 Document Upload")
    st.caption("Upload documents for ingestion into the vector store with RBAC metadata.")

    col1, col2 = st.columns([2, 1])

    with col1:
        _render_upload_form()

    with col2:
        _render_collection_stats()

    st.divider()
    _render_document_manager()

    st.divider()
    _render_recent_uploads()


def _render_document_manager() -> None:
    """Render document management section with delete and update capabilities."""
    st.subheader("🗂️ Document Manager")
    st.caption("View, update, and delete documents from the vector store.")

    user = st.session_state.current_user
    qdrant = _get_qdrant_manager()

    # Fetch documents for the user's org
    try:
        from qdrant_client import models
        filter_ = models.Filter(
            must=[
                models.FieldCondition(
                    key="org_id",
                    match=models.MatchValue(value=user["org_id"]),
                ),
            ]
        )
        records = qdrant.scroll_documents(filter_=filter_, limit=100)
    except Exception as exc:
        st.warning(f"Could not load documents: {exc}")
        return

    if not records:
        st.info("No documents found in the vector store.")
        return

    # Group records by source_file
    docs_by_source: dict[str, list] = {}
    for rec in records:
        source = rec.payload.get("source_file", "Unknown") if rec.payload else "Unknown"
        docs_by_source.setdefault(source, []).append(rec)

    # Display documents in an expandable list
    for source_file, recs in sorted(docs_by_source.items()):
        with st.expander(f"📄 {source_file} ({len(recs)} chunks)"):
            cols = st.columns([3, 1, 1])
            with cols[0]:
                st.caption(f"Point IDs: {len(recs)} chunks")
                first_payload = recs[0].payload or {}
                st.caption(
                    f"Sensitivity: {first_payload.get('sensitivity_level', 'N/A')} | "
                    f"Roles: {', '.join(first_payload.get('roles', []))}"
                )

            with cols[1]:
                # Update metadata
                new_sensitivity = st.selectbox(
                    "Sensitivity",
                    ["low", "medium", "high"],
                    index=["low", "medium", "high"].index(
                        first_payload.get("sensitivity_level", "low")
                    ),
                    key=f"sens_{source_file}",
                )
                new_roles = st.multiselect(
                    "Roles",
                    ["admin", "analyst", "viewer"],
                    default=first_payload.get("roles", ["viewer"]),
                    key=f"roles_{source_file}",
                )
                if st.button("Update", key=f"upd_{source_file}", use_container_width=True):
                    _update_document_batch(recs, new_sensitivity, new_roles, qdrant)

            with cols[2]:
                if st.button(
                    "🗑️ Delete",
                    key=f"del_{source_file}",
                    use_container_width=True,
                    type="secondary",
                ):
                    _delete_document_batch(recs, source_file, qdrant)


def _update_document_batch(
    records: list,
    sensitivity: str,
    roles: list[str],
    qdrant: QdrantManager,
) -> None:
    """Update metadata for a batch of document chunks.

    Args:
        records: List of Qdrant records to update.
        sensitivity: New sensitivity level.
        roles: New roles list.
        qdrant: QdrantManager instance.
    """
    from ingestion.metadata import SensitivityLevel, sensitivity_to_int

    metadata = {
        "sensitivity_level": sensitivity,
        "sensitivity_level_int": sensitivity_to_int(SensitivityLevel(sensitivity)),
        "roles": roles,
    }
    updated = 0
    for rec in records:
        if qdrant.update_document_metadata(str(rec.id), metadata):
            updated += 1

    if updated:
        st.success(f"Updated {updated} chunks for {records[0].payload.get('source_file', 'document')}")
        st.rerun()
    else:
        st.error("Failed to update document metadata.")


def _delete_document_batch(
    records: list,
    source_file: str,
    qdrant: QdrantManager,
) -> None:
    """Delete a batch of document chunks from Qdrant.

    Args:
        records: List of Qdrant records to delete.
        source_file: Source filename for logging.
        qdrant: QdrantManager instance.
    """
    deleted = 0
    for rec in records:
        if qdrant.delete_document_by_id(str(rec.id)):
            deleted += 1

    if deleted:
        st.success(f"Deleted {deleted} chunks for {source_file}")
        # Also log to audit
        user = st.session_state.current_user
        st.session_state.audit_log.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "user": user["display_name"],
                "user_id": user["user_id"],
                "action": "delete",
                "query": source_file,
                "details": f"deleted_chunks={deleted}",
                "sensitivity": "high",
                "status": "success",
                "latency_ms": 0.0,
                "confidence": 0.0,
            }
        )
        st.rerun()
    else:
        st.error("Failed to delete document.")


def _render_upload_form() -> None:
    """Render the file upload form with metadata options."""
    uploaded_file = st.file_uploader(
        "Choose a document",
        type=ACCEPTED_TYPES,
        help="Supported: PDF, DOCX, TXT, PNG, JPG, JPEG, TIFF",
    )

    if uploaded_file is not None:
        st.info(f"📄 **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")

        with st.form("upload_metadata_form", clear_on_submit=True):
            st.subheader("Document Metadata")

            sensitivity = st.selectbox(
                "Sensitivity Level",
                options=["Low", "Medium", "High"],
                index=0,
                help="Controls who can access this document based on clearance level.",
            )

            roles = st.multiselect(
                "Access Roles",
                options=["admin", "analyst", "viewer"],
                default=["viewer"],
                help="Roles that will have access to this document.",
            )

            tags = st.text_input(
                "Additional Tags",
                placeholder="finance, quarterly-report, confidential",
                help="Comma-separated tags for document categorization.",
            )

            submitted = st.form_submit_button("🚀 Ingest Document", use_container_width=True)

            if submitted:
                _process_upload(
                    uploaded_file=uploaded_file,
                    sensitivity=sensitivity,
                    roles=roles,
                    tags=tags,
                )


def _process_upload(
    uploaded_file,
    sensitivity: str,
    roles: list[str],
    tags: str,
) -> None:
    """Process the uploaded file through the ingestion pipeline.

    Args:
        uploaded_file: Streamlit UploadedFile object.
        sensitivity: Sensitivity level string (Low, Medium, High).
        roles: List of access roles.
        tags: Comma-separated tags string.
    """
    user = st.session_state.current_user

    # Rate limit check
    allowed, rl_meta = check_upload_rate_limit(user["user_id"])
    if not allowed:
        st.warning(
            f"⏳ Upload rate limit exceeded. Please wait {rl_meta['retry_after']} seconds.",
            icon="⏳",
        )
        return

    # Map sensitivity string to enum
    sensitivity_map = {
        "Low": SensitivityLevel.LOW,
        "Medium": SensitivityLevel.MEDIUM,
        "High": SensitivityLevel.HIGH,
    }
    sensitivity_level = sensitivity_map.get(sensitivity, SensitivityLevel.LOW)

    # Save file to temp location
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    # Create ingestion request
    request = IngestRequest(
        file_path=tmp_path,
        user_id=user["user_id"],
        org_id=user["org_id"],
        sensitivity_level=sensitivity_level,
        roles=roles if roles else ["viewer"],
    )

    # Run ingestion
    with st.spinner("⏳ Ingesting document..."):
        try:
            qdrant = _get_qdrant_manager()
            embeddings = _get_embedding_service()
            bm25_index = BM25Index()
            pipeline = IngestionPipeline(
                qdrant_manager=qdrant,
                embedding_service=embeddings,
                bm25_index=bm25_index,
            )

            result = run_async(pipeline.ingest_document(request))

            # Display result
            if result.status == "success":
                st.success(
                    f"✅ **Ingestion successful!**\n\n"
                    f"- Chunks created: **{result.num_chunks}**\n"
                    f"- Processing time: **{result.processing_time_seconds:.2f}s**",
                    icon="🎉",
                )
            elif result.status == "partial":
                st.warning(
                    f"⚠️ **Partial ingestion**\n\n"
                    f"- Chunks created: **{result.num_chunks}**\n"
                    f"- Errors: {', '.join(result.errors)}",
                    icon="⚠️",
                )
            else:
                st.error(
                    f"❌ **Ingestion failed**\n\n- Errors: {', '.join(result.errors)}",
                    icon="🚫",
                )

            # Store in recent uploads
            if "recent_uploads" not in st.session_state:
                st.session_state.recent_uploads = []

            st.session_state.recent_uploads.append(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "file_name": uploaded_file.name,
                    "status": result.status,
                    "chunks": result.num_chunks,
                    "sensitivity": sensitivity,
                    "roles": ", ".join(roles),
                    "processing_time": f"{result.processing_time_seconds:.2f}s",
                    "user": user["display_name"],
                }
            )

            # Audit log
            st.session_state.audit_log.append(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "user": user["display_name"],
                    "user_id": user["user_id"],
                    "action": "upload",
                    "query": uploaded_file.name,
                    "details": f"chunks={result.num_chunks}, status={result.status}",
                    "sensitivity": sensitivity,
                    "status": result.status,
                    "latency_ms": round(result.processing_time_seconds * 1000, 1),
                    "confidence": 0.0,
                }
            )

        except Exception as exc:
            logger.error("upload_ingestion_error", error=str(exc))
            st.error(f"❌ Ingestion failed: {exc}")

    # Clean up temp file
    try:
        Path(tmp_path).unlink(missing_ok=True)
    except Exception as e:
        logger.warning("Failed to clean up temp file", error=str(e), path=tmp_path)


def _render_collection_stats() -> None:
    """Render collection statistics from Qdrant."""
    st.subheader("📊 Collection Stats")

    try:
        qdrant = _get_qdrant_manager()
        info = qdrant.get_collection_info()

        if info:
            st.metric("Documents (chunks)", info.get("points_count", 0))
            st.metric("Vectors", info.get("vectors_count", 0))
            st.caption(f"Collection: `{info.get('name', 'N/A')}`")
            status = info.get("status", "unknown")
            if status == "green":
                st.success("Collection healthy", icon="✅")
            else:
                st.info(f"Status: {status}")
        else:
            st.info("No collection found. Upload a document to create one.")

    except Exception as exc:
        st.warning(f"Cannot connect to Qdrant: {exc}")


def _render_recent_uploads() -> None:
    """Render a table of recent upload results."""
    st.subheader("📋 Recent Uploads")

    recent = st.session_state.get("recent_uploads", [])
    if not recent:
        st.info("No uploads yet. Upload a document above to get started.")
        return

    # Display as dataframe (most recent first)
    import pandas as pd

    df = pd.DataFrame(reversed(recent))
    display_cols = [
        "timestamp",
        "file_name",
        "status",
        "chunks",
        "sensitivity",
        "processing_time",
        "user",
    ]
    available_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available_cols], use_container_width=True, hide_index=True)
