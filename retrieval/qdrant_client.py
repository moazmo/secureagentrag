"""Qdrant vector database manager with RBAC-aware operations."""

from __future__ import annotations

import uuid

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from config.settings import settings
from ingestion.metadata import SensitivityLevel, UserContext, sensitivity_to_int
from utils.logging import get_logger

logger = get_logger(__name__)


class QdrantManager:
    """Manages Qdrant vector database operations including collection lifecycle and document upsert.

    Provides methods for collection management and RBAC-aware document storage.

    Args:
        url: Qdrant server URL. Defaults to settings.qdrant_url.
        collection_name: Target collection name. Defaults to settings.qdrant_collection.
        api_key: Optional API key for Qdrant Cloud authentication.
    """

    def __init__(
        self,
        url: str | None = None,
        collection_name: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize the Qdrant manager.

        Args:
            url: Qdrant server URL. Falls back to settings.qdrant_url.
            collection_name: Collection name. Falls back to settings.qdrant_collection.
            api_key: API key for authentication. Falls back to settings.qdrant_api_key.
        """
        self._url = url if url is not None else settings.qdrant_url
        self._collection_name = (
            collection_name if collection_name is not None else settings.qdrant_collection
        )
        self._api_key = api_key if api_key is not None else settings.qdrant_api_key

        self._client = QdrantClient(
            url=self._url,
            api_key=self._api_key,
            timeout=30,
        )

        logger.info(
            "qdrant_manager_initialized",
            url=self._url,
            collection=self._collection_name,
        )

    @property
    def collection_name(self) -> str:
        """Return the current collection name."""
        return self._collection_name

    @property
    def client(self) -> QdrantClient:
        """Return the underlying QdrantClient instance."""
        return self._client

    def ensure_collection(self, vector_size: int | None = None) -> None:
        """Create the collection if it does not already exist.

        Uses Cosine distance metric and the configured vector size.

        Args:
            vector_size: Dimension of the embedding vectors.
                Defaults to settings.embedding_dim.
        """
        size = vector_size if vector_size is not None else settings.embedding_dim

        try:
            collections = self._client.get_collections().collections
            existing_names = {c.name for c in collections}

            if self._collection_name in existing_names:
                logger.info(
                    "collection_already_exists",
                    collection=self._collection_name,
                )
                return

            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                "collection_created",
                collection=self._collection_name,
                vector_size=size,
                distance="Cosine",
            )

        except Exception as exc:
            logger.error(
                "collection_ensure_failed",
                collection=self._collection_name,
                error=str(exc),
            )
            raise

    async def upsert_documents(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> list[str]:
        """Upsert document chunks with embeddings and metadata into Qdrant.

        Generates UUID for each point and stores the chunk text in the payload
        alongside the provided metadata.

        Args:
            chunks: List of text chunks.
            embeddings: Corresponding embedding vectors.
            metadatas: Corresponding metadata dictionaries.

        Returns:
            List of point ID strings (UUIDs).

        Raises:
            ValueError: If input lists have mismatched lengths.
            Exception: On Qdrant upsert failure.
        """
        if not (len(chunks) == len(embeddings) == len(metadatas)):
            raise ValueError(
                f"Input length mismatch: chunks={len(chunks)}, "
                f"embeddings={len(embeddings)}, metadatas={len(metadatas)}"
            )

        if not chunks:
            return []

        point_ids: list[str] = []
        points: list[PointStruct] = []

        for chunk_text, embedding, metadata in zip(chunks, embeddings, metadatas, strict=False):
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)

            payload = {
                "text": chunk_text,
                **metadata,
            }

            # Ensure sensitivity_level_int is present for RBAC range filtering
            if "sensitivity_level_int" not in payload:
                sl = payload.get("sensitivity_level")
                if sl is not None:
                    try:
                        payload["sensitivity_level_int"] = sensitivity_to_int(SensitivityLevel(sl))
                    except (ValueError, KeyError):
                        payload["sensitivity_level_int"] = 1

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        try:
            self._client.upsert(
                collection_name=self._collection_name,
                points=points,
            )
            logger.info(
                "documents_upserted",
                collection=self._collection_name,
                count=len(points),
            )
        except Exception as exc:
            logger.error(
                "upsert_failed",
                collection=self._collection_name,
                count=len(points),
                error=str(exc),
            )
            raise

        return point_ids

    def get_collection_info(self) -> dict | None:
        """Retrieve information about the current collection.

        Returns:
            Dictionary with collection info, or None if collection doesn't exist.
        """
        try:
            info = self._client.get_collection(self._collection_name)
            return {
                "name": self._collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": info.status.value if info.status else None,
            }
        except Exception as exc:
            logger.warning(
                "collection_info_failed",
                collection=self._collection_name,
                error=str(exc),
            )
            return None

    def delete_collection(self) -> None:
        """Delete the current collection from Qdrant.

        Logs a warning if the collection doesn't exist.
        """
        try:
            self._client.delete_collection(self._collection_name)
            logger.info("collection_deleted", collection=self._collection_name)
        except Exception as exc:
            logger.warning(
                "collection_delete_failed",
                collection=self._collection_name,
                error=str(exc),
            )

    def build_rbac_filter(self, user_context: UserContext) -> models.Filter:
        """Build a Qdrant filter that enforces role-based access control.

        The filter ensures:
        - User belongs to the same organization as the document.
        - Document sensitivity level is within the user's clearance.
        - At least one of the user's roles matches the document's roles.

        Args:
            user_context: Authenticated user context with org, roles, and clearance.

        Returns:
            A Qdrant Filter object ready for use in search queries.
        """
        must_conditions = [
            models.FieldCondition(
                key="org_id",
                match=models.MatchValue(value=user_context.org_id),
            ),
            models.FieldCondition(
                key="sensitivity_level_int",
                range=models.Range(lte=user_context.clearance_level),
            ),
            models.FieldCondition(
                key="roles",
                match=models.MatchAny(any=user_context.roles),
            ),
        ]
        return models.Filter(must=must_conditions)

    def search_with_rbac(
        self,
        query_embedding: list[float],
        user_context: UserContext,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[models.ScoredPoint]:
        """Search the collection with RBAC filter applied.

        Args:
            query_embedding: Query vector for similarity search.
            user_context: Authenticated user context for RBAC filtering.
            top_k: Maximum number of results. Defaults to settings.top_k.
            score_threshold: Minimum score threshold. Defaults to None.

        Returns:
            List of scored points matching the query with RBAC constraints.
        """
        k = top_k if top_k is not None else settings.top_k
        rbac_filter = self.build_rbac_filter(user_context)

        try:
            results = self._client.search(
                collection_name=self._collection_name,
                query_vector=query_embedding,
                query_filter=rbac_filter,
                limit=k,
                score_threshold=score_threshold,
            )
            logger.info(
                "search_with_rbac_completed",
                collection=self._collection_name,
                results_count=len(results),
                user_id=user_context.user_id,
                org_id=user_context.org_id,
            )
            return results
        except Exception as exc:
            logger.error(
                "search_with_rbac_failed",
                collection=self._collection_name,
                error=str(exc),
            )
            return []

    def search_without_rbac(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        score_threshold: float | None = None,
        admin_context: UserContext | None = None,
    ) -> list[models.ScoredPoint]:
        """Search the collection without RBAC filtering (admin/debug use).

        Requires admin role for security. Logs a warning when invoked.

        Args:
            query_embedding: Query vector for similarity search.
            top_k: Maximum number of results. Defaults to settings.top_k.
            score_threshold: Minimum score threshold. Defaults to None.
            admin_context: UserContext that must contain 'admin' role.

        Returns:
            List of scored points matching the query.

        Raises:
            PermissionError: If admin_context is missing or lacks admin role.
        """
        if admin_context is None or "admin" not in admin_context.roles:
            logger.warning(
                "search_without_rbac_called_without_admin",
                admin_context_provided=admin_context is not None,
            )
            raise PermissionError("Admin role required for unfiltered search")

        logger.warning(
            "search_without_rbac_invoked",
            user_id=admin_context.user_id,
            org_id=admin_context.org_id,
        )

        k = top_k if top_k is not None else settings.top_k

        try:
            results = self._client.search(
                collection_name=self._collection_name,
                query_vector=query_embedding,
                limit=k,
                score_threshold=score_threshold,
            )
            logger.info(
                "search_without_rbac_completed",
                collection=self._collection_name,
                results_count=len(results),
            )
            return results
        except Exception as exc:
            logger.error(
                "search_without_rbac_failed",
                collection=self._collection_name,
                error=str(exc),
            )
            return []

    def get_document_count(self) -> int:
        """Return total number of points in the collection.

        Returns:
            Integer count of documents, or 0 if collection info unavailable.
        """
        try:
            info = self._client.get_collection(self._collection_name)
            return info.points_count or 0
        except Exception as exc:
            logger.warning(
                "get_document_count_failed",
                collection=self._collection_name,
                error=str(exc),
            )
            return 0

    def scroll_documents(
        self,
        filter_: models.Filter | None = None,
        limit: int = 100,
    ) -> list[models.Record]:
        """Scroll/list documents from the collection with optional filtering.

        Args:
            filter_: Optional Qdrant filter to apply.
            limit: Maximum number of documents to return.

        Returns:
            List of point records from the collection.
        """
        try:
            results, _ = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=filter_,
                limit=limit,
            )
            return results
        except Exception as exc:
            logger.error(
                "scroll_documents_failed",
                collection=self._collection_name,
                error=str(exc),
            )
            return []

    def delete_documents_by_filter(
        self,
        filter_: models.Filter | None = None,
    ) -> int:
        """Delete documents matching the given filter.

        If no filter is provided, deletes ALL documents in the collection.
        Use with caution.

        Args:
            filter_: Qdrant filter to match documents for deletion.

        Returns:
            Number of documents deleted.
        """
        try:
            result = self._client.delete(
                collection_name=self._collection_name,
                points_selector=models.FilterSelector(filter=filter_)
                if filter_
                else models.PointIdsList(points=[]),
            )
            deleted = getattr(result, "operation_id", 0)
            logger.info(
                "documents_deleted",
                collection=self._collection_name,
                deleted=deleted,
                filter_applied=filter_ is not None,
            )
            return deleted
        except Exception as exc:
            logger.error(
                "delete_documents_failed",
                collection=self._collection_name,
                error=str(exc),
            )
            return 0

    def delete_document_by_id(self, point_id: str) -> bool:
        """Delete a single document by its point ID.

        Args:
            point_id: The UUID of the point to delete.

        Returns:
            True if deletion was successful, False otherwise.
        """
        try:
            self._client.delete(
                collection_name=self._collection_name,
                points_selector=models.PointIdsList(points=[point_id]),
            )
            logger.info("document_deleted", point_id=point_id)
            return True
        except Exception as exc:
            logger.error("delete_document_failed", point_id=point_id, error=str(exc))
            return False

    def update_document_metadata(
        self,
        point_id: str,
        metadata: dict,
    ) -> bool:
        """Update metadata for a specific document.

        Args:
            point_id: The UUID of the point to update.
            metadata: Dict of metadata fields to update.

        Returns:
            True if update was successful, False otherwise.
        """
        try:
            # Ensure sensitivity_level_int is updated if sensitivity_level changed
            if "sensitivity_level" in metadata and "sensitivity_level_int" not in metadata:
                try:
                    metadata["sensitivity_level_int"] = sensitivity_to_int(
                        SensitivityLevel(metadata["sensitivity_level"])
                    )
                except (ValueError, KeyError):
                    metadata["sensitivity_level_int"] = 1

            self._client.set_payload(
                collection_name=self._collection_name,
                payload=metadata,
                points=[point_id],
            )
            logger.info("document_metadata_updated", point_id=point_id)
            return True
        except Exception as exc:
            logger.error(
                "update_document_metadata_failed",
                point_id=point_id,
                error=str(exc),
            )
            return False

    def get_documents_by_source(
        self,
        source_file: str,
        org_id: str | None = None,
    ) -> list[models.Record]:
        """Get all documents originating from a specific source file.

        Args:
            source_file: The source filename to search for.
            org_id: Optional org_id filter.

        Returns:
            List of matching point records.
        """
        conditions = [
            models.FieldCondition(
                key="source_file",
                match=models.MatchValue(value=source_file),
            ),
        ]
        if org_id:
            conditions.append(
                models.FieldCondition(
                    key="org_id",
                    match=models.MatchValue(value=org_id),
                )
            )
        filter_ = models.Filter(must=conditions)
        return self.scroll_documents(filter_=filter_, limit=1000)
