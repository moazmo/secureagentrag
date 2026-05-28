"""Qdrant vector database manager with RBAC-aware operations."""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
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
        # Per-tenant manager cache. In multi-tenant mode each `for_org(org_id)`
        # call previously created a fresh QdrantManager (new HTTP client +
        # extra `get_collections` round-trip via `ensure_collection`). Caching
        # by collection name turns repeat calls into pure dict lookups so the
        # per-request overhead disappears. Stays bound to *this* root manager
        # — distinct roots (different URLs) keep distinct caches.
        self._tenant_cache: dict[str, QdrantManager] = {}

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

    def for_org(self, org_id: str) -> QdrantManager:
        """Return a QdrantManager scoped to an organization-specific collection.

        When ``settings.multi_tenant_collections`` is True, this returns a
        per-org manager bound to ``documents_{org_id}``. Each tenant collection
        is created the first time it is requested (with the same dense + sparse
        vector configuration as the global collection — sparse isolation is
        therefore structural: org A's sparse vectors live in
        ``documents_acme_corp.sparse``, org B's in ``documents_partner_inc.sparse``,
        and Qdrant cannot cross collections in a single query) and the manager
        is cached on the root instance so repeat requests are O(1) dict lookups
        rather than fresh HTTP-client + ``get_collections`` round-trips.

        When ``multi_tenant_collections`` is False, returns ``self``.

        Args:
            org_id: Organization identifier.

        Returns:
            A QdrantManager instance (new, cached, or self).
        """
        if not settings.multi_tenant_collections:
            return self
        from retrieval.multitenancy import get_collection_name

        org_collection = get_collection_name(org_id)
        if org_collection == self._collection_name:
            return self
        cached = self._tenant_cache.get(org_collection)
        if cached is not None:
            return cached
        mgr = QdrantManager(
            url=self._url,
            collection_name=org_collection,
            api_key=self._api_key,
        )
        mgr.ensure_collection()
        self._tenant_cache[org_collection] = mgr
        logger.info(
            "tenant_collection_cached",
            collection=org_collection,
            cache_size=len(self._tenant_cache),
        )
        return mgr

    def for_session(self, session_id: str) -> QdrantManager:
        """Return a QdrantManager bound to a BYOK visitor's session collection.

        Mirrors ``for_org`` but uses the session-scoped naming convention
        ``documents_sess_<sanitized_session>``. The collection is created
        on first request with the same dense + sparse vector configuration
        as the base collection, then cached on this root manager so repeat
        requests are O(1) dict lookups.

        BYOK uploads live in the visitor's session collection only; the
        24-hour purge cron drops abandoned collections so 1 GB Qdrant
        Cloud quota stays bounded.

        Args:
            session_id: Per-visitor session UUID (BYOK mode).

        Returns:
            A QdrantManager scoped to the session collection.
        """
        if not session_id:
            return self
        from retrieval.multitenancy import get_collection_name

        # Force BYOK-style naming even when settings.byok_mode is False so
        # tests can exercise the path without flipping the global flag.
        base = self._collection_name
        sanitized = "".join(c if c.isalnum() else "_" for c in session_id)
        sess_collection = f"{base}_sess_{sanitized}"
        # Honour the canonical helper when both flags align so a future rename
        # of the prefix has a single source of truth.
        if settings.byok_mode:
            try:
                sess_collection = get_collection_name(session_id=session_id)
            except Exception:
                pass

        if sess_collection == self._collection_name:
            return self
        cached = self._tenant_cache.get(sess_collection)
        if cached is not None:
            return cached
        mgr = QdrantManager(
            url=self._url,
            collection_name=sess_collection,
            api_key=self._api_key,
        )
        mgr.ensure_collection()
        # Qdrant Cloud requires explicit payload indexes on filterable fields.
        # Mirror the indexes created on the base demo collection.
        for field, schema in (
            ("org_id", "keyword"),
            ("sensitivity_level_int", "integer"),
            ("roles", "keyword"),
            ("user_id", "keyword"),
            ("source_file", "keyword"),
            # Per-upload group key used by the BYOK delete endpoint to drop
            # all chunks of one upload. Qdrant Cloud refuses Filter()
            # predicates on un-indexed payload keys, so this is mandatory.
            ("source_file_id", "keyword"),
        ):
            try:
                mgr._client.create_payload_index(
                    collection_name=sess_collection,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception:
                # Index may already exist; safe to ignore.
                pass
        self._tenant_cache[sess_collection] = mgr
        logger.info(
            "byok_session_collection_cached",
            collection=sess_collection,
            cache_size=len(self._tenant_cache),
        )
        return mgr

    def ensure_collection(self, vector_size: int | None = None) -> None:
        """Create the collection if it does not already exist.

        Creates both dense and sparse vector configurations so that hybrid
        search (dense + sparse) works out of the box.

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

            sparse_name = getattr(settings, "sparse_vector_name", "sparse")
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=size,
                    distance=Distance.COSINE,
                ),
                sparse_vectors_config={sparse_name: SparseVectorParams()},
            )
            logger.info(
                "collection_created",
                collection=self._collection_name,
                vector_size=size,
                distance="Cosine",
                sparse_vector=sparse_name,
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
        sparse_vectors: list[SparseVector] | None = None,
    ) -> list[str]:
        """Upsert document chunks with embeddings and metadata into Qdrant.

        Generates UUID for each point and stores the chunk text in the payload
        alongside the provided metadata. When *sparse_vectors* are supplied
        they are written to the named sparse vector field configured by
        ``settings.sparse_vector_name``.

        Args:
            chunks: List of text chunks.
            embeddings: Corresponding dense embedding vectors.
            metadatas: Corresponding metadata dictionaries.
            sparse_vectors: Optional sparse vectors for hybrid search.

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
        if sparse_vectors is not None and len(sparse_vectors) != len(chunks):
            raise ValueError(
                f"Sparse vector length mismatch: sparse={len(sparse_vectors)}, chunks={len(chunks)}"
            )

        if not chunks:
            return []

        point_ids: list[str] = []
        points: list[PointStruct] = []
        sparse_name = getattr(settings, "sparse_vector_name", "sparse")
        has_sparse = sparse_vectors is not None

        for idx, (chunk_text, embedding, metadata) in enumerate(
            zip(chunks, embeddings, metadatas, strict=False)
        ):
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)

            payload = {
                "text": chunk_text,
                **metadata,
            }

            # Defensive: ensure sensitivity_level_int present even if caller
            # passed metadata not produced by DocumentMetadata.to_qdrant_payload.
            if "sensitivity_level_int" not in payload:
                sl = payload.get("sensitivity_level")
                if sl is not None:
                    try:
                        payload["sensitivity_level_int"] = sensitivity_to_int(SensitivityLevel(sl))
                    except (ValueError, KeyError):
                        payload["sensitivity_level_int"] = 1

            vector: dict[str, Any] | list[float] = embedding
            if has_sparse:
                vector = {
                    "": embedding,
                    sparse_name: sparse_vectors[idx],
                }

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
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
                has_sparse=has_sparse,
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
            # vectors_count was removed from CollectionInfo in qdrant-client >= 1.10;
            # use getattr so this stays forward-compatible.
            return {
                "name": self._collection_name,
                "points_count": info.points_count,
                "vectors_count": getattr(info, "vectors_count", info.points_count),
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

    def build_combined_filter(
        self,
        user_context: UserContext,
        extra_conditions: list[dict[str, Any]] | None = None,
    ) -> models.Filter:
        """Build a Qdrant filter combining RBAC with self-query conditions.

        Args:
            user_context: Authenticated user context for RBAC.
            extra_conditions: List of condition dicts from
                ``self_query.build_qdrant_filter_conditions``.

        Returns:
            A Qdrant Filter with RBAC must-conditions plus any extra conditions.
        """
        rbac = self.build_rbac_filter(user_context)
        if not extra_conditions:
            return rbac

        combined_must = list(rbac.must or [])
        for cond in extra_conditions:
            if "match" in cond:
                combined_must.append(
                    models.FieldCondition(
                        key=cond["key"],
                        match=cond["match"],
                    )
                )
            elif "range" in cond:
                combined_must.append(
                    models.FieldCondition(
                        key=cond["key"],
                        range=cond["range"],
                    )
                )
        return models.Filter(must=combined_must)

    def search_with_rbac(
        self,
        query_embedding: list[float],
        user_context: UserContext,
        top_k: int | None = None,
        score_threshold: float | None = None,
        extra_filter: models.Filter | None = None,
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
        rbac_filter = extra_filter or self.build_rbac_filter(user_context)

        try:
            # qdrant-client >= 1.13 replaced .search() with .query_points()
            # which returns a QueryResponse wrapping a list of ScoredPoint.
            response = self._client.query_points(
                collection_name=self._collection_name,
                query=query_embedding,
                query_filter=rbac_filter,
                limit=k,
                score_threshold=score_threshold,
            )
            results = response.points
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

    def search_sparse_with_rbac(
        self,
        sparse_vector: models.SparseVector,
        user_context: UserContext,
        top_k: int | None = None,
        score_threshold: float | None = None,
        extra_filter: models.Filter | None = None,
    ) -> list[models.ScoredPoint]:
        """Search the sparse vector field with RBAC filter applied.

        Args:
            sparse_vector: Query sparse vector (indices + values).
            user_context: Authenticated user context for RBAC filtering.
            top_k: Maximum number of results. Defaults to settings.top_k.
            score_threshold: Minimum score threshold. Defaults to None.
            extra_filter: Optional additional Qdrant filter.

        Returns:
            List of scored points from the sparse vector index.
        """
        k = top_k if top_k is not None else settings.top_k
        rbac_filter = extra_filter or self.build_rbac_filter(user_context)
        sparse_name = getattr(settings, "sparse_vector_name", "sparse")

        try:
            response = self._client.query_points(
                collection_name=self._collection_name,
                query=sparse_vector,
                using=sparse_name,
                query_filter=rbac_filter,
                limit=k,
                score_threshold=score_threshold,
            )
            results = response.points
            logger.info(
                "search_sparse_with_rbac_completed",
                collection=self._collection_name,
                results_count=len(results),
                user_id=user_context.user_id,
                org_id=user_context.org_id,
            )
            return results
        except Exception as exc:
            logger.error(
                "search_sparse_with_rbac_failed",
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
            response = self._client.query_points(
                collection_name=self._collection_name,
                query=query_embedding,
                limit=k,
                score_threshold=score_threshold,
            )
            results = response.points
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
