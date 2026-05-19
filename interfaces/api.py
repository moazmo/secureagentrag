"""FastAPI surface for SecureAgentRAG.

Run with::

    uv run uvicorn interfaces.api:app --host 0.0.0.0 --port 8080

Endpoints
---------
- ``GET  /healthz``     — liveness probe (no auth).
- ``GET  /readyz``      — readiness — pings Qdrant + Ollama.
- ``POST /query``       — run the RAG pipeline; returns ``QueryResponse``.
- ``POST /ingest``      — ingest a local file; requires ``user`` role.
- ``GET  /audit``       — read paginated audit entries; requires ``admin``.
- ``POST /audit/verify``— verify the hash-chain; requires ``admin``.

Auth uses a stateless bearer token. The token payload is a base64-encoded JSON
``UserContext`` so the API has no session store — caller provides identity on
every request. Production deployments should swap this for Keycloak/Auth0 JWT
verification (left as a hook in ``_resolve_user``).
"""

from __future__ import annotations

import base64
import json
from datetime import date
from typing import Annotated

from utils.logging import get_logger

logger = get_logger(__name__)

try:
    from fastapi import Depends, FastAPI, Header, HTTPException, status
    from fastapi.responses import JSONResponse

    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FASTAPI_AVAILABLE = False
    Depends = Header = FastAPI = HTTPException = JSONResponse = status = None  # type: ignore[assignment]

if _FASTAPI_AVAILABLE:
    from core.graph import run_rag_pipeline
    from core.schemas import (
        IngestRequestModel,
        IngestResponseModel,
        QueryRequest,
        QueryResponse,
    )
    from ingestion.metadata import IngestRequest, SensitivityLevel, UserContext
    from utils.audit import audit_logger
    from utils.health import run_health_checks
    from utils.rate_limiter import RateLimiter

    rate_limiter = RateLimiter()  # uses default token-bucket config

    def _resolve_user(authorization: Annotated[str | None, Header()] = None) -> UserContext:
        """Decode the bearer token into a ``UserContext``.

        Token format: ``Bearer <base64(json(UserContext))>``. Production
        replacement: validate a Keycloak / Auth0 JWT and map claims to roles.
        """
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
        token = authorization.split(" ", 1)[1]
        try:
            payload = json.loads(base64.b64decode(token).decode("utf-8"))
            return UserContext(**payload)
        except Exception as exc:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"invalid token: {exc}") from exc

    def _require_role(required: str):
        def _dep(user: Annotated[UserContext, Depends(_resolve_user)]) -> UserContext:
            if required not in user.roles and "admin" not in user.roles:
                raise HTTPException(status.HTTP_403_FORBIDDEN, f"role '{required}' required")
            return user

        return _dep

    app = FastAPI(
        title="SecureAgentRAG API",
        version="0.1.0",
        description="Privacy-first multi-agent RAG with RBAC, guardrails, and audit chain.",
    )

    @app.get("/healthz", tags=["ops"])
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", tags=["ops"])
    async def readyz() -> JSONResponse:
        report = await run_health_checks()
        code = 200 if report.overall_healthy else 503
        return JSONResponse(report.to_dict(), status_code=code)

    @app.post("/query", response_model=QueryResponse, tags=["rag"])
    async def query_endpoint(
        body: QueryRequest,
        user: Annotated[UserContext, Depends(_resolve_user)],
    ) -> QueryResponse:
        if not rate_limiter.is_allowed(f"{user.user_id}:query"):
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "rate limit exceeded")
        # Caller-supplied user_id must match the bearer-token identity.
        if body.user_id != user.user_id:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "user_id mismatch")
        state = await run_rag_pipeline(
            query=body.query,
            user_context=user,
            thread_id=f"api-{user.user_id}",
            prefer_cloud=body.prefer_cloud,
            override_provider=body.override_provider,
        )
        return QueryResponse.from_state(state)

    @app.post("/ingest", response_model=IngestResponseModel, tags=["rag"])
    async def ingest_endpoint(
        body: IngestRequestModel,
        user: Annotated[UserContext, Depends(_require_role("user"))],
    ) -> IngestResponseModel:
        if body.user_id != user.user_id:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "user_id mismatch")
        from core.agents.retriever import _get_hybrid_searcher
        from ingestion.pipeline import IngestionPipeline

        searcher = _get_hybrid_searcher()
        pipeline = IngestionPipeline(
            qdrant_manager=searcher._qdrant,  # type: ignore[attr-defined]
            embedding_service=searcher._embeddings,  # type: ignore[attr-defined]
            bm25_index=searcher._bm25_index,  # type: ignore[attr-defined]
        )
        req = IngestRequest(
            file_path=body.file_path,
            user_id=body.user_id,
            org_id=body.org_id,
            sensitivity_level=SensitivityLevel(body.sensitivity_level),
            roles=body.roles,
        )
        result = await pipeline.ingest_document(req)
        return IngestResponseModel(
            file_path=result.file_path,
            status=result.status,
            num_chunks=result.num_chunks,
            point_ids=result.point_ids,
            errors=result.errors,
            processing_time_seconds=result.processing_time_seconds,
        )

    @app.get("/audit", tags=["audit"])
    async def audit_list(
        user: Annotated[UserContext, Depends(_require_role("admin"))],
        start: str | None = None,
        end: str | None = None,
        limit: int = 100,
    ) -> dict:
        today = date.today().isoformat()
        entries = audit_logger.get_entries(
            start_date=start or today,
            end_date=end or today,
            user_id=None,
            action=None,
        )
        return {
            "total": len(entries),
            "items": [e.model_dump(mode="json") for e in entries[:limit]],
        }

    @app.post("/audit/verify", tags=["audit"])
    async def audit_verify(
        user: Annotated[UserContext, Depends(_require_role("admin"))],
        start: str | None = None,
        end: str | None = None,
    ) -> dict:
        result = audit_logger.verify_chain(start_date=start, end_date=end)
        return result

else:  # pragma: no cover
    app = None  # type: ignore[assignment]


def mint_dev_token(user: dict) -> str:
    """Convenience for local testing — build a bearer token for a UserContext dict."""
    payload = json.dumps(user).encode("utf-8")
    return base64.b64encode(payload).decode("ascii")
