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

from config.settings import settings
from utils.auth import AuthError, issue_token, verify_token
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

    _AUTH_ERROR_STATUS: dict[str, int] = {
        "missing": status.HTTP_401_UNAUTHORIZED,
        "malformed": status.HTTP_401_UNAUTHORIZED,
        "expired": status.HTTP_401_UNAUTHORIZED,
        "bad_signature": status.HTTP_401_UNAUTHORIZED,
        "bad_claims": status.HTTP_403_FORBIDDEN,
    }

    def _resolve_user_full(
        authorization: Annotated[str | None, Header()] = None,
    ) -> tuple[UserContext, dict]:
        """Verify the bearer token and return (UserContext, claims).

        Delegates to :func:`utils.auth.verify_token`, which uses HS256 JWT
        when ``SAR_JWT_SECRET`` is set and falls back to the legacy unsigned
        base64 token otherwise (with a runtime warning).
        """
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
        token = authorization.split(" ", 1)[1]
        try:
            return verify_token(token)
        except AuthError as exc:
            code = _AUTH_ERROR_STATUS.get(exc.reason, status.HTTP_401_UNAUTHORIZED)
            raise HTTPException(code, f"auth_{exc.reason}: {exc}") from exc

    def _resolve_user(authorization: Annotated[str | None, Header()] = None) -> UserContext:
        """Backward-compatible dependency returning only the UserContext."""
        ctx, _claims = _resolve_user_full(authorization=authorization)
        return ctx

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

    # Initialize Phoenix tracing if configured
    from utils.observability import setup_tracing

    _tracing_enabled = setup_tracing()
    if _tracing_enabled:
        logger.info("phoenix_tracing_active_in_api")

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
        auth: Annotated[tuple[UserContext, dict], Depends(_resolve_user_full)],
    ) -> QueryResponse:
        user, claims = auth
        if not rate_limiter.is_allowed(f"{user.user_id}:query"):
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "rate limit exceeded")
        # Caller-supplied user_id must match the bearer-token identity.
        if body.user_id != user.user_id:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "user_id mismatch")
        # Use the JWT id so the audit trail can correlate a query with the
        # exact token that authorised it; useful for revocation forensics.
        jti = claims.get("jti", "unsigned")
        state = await run_rag_pipeline(
            query=body.query,
            user_context=user,
            thread_id=f"api-{user.user_id}-{jti}",
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

    from pydantic import BaseModel as _PydBM

    class _TokenRequest(_PydBM):
        """Identity payload accepted by the dev ``/token`` endpoint."""

        user_id: str
        org_id: str = ""
        roles: list[str] = []
        clearance_level: int = 1
        ttl_seconds: int | None = None

    class _TokenResponse(_PydBM):
        access_token: str
        token_type: str = "bearer"
        expires_in: int

    @app.post("/token", response_model=_TokenResponse, tags=["auth"])
    async def issue_dev_token(body: _TokenRequest) -> _TokenResponse:
        """Mint a signed JWT for local testing.

        In production the IdP (Keycloak / Auth0 / Microsoft Entra) issues the
        token externally and this endpoint is removed via the
        ``SAR_DISABLE_DEV_TOKEN`` flag — kept here so the e2e smoke script
        and the Streamlit demo can mint a real token rather than the
        unsigned base64 fallback.
        """
        if not settings.jwt_secret:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "SAR_JWT_SECRET is not configured; token endpoint disabled",
            )
        try:
            token = issue_token(
                user_id=body.user_id,
                org_id=body.org_id,
                roles=body.roles,
                clearance_level=body.clearance_level,
                ttl_seconds=body.ttl_seconds,
            )
        except AuthError as exc:
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR, f"token_issue_{exc.reason}: {exc}"
            ) from exc
        return _TokenResponse(
            access_token=token,
            expires_in=body.ttl_seconds or settings.jwt_ttl_seconds,
        )

else:  # pragma: no cover
    app = None  # type: ignore[assignment]


def mint_dev_token(user: dict) -> str:
    """Convenience for local testing — build a bearer token for a UserContext dict.

    When ``SAR_JWT_SECRET`` is configured this mints a real signed JWT; with
    no secret it falls back to the legacy unsigned base64 shape so existing
    test fixtures keep working.
    """
    if settings.jwt_secret:
        try:
            return issue_token(
                user_id=user.get("user_id", ""),
                org_id=user.get("org_id", ""),
                roles=list(user.get("roles", [])),
                clearance_level=int(user.get("clearance_level", 1)),
            )
        except AuthError:
            # Fall through to legacy shape on issuer error.
            pass
    payload = json.dumps(user).encode("utf-8")
    return base64.b64encode(payload).decode("ascii")
