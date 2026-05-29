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
import contextlib
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

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        """Start background jobs on boot, stop them cleanly on shutdown.

        - Periodic audit hash-chain verification (always; local-only).
        - BYOK session-collection purge (only when ``byok_mode`` is on).
        Both degrade gracefully when APScheduler is absent and never block
        startup on failure.
        """
        schedulers: list = []
        try:
            from utils.audit_verify import schedule_audit_verification

            s = schedule_audit_verification()
            if s is not None:
                schedulers.append(s)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("audit_verify_schedule_failed", error=str(exc))

        if settings.byok_mode:
            try:
                from retrieval.qdrant_client import QdrantManager
                from retrieval.session_purge import schedule_session_purge

                s = schedule_session_purge(QdrantManager().client)
                if s is not None:
                    schedulers.append(s)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("session_purge_schedule_failed", error=str(exc))

        try:
            yield
        finally:
            for s in schedulers:
                with contextlib.suppress(Exception):
                    s.shutdown(wait=False)

    app = FastAPI(
        title="SecureAgentRAG API",
        version="0.1.0",
        description="Privacy-first multi-agent RAG with RBAC, guardrails, and audit chain.",
        lifespan=_lifespan,
    )

    # Initialize Phoenix tracing if configured.
    # When ``settings.byok_mode`` is on, ``setup_tracing`` short-circuits to
    # False regardless of phoenix_endpoint (see utils/observability.py).
    from utils.observability import setup_tracing

    _tracing_enabled = setup_tracing()
    if _tracing_enabled:
        logger.info("phoenix_tracing_active_in_api")

    # ── Prometheus metrics ───────────────────────────────────────────────
    # Aggregate counters/histograms only — no prompts, completions, keys, or
    # user text ever lands in a label, so this is safe to expose even under
    # BYOK (unlike Phoenix tracing, which is hard-disabled above). When the
    # ``[metrics]`` extra is installed we mount prometheus-fastapi-
    # instrumentator for HTTP-level metrics and let it serve ``/metrics``;
    # the custom RAG metrics in utils.metrics share the same default registry
    # so they appear in the same exposition. Without the extra we fall back to
    # a manual ``/metrics`` route that 501s until prometheus_client is present.
    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        Instrumentator(
            should_group_status_codes=True,
            excluded_handlers=["/metrics", "/healthz"],
        ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
        logger.info("prometheus_metrics_enabled", mode="instrumentator")
    except ImportError:
        from fastapi import Response

        from utils.metrics import render_latest

        @app.get("/metrics", include_in_schema=False, tags=["ops"])
        async def metrics() -> Response:
            try:
                payload, content_type = render_latest()
            except RuntimeError:
                return Response(
                    "prometheus_client not installed; install the [metrics] extra",
                    status_code=status.HTTP_501_NOT_IMPLEMENTED,
                    media_type="text/plain",
                )
            return Response(payload, media_type=content_type)

        logger.info("prometheus_metrics_enabled", mode="manual_fallback")

    # ── BYOK CORS middleware ─────────────────────────────────────────────
    # Only mount CORS when:
    #   1) BYOK mode is on (public demo path), AND
    #   2) an explicit allowlist is configured via SAR_CORS_ALLOW_ORIGINS.
    # Empty allowlist + BYOK = wildcard would be a footgun (CSRF surface).
    # Empty allowlist + dev = no CORS needed (local same-origin).
    if settings.byok_mode and settings.cors_allow_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_allow_origins),
            allow_credentials=False,  # BYOK never uses cookies
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )
        logger.info("byok_cors_enabled", origins=list(settings.cors_allow_origins))

    @app.get("/healthz", tags=["ops"])
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", tags=["ops"])
    async def readyz() -> JSONResponse:
        # In BYOK production mode the deploy has no local Ollama -- pinging
        # it would always fail and the cron keepalive would mistake the
        # Space for down. Skip Ollama; ping Groq /models with the owner key
        # instead (when configured). Postgres + Redis remain optional.
        if settings.byok_mode:
            report = await run_health_checks(include_ollama=False)
            # Optionally surface Groq reachability when an owner key is set.
            if settings.groq_api_key:
                import time

                from utils.health import HealthStatus

                t0 = time.perf_counter()
                try:
                    import httpx

                    async with httpx.AsyncClient(timeout=5.0) as client:
                        r = await client.get(
                            f"{settings.groq_api_base}/models",
                            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                        )
                        latency_ms = (time.perf_counter() - t0) * 1000
                        if r.status_code == 200:
                            report.services.append(
                                HealthStatus(
                                    name="groq",
                                    healthy=True,
                                    latency_ms=latency_ms,
                                    message="Owner key reachable.",
                                    optional=True,
                                )
                            )
                        else:
                            report.services.append(
                                HealthStatus(
                                    name="groq",
                                    healthy=False,
                                    latency_ms=latency_ms,
                                    message=f"HTTP {r.status_code}",
                                    optional=True,
                                )
                            )
                except Exception as exc:
                    latency_ms = (time.perf_counter() - t0) * 1000
                    report.services.append(
                        HealthStatus(
                            name="groq",
                            healthy=False,
                            latency_ms=latency_ms,
                            message=f"Connection failed: {exc!s}",
                            optional=True,
                        )
                    )
            code = 200 if report.overall_healthy else 503
            return JSONResponse(report.to_dict(), status_code=code)
        report = await run_health_checks()
        code = 200 if report.overall_healthy else 503
        return JSONResponse(report.to_dict(), status_code=code)

    # ── BYOK demo endpoint ───────────────────────────────────────────────
    # Mounted only when ``settings.byok_mode`` is on. Bypasses JWT auth and
    # uses per-request BYOK credentials instead. Isolation is enforced via
    # session-scoped Qdrant collections, not JWT identity.
    if settings.byok_mode:
        from interfaces.byok import ByokCreds, client_ip_from_request, extract_byok
        from utils.rate_limiter import get_owner_key_throttle

        # All demo personas share ``org_id="demo"`` so they query the same
        # ingested corpus. RBAC differentiation is enforced via clearance
        # level + roles at the payload-filter layer -- exactly the production
        # invariant we want to demonstrate.
        _DEMO_ORG_ID = "demo"
        # Sensitivity levels are LOW=1, MEDIUM=2, HIGH=3 (see
        # ``ingestion/metadata.py::sensitivity_to_int``). Clearance levels must
        # be in the same range so the Qdrant range filter passes the right
        # chunks. Engineer < Compliance == Executive, but executive carries
        # a wider role set (sees both engineering + compliance content).
        # ``style`` is a short tone hint injected into the synthesizer's
        # system prompt so the three demo personas produce visibly distinct
        # answers from the same retrieved chunks.
        _DEMO_PERSONAS: dict[str, dict] = {
            "engineer": {
                "clearance_level": 2,
                "roles": ["engineering"],
                "style": (
                    "Write for a senior engineer. Use precise technical language, "
                    "name the underlying mechanism, and prefer concrete code "
                    "snippets or commands over abstract description. Skip the "
                    "executive summary -- this reader wants the wire-level detail."
                ),
            },
            "compliance": {
                "clearance_level": 3,
                "roles": ["compliance", "legal"],
                "style": (
                    "Write for a compliance / legal reviewer. Foreground the "
                    "regulatory citations, control IDs, and risk vocabulary. "
                    "Highlight gaps and required attestations. Hedge claims that "
                    "are not directly supported by an authoritative source."
                ),
            },
            "executive": {
                "clearance_level": 3,
                "roles": ["executive", "compliance", "engineering"],
                "style": (
                    "Write for a busy executive. Lead with the bottom line in "
                    "two sentences. Quantify impact in dollars / risk percentage / "
                    "headcount where the source supports it. Skip implementation "
                    "detail unless it changes the decision."
                ),
            },
        }

        def _persona_to_user_ctx(creds: ByokCreds) -> UserContext:
            """Translate ``creds.demo_persona`` into a synthetic UserContext.

            Unknown / missing persona → minimal read-only profile so the demo
            still answers but cannot escalate beyond the lowest clearance.
            """
            preset = _DEMO_PERSONAS.get((creds.demo_persona or "").lower())
            if preset is None:
                preset = {"clearance_level": 1, "roles": ["viewer"], "style": ""}
            return UserContext(
                user_id=f"demo-{creds.session_id}",
                org_id=_DEMO_ORG_ID,
                clearance_level=preset["clearance_level"],
                roles=preset["roles"],
            )

        def _persona_style(creds: ByokCreds) -> str:
            preset = _DEMO_PERSONAS.get((creds.demo_persona or "").lower())
            return (preset or {}).get("style", "") if preset else ""

        # ── Public demo metadata endpoints (no auth, no BYOK key needed) ──
        # Lightweight read-only endpoints that power the public corpus +
        # personas + status pages in the frontend. They expose only
        # metadata that is already implied by the demo (filename, roles,
        # sensitivity, chunk counts) -- nothing that could leak content.
        @app.get("/byok/personas", tags=["byok"])
        async def byok_personas() -> dict:
            """Return the three preset RBAC personas + their synth styles.

            Single source of truth for the frontend ``/personas`` page so
            the UI never drifts from the actual server-side dispatch in
            ``_persona_to_user_ctx``.
            """
            return {
                "items": [
                    {
                        "key": key,
                        "label": key.capitalize(),
                        "clearance_level": preset["clearance_level"],
                        "roles": list(preset["roles"]),
                        "style": preset["style"],
                    }
                    for key, preset in _DEMO_PERSONAS.items()
                ],
                "default": "engineer",
                "org_id": _DEMO_ORG_ID,
            }

        @app.get("/byok/corpus", tags=["byok"])
        async def byok_corpus() -> dict:
            """Summarise the base demo corpus -- source files + metadata.

            Scrolls the root tenant Qdrant collection (the 10 hand-curated
            demo docs) and groups points by ``source_file``. Returns one
            row per file with the chunk count, sensitivity label, and
            roles -- never the chunk text. Visitor uploads under
            ``documents_sess_<sid>`` are NOT included (those live in the
            session collection and are surfaced via ``/byok/uploads``).
            """
            from core.agents.retriever import _get_hybrid_searcher

            files: dict[str, dict] = {}
            try:
                searcher = _get_hybrid_searcher()
                # The root manager (``for_org`` of the demo org) points at
                # the base demo collection. Multi-tenant mode wraps the
                # name as ``documents_demo``.
                qdrant = searcher._qdrant.for_org(_DEMO_ORG_ID)  # type: ignore[attr-defined]
                client = qdrant.client
                collection = qdrant.collection_name
                next_offset = None
                pages = 0
                # Scroll up to 4 pages of 256 -- the demo corpus is ~140
                # chunks today so one page covers it. Guard against a
                # runaway scroll if someone later expands the corpus past
                # ~1k chunks.
                while pages < 4:
                    points, next_offset = client.scroll(
                        collection_name=collection,
                        limit=256,
                        with_payload=True,
                        with_vectors=False,
                        offset=next_offset,
                    )
                    for p in points:
                        payload = p.payload or {}
                        src = payload.get("source_file") or "unknown"
                        base_name = src.split("/")[-1].split("\\")[-1]
                        roles = payload.get("roles") or []
                        sensitivity = payload.get("sensitivity_level") or "low"
                        item = files.setdefault(
                            src,
                            {
                                "source_file": base_name,
                                "chunks": 0,
                                "sensitivity_level": sensitivity,
                                "roles": list(roles),
                            },
                        )
                        item["chunks"] += 1
                        # Roles can vary across chunks of the same file in
                        # principle; union them so the visitor sees the
                        # widest access required to retrieve the file.
                        for r in roles:
                            if r not in item["roles"]:
                                item["roles"].append(r)
                    pages += 1
                    if not next_offset:
                        break
            except Exception as exc:
                logger.warning("byok_corpus_list_failed", error=str(exc))
            sorted_items = sorted(files.values(), key=lambda f: f["source_file"].lower())
            return {
                "collection": "documents",
                "count": len(sorted_items),
                "total_chunks": sum(f["chunks"] for f in sorted_items),
                "items": sorted_items,
            }

        from pydantic import BaseModel as _ByokBaseModel

        class _ByokChatBody(_ByokBaseModel):
            """Public-demo chat payload — no auth fields, only the question text."""

            query: str
            prefer_cloud: bool = True

        # Runtime import — FastAPI dependency injection reads the annotation
        # at request time, so this must NOT be a TYPE_CHECKING-only import.
        from fastapi import Request as _FastApiRequest

        @app.post("/byok/chat", tags=["byok"])
        async def byok_chat_endpoint(
            request: _FastApiRequest,
            body: _ByokChatBody,
            creds: Annotated[ByokCreds, Depends(extract_byok)],
        ) -> dict:
            """Public-demo chat endpoint backed by BYOK credentials.

            Routing:
            - Visitor brought a key (``creds.has_user_key()``): pipeline uses
              the visitor's provider + key. No throttle.
            - Visitor did NOT bring a key: pipeline falls back to the owner's
              configured cloud provider key, gated by ``OwnerKeyHourThrottle``.
              When exhausted, returns 429 with copy nudging BYOK.

            Persona maps to a synthetic ``UserContext`` so the existing RBAC
            filter still runs end-to-end — same code path as authenticated
            queries, just with demo identities.
            """
            if not creds.has_user_key():
                throttle = get_owner_key_throttle()
                client_ip = client_ip_from_request(request)
                ok, meta = throttle.allow(client_ip)
                if not ok:
                    raise HTTPException(
                        status.HTTP_429_TOO_MANY_REQUESTS,
                        detail={
                            "reason": meta["reason"],
                            "retry_after_seconds": meta["retry_after"],
                            "hint": (
                                "Owner-key fallback exhausted for this IP. "
                                "Paste your own LLM key to continue — your key "
                                "is never stored server-side."
                            ),
                        },
                    )
            user_ctx = _persona_to_user_ctx(creds)
            import time as _t

            _t0 = _t.perf_counter()
            state = await run_rag_pipeline(
                query=body.query,
                user_context=user_ctx,
                thread_id=f"byok-{creds.session_id}",
                prefer_cloud=body.prefer_cloud,
                # Visitor's chosen provider when present; falls back to env.
                override_provider=creds.safe_provider(),
                persona_style=_persona_style(creds),
                byok_session_id=creds.session_id,
            )
            elapsed_ms = (_t.perf_counter() - _t0) * 1000
            response = QueryResponse.from_state(state)
            # Persist a single audit-log row so /byok/audit can surface the
            # session's history. utils.pii.redact strips key shapes from the
            # query/answer before write.
            try:
                audit_logger.log_query(
                    user_id=user_ctx.user_id,
                    org_id=user_ctx.org_id,
                    query=body.query,
                    response_summary=(response.answer or "")[:200],
                    sensitivity=state.get("query_sensitivity", "low"),
                    status="blocked" if response.blocked else "success",
                    latency_ms=elapsed_ms,
                    persona=creds.demo_persona or "anonymous",
                    byok_used=creds.has_user_key(),
                    synth_provider=response.provenance.provider,
                    synth_model=response.provenance.model,
                    faithfulness_ratio=state.get("faithfulness_ratio", 1.0),
                    documents_used=len(state.get("relevant_documents", [])),
                )
            except Exception as exc:  # pragma: no cover -- defensive
                logger.exception("byok_audit_persist_failed", error=str(exc))
            return {
                "session_id": creds.session_id,
                "persona": creds.demo_persona or "anonymous",
                "byok_used": creds.has_user_key(),
                "response": response.model_dump(mode="json"),
                # Extra payload surfaced for the new frontend UX -- raw audit
                # trail of nodes executed (used for the "trace pills" strip)
                # and the rewriter's output if the rewriter fired.
                "trace": [
                    {k: v for k, v in entry.items() if k in {"node", "action"}}
                    for entry in state.get("audit_trail", [])
                ],
                "rewritten_query": state.get("rewritten_query", ""),
                "query_sensitivity": state.get("query_sensitivity", "low"),
                "faithfulness_ratio": float(state.get("faithfulness_ratio", 1.0)),
                "documents_seen_total": len(state.get("documents", [])),
                "documents_used_total": len(state.get("relevant_documents", [])),
            }

        # ── BYOK streaming endpoint (SSE) ────────────────────────────────
        from fastapi.responses import StreamingResponse as _StreamingResponse

        from core.graph import run_rag_pipeline_stream

        @app.post("/byok/chat/stream", tags=["byok"])
        async def byok_chat_stream_endpoint(
            request: _FastApiRequest,
            body: _ByokChatBody,
            creds: Annotated[ByokCreds, Depends(extract_byok)],
        ):
            """Server-Sent Events variant of ``/byok/chat``.

            Emits ``event: <type>\\ndata: <json>\\n\\n`` frames mirroring the
            event dicts produced by ``run_rag_pipeline_stream``:

            - ``phase``   — graph node fired (with merged state snapshot)
            - ``token``   — synthesizer streaming token
            - ``blocked`` — guardrails / security / timeout refusal
            - ``final``   — last state + total latency

            CORS is already mounted on the app when ``byok_mode`` is on.
            """
            if not creds.has_user_key():
                throttle = get_owner_key_throttle()
                client_ip = client_ip_from_request(request)
                ok, meta = throttle.allow(client_ip)
                if not ok:
                    raise HTTPException(
                        status.HTTP_429_TOO_MANY_REQUESTS,
                        detail={
                            "reason": meta["reason"],
                            "retry_after_seconds": meta["retry_after"],
                            "hint": (
                                "Owner-key fallback exhausted for this IP. "
                                "Paste your own LLM key to continue — your key "
                                "is never stored server-side."
                            ),
                        },
                    )
            user_ctx = _persona_to_user_ctx(creds)

            async def _gen():
                import time as _t

                _t0 = _t.perf_counter()
                # Replay the session_id up front so the client can stitch
                # token deltas to a known turn without waiting for `final`.
                yield (
                    "event: open\n"
                    f"data: {json.dumps({'session_id': creds.session_id, 'persona': creds.demo_persona or 'anonymous', 'byok_used': creds.has_user_key()})}\n\n"
                )
                final_state: dict | None = None
                try:
                    async for evt in run_rag_pipeline_stream(
                        query=body.query,
                        user_context=user_ctx,
                        thread_id=f"byok-{creds.session_id}",
                        prefer_cloud=body.prefer_cloud,
                        override_provider=creds.safe_provider(),
                        persona_style=_persona_style(creds),
                        byok_session_id=creds.session_id,
                    ):
                        etype = evt.get("type", "unknown")
                        # Reshape the heavy phase/final payload -- raw GraphState
                        # is large; the frontend only needs the public-facing
                        # bits. token frames pass through verbatim.
                        if etype == "token":
                            payload = {"text": evt.get("text", "")}
                        elif etype in ("phase", "blocked", "final"):
                            st = evt.get("state", {}) or {}
                            if etype == "final":
                                final_state = st
                            payload = {
                                "name": evt.get("name", ""),
                                "message": evt.get("message", ""),
                                "latency_ms": evt.get("latency_ms", 0.0),
                                "rewritten_query": st.get("rewritten_query", ""),
                                "query_sensitivity": st.get("query_sensitivity", "low"),
                                "guardrails_passed": st.get("guardrails_passed", True),
                                "security_passed": st.get("security_passed", True),
                                "documents_seen_total": len(st.get("documents", [])),
                                "documents_used_total": len(st.get("relevant_documents", [])),
                                "faithfulness_ratio": float(st.get("faithfulness_ratio", 1.0)),
                                "confidence_score": float(st.get("confidence_score", 0.0)),
                                "synth_provider": st.get("synth_provider", ""),
                                "synth_model": st.get("synth_model", ""),
                                "synth_latency_ms": float(st.get("synth_latency_ms", 0.0)),
                                "trace": [
                                    {k: v for k, v in e.items() if k in {"node", "action"}}
                                    for e in st.get("audit_trail", [])
                                ],
                            }
                            if etype == "final":
                                # Include the full QueryResponse shape so the
                                # frontend reaches parity with the non-stream
                                # endpoint (citations + provenance + blocked).
                                payload["response"] = QueryResponse.from_state(st).model_dump(
                                    mode="json"
                                )
                        else:
                            payload = evt
                        yield f"event: {etype}\ndata: {json.dumps(payload)}\n\n"
                except Exception as exc:  # pragma: no cover -- defensive
                    logger.exception("byok_stream_failed", error=str(exc))
                    yield (f"event: error\ndata: {json.dumps({'message': 'stream_failed'})}\n\n")
                # Persist audit row at the end of the stream so /byok/audit
                # surfaces the session's history even when the visitor
                # disconnects before the final frame.
                if final_state is not None:
                    elapsed_ms = (_t.perf_counter() - _t0) * 1000
                    resp = QueryResponse.from_state(final_state)
                    try:
                        audit_logger.log_query(
                            user_id=user_ctx.user_id,
                            org_id=user_ctx.org_id,
                            query=body.query,
                            response_summary=(resp.answer or "")[:200],
                            sensitivity=final_state.get("query_sensitivity", "low"),
                            status="blocked" if resp.blocked else "success",
                            latency_ms=elapsed_ms,
                            persona=creds.demo_persona or "anonymous",
                            byok_used=creds.has_user_key(),
                            synth_provider=resp.provenance.provider,
                            synth_model=resp.provenance.model,
                            faithfulness_ratio=final_state.get("faithfulness_ratio", 1.0),
                            documents_used=len(final_state.get("relevant_documents", [])),
                            stream=True,
                        )
                    except Exception as exc:  # pragma: no cover
                        logger.exception("byok_stream_audit_persist_failed", error=str(exc))

            return _StreamingResponse(
                _gen(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                },
            )

        # ── BYOK public audit export ─────────────────────────────────────
        @app.get("/byok/audit", tags=["byok"])
        async def byok_audit_endpoint(
            request: _FastApiRequest,
            creds: Annotated[ByokCreds, Depends(extract_byok)],
        ) -> dict:
            """Return the last N PII-redacted audit entries for this demo.

            Public, session-scoped, capped by ``settings.byok_audit_max_entries``.
            The audit log file is shared across sessions inside the HF Space --
            we filter by the visitor's demo user_id (``demo-<session_id>``) so
            visitors only see their own turns.
            """
            limit = max(0, int(settings.byok_audit_max_entries))
            if limit == 0:
                raise HTTPException(status.HTTP_404_NOT_FOUND, detail="audit export disabled")
            today = date.today().isoformat()
            entries = audit_logger.get_entries(
                start_date=today,
                end_date=today,
                user_id=f"demo-{creds.session_id}",
            )
            # Newest first, then cap.
            entries = list(reversed(entries))[:limit]
            return {
                "session_id": creds.session_id,
                "count": len(entries),
                "items": [e.model_dump(mode="json") for e in entries],
            }

        # ── BYOK upload endpoints ────────────────────────────────────────
        from fastapi import File, UploadFile
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        from core.agents.retriever import _get_hybrid_searcher
        from ingestion.metadata import IngestRequest, SensitivityLevel
        from ingestion.pipeline import IngestionPipeline

        def _session_qdrant_for_creds(creds: ByokCreds):
            """Return a QdrantManager bound to the visitor's session collection."""
            searcher = _get_hybrid_searcher()
            return searcher._qdrant.for_session(creds.session_id)  # type: ignore[attr-defined]

        def _list_session_uploads(creds: ByokCreds) -> list[dict]:
            """Group session-collection points by `source_file` -> upload rows."""
            qdrant = _session_qdrant_for_creds(creds)
            client = qdrant.client
            collection = qdrant.collection_name
            files: dict[str, dict] = {}
            try:
                # Single scroll; the cap is 5 files * O(low chunks) so a single
                # page covers it. Limit at 1000 points is defensive only.
                points, _next = client.scroll(
                    collection_name=collection,
                    limit=1000,
                    with_payload=True,
                    with_vectors=False,
                )
                for p in points:
                    payload = p.payload or {}
                    src = payload.get("source_file") or ""
                    # Reduce absolute paths down to a basename + sha so the
                    # visitor sees the filename they uploaded, not the
                    # container's tmp path.
                    base_name = src.split("/")[-1].split("\\")[-1]
                    item = files.setdefault(
                        src,
                        {
                            "file_id": payload.get("source_file_id", base_name),
                            "filename": base_name,
                            "source_file": src,
                            "chunks": 0,
                            "first_ingested": payload.get("ingested_at"),
                        },
                    )
                    item["chunks"] += 1
            except Exception as exc:
                logger.warning("byok_uploads_list_failed", error=str(exc))
            return list(files.values())

        @app.get("/byok/uploads", tags=["byok"])
        async def byok_uploads_list(
            request: _FastApiRequest,
            creds: Annotated[ByokCreds, Depends(extract_byok)],
        ) -> dict:
            uploads = _list_session_uploads(creds)
            return {
                "session_id": creds.session_id,
                "count": len(uploads),
                "max_files": settings.byok_upload_max_files,
                "max_bytes": settings.byok_upload_max_bytes,
                "max_chunks_per_file": settings.byok_upload_max_chunks_per_file,
                "allowed_extensions": list(settings.byok_upload_allowed_extensions),
                "items": uploads,
            }

        @app.post("/byok/uploads", tags=["byok"])
        async def byok_uploads_ingest(
            request: _FastApiRequest,
            file: Annotated[UploadFile, File(...)],
            creds: Annotated[ByokCreds, Depends(extract_byok)],
        ) -> dict:
            """Accept a multipart upload from the BYOK visitor.

            The file is parsed by the existing ``ingestion.pipeline``, chunked,
            embedded, and upserted into the visitor's session-scoped Qdrant
            collection (``documents_sess_<sid>``). Visitor uploads are tagged
            ``org_id="demo"`` + roles=["viewer", ...all personas] +
            sensitivity_level=LOW so every demo persona can see them; the
            visitor's session collection is the isolation boundary.
            """
            # ── 1. Validate ext + size ──────────────────────────────────
            filename = file.filename or "upload"
            ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
            allowed = {e.lower() for e in settings.byok_upload_allowed_extensions}
            if ext not in allowed:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST,
                    detail={
                        "reason": "unsupported_extension",
                        "extension": ext,
                        "allowed": sorted(allowed),
                    },
                )
            # Read with a hard cap; abort on first byte over the limit.
            max_bytes = int(settings.byok_upload_max_bytes)
            buf = bytearray()
            while True:
                chunk = await file.read(64 * 1024)
                if not chunk:
                    break
                buf.extend(chunk)
                if len(buf) > max_bytes:
                    raise HTTPException(
                        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail={
                            "reason": "file_too_large",
                            "limit_bytes": max_bytes,
                        },
                    )
            if len(buf) == 0:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, detail={"reason": "empty_file"})

            # ── 2. Enforce per-session file-count cap ───────────────────
            existing = _list_session_uploads(creds)
            if len(existing) >= int(settings.byok_upload_max_files):
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    detail={
                        "reason": "upload_quota_exceeded",
                        "max_files": settings.byok_upload_max_files,
                        "hint": "Delete an existing upload first.",
                    },
                )

            # ── 3. Spool to temp file -- the ingestion pipeline reads from disk.
            import os as _os
            import tempfile as _tempfile
            import uuid as _uuid
            from datetime import UTC as _UTC
            from datetime import datetime as _datetime

            file_id = _uuid.uuid4().hex
            safe_name = (
                "".join(c if (c.isalnum() or c in "._-") else "_" for c in filename) or "upload"
            )
            tmp_dir = _tempfile.mkdtemp(prefix=f"byok_{creds.session_id}_")
            tmp_path = _os.path.join(tmp_dir, safe_name)
            try:
                with open(tmp_path, "wb") as fh:
                    fh.write(bytes(buf))

                # ── 4. Build session-scoped pipeline + ingest ───────────
                searcher = _get_hybrid_searcher()
                sess_qdrant = searcher._qdrant.for_session(creds.session_id)  # type: ignore[attr-defined]
                pipeline = IngestionPipeline(
                    qdrant_manager=sess_qdrant,
                    embedding_service=searcher._embedder,  # type: ignore[attr-defined]
                    sparse_service=searcher._sparse,  # type: ignore[attr-defined]
                )
                req = IngestRequest(
                    file_path=tmp_path,
                    user_id=f"demo-{creds.session_id}",
                    org_id=_DEMO_ORG_ID,
                    sensitivity_level=SensitivityLevel.LOW,
                    # Visible to every demo persona inside the visitor's session.
                    roles=[
                        "viewer",
                        "engineering",
                        "compliance",
                        "legal",
                        "executive",
                    ],
                )
                result = await pipeline.ingest_document(req)
                # Reject documents that chunk too aggressively. Without this
                # cap a 100-page PDF dominates the dual-collection retrieval
                # and the grader/faithfulness loop times out under the 180 s
                # SLO. Delete what we just upserted and return 413 so the
                # visitor knows exactly what failed.
                max_chunks = int(settings.byok_upload_max_chunks_per_file)
                if max_chunks > 0 and result.num_chunks > max_chunks:
                    try:
                        if result.point_ids:
                            sess_qdrant.client.delete(
                                collection_name=sess_qdrant.collection_name,
                                points_selector=list(result.point_ids),
                            )
                    except Exception as exc:
                        logger.warning(
                            "byok_upload_oversize_cleanup_failed",
                            error=str(exc),
                            chunks=result.num_chunks,
                        )
                    raise HTTPException(
                        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail={
                            "reason": "too_many_chunks",
                            "chunk_count": result.num_chunks,
                            "max_chunks": max_chunks,
                            "hint": (
                                f"This file chunks to {result.num_chunks} pieces "
                                f"but the BYOK demo caps single uploads at {max_chunks}. "
                                "Try a shorter document or a section instead."
                            ),
                        },
                    )
                # Tag every newly-upserted chunk with the file_id + ingested_at
                # so the list and delete endpoints can group by file.
                if result.point_ids:
                    try:
                        sess_qdrant.client.set_payload(
                            collection_name=sess_qdrant.collection_name,
                            payload={
                                "source_file_id": file_id,
                                "ingested_at": _datetime.now(_UTC).isoformat(),
                                "original_filename": safe_name,
                            },
                            points=list(result.point_ids),
                        )
                    except Exception as exc:
                        logger.warning("byok_upload_set_payload_failed", error=str(exc))

                # ── 5. Audit ────────────────────────────────────────────
                try:
                    audit_logger.log_query(
                        user_id=f"demo-{creds.session_id}",
                        org_id=_DEMO_ORG_ID,
                        query=f"[upload] {safe_name}",
                        response_summary=(
                            f"ingested {result.num_chunks} chunks from {len(buf)} bytes"
                        ),
                        sensitivity="low",
                        status=result.status,
                        latency_ms=result.processing_time_seconds * 1000,
                        action_hint="upload",
                        file_id=file_id,
                        filename=safe_name,
                        chunks=result.num_chunks,
                    )
                except Exception as exc:
                    logger.warning("byok_upload_audit_failed", error=str(exc))

                return {
                    "session_id": creds.session_id,
                    "file_id": file_id,
                    "filename": safe_name,
                    "status": result.status,
                    "chunks": result.num_chunks,
                    "errors": result.errors,
                    "processing_time_seconds": result.processing_time_seconds,
                }
            finally:
                # Always purge the temp file -- visitor content never lingers on disk.
                try:
                    _os.remove(tmp_path)
                    _os.rmdir(tmp_dir)
                except OSError:
                    pass

        @app.delete("/byok/uploads/{file_id}", tags=["byok"])
        async def byok_uploads_delete(
            request: _FastApiRequest,
            file_id: str,
            creds: Annotated[ByokCreds, Depends(extract_byok)],
        ) -> dict:
            qdrant = _session_qdrant_for_creds(creds)
            client = qdrant.client
            collection = qdrant.collection_name
            flt = Filter(
                must=[FieldCondition(key="source_file_id", match=MatchValue(value=file_id))]
            )
            try:
                # Count before delete so we can report what was dropped.
                count_resp = client.count(collection_name=collection, count_filter=flt, exact=True)
                deleted = int(getattr(count_resp, "count", 0))
                client.delete(collection_name=collection, points_selector=flt)
            except Exception as exc:
                raise HTTPException(
                    status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"delete_failed: {exc!s}",
                ) from exc
            return {
                "session_id": creds.session_id,
                "file_id": file_id,
                "deleted_chunks": deleted,
            }

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
            embedding_service=searcher._embedder,  # type: ignore[attr-defined]
            sparse_service=searcher._sparse,  # type: ignore[attr-defined]
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
        if settings.jwt_algorithm.upper() == "RS256":
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                "Dev token endpoint disabled in RS256 mode — use the external IdP",
            )
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
            token_type="bearer",
            expires_in=body.ttl_seconds or settings.jwt_ttl_seconds,
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

    When ``SAR_JWT_SECRET`` is configured this mints a real signed JWT. With
    no secret it emits the legacy unsigned base64 shape *only* when
    ``SAR_ALLOW_UNSIGNED_TOKENS`` is on (matching the verifier's fail-closed
    policy); otherwise it raises so callers are forced to configure auth.
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
            # Fall through to legacy shape on issuer error (only if allowed).
            pass
    if not settings.allow_unsigned_tokens:
        raise AuthError(
            "missing",
            "cannot mint a token: set SAR_JWT_SECRET, or SAR_ALLOW_UNSIGNED_TOKENS=true for dev",
        )
    payload = json.dumps(user).encode("utf-8")
    return base64.b64encode(payload).decode("ascii")
