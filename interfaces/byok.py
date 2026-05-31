"""BYOK (Bring Your Own Key) request extraction for the public demo.

Mounted on the FastAPI surface only when ``settings.byok_mode=True`` (production
HF Space image). Extracts per-request LLM credentials and session identity from
HTTP headers so the RAG pipeline can route to the visitor's own LLM provider
and Qdrant collection.

The extracted ``ByokCreds`` is **never persisted**:

- API keys live only in the request scope (FastAPI dep dies after response)
- ``utils.pii.redact`` strips key-shaped substrings from audit log entries
- The frontend stores the key in ``localStorage`` and forwards it as a header;
  cookies are forbidden (CSRF surface).

See ``launch-plan/03-backend-byok.md`` and ``launch-plan/11-security-checklist.md``.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from fastapi import Request


# Header names the frontend sends.
HDR_USER_KEY = "X-User-LLM-Key"
HDR_USER_PROVIDER = "X-User-Provider"
HDR_USER_OLLAMA_URL = "X-User-Ollama-URL"
HDR_SESSION_ID = "X-Session-ID"
HDR_DEMO_PERSONA = "X-Demo-Persona"

# Supported provider literals carried in X-User-Provider.
SUPPORTED_PROVIDERS: frozenset[str] = frozenset({"groq", "openai", "anthropic", "ollama"})


class ByokCreds(BaseModel):
    """Per-request BYOK credentials and session identity.

    Attributes:
        user_key: Visitor's own LLM provider API key. None means owner-key
            fallback (subject to ``OwnerKeyHourThrottle``).
        provider: Which LLM provider the ``user_key`` is for. Validated
            against ``SUPPORTED_PROVIDERS``. None defaults to the platform
            owner's configured ``cloud_provider``.
        ollama_url: Visitor's Ollama instance URL when provider == "ollama".
            Ignored otherwise.
        session_id: Per-visitor session identifier. Drives the per-session
            Qdrant collection name. Generated server-side when the visitor
            does not provide one (first request of a session).
        demo_persona: Optional preset RBAC profile for the public demo —
            ``engineer`` / ``compliance`` / ``executive``. Translated to
            ``UserContext`` downstream.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    user_key: str | None = None
    provider: str | None = None
    ollama_url: str | None = None
    session_id: str = Field(..., min_length=1, max_length=128)
    demo_persona: str | None = None

    def has_user_key(self) -> bool:
        """True when the visitor brought their own LLM key.

        Owner-key fallback (False) goes through the per-IP throttle; visitor
        BYOK (True) bypasses it. Callers MUST consult this before deciding to
        consume the owner-key quota.
        """
        return bool(self.user_key and self.user_key.strip())

    def safe_provider(self) -> str | None:
        """Return ``provider`` if it is in the allowlist, else None."""
        if self.provider and self.provider.lower() in SUPPORTED_PROVIDERS:
            return self.provider.lower()
        return None

    def byok_active(self) -> bool:
        """True when the visitor's creds can actually power a per-request LLM call.

        Stricter than :meth:`has_user_key`: a key-based provider (groq / openai /
        anthropic) needs a non-empty key AND a valid provider; an Ollama BYOK
        needs a reachable URL. This is the gate the chat endpoints use both to
        (a) bypass the owner-key throttle and (b) bind the per-request client —
        so a bare ``X-User-LLM-Key`` with no usable provider can no longer skip
        the throttle while still spending the owner key.
        """
        prov = self.safe_provider()
        if prov in ("groq", "openai", "anthropic"):
            return self.has_user_key()
        if prov == "ollama":
            return bool(self.ollama_url and self.ollama_url.strip())
        return False


def client_ip_from_request(request: Request) -> str:
    """Resolve the visitor IP for throttling, honouring ``X-Forwarded-For``.

    HF Spaces, Vercel, and most cloud reverse proxies set ``X-Forwarded-For``
    with the chain ``client, proxy1, proxy2`` (each trusted proxy *appends* the
    peer it saw). Falls back to the socket peer (``request.client.host``) when
    XFF is absent, then to ``"anon"`` so the throttle never crashes on a missing
    source.

    Trust model: XFF is a client-appendable header, so the leftmost token is
    attacker-controlled and can be spoofed to bypass the per-IP owner-key
    throttle. When ``settings.byok_xff_trusted_hops`` (``SAR_BYOK_XFF_TRUSTED_HOPS``)
    is set to the number of trusted proxies in front of the app, we take the
    entry that many positions from the right — the address the *innermost
    trusted* proxy observed, which a client cannot forge. With the default 0 we
    keep the legacy leftmost behaviour (best-effort; the provider's own per-key
    quota is the real ceiling). Only enable ``SAR_BYOK_MODE`` behind a trusted
    proxy.

    ``X-Real-IP`` is honoured second as a courtesy for proxies that set it
    instead of XFF.
    """
    xff = request.headers.get("x-forwarded-for") or request.headers.get("X-Forwarded-For")
    if xff:
        parts = [p.strip() for p in xff.split(",") if p.strip()]
        if parts:
            from config.settings import settings

            hops = max(0, int(settings.byok_xff_trusted_hops))
            # With N trusted appending proxies, the spoof-resistant client IP is
            # parts[-(N+1)]. Fall back to leftmost when the chain is shorter than
            # the configured trust depth (misconfiguration / direct hit).
            if hops > 0 and len(parts) >= hops + 1:
                return parts[-(hops + 1)]
            return parts[0]
    real_ip = request.headers.get("x-real-ip") or request.headers.get("X-Real-IP")
    if real_ip and real_ip.strip():
        return real_ip.strip()
    if request.client and request.client.host:
        return request.client.host
    return "anon"


def _derive_session_id(client_host: str | None) -> str:
    """Generate a deterministic-but-non-identifying session ID.

    Falls back to a short hash of the client host + a random UUID. The hash
    keeps the same session sticky if the visitor reconnects within the same
    UVicorn worker; the random UUID ensures cross-worker / cross-restart
    isolation. The full UUID flavour stays server-side — we never expose
    raw IP addresses in the collection name.
    """
    host = (client_host or "anon").strip() or "anon"
    digest = hashlib.sha256(host.encode("utf-8")).hexdigest()[:8]
    # Full UUID4 (122 bits) for the random component — the session id guards one
    # visitor's session-scoped uploads / audit from another, so it must be hard
    # to guess. The host digest only adds reconnect stickiness within a worker.
    return f"{digest}-{uuid.uuid4().hex}"


def build_creds(
    *,
    user_key: str | None,
    provider: str | None,
    ollama_url: str | None,
    session_id: str | None,
    demo_persona: str | None,
    client_host: str | None,
) -> ByokCreds:
    """Pure factory — builds ``ByokCreds`` from raw header values.

    Separated from the FastAPI dependency so it is unit-testable without
    spinning up a Request object. Whitespace-trims every input; generates
    ``session_id`` server-side when the client omitted it.
    """
    return ByokCreds(
        user_key=(user_key or None),
        provider=(provider or None),
        ollama_url=(ollama_url or None),
        session_id=(session_id or "").strip() or _derive_session_id(client_host),
        demo_persona=(demo_persona or None),
    )


# ── FastAPI integration ──────────────────────────────────────────────────────
# Header annotations live in this branch so the module can be imported in
# environments where fastapi is not installed (e.g. lightweight unit tests).

try:
    # Runtime imports — FastAPI dependency injection reads annotations at
    # request time, so these must NOT live in a TYPE_CHECKING-only block.
    from fastapi import Header, Request  # noqa: TC002

    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FASTAPI_AVAILABLE = False

    def Header(*_a: object, **_kw: object) -> None:  # type: ignore[no-redef]  # noqa: N802 — keep FastAPI's name
        """No-op shim when FastAPI is not installed (lint-only env)."""
        return None


if _FASTAPI_AVAILABLE:
    from typing import Annotated

    def extract_byok(
        request: Request,
        x_user_llm_key: Annotated[str | None, Header()] = None,
        x_user_provider: Annotated[str | None, Header()] = None,
        x_user_ollama_url: Annotated[str | None, Header()] = None,
        x_session_id: Annotated[str | None, Header()] = None,
        x_demo_persona: Annotated[str | None, Header()] = None,
    ) -> ByokCreds:
        """FastAPI dependency: extract per-request BYOK credentials.

        Pure data extraction — authentication, throttling, and routing
        decisions happen downstream so they can be unit-tested independently
        of FastAPI's request lifecycle.
        """
        host = request.client.host if request.client else None
        return build_creds(
            user_key=x_user_llm_key,
            provider=x_user_provider,
            ollama_url=x_user_ollama_url,
            session_id=x_session_id,
            demo_persona=x_demo_persona,
            client_host=host,
        )
