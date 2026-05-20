"""JWT issuance + verification for the FastAPI / MCP surfaces.

Why
---
Until now the only auth was a base64-encoded JSON ``UserContext`` carried as
a bearer token. That proves nothing — any caller can craft any identity. This
module replaces it with HS256-signed JWTs:

- ``issue_token(user_id, org_id, roles, clearance_level)`` mints a token
  signed with ``settings.jwt_secret``. Suitable for the dev ``/token``
  endpoint and for tests.
- ``verify_token(token)`` validates signature, expiry, and (when configured)
  ``iss`` / ``aud`` claims, then returns a ``UserContext`` plus the raw
  claims (for audit logging the ``jti``).

Backwards compatibility
-----------------------
When ``settings.jwt_secret`` is unset the verifier falls back to the legacy
base64 path and logs a warning. This keeps existing tests/smoke scripts
running. Production deployments must set ``SAR_JWT_SECRET``.

Production replacement
----------------------
For real IdP integration (Keycloak / Auth0 / Microsoft Entra) replace the
``_verify_jwt`` body with the IdP's JWKS-driven public-key verification — the
function signature and the surrounding plumbing stay the same.
"""

from __future__ import annotations

import base64
import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from config.settings import settings
from ingestion.metadata import UserContext
from utils.logging import get_logger

logger = get_logger(__name__)


class AuthError(Exception):
    """Raised when token verification fails for any reason.

    Carries a short machine-readable ``reason`` (``missing`` / ``expired`` /
    ``bad_signature`` / ``bad_claims`` / ``malformed``) so callers can map
    consistently to HTTP status codes.
    """

    def __init__(self, reason: str, message: str = "") -> None:
        super().__init__(message or reason)
        self.reason = reason


def _jose_available() -> bool:
    """Best-effort detection of python-jose. Cached on import would be
    nicer but a function makes the unit tests cleaner.
    """
    try:
        import jose  # noqa: F401

        return True
    except ImportError:
        return False


def issue_token(
    user_id: str,
    org_id: str,
    roles: list[str],
    clearance_level: int = 1,
    ttl_seconds: int | None = None,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """Mint a signed JWT for the given identity.

    Args:
        user_id: Stable principal identifier.
        org_id: Organization the principal belongs to. Drives multi-tenant
            collection routing.
        roles: List of role strings carried into the ``UserContext``.
        clearance_level: Numeric clearance (1=low / 3=high).
        ttl_seconds: Lifetime override; defaults to ``settings.jwt_ttl_seconds``.
        extra_claims: Optional extra claims to merge into the token payload.

    Returns:
        Compact JWT string.

    Raises:
        AuthError: If ``settings.jwt_secret`` is not configured or python-jose
            is missing.
    """
    if not settings.jwt_secret:
        raise AuthError("missing", "SAR_JWT_SECRET is not configured")
    if not _jose_available():
        raise AuthError("missing", "python-jose is not installed (install the [api] extra)")

    from jose import jwt  # type: ignore[import-not-found]

    now = datetime.now(UTC)
    ttl = ttl_seconds if ttl_seconds is not None else settings.jwt_ttl_seconds
    payload: dict[str, Any] = {
        "sub": user_id,
        "user_id": user_id,
        "org_id": org_id,
        "roles": list(roles),
        "clearance_level": int(clearance_level),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=ttl)).timestamp()),
        "jti": str(uuid.uuid4()),
    }
    if settings.jwt_issuer:
        payload["iss"] = settings.jwt_issuer
    if settings.jwt_audience:
        payload["aud"] = settings.jwt_audience
    if extra_claims:
        payload.update(extra_claims)

    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    logger.info(
        "jwt_issued",
        user_id=user_id,
        org_id=org_id,
        roles=roles,
        jti=payload["jti"],
        ttl_seconds=ttl,
    )
    return token


def _verify_legacy_base64(token: str) -> tuple[UserContext, dict[str, Any]]:
    """Decode the legacy unsigned base64(json(UserContext)) shape.

    Used only when ``settings.jwt_secret`` is unset. Always raises a runtime
    warning so deployments don't forget to flip on signed JWTs.
    """
    logger.warning(
        "auth_unsigned_token",
        message=(
            "SAR_JWT_SECRET unset — accepting unsigned base64 tokens. Anyone "
            "with network access can impersonate any user. Configure "
            "SAR_JWT_SECRET in production."
        ),
    )
    try:
        payload = json.loads(base64.b64decode(token).decode("utf-8"))
    except Exception as exc:
        raise AuthError("malformed", f"base64/json decode failed: {exc}") from exc
    try:
        ctx = UserContext(**payload)
    except Exception as exc:
        raise AuthError("bad_claims", f"UserContext build failed: {exc}") from exc
    return ctx, {"sub": ctx.user_id, "jti": "unsigned"}


def _verify_jwt(token: str) -> tuple[UserContext, dict[str, Any]]:
    """Verify a signed JWT against the configured secret.

    Production swap-in point: replace this body with JWKS-driven RS256
    verification against an external IdP.
    """
    if not _jose_available():
        raise AuthError("missing", "python-jose is not installed")

    from jose import JWTError, jwt  # type: ignore[import-not-found]

    options: dict[str, Any] = {"require": ["exp", "iat", "sub"]}
    audience = settings.jwt_audience or None
    if not audience:
        # When no aud configured, disable that check so the token decodes.
        options["verify_aud"] = False

    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            audience=audience,
            issuer=settings.jwt_issuer or None,
            options=options,
        )
    except JWTError as exc:
        msg = str(exc).lower()
        # Map the jose exception text into a stable reason code.
        if "expired" in msg:
            reason = "expired"
        elif "signature" in msg:
            reason = "bad_signature"
        elif "claim" in msg or "audience" in msg or "issuer" in msg:
            reason = "bad_claims"
        else:
            reason = "malformed"
        raise AuthError(reason, f"jwt decode failed: {exc}") from exc

    try:
        ctx = UserContext(
            user_id=claims.get("user_id") or claims["sub"],
            org_id=claims.get("org_id", ""),
            roles=list(claims.get("roles", [])),
            clearance_level=int(claims.get("clearance_level", 1)),
        )
    except Exception as exc:
        raise AuthError("bad_claims", f"UserContext build failed: {exc}") from exc

    return ctx, claims


def verify_token(token: str) -> tuple[UserContext, dict[str, Any]]:
    """Resolve a bearer token to a ``UserContext`` plus the raw claims.

    Args:
        token: Raw bearer token (no ``Bearer `` prefix).

    Returns:
        ``(user_context, claims)``. ``claims`` includes at minimum ``sub`` and
        ``jti``; the latter is used in audit-trail entries so a tampered or
        replayed token is traceable.

    Raises:
        AuthError: With ``.reason`` set to one of
            ``missing`` / ``malformed`` / ``expired`` / ``bad_signature`` /
            ``bad_claims``.
    """
    if not token or not isinstance(token, str):
        raise AuthError("missing", "empty token")
    if settings.jwt_secret:
        return _verify_jwt(token)
    return _verify_legacy_base64(token)
