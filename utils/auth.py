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

RS256 / JWKS mode
-----------------
When ``settings.jwt_algorithm`` is ``"RS256"`` tokens are verified against a
remote JWKS endpoint (Keycloak / Auth0 / etc.). The local ``/token`` endpoint
returns 404 because we do not hold the IdP's private key. See ``utils/jwks_cache``.

Fail-closed default
-------------------
When ``settings.jwt_secret`` is unset the verifier rejects every token. The
legacy unsigned base64 shape is accepted *only* when
``settings.allow_unsigned_tokens`` is explicitly turned on (dev/test). An
unsigned token proves no identity, so it is never honoured silently in
production. Deployments must set ``SAR_JWT_SECRET`` or ``SAR_JWKS_URL``.
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


def _decode_header(token: str) -> dict[str, Any]:
    """Decode the JWT header without verification."""
    parts = token.split(".")
    if len(parts) != 3:
        raise AuthError("malformed", "token must have 3 segments")
    padding = 4 - len(parts[0]) % 4
    if padding != 4:
        parts[0] += "=" * padding
    try:
        return json.loads(base64.urlsafe_b64decode(parts[0]).decode("utf-8"))
    except Exception as exc:
        raise AuthError("malformed", f"invalid header: {exc}") from exc


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
    if settings.jwt_algorithm.upper() == "RS256":
        raise AuthError("missing", "Cannot issue RS256 tokens locally — use the external IdP")
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
    """Verify a signed JWT (HS256 or RS256)."""
    if not _jose_available():
        raise AuthError("missing", "python-jose is not installed")

    from jose import JWTError, jwt  # type: ignore[import-not-found]

    options: dict[str, Any] = {"require": ["exp", "iat", "sub"]}
    audience = settings.jwt_audience or None
    if not audience:
        options["verify_aud"] = False

    algorithm = settings.jwt_algorithm.upper()

    # Resolve the verification key.
    if algorithm == "RS256":
        try:
            header = _decode_header(token)
            kid = header.get("kid")
            if not kid:
                raise AuthError("bad_claims", "RS256 token missing 'kid' header")
            from utils.jwks_cache import get_signing_key

            key = get_signing_key(kid)
        except AuthError:
            raise
        except Exception as exc:
            raise AuthError("bad_signature", f"JWKS lookup failed: {exc}") from exc
    else:
        key = settings.jwt_secret
        if not key:
            raise AuthError("missing", "SAR_JWT_SECRET is not configured")

    try:
        claims = jwt.decode(
            token,
            key,
            algorithms=[algorithm],
            audience=audience,
            issuer=settings.jwt_issuer or None,
            options=options,
        )
    except JWTError as exc:
        msg = str(exc).lower()
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
    # When jwt_secret is set OR we're in RS256 mode, use JWT verification.
    if settings.jwt_secret or settings.jwt_algorithm.upper() == "RS256":
        return _verify_jwt(token)
    # No signing key configured. Fail closed unless the legacy unsigned shape
    # is explicitly opted into (dev/test only).
    if settings.allow_unsigned_tokens:
        return _verify_legacy_base64(token)
    raise AuthError(
        "missing",
        "no JWT secret configured and unsigned tokens are disabled "
        "(set SAR_JWT_SECRET, or SAR_ALLOW_UNSIGNED_TOKENS=true for dev only)",
    )
