"""In-memory JWKS cache with TTL for RS256 token verification.

Fetches the JSON Web Key Set from a remote endpoint and caches it for
``settings.jwks_cache_ttl_seconds`` (default 5 min). When a token arrives
with a ``kid`` not present in the cache, the cache is refreshed once
before giving up.

Usage::

    from utils.jwks_cache import get_signing_key
    key = get_signing_key(token_kid, jwks_url="https://idp/realm/protocol/openid-connect/certs")
"""

from __future__ import annotations

import json
import time
from typing import Any

from config.settings import settings
from utils.logging import get_logger

logger = get_logger(__name__)

# Module-level cache: {jwks_url: {"fetched_at": float, "keys": dict[str, str]}}
_cache: dict[str, dict[str, Any]] = {}


def _fetch_jwks(url: str) -> dict[str, str]:
    """Download JWKS and return a mapping kid -> PEM-encoded public key."""
    import urllib.request

    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    keys: dict[str, str] = {}
    for jwk in data.get("keys", []):
        kid = jwk.get("kid")
        if not kid:
            continue
        # Only RSA keys are supported for RS256.
        if jwk.get("kty") != "RSA":
            continue
        pem = _jwk_to_pem(jwk)
        if pem:
            keys[kid] = pem

    logger.info("jwks_fetched", url=url, key_count=len(keys))
    return keys


def _jwk_to_pem(jwk: dict[str, Any]) -> str | None:
    """Convert a single RSA JWK to PEM using python-jose if available."""
    try:
        from jose.backends import RSAKey
        from jose.utils import base64url_decode
    except ImportError:
        logger.warning("python_jose_missing_for_jwk_conversion")
        return None

    try:
        n = int.from_bytes(base64url_decode(jwk["n"]), "big")
        e = int.from_bytes(base64url_decode(jwk["e"]), "big")
        rsa_key = RSAKey({"n": n, "e": e})
        return rsa_key.to_pem().decode("utf-8")
    except Exception as exc:
        logger.warning("jwk_to_pem_failed", error=str(exc), kid=jwk.get("kid"))
        return None


def get_signing_key(
    kid: str,
    jwks_url: str | None = None,
    force_refresh: bool = False,
) -> str:
    """Return the PEM-encoded RSA public key for the given ``kid``.

    Args:
        kid: Key ID from the JWT header.
        jwks_url: JWKS endpoint. Defaults to ``settings.jwks_url``.
        force_refresh: If True, bypass the cache and re-fetch.

    Returns:
        PEM-encoded public key string.

    Raises:
        RuntimeError: If the key cannot be found after refresh.
    """
    url = jwks_url or settings.jwks_url
    if not url:
        raise RuntimeError("jwks_url is not configured")

    ttl = getattr(settings, "jwks_cache_ttl_seconds", 300)
    now = time.monotonic()

    entry = _cache.get(url)
    if not force_refresh and entry and (now - entry["fetched_at"]) < ttl and kid in entry["keys"]:
        return entry["keys"][kid]

    # Cache miss or stale — fetch.
    try:
        keys = _fetch_jwks(url)
    except Exception as exc:
        # On fetch failure, try to serve stale if we have it.
        if entry and kid in entry.get("keys", {}):
            logger.warning("jwks_fetch_failed_serving_stale", error=str(exc), url=url)
            return entry["keys"][kid]
        raise RuntimeError(f"JWKS fetch failed: {exc}") from exc

    _cache[url] = {"fetched_at": now, "keys": keys}

    if kid not in keys:
        raise RuntimeError(f"kid '{kid}' not found in JWKS")
    return keys[kid]


def clear_cache(url: str | None = None) -> None:
    """Clear the JWKS cache. Used in tests."""
    global _cache
    if url is None:
        _cache.clear()
    else:
        _cache.pop(url, None)
