"""Tests for JWT auth layer (HS256 + RS256)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from config.settings import settings
from utils.auth import AuthError, issue_token, verify_token

pytest.importorskip("jose")

SECRET = "test-secret-do-not-use-in-prod"


def _make_rs256_token(claims: dict, kid: str | None = "test-kid") -> str:
    """Build an unsigned RS256-shaped JWT (header.payload.sig) for testing.

    The signature is a dummy — tests that need a valid signature mock
    ``get_signing_key`` to return a real RSA public key.
    """
    import base64
    import json

    header_dict: dict[str, str] = {"alg": "RS256", "typ": "JWT"}
    if kid is not None:
        header_dict["kid"] = kid
    header = base64.urlsafe_b64encode(json.dumps(header_dict).encode()).decode().rstrip("=")
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"{header}.{payload}.dummysig"


def _generate_rsa_key_pair() -> tuple[str, str]:
    """Return (private_pem, public_pem) for testing RS256."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    public_pem = (
        private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("utf-8")
    )
    return private_pem, public_pem


# ── HS256 tests (existing behaviour) ──────────────────────────────────────────


def test_issue_then_verify_round_trip():
    with patch.object(settings, "jwt_secret", SECRET):
        token = issue_token(
            user_id="alice",
            org_id="acme",
            roles=["admin", "user"],
            clearance_level=3,
        )
        ctx, claims = verify_token(token)

    assert ctx.user_id == "alice"
    assert ctx.org_id == "acme"
    assert ctx.roles == ["admin", "user"]
    assert ctx.clearance_level == 3
    assert claims["jti"]
    assert claims["sub"] == "alice"
    assert "iat" in claims and "exp" in claims


def test_expired_token_rejected():
    with patch.object(settings, "jwt_secret", SECRET):
        token = issue_token(
            user_id="alice",
            org_id="acme",
            roles=["user"],
            ttl_seconds=-30,
        )
        with pytest.raises(AuthError) as exc:
            verify_token(token)
    assert exc.value.reason == "expired"


def test_bad_signature_rejected():
    with patch.object(settings, "jwt_secret", SECRET):
        token = issue_token(user_id="alice", org_id="acme", roles=["user"])
    parts = token.split(".")
    parts[-1] = parts[-1][:-5] + "AAAAA"
    tampered = ".".join(parts)
    with patch.object(settings, "jwt_secret", SECRET), pytest.raises(AuthError) as exc:
        verify_token(tampered)
    assert exc.value.reason == "bad_signature"


def test_wrong_secret_rejected():
    with patch.object(settings, "jwt_secret", SECRET):
        token = issue_token(user_id="alice", org_id="acme", roles=["user"])
    with patch.object(settings, "jwt_secret", "OTHER"), pytest.raises(AuthError) as exc:
        verify_token(token)
    assert exc.value.reason == "bad_signature"


def test_wrong_audience_rejected():
    with (
        patch.object(settings, "jwt_secret", SECRET),
        patch.object(settings, "jwt_audience", "first-aud"),
    ):
        token = issue_token(user_id="alice", org_id="acme", roles=["user"])

    with (
        patch.object(settings, "jwt_secret", SECRET),
        patch.object(settings, "jwt_audience", "different-aud"),
        pytest.raises(AuthError) as exc,
    ):
        verify_token(token)
    assert exc.value.reason == "bad_claims"


def test_empty_token_raises_missing():
    with patch.object(settings, "jwt_secret", SECRET), pytest.raises(AuthError) as exc:
        verify_token("")
    assert exc.value.reason == "missing"


def test_issue_without_secret_raises_missing():
    with patch.object(settings, "jwt_secret", None), pytest.raises(AuthError) as exc:
        issue_token(user_id="alice", org_id="acme", roles=["user"])
    assert exc.value.reason == "missing"


def test_legacy_unsigned_token_accepted_when_secret_unset():
    import base64
    import json

    payload = {
        "user_id": "alice",
        "org_id": "acme",
        "roles": ["user"],
        "clearance_level": 1,
    }
    token = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")

    with patch.object(settings, "jwt_secret", None):
        ctx, claims = verify_token(token)
    assert ctx.user_id == "alice"
    assert claims["jti"] == "unsigned"


def test_malformed_legacy_token_rejected_when_secret_unset():
    with patch.object(settings, "jwt_secret", None), pytest.raises(AuthError) as exc:
        verify_token("not-base64!@#$")
    assert exc.value.reason in {"malformed", "bad_claims"}


# ── RS256 tests ───────────────────────────────────────────────────────────────


def test_rs256_token_verified_with_mocked_jwks():
    """RS256 token is verified when get_signing_key returns the correct RSA key."""
    from jose import jwt as jose_jwt

    private_pem, public_pem = _generate_rsa_key_pair()

    claims = {
        "sub": "alice",
        "user_id": "alice",
        "org_id": "acme",
        "roles": ["admin"],
        "clearance_level": 3,
        "iat": 1609459200,
        "exp": 4102444800,
        "jti": "test-jti-123",
    }
    token = jose_jwt.encode(claims, private_pem, algorithm="RS256", headers={"kid": "test-kid"})

    with (
        patch.object(settings, "jwt_algorithm", "RS256"),
        patch.object(settings, "jwt_secret", None),
        patch.object(settings, "jwt_issuer", None),
        patch("utils.jwks_cache.get_signing_key", return_value=public_pem),
    ):
        ctx, decoded = verify_token(token)

    assert ctx.user_id == "alice"
    assert ctx.org_id == "acme"
    assert decoded["jti"] == "test-jti-123"


def test_rs256_missing_kid_raises_bad_claims():
    """RS256 token without kid header is rejected."""
    token = _make_rs256_token({"sub": "alice", "iat": 1, "exp": 9999999999}, kid=None)
    with (
        patch.object(settings, "jwt_algorithm", "RS256"),
        patch.object(settings, "jwt_secret", None),
        pytest.raises(AuthError) as exc,
    ):
        verify_token(token)
    assert exc.value.reason == "bad_claims"


def test_rs256_jwks_fetch_failure_raises_bad_signature():
    """When JWKS lookup fails, the error maps to bad_signature."""
    token = _make_rs256_token({"sub": "alice", "iat": 1, "exp": 9999999999}, kid="unknown-kid")
    with (
        patch.object(settings, "jwt_algorithm", "RS256"),
        patch.object(settings, "jwt_secret", None),
        patch("utils.jwks_cache.get_signing_key", side_effect=RuntimeError("network down")),
        pytest.raises(AuthError) as exc,
    ):
        verify_token(token)
    assert exc.value.reason == "bad_signature"


def test_rs256_expired_token_rejected():
    """Expired RS256 token is rejected even with valid JWKS."""
    from jose import jwt as jose_jwt

    private_pem, public_pem = _generate_rsa_key_pair()

    claims = {
        "sub": "alice",
        "user_id": "alice",
        "org_id": "acme",
        "roles": ["user"],
        "clearance_level": 1,
        "iat": 1609459200,
        "exp": 1609459201,  # expired
        "jti": "expired-jti",
    }
    token = jose_jwt.encode(claims, private_pem, algorithm="RS256", headers={"kid": "k1"})

    with (
        patch.object(settings, "jwt_algorithm", "RS256"),
        patch.object(settings, "jwt_secret", None),
        patch("utils.jwks_cache.get_signing_key", return_value=public_pem),
        pytest.raises(AuthError) as exc,
    ):
        verify_token(token)
    assert exc.value.reason == "expired"


def test_issue_token_raises_in_rs256_mode():
    """Local token issuance is blocked when RS256 is configured."""
    with (
        patch.object(settings, "jwt_algorithm", "RS256"),
        patch.object(settings, "jwt_secret", SECRET),
        pytest.raises(AuthError) as exc,
    ):
        issue_token(user_id="alice", org_id="acme", roles=["user"])
    assert exc.value.reason == "missing"
