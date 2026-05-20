"""Tests for the HS256 JWT auth layer."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from config.settings import settings
from utils.auth import AuthError, issue_token, verify_token

# Skip the whole module if python-jose isn't installed in this env.
pytest.importorskip("jose")


SECRET = "test-secret-do-not-use-in-prod"


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
    assert claims["jti"]  # uuid was set
    assert claims["sub"] == "alice"
    assert "iat" in claims and "exp" in claims


def test_expired_token_rejected():
    """An already-expired token (negative TTL bakes exp into the past)
    must be rejected with reason=expired without us sleeping."""
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
    # Tamper the last 5 chars of the signature segment.
    parts = token.split(".")
    parts[-1] = parts[-1][:-5] + "AAAAA"
    tampered = ".".join(parts)
    with patch.object(settings, "jwt_secret", SECRET), pytest.raises(AuthError) as exc:
        verify_token(tampered)
    assert exc.value.reason == "bad_signature"


def test_wrong_secret_rejected():
    # Issuer uses one secret, verifier uses a different one.
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

    # Verifier expects a different audience claim.
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
    """Backwards-compat path: with SAR_JWT_SECRET unset the verifier accepts
    base64(json(UserContext)). A runtime warning is logged."""
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
