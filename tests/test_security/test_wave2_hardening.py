"""Regression tests for the wave-2 review hardening.

Covers: schema input bounds (clearance cap, path-traversal guard), the
guardrails gate failing closed, and issue_token refusing to let extra claims
overwrite reserved/registered claims.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from config.settings import settings
from core.agents.guardrails import guardrails_gate
from core.schemas import IngestRequestModel, QueryRequest


class TestSchemaBounds:
    def test_clearance_level_is_bounded_1_to_3(self) -> None:
        QueryRequest(query="q", user_id="u", clearance_level=3)  # ok
        with pytest.raises(ValidationError):
            QueryRequest(query="q", user_id="u", clearance_level=99)
        with pytest.raises(ValidationError):
            QueryRequest(query="q", user_id="u", clearance_level=0)

    def test_ingest_file_path_blocks_traversal_and_null(self) -> None:
        IngestRequestModel(file_path="docs/report.pdf", user_id="u")  # ok
        with pytest.raises(ValidationError):
            IngestRequestModel(file_path="../../etc/passwd", user_id="u")
        with pytest.raises(ValidationError):
            IngestRequestModel(file_path="ok\x00.pdf", user_id="u")
        with pytest.raises(ValidationError):
            IngestRequestModel(file_path="   ", user_id="u")


class TestGuardrailsGateFailsClosed:
    def test_missing_flag_blocks(self) -> None:
        assert guardrails_gate({}) == "blocked"

    def test_explicit_flags(self) -> None:
        assert guardrails_gate({"guardrails_passed": True}) == "proceed"
        assert guardrails_gate({"guardrails_passed": False}) == "blocked"


class TestIssueTokenReservedClaims:
    def test_extra_claims_cannot_overwrite_reserved(self) -> None:
        from utils.auth import issue_token, verify_token

        with patch.object(settings, "jwt_secret", "unit-test-secret-please-ignore"):
            # exp=0 would make the token instantly expired IF it were honoured;
            # sub="attacker" would spoof identity. Neither must take effect.
            token = issue_token(
                "u1",
                "o1",
                ["viewer"],
                extra_claims={"exp": 0, "sub": "attacker", "jti": "x", "team": "blue"},
            )
            ctx, claims = verify_token(token)

        assert ctx.user_id == "u1"
        assert claims["sub"] == "u1"  # not "attacker"
        assert claims["exp"] != 0  # real expiry kept
        assert claims.get("team") == "blue"  # non-reserved claim preserved
