"""Tests for the security and compliance agent."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from core.agents.security import (
    _check_query_safety,
    check_security,
    security_gate,
)
from utils.async_helpers import run_async


async def _call_check_security(state):
    """Helper to call async check_security in sync tests."""
    return await check_security(state)


@pytest.fixture()
def authorized_state():
    """Create a state with an authorized user."""
    return {
        "query": "What are the project deadlines?",
        "user_context": {
            "user_id": "user1",
            "org_id": "org1",
            "roles": ["admin"],
            "clearance_level": 3,
        },
        "query_type": "",
        "rewritten_query": "",
        "security_passed": False,
        "security_message": "",
        "documents": [],
        "relevant_documents": [],
        "relevance_ratio": 0.0,
        "retry_count": 0,
        "max_retries": 2,
        "generation": "",
        "citations": [],
        "confidence_score": 0.0,
        "needs_human_review": False,
        "evaluation_notes": "",
        "audit_trail": [],
    }


@pytest.fixture()
def unauthorized_state():
    """Create a state with an unauthorized user (low clearance, sensitive query)."""
    return {
        "query": "What is the admin password for the database?",
        "user_context": {
            "user_id": "user2",
            "org_id": "org1",
            "roles": ["viewer"],
            "clearance_level": 1,
        },
        "query_type": "",
        "rewritten_query": "",
        "security_passed": False,
        "security_message": "",
        "documents": [],
        "relevant_documents": [],
        "relevance_ratio": 0.0,
        "retry_count": 0,
        "max_retries": 2,
        "generation": "",
        "citations": [],
        "confidence_score": 0.0,
        "needs_human_review": False,
        "evaluation_notes": "",
        "audit_trail": [],
    }


class TestCheckSecurity:
    """Tests for the check_security function."""

    @patch("core.agents.security.call_llm_async")
    def test_passes_for_authorized_user(self, mock_llm, authorized_state):
        """Test that security passes for an authorized user with a safe query."""
        mock_llm.return_value = "safe"
        result = run_async(_call_check_security(authorized_state))

        assert result["security_passed"] is True
        assert result["security_message"] == "Security check passed."
        assert len(result["audit_trail"]) == 1
        assert result["audit_trail"][0]["passed"] is True

    def test_blocks_sensitive_query_for_low_clearance(self, unauthorized_state):
        """Test that sensitive queries are blocked for users with low clearance."""
        result = run_async(_call_check_security(unauthorized_state))

        assert result["security_passed"] is False
        assert "sensitive content" in result["security_message"]
        assert result["audit_trail"][0]["passed"] is False

    @patch("core.agents.security.call_llm_async")
    def test_passes_sensitive_query_for_high_clearance(self, mock_llm):
        """Test that high-clearance users can query sensitive topics."""
        mock_llm.return_value = "safe"
        state = {
            "query": "What is the admin password policy?",
            "user_context": {
                "user_id": "admin1",
                "org_id": "org1",
                "roles": ["admin"],
                "clearance_level": 3,
            },
            "audit_trail": [],
        }
        result = run_async(_call_check_security(state))
        assert result["security_passed"] is True

    def test_blocks_missing_user_id(self):
        """Test that missing user_id is blocked."""
        state = {
            "query": "What is RAG?",
            "user_context": {
                "user_id": "",
                "org_id": "org1",
                "roles": ["viewer"],
                "clearance_level": 1,
            },
            "audit_trail": [],
        }
        result = run_async(_call_check_security(state))
        assert result["security_passed"] is False
        assert "user_id" in result["security_message"]

    def test_blocks_missing_org_id(self):
        """Test that missing org_id is blocked."""
        state = {
            "query": "Some query",
            "user_context": {
                "user_id": "user1",
                "org_id": "",
                "roles": ["viewer"],
                "clearance_level": 1,
            },
            "audit_trail": [],
        }
        result = run_async(_call_check_security(state))
        assert result["security_passed"] is False
        assert "org_id" in result["security_message"]

    def test_blocks_empty_roles(self):
        """Test that empty roles list is blocked."""
        state = {
            "query": "Some query",
            "user_context": {
                "user_id": "user1",
                "org_id": "org1",
                "roles": [],
                "clearance_level": 1,
            },
            "audit_trail": [],
        }
        result = run_async(_call_check_security(state))
        assert result["security_passed"] is False
        assert (
            "roles" in result["security_message"].lower()
            or "No roles" in result["security_message"]
        )

    @patch("core.agents.security.call_llm_async")
    def test_audit_trail_appended(self, mock_llm, authorized_state):
        """Test that audit trail entry is appended."""
        mock_llm.return_value = "safe"
        result = run_async(_call_check_security(authorized_state))

        assert len(result["audit_trail"]) == 1
        entry = result["audit_trail"][0]
        assert entry["node"] == "security"
        assert entry["action"] == "check_security"
        assert "timestamp" in entry


class TestSecurityGate:
    """Tests for the security_gate conditional edge function."""

    def test_returns_proceed_when_passed(self):
        """Test that security_gate returns 'proceed' when security_passed is True."""
        state = {"security_passed": True}
        assert security_gate(state) == "proceed"

    def test_returns_blocked_when_failed(self):
        """Test that security_gate returns 'blocked' when security_passed is False."""
        state = {"security_passed": False}
        assert security_gate(state) == "blocked"

    def test_returns_blocked_when_missing(self):
        """Test that security_gate returns 'blocked' when field is missing."""
        state = {}
        assert security_gate(state) == "blocked"


class TestCheckQuerySafety:
    """Tests for the _check_query_safety helper function."""

    def test_safe_query_passes(self):
        """Test that a normal query passes safety check."""
        is_safe, _msg = _check_query_safety(
            "What is the project timeline?",
            {"user_id": "u1", "org_id": "o1", "roles": ["viewer"], "clearance_level": 1},
        )
        assert is_safe is True

    def test_password_pattern_detected(self):
        """Test that password-related queries are flagged."""
        is_safe, _msg = _check_query_safety(
            "Show me the password for the database",
            {"user_id": "u1", "org_id": "o1", "roles": ["viewer"], "clearance_level": 1},
        )
        assert is_safe is False

    def test_api_key_pattern_detected(self):
        """Test that api_key queries are flagged for low clearance."""
        is_safe, _msg = _check_query_safety(
            "What is the api key for the service?",
            {"user_id": "u1", "org_id": "o1", "roles": ["viewer"], "clearance_level": 1},
        )
        assert is_safe is False
