"""Tests for the scheduled audit-chain verification helper."""

from __future__ import annotations

from unittest.mock import patch

from config.settings import settings
from utils import audit_verify


class _FakeAudit:
    def __init__(self, result):
        self._result = result
        self.calls = 0

    def verify_chain(self):
        self.calls += 1
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def test_run_verification_valid_chain():
    fake = _FakeAudit({"valid": True, "checked": 5, "broken_at": [], "last_hash": "abc"})
    result = audit_verify.run_audit_verification(fake)
    assert result["valid"] is True
    assert fake.calls == 1


def test_run_verification_broken_chain_does_not_raise():
    fake = _FakeAudit(
        {
            "valid": False,
            "checked": 2,
            "broken_at": ["audit_x.jsonl:3:hash_mismatch"],
            "last_hash": "",
        }
    )
    result = audit_verify.run_audit_verification(fake)
    assert result["valid"] is False
    assert result["broken_at"]


def test_run_verification_swallows_exceptions():
    fake = _FakeAudit(RuntimeError("disk gone"))
    result = audit_verify.run_audit_verification(fake)
    assert result["valid"] is False
    assert any("error" in b for b in result["broken_at"])


def test_schedule_disabled_returns_none():
    with patch.object(settings, "audit_verify_enabled", False):
        assert audit_verify.schedule_audit_verification() is None


def test_metric_recorded_on_valid_run():
    import pytest

    pytest.importorskip("prometheus_client")
    from prometheus_client import REGISTRY

    before = (
        REGISTRY.get_sample_value("audit_chain_verifications_total", {"result": "valid"}) or 0.0
    )
    audit_verify.run_audit_verification(
        _FakeAudit({"valid": True, "checked": 1, "broken_at": [], "last_hash": "z"})
    )
    after = REGISTRY.get_sample_value("audit_chain_verifications_total", {"result": "valid"})
    assert after == before + 1
    assert REGISTRY.get_sample_value("audit_chain_valid") == 1
