"""Tests for the per-IP hourly owner-key fallback throttle (BYOK mode).

See ``launch-plan/03-backend-byok.md`` § OwnerKeyThrottle and
``launch-plan/11-security-checklist.md`` § Per-IP throttle.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from utils.rate_limiter import (
    OwnerKeyHourThrottle,
    get_owner_key_throttle,
    reset_owner_key_throttle,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Each test starts with a fresh singleton."""
    reset_owner_key_throttle()
    yield
    reset_owner_key_throttle()


def test_first_three_requests_allowed_then_429() -> None:
    """Default quota=3 — fourth request from same IP returns 429."""
    t = OwnerKeyHourThrottle(quota_per_hour=3)
    allowed1, _ = t.allow("1.2.3.4", now=0.0)
    allowed2, _ = t.allow("1.2.3.4", now=1.0)
    allowed3, _ = t.allow("1.2.3.4", now=2.0)
    allowed4, meta4 = t.allow("1.2.3.4", now=3.0)
    assert allowed1 is True
    assert allowed2 is True
    assert allowed3 is True
    assert allowed4 is False
    assert meta4["reason"] == "owner_key_hourly_quota_exhausted"
    assert meta4["retry_after"] > 0
    assert meta4["remaining"] == 0


def test_different_ips_get_independent_quotas() -> None:
    """Throttle must not leak quota across visitors."""
    t = OwnerKeyHourThrottle(quota_per_hour=3)
    for _ in range(3):
        assert t.allow("1.2.3.4", now=0.0)[0] is True
    # Same wall-clock instant, different IP — quota intact.
    allowed, _ = t.allow("5.6.7.8", now=0.0)
    assert allowed is True


def test_quota_resets_after_hour() -> None:
    """Old timestamps expire — bucket re-opens once they age past 3600s."""
    t = OwnerKeyHourThrottle(quota_per_hour=3)
    for _ in range(3):
        t.allow("1.2.3.4", now=0.0)
    assert t.allow("1.2.3.4", now=100.0)[0] is False
    # Just past one-hour boundary.
    assert t.allow("1.2.3.4", now=3601.0)[0] is True


def test_retry_after_decreases_as_window_advances() -> None:
    """``retry_after`` shrinks toward zero as the oldest entry ages out."""
    t = OwnerKeyHourThrottle(quota_per_hour=1)
    t.allow("1.2.3.4", now=0.0)
    _, meta_early = t.allow("1.2.3.4", now=10.0)
    _, meta_late = t.allow("1.2.3.4", now=3590.0)
    assert meta_early["retry_after"] > meta_late["retry_after"]


def test_zero_quota_blocks_everything() -> None:
    """Quota=0 is the kill switch for owner-key fallback."""
    t = OwnerKeyHourThrottle(quota_per_hour=0)
    allowed, _ = t.allow("1.2.3.4", now=0.0)
    assert allowed is False


def test_negative_quota_rejected() -> None:
    """Misconfiguration must surface loudly, not silently allow all."""
    with pytest.raises(ValueError):
        OwnerKeyHourThrottle(quota_per_hour=-1)


def test_reset_clears_specific_ip() -> None:
    t = OwnerKeyHourThrottle(quota_per_hour=2)
    t.allow("1.2.3.4", now=0.0)
    t.allow("1.2.3.4", now=1.0)
    assert t.allow("1.2.3.4", now=2.0)[0] is False  # over quota
    t.reset("1.2.3.4")
    assert t.allow("1.2.3.4", now=3.0)[0] is True


def test_reset_all_clears_every_bucket() -> None:
    t = OwnerKeyHourThrottle(quota_per_hour=1)
    t.allow("1.2.3.4", now=0.0)
    t.allow("5.6.7.8", now=0.0)
    t.reset_all()
    assert t.allow("1.2.3.4", now=1.0)[0] is True
    assert t.allow("5.6.7.8", now=1.0)[0] is True


def test_singleton_reads_settings_lazily() -> None:
    """Quota change after import must take effect via ``reset_owner_key_throttle``."""
    from config.settings import settings

    with patch.object(settings, "byok_owner_key_quota_per_hour", 5):
        reset_owner_key_throttle()
        throttle = get_owner_key_throttle()
        assert throttle._quota_per_hour == 5


def test_singleton_is_stable_between_calls() -> None:
    """Same instance returned across calls — needed so quotas accumulate."""
    a = get_owner_key_throttle()
    b = get_owner_key_throttle()
    assert a is b


def test_anonymous_ip_still_throttled() -> None:
    """When the request has no client.host we use 'anon' as the key.

    The throttle must NOT special-case 'anon' to bypass — otherwise an
    attacker behind a proxy stripping X-Forwarded-For would burn unlimited
    owner-key quota.
    """
    t = OwnerKeyHourThrottle(quota_per_hour=2)
    assert t.allow("anon", now=0.0)[0] is True
    assert t.allow("anon", now=1.0)[0] is True
    assert t.allow("anon", now=2.0)[0] is False
