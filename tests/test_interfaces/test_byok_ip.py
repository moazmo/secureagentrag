"""Tests for ``client_ip_from_request`` X-Forwarded-For resolution.

XFF is a client-appendable header, so the leftmost token is spoofable. These
tests pin both the legacy leftmost behaviour (hops=0) and the spoof-resistant
trusted-hops resolution that picks the address the innermost trusted proxy saw.
"""

from __future__ import annotations

from unittest.mock import patch

from config.settings import settings
from interfaces.byok import client_ip_from_request


class _FakeClient:
    def __init__(self, host: str | None) -> None:
        self.host = host


class _FakeRequest:
    """Minimal stand-in for starlette.Request (headers + client only)."""

    def __init__(self, headers: dict[str, str], host: str | None = None) -> None:
        self.headers = headers
        self.client = _FakeClient(host) if host is not None else None


def test_xff_leftmost_by_default() -> None:
    req = _FakeRequest({"x-forwarded-for": "1.1.1.1, 2.2.2.2, 3.3.3.3"})
    with patch.object(settings, "byok_xff_trusted_hops", 0):
        assert client_ip_from_request(req) == "1.1.1.1"


def test_xff_one_trusted_hop_picks_second_from_right() -> None:
    req = _FakeRequest({"x-forwarded-for": "1.1.1.1, 2.2.2.2, 3.3.3.3"})
    with patch.object(settings, "byok_xff_trusted_hops", 1):
        # Innermost trusted proxy (the one that appended 3.3.3.3) saw 2.2.2.2.
        assert client_ip_from_request(req) == "2.2.2.2"


def test_xff_two_trusted_hops() -> None:
    req = _FakeRequest({"x-forwarded-for": "1.1.1.1, 2.2.2.2, 3.3.3.3"})
    with patch.object(settings, "byok_xff_trusted_hops", 2):
        assert client_ip_from_request(req) == "1.1.1.1"


def test_xff_hops_longer_than_chain_falls_back_to_leftmost() -> None:
    req = _FakeRequest({"x-forwarded-for": "9.9.9.9"})
    with patch.object(settings, "byok_xff_trusted_hops", 3):
        assert client_ip_from_request(req) == "9.9.9.9"


def test_falls_back_to_real_ip_then_client_then_anon() -> None:
    with patch.object(settings, "byok_xff_trusted_hops", 0):
        assert client_ip_from_request(_FakeRequest({"x-real-ip": "8.8.8.8"})) == "8.8.8.8"
        assert client_ip_from_request(_FakeRequest({}, host="7.7.7.7")) == "7.7.7.7"
        assert client_ip_from_request(_FakeRequest({})) == "anon"
