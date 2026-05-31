"""SSRF regression tests for the BYOK Ollama URL validator (H2).

A visitor-supplied ``X-User-Ollama-URL`` is attacker-controlled. Without
validation the backend could be coerced into fetching internal targets
(cloud metadata, the app's own Qdrant, RFC-1918 hosts). These tests pin the
allow/deny behaviour.
"""

from __future__ import annotations

import pytest

from inference.ollama_client import _assert_safe_ollama_url, make_byok_ollama_client


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://localhost:6333",  # the app's own Qdrant
        "http://127.0.0.1:11434",  # loopback
        "https://10.0.0.5",  # RFC-1918
        "http://192.168.1.10:11434",  # RFC-1918
        "http://172.16.0.1",  # RFC-1918
        "http://[::1]:11434",  # IPv6 loopback
        "ftp://example.com",  # wrong scheme
        "http://foo.internal",  # internal hostname
        "http://db.local",  # mDNS/internal
        "http://0.0.0.0:11434",  # unspecified
    ],
)
def test_blocks_internal_and_bad_scheme(url: str) -> None:
    with pytest.raises(ValueError):
        _assert_safe_ollama_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://ollama.example.com",
        "http://8.8.8.8:11434",  # public IP literal
        "https://my-ollama.fly.dev/api",
    ],
)
def test_allows_public_targets(url: str) -> None:
    _assert_safe_ollama_url(url)  # must not raise


def test_make_byok_rejects_metadata_ip() -> None:
    with pytest.raises(ValueError):
        make_byok_ollama_client(base_url="http://169.254.169.254")


def test_make_byok_allows_public_host() -> None:
    client = make_byok_ollama_client(base_url="https://ollama.example.com")
    assert client is not None
