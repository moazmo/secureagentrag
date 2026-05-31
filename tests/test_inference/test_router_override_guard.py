"""C1 regression: override_provider must never move HIGH off local (self-hosted).

The router honours an admin ``override_provider`` for LOW/MEDIUM, but on a
self-hosted deploy (``allow_cloud_for_high`` off) it must NOT let an override
route HIGH-sensitivity content to a cloud provider — that would bypass the
"HIGH stays local" privacy guarantee. (override_provider is not currently wired
into the pipeline; this guard is defence-in-depth.)
"""

from __future__ import annotations

from unittest.mock import patch

from config.settings import settings
from inference.router import InferenceRouter


def test_override_cannot_move_high_off_local_when_selfhosted() -> None:
    r = InferenceRouter(cloud_provider="groq", force_local_for_sensitive=True)
    with patch.object(settings, "allow_cloud_for_high", False):
        decision = r.route(sensitivity_level="high", override_provider="openai")
    assert decision.provider == "ollama"
    assert decision.forced_local is True


def test_override_to_ollama_for_high_is_fine() -> None:
    r = InferenceRouter(cloud_provider="groq", force_local_for_sensitive=True)
    with patch.object(settings, "allow_cloud_for_high", False):
        decision = r.route(sensitivity_level="high", override_provider="ollama")
    assert decision.provider == "ollama"
    assert decision.forced_local is True


def test_override_honoured_for_low_sensitivity() -> None:
    r = InferenceRouter(cloud_provider="groq", force_local_for_sensitive=True)
    with patch.object(settings, "allow_cloud_for_high", False):
        decision = r.route(sensitivity_level="low", override_provider="openai")
    assert decision.provider == "openai"
    assert decision.forced_local is False


def test_override_high_allowed_when_cloud_for_high_enabled() -> None:
    """The GPU-less public demo opts into cloud-for-HIGH; override is then OK."""
    r = InferenceRouter(cloud_provider="groq", force_local_for_sensitive=True)
    with patch.object(settings, "allow_cloud_for_high", True):
        decision = r.route(sensitivity_level="high", override_provider="openai")
    assert decision.provider == "openai"
