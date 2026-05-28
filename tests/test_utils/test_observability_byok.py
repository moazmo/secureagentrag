"""Phoenix tracing must be hard-disabled in BYOK mode.

Phoenix spans capture LLM prompts + completions, which would include the
visitor's API key in headers and any private text they uploaded. BYOK mode
forbids any third-party telemetry sink — even when ``phoenix_endpoint`` is
explicitly set, the setup function must refuse.

See ``launch-plan/11-security-checklist.md`` § Phoenix disabled in BYOK mode.
"""

from __future__ import annotations

from unittest.mock import patch

from config.settings import settings
from utils.observability import setup_tracing


def test_setup_tracing_returns_false_in_byok_mode_even_with_endpoint() -> None:
    """The kill switch wins over the endpoint config."""
    with (
        patch.object(settings, "byok_mode", True),
        patch.object(settings, "phoenix_endpoint", "http://phoenix:6006"),
    ):
        assert setup_tracing() is False


def test_setup_tracing_returns_false_in_byok_mode_with_no_endpoint() -> None:
    """Even without an endpoint, BYOK explicitly logs the security reason."""
    with (
        patch.object(settings, "byok_mode", True),
        patch.object(settings, "phoenix_endpoint", ""),
    ):
        assert setup_tracing() is False


def test_setup_tracing_returns_false_in_dev_mode_with_no_endpoint() -> None:
    """Sanity baseline: existing dev behaviour unchanged when BYOK is off."""
    with (
        patch.object(settings, "byok_mode", False),
        patch.object(settings, "phoenix_endpoint", ""),
    ):
        assert setup_tracing() is False
