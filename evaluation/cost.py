"""Cost estimator — convert per-query provider usage into USD.

Reads ``synth_provider`` / ``synth_model`` / ``synth_usage`` /
``synth_latency_ms`` from a final ``GraphState`` and returns the estimated
inference cost. Local inference is priced as electricity (kWh x $/kWh)
rather than zero so the dashboard can show the true compute baseline.

Prices live in ``config/settings.py`` and are USD per 1M tokens.
"""

from __future__ import annotations

from typing import Any

from config.settings import settings


def _provider_prices(provider: str) -> tuple[float, float] | None:
    """Return (input_per_1m, output_per_1m) for known cloud providers."""
    table = {
        "groq": (settings.price_groq_input_per_1m, settings.price_groq_output_per_1m),
        "openai": (settings.price_openai_input_per_1m, settings.price_openai_output_per_1m),
        "anthropic": (
            settings.price_anthropic_input_per_1m,
            settings.price_anthropic_output_per_1m,
        ),
    }
    return table.get(provider.lower())


def estimate_query_cost(
    provider: str,
    usage: dict[str, Any],
    latency_ms: float,
) -> dict[str, Any]:
    """Estimate the cost in USD for a single synthesizer call.

    Args:
        provider: ``"ollama"`` / ``"groq"`` / ``"openai"`` / ``"anthropic"``.
        usage: Dict with optional ``prompt_tokens`` / ``completion_tokens``
            / ``total_tokens``. Missing keys default to 0.
        latency_ms: Wall-clock latency of the call (used only for local
            electricity estimate).

    Returns:
        Dict ``{cost_usd, input_tokens, output_tokens, mode, provider}``.
    """
    p = provider.lower() if provider else ""
    prices = _provider_prices(p)

    in_tok = int(usage.get("prompt_tokens", 0) or 0)
    out_tok = int(usage.get("completion_tokens", 0) or 0)
    if not in_tok and not out_tok:
        total = int(usage.get("total_tokens", 0) or 0)
        # Best-effort 25/75 split when only total is reported.
        in_tok = int(total * 0.25)
        out_tok = total - in_tok

    if prices is None:
        # Local — pay only electricity, proportional to wall-clock seconds.
        seconds = max(latency_ms, 0.0) / 1000.0
        cost = seconds * settings.price_local_per_second
        mode = "local"
    else:
        cost = (in_tok / 1_000_000) * prices[0] + (out_tok / 1_000_000) * prices[1]
        mode = "cloud"

    return {
        "cost_usd": round(cost, 6),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "mode": mode,
        "provider": p or "unknown",
    }


def estimate_state_cost(state: dict[str, Any]) -> dict[str, Any]:
    """Estimate cost from a final ``GraphState`` dict."""
    return estimate_query_cost(
        provider=state.get("synth_provider", ""),
        usage=state.get("synth_usage", {}),
        latency_ms=state.get("synth_latency_ms", 0.0),
    )
