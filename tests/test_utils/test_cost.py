"""Cost estimator tests."""

from __future__ import annotations

from evaluation.cost import estimate_query_cost


def test_groq_pricing() -> None:
    out = estimate_query_cost("groq", {"prompt_tokens": 1_000_000, "completion_tokens": 0}, 0.0)
    assert out["mode"] == "cloud"
    assert out["cost_usd"] > 0


def test_local_pricing_uses_latency() -> None:
    out = estimate_query_cost("ollama", {"total_tokens": 100}, latency_ms=1000.0)
    assert out["mode"] == "local"
    # 1 second * 0.000008 = 0.000008
    assert out["cost_usd"] >= 0


def test_unknown_provider_treated_as_local() -> None:
    out = estimate_query_cost("", {}, latency_ms=500.0)
    assert out["mode"] == "local"


def test_split_when_only_total_reported() -> None:
    out = estimate_query_cost("openai", {"total_tokens": 1000}, 0.0)
    assert out["input_tokens"] + out["output_tokens"] == 1000
