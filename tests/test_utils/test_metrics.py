"""Unit tests for utils.metrics — custom RAG Prometheus metrics.

Skipped wholesale when the ``[metrics]`` extra (prometheus_client) is not
installed, since the module degrades to no-ops in that case.
"""

from __future__ import annotations

import pytest

pytest.importorskip("prometheus_client")

from prometheus_client import REGISTRY

from utils import metrics


def _sample(name: str, labels: dict[str, str] | None = None) -> float:
    """Return the current registry value of a sample, or 0.0 if absent."""
    value = REGISTRY.get_sample_value(name, labels or {})
    return value if value is not None else 0.0


def test_metrics_enabled_with_extra():
    assert metrics.METRICS_ENABLED is True


def test_success_run_records_outcome_provider_latency():
    before_req = _sample("rag_pipeline_requests_total", {"outcome": "success"})
    before_provider = _sample("inference_routed_by_provider_total", {"provider": "groq"})
    before_count = _sample("rag_pipeline_latency_seconds_count", {"outcome": "success"})

    metrics.record_pipeline_run(
        {
            "guardrails_passed": True,
            "security_passed": True,
            "needs_human_review": False,
            "synth_provider": "groq",
            "faithfulness_unsupported": [],
        },
        latency_ms=1234.0,
    )

    assert _sample("rag_pipeline_requests_total", {"outcome": "success"}) == before_req + 1
    assert (
        _sample("inference_routed_by_provider_total", {"provider": "groq"}) == before_provider + 1
    )
    assert _sample("rag_pipeline_latency_seconds_count", {"outcome": "success"}) == before_count + 1


def test_blocked_run_records_guardrails_reason():
    before = _sample(
        "guardrails_blocked_total",
        {"gate": "guardrails", "reason": "prompt_injection"},
    )
    metrics.record_pipeline_run(
        {
            "guardrails_passed": False,
            "guardrails_reason": "prompt_injection",
            "security_passed": True,
        },
        latency_ms=12.0,
    )
    assert (
        _sample(
            "guardrails_blocked_total",
            {"gate": "guardrails", "reason": "prompt_injection"},
        )
        == before + 1
    )


def test_unknown_reason_collapses_to_other():
    before = _sample("guardrails_blocked_total", {"gate": "guardrails", "reason": "other"})
    metrics.record_pipeline_run(
        {
            "guardrails_passed": False,
            "guardrails_reason": "some weird attacker-controlled string!!!",
            "security_passed": True,
        },
        latency_ms=5.0,
    )
    assert (
        _sample("guardrails_blocked_total", {"gate": "guardrails", "reason": "other"}) == before + 1
    )


def test_faithfulness_dropped_increments_by_count():
    before = _sample("faithfulness_dropped_total")
    metrics.record_pipeline_run(
        {
            "guardrails_passed": True,
            "security_passed": True,
            "synth_provider": "ollama",
            "faithfulness_unsupported": ["s1", "s2", "s3"],
        },
        latency_ms=900.0,
    )
    assert _sample("faithfulness_dropped_total") == before + 3


def test_timeout_outcome_label():
    before = _sample("rag_pipeline_requests_total", {"outcome": "timeout"})
    metrics.record_pipeline_run(
        {
            "guardrails_passed": True,
            "security_passed": True,
            "evaluation_notes": "request_timeout",
            "needs_human_review": True,
        },
        latency_ms=180000.0,
    )
    assert _sample("rag_pipeline_requests_total", {"outcome": "timeout"}) == before + 1


def test_render_latest_returns_exposition():
    payload, content_type = metrics.render_latest()
    assert isinstance(payload, bytes)
    assert b"rag_pipeline_requests_total" in payload
    assert "text/plain" in content_type
