"""Prometheus metrics for the RAG pipeline.

Self-hosted observability. Unlike Phoenix tracing (which captures prompts and
completions and is therefore *hard-disabled* under BYOK), Prometheus metrics
are aggregate counters and histograms — no request content, no keys, no user
text ever lands in a label. Only low-cardinality categorical dimensions
(provider name, gate verdict, outcome) are recorded, so they are safe to emit
even on the public BYOK demo.

Graceful degradation: if ``prometheus_client`` is not installed the module
exposes no-op stubs and ``METRICS_ENABLED`` is ``False``. Nothing in the hot
path raises — a missing dependency just means ``/metrics`` returns 501.

The custom RAG metrics live on the global ``prometheus_client.REGISTRY`` so
they share the same ``/metrics`` exposition as the HTTP-level metrics emitted
by ``prometheus-fastapi-instrumentator`` (which also defaults to that
registry).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from utils.logging import get_logger

if TYPE_CHECKING:
    from core.state import GraphState

_log = get_logger(__name__)

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    METRICS_ENABLED = True
except ImportError:  # pragma: no cover - exercised only without the extra
    METRICS_ENABLED = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    Counter = Gauge = Histogram = None  # type: ignore[assignment]
    generate_latest = None  # type: ignore[assignment]


# ── Metric definitions ───────────────────────────────────────────────────
# Buckets tuned for the RAG SLO: sub-second cache-warm answers up to the
# 180 s wall-clock budget (SAR_REQUEST_TIMEOUT_S). The wide tail captures
# Groq free-tier rate-limit waits + reranker cold loads without smearing the
# fast path across too few buckets.
_LATENCY_BUCKETS = (0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 180.0)

if METRICS_ENABLED:
    PIPELINE_LATENCY = Histogram(
        "rag_pipeline_latency_seconds",
        "End-to-end RAG pipeline wall-clock latency, by terminal outcome.",
        labelnames=("outcome",),
        buckets=_LATENCY_BUCKETS,
    )
    PIPELINE_REQUESTS = Counter(
        "rag_pipeline_requests_total",
        "RAG pipeline invocations, by terminal outcome.",
        labelnames=("outcome",),
    )
    GUARDRAILS_BLOCKED = Counter(
        "guardrails_blocked_total",
        "Requests blocked at a safety gate, by gate and reason category.",
        labelnames=("gate", "reason"),
    )
    INFERENCE_ROUTED = Counter(
        "inference_routed_by_provider_total",
        "Synthesis inference calls routed, by provider.",
        labelnames=("provider",),
    )
    FAITHFULNESS_DROPPED = Counter(
        "faithfulness_dropped_total",
        "Total sentences dropped/flagged by the NLI faithfulness gate.",
    )
    AUDIT_VERIFICATIONS = Counter(
        "audit_chain_verifications_total",
        "Scheduled audit hash-chain verification runs, by result.",
        labelnames=("result",),
    )
    AUDIT_CHAIN_VALID = Gauge(
        "audit_chain_valid",
        "1 if the last audit-chain verification passed, 0 if tamper/error detected.",
    )
else:  # pragma: no cover
    PIPELINE_LATENCY = None
    PIPELINE_REQUESTS = None
    GUARDRAILS_BLOCKED = None
    INFERENCE_ROUTED = None
    FAITHFULNESS_DROPPED = None
    AUDIT_VERIFICATIONS = None
    AUDIT_CHAIN_VALID = None


# ── Recording helpers ──────────────────────────────────────────────────────
# Each helper is a hard no-op when the extra is absent, and every increment is
# wrapped so a metrics failure can never break a request. Labels are clamped
# to a known vocabulary to keep cardinality bounded.

_KNOWN_PROVIDERS = {"ollama", "groq", "openai", "anthropic"}


def _outcome_from_state(state: GraphState | dict) -> str:
    """Derive a terminal-outcome label from a final pipeline state.

    One of: ``blocked`` (a safety gate stopped the request), ``timeout``
    (wall-clock budget exceeded), ``review`` (answered but flagged for human
    review), or ``success``.
    """
    if state.get("guardrails_passed") is False or state.get("security_passed") is False:
        return "blocked"
    if state.get("evaluation_notes") == "request_timeout":
        return "timeout"
    if state.get("needs_human_review"):
        return "review"
    return "success"


def record_pipeline_run(state: GraphState | dict, latency_ms: float) -> None:
    """Record latency, outcome, provider, and faithfulness drops for one run.

    Safe to call unconditionally at the end of every pipeline invocation
    (streaming and non-streaming). No-op without the ``prometheus_client``
    extra; never raises.
    """
    if not METRICS_ENABLED:
        return
    try:
        outcome = _outcome_from_state(state)
        PIPELINE_LATENCY.labels(outcome=outcome).observe(latency_ms / 1000.0)
        PIPELINE_REQUESTS.labels(outcome=outcome).inc()

        provider = (state.get("synth_provider") or "").lower()
        if provider:
            label = provider if provider in _KNOWN_PROVIDERS else "other"
            INFERENCE_ROUTED.labels(provider=label).inc()

        dropped = len(state.get("faithfulness_unsupported") or [])
        if dropped:
            FAITHFULNESS_DROPPED.inc(dropped)

        if outcome == "blocked":
            if state.get("guardrails_passed") is False:
                _record_block("guardrails", state.get("guardrails_reason"))
            elif state.get("security_passed") is False:
                _record_block("security", state.get("security_message"))
    except Exception as exc:  # pragma: no cover - defensive
        _log.debug("metrics_record_failed", error=str(exc))


# Bounded vocabulary for the guardrails reason label. Anything outside this
# set collapses to "other" so an attacker-controlled rejection message can
# never explode metric cardinality.
_KNOWN_GATE_REASONS = {
    "prompt_injection",
    "jailbreak",
    "pii_detected",
    "toxic",
    "off_topic",
    "rbac_denied",
    "sensitivity_block",
    "policy_violation",
}


def _record_block(gate: str, reason: str | None) -> None:
    if not METRICS_ENABLED:
        return
    key = (reason or "unknown").strip().lower().replace(" ", "_")[:40]
    label = key if key in _KNOWN_GATE_REASONS else "other"
    GUARDRAILS_BLOCKED.labels(gate=gate, reason=label).inc()


def record_audit_verification(result: str, valid: bool) -> None:
    """Record one scheduled audit-chain verification outcome.

    ``result`` is one of ``valid`` / ``broken`` / ``error``. No-op without the
    extra; never raises.
    """
    if not METRICS_ENABLED:
        return
    try:
        AUDIT_VERIFICATIONS.labels(result=result).inc()
        AUDIT_CHAIN_VALID.set(1 if valid else 0)
    except Exception as exc:  # pragma: no cover - defensive
        _log.debug("metrics_audit_record_failed", error=str(exc))


def render_latest() -> tuple[bytes, str]:
    """Return ``(payload, content_type)`` for a ``/metrics`` response.

    Raises ``RuntimeError`` when the extra is absent so the API layer can map
    that to a 501.
    """
    if not METRICS_ENABLED:
        raise RuntimeError("prometheus_client not installed")
    return generate_latest(), CONTENT_TYPE_LATEST
