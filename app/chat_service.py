"""Service helpers for the Streamlit chat view.

Split out of ``app/views/chat.py`` so the view stays focused on UI plumbing
(streaming state, layout, widget wiring) while audit / Ragas / metrics
persistence live here. All three helpers still depend on Streamlit's
session state because they read the active user and update on-screen
session-only caches; keeping that coupling explicit avoids passing four
or five session-state slices through every call site.

Functions exported:

* :func:`log_audit_entry`  — append one row to ``session_state.audit_log``.
* :func:`run_ragas_evaluation` — fire-and-forget Ragas scoring with
  graceful degradation when the ``[evaluation]`` extras are missing.
* :func:`store_evaluation_data` — persist a query's metrics to both
  session state and the SQLite metrics store for cross-restart trends.
"""

from __future__ import annotations

from datetime import UTC, datetime

import streamlit as st

from evaluation.ragas_eval import EvalSample, RagasEvaluator
from utils.async_helpers import run_async
from utils.logging import get_logger
from utils.metrics_store import store_metric

logger = get_logger(__name__)


def log_audit_entry(
    action: str,
    query: str,
    details: str,
    latency_ms: float,
    confidence: float = 0.0,
) -> None:
    """Append one row to the in-memory session audit log.

    The disk-backed audit chain (``utils.audit``) is written by the
    pipeline itself; this entry is the session-state copy that powers
    the Audit Log tab.

    Args:
        action: Event category — ``"query"``, ``"query_blocked"``, etc.
        query: Raw query text (clipped to 100 chars).
        details: Free-form details string for the row.
        latency_ms: End-to-end latency in milliseconds.
        confidence: Confidence score 0.0-1.0; defaults to 0.
    """
    user = st.session_state.current_user
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "user": user.get("display_name", user.get("user_id", "unknown")),
        "user_id": user.get("user_id", "unknown"),
        "action": action,
        "query": query[:100],
        "details": details,
        "sensitivity": f"Level {user.get('clearance_level', 1)}",
        "status": "blocked" if "blocked" in action else "success",
        "latency_ms": round(latency_ms, 1),
        "confidence": confidence,
    }
    st.session_state.audit_log.append(entry)


def run_ragas_evaluation(
    query: str,
    generation: str,
    citations: list[dict],
) -> None:
    """Fire-and-forget Ragas scoring; stash results in session state.

    Fails silently when the ``[evaluation]`` extras (``ragas`` + ``pandas``)
    are not installed — Ragas is optional and the chat path should not be
    blocked when it is unavailable.

    Args:
        query: The user's query.
        generation: The synthesizer's final answer.
        citations: List of citation dicts; only entries with non-empty
            ``chunk_text`` contribute as Ragas contexts.
    """
    try:
        contexts = [c.get("chunk_text", "") for c in citations if c.get("chunk_text")]
        if not contexts:
            return

        sample = EvalSample(query=query, response=generation, contexts=contexts)
        evaluator = RagasEvaluator()
        if not evaluator.is_available():
            logger.debug("ragas_not_available_skipping")
            return

        result = run_async(evaluator.evaluate_single(sample))

        if "ragas_scores" not in st.session_state:
            st.session_state.ragas_scores = []
        st.session_state.ragas_scores.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "query": query[:80],
                "faithfulness": result.faithfulness,
                "answer_relevancy": result.answer_relevancy,
                "context_precision": result.context_precision,
                "overall_score": result.overall_score,
                "latency_ms": result.latency_ms,
            }
        )
        logger.info(
            "ragas_evaluation_completed",
            query_len=len(query),
            overall_score=result.overall_score,
        )
    except Exception as exc:
        logger.debug("ragas_evaluation_failed", error=str(exc))


def store_evaluation_data(
    query: str,
    confidence: float,
    latency_ms: float,
    query_type: str,
    security_passed: bool,
    provider: str | None = None,
    model: str | None = None,
    synth_latency_ms: float | None = None,
    tokens: int | None = None,
) -> None:
    """Persist evaluation metrics in both session state and SQLite.

    Session state powers the Evaluation tab's recent-queries widget;
    SQLite persistence (``utils.metrics_store``) gives the cost dashboard
    and the nightly Ragas regression gate something to chart across
    restarts.

    Args:
        query: User query (clipped to 80 chars in session, 200 in SQLite).
        confidence: Confidence score 0.0-1.0.
        latency_ms: End-to-end pipeline latency.
        query_type: Router classification (``simple`` / ``complex`` / …).
        security_passed: Result of the security gate.
        provider: Synthesizer provider name; falls back to inference mode.
        model: Synthesizer model name; falls back to sidebar selection.
        synth_latency_ms: Synth-only latency (excludes retrieval / grading).
        tokens: Total token usage if reported by the provider.
    """
    if "evaluation_data" not in st.session_state:
        st.session_state.evaluation_data = []

    actual_model = model or st.session_state.selected_model
    actual_provider = provider or (
        "ollama" if st.session_state.inference_mode == "local" else "cloud"
    )

    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "query": query[:80],
        "confidence": confidence,
        "latency_ms": round(latency_ms, 1),
        "synth_latency_ms": round(synth_latency_ms or 0.0, 1),
        "tokens": tokens or 0,
        "query_type": query_type,
        "security_passed": security_passed,
        "user": st.session_state.current_user.get("display_name", "unknown"),
        "provider": actual_provider,
        "model": actual_model,
        "mode": st.session_state.inference_mode,
    }
    st.session_state.evaluation_data.append(entry)

    try:
        store_metric(
            query=query[:200],
            confidence=confidence,
            latency_ms=latency_ms,
            query_type=query_type,
            user_id=st.session_state.current_user.get("display_name", "unknown"),
            model=actual_model,
            security_passed=security_passed,
            metadata={
                "mode": st.session_state.inference_mode,
                "provider": actual_provider,
                "synth_latency_ms": synth_latency_ms,
                "tokens": tokens,
            },
        )
    except Exception as exc:
        logger.debug("metric_persistence_failed", error=str(exc))
