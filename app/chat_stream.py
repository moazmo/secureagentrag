"""Streaming RAG chat path — extracted from ``app/views/chat.py``.

Drives ``run_rag_pipeline_stream`` end-to-end: status widget per phase,
token writer into a single Streamlit placeholder, blocked-gate / timeout
short-circuits, then renders confidence + provenance + citation strip
once the final state lands. Audit + Ragas + metrics + cache hand-off go
through ``app.chat_service`` so persistence stays in one place.
"""

from __future__ import annotations

import time
import uuid

import streamlit as st

from app.chat_service import (
    log_audit_entry,
    persist_message,
    run_ragas_evaluation,
    store_evaluation_data,
)
from app.components.chat_message import render_security_blocked
from app.components.sidebar import get_current_user_context
from core.graph import run_rag_pipeline_stream
from utils.async_helpers import run_async
from utils.logging import get_logger
from utils.query_cache import set_cached_result

logger = get_logger(__name__)


_PHASE_LABELS = {
    "router": "🧭 Classifying query",
    "security": "🛡️ Security check",
    "retriever": "🔍 Retrieving documents",
    "grader": "🎯 Grading relevance",
    "rewriter": "✏️ Rewriting query (corrective loop)",
}


def _confidence_badge_color(confidence: float) -> tuple[str, str]:
    """Pick a (hex, glyph) pair to colour the confidence chip."""
    if confidence >= 0.8:
        return "#22c55e", "🟢"
    if confidence >= 0.6:
        return "#eab308", "🟡"
    return "#ef4444", "🔴"


def _render_result_strip(
    placeholder, citations: list[dict], confidence: float, routing_info: dict
) -> None:
    """Render the confidence + provenance + sources block under the answer."""
    provider = routing_info["provider"]
    model = routing_info["model"]
    synth_latency = routing_info["latency_ms"]
    total_tokens = routing_info["tokens"]
    color, icon = _confidence_badge_color(confidence)
    prov_badge = f"☁️ {provider}" if provider != "ollama" else f"🏠 {provider}"
    with placeholder.container():
        st.markdown(
            f"<div style='display:flex;gap:8px;flex-wrap:wrap;align-items:center;'>"
            f"<span style='padding:4px 10px;border-radius:12px;background:{color}22;"
            f"color:{color};font-weight:600;'>{icon} Confidence: {confidence:.0%}</span>"
            f"<span style='padding:4px 10px;border-radius:12px;background:#3b82f622;"
            f"color:#3b82f6;font-weight:500;'>{prov_badge} · {model}</span>"
            f"<span style='padding:4px 10px;border-radius:12px;background:#8b5cf622;"
            f"color:#8b5cf6;font-weight:500;'>⚡ {synth_latency / 1000:.1f}s synth"
            f"{f' · {total_tokens} tok' if total_tokens else ''}</span></div>",
            unsafe_allow_html=True,
        )
        if citations:
            st.markdown(f"\n**Sources ({len(citations)} citations):**")
            for i, cite in enumerate(citations, 1):
                st.markdown(
                    f"{i}. *{cite['source_file']}* (p. {cite['page_number']}) — "
                    f"score: {cite['relevance_score']:.2f}"
                )


def process_query_streaming(query: str) -> None:
    """Drive the streaming pipeline for a single chat turn.

    Token chunks are concatenated into ``placeholder.markdown(...)`` as
    they arrive; phase events drive a ``st.status(...)`` label. On
    ``"blocked"`` or ``"final"`` we exit the consume loop and render the
    appropriate UI hand-off.

    Args:
        query: The user's natural language question (already validated).
    """
    start_time = time.time()
    user_context = get_current_user_context()
    thread_id = st.session_state.get("thread_id", str(uuid.uuid4()))

    status_widget = st.status("Starting pipeline…", expanded=False)
    placeholder = st.empty()
    citations_placeholder = st.empty()

    collected_text: list[str] = []
    final_state: dict | None = None
    blocked_message: str | None = None

    prefer_cloud = st.session_state.get("inference_mode") == "cloud"

    async def _consume() -> None:
        nonlocal final_state, blocked_message
        first_token_seen = False
        async for event in run_rag_pipeline_stream(
            query=query,
            user_context=user_context,
            thread_id=thread_id,
            prefer_cloud=prefer_cloud,
        ):
            etype = event["type"]
            if etype == "phase":
                label = _PHASE_LABELS.get(event["name"], event["name"])
                status_widget.update(label=label, state="running")
            elif etype == "blocked":
                blocked_message = event["message"]
                status_widget.update(label="🚫 Blocked by security gate", state="error")
                return
            elif etype == "token":
                if not first_token_seen:
                    status_widget.update(label="💬 Generating answer…", state="running")
                    first_token_seen = True
                collected_text.append(event["text"])
                placeholder.markdown("".join(collected_text))
            elif etype == "final":
                final_state = event["state"]
                status_widget.update(label="✅ Done", state="complete")

    try:
        run_async(_consume())
        latency_ms = (time.time() - start_time) * 1000

        if blocked_message is not None:
            render_security_blocked(blocked_message)
            st.session_state.chat_history.append({"role": "blocked", "content": blocked_message})
            persist_message("blocked", blocked_message)
            log_audit_entry(
                action="query_blocked",
                query=query,
                details=blocked_message,
                latency_ms=latency_ms,
            )
            return

        if final_state is None:
            logger.error("chat_streaming_no_final_state", query_len=len(query))
            st.error("Streaming pipeline did not produce a final state.")
            return

        generation = final_state.get("generation", "")
        citations = final_state.get("citations", [])
        confidence = final_state.get("confidence_score", 0.0)
        needs_human_review = final_state.get("needs_human_review", False)
        evaluation_notes = final_state.get("evaluation_notes", "")
        query_type = final_state.get("query_type", "unknown")
        security_passed = final_state.get("security_passed", False)

        # Ensure the final placeholder shows the disclaimer-augmented text
        if generation and generation != "".join(collected_text):
            placeholder.markdown(generation)

        if needs_human_review:
            st.warning(
                f"⚠️ **Low Confidence Response** — Human review recommended\n\n{evaluation_notes}",
                icon="⚠️",
            )

        provider = final_state.get("synth_provider", "ollama")
        model = final_state.get("synth_model", st.session_state.selected_model)
        synth_latency = final_state.get("synth_latency_ms", 0.0)
        usage = final_state.get("synth_usage") or {}
        total_tokens = usage.get("total_tokens", 0)

        routing_info = {
            "provider": provider,
            "model": model,
            "forced_local": provider == "ollama" and st.session_state.inference_mode == "cloud",
            "latency_ms": synth_latency,
            "tokens": total_tokens,
        }
        _render_result_strip(citations_placeholder, citations, confidence, routing_info)

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": generation,
                "citations": citations,
                "confidence": confidence,
                "routing_info": routing_info,
            }
        )
        persist_message(
            "assistant",
            generation,
            metadata={
                "citations": citations,
                "confidence": confidence,
                "routing_info": routing_info,
            },
        )
        log_audit_entry(
            action="query",
            query=query,
            details=f"confidence={confidence:.2f}, citations={len(citations)}, streamed=True",
            latency_ms=latency_ms,
            confidence=confidence,
        )
        store_evaluation_data(
            query=query,
            confidence=confidence,
            latency_ms=latency_ms,
            query_type=query_type,
            security_passed=security_passed,
            provider=provider,
            model=model,
            synth_latency_ms=synth_latency,
            tokens=total_tokens,
        )
        run_ragas_evaluation(query, generation, citations)
        set_cached_result(
            user_id=user_context.user_id,
            query=query,
            result={
                "generation": generation,
                "citations": citations,
                "confidence": confidence,
                "routing_info": routing_info,
            },
        )

    except Exception as exc:
        logger.error("chat_streaming_error", error=str(exc), query_len=len(query))
        st.error(f"⚠️ An error occurred while processing your query: {exc}")
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": f"❌ Error: {exc}",
                "citations": [],
                "confidence": 0.0,
            }
        )
