"""Non-streaming RAG chat path — extracted from ``app/views/chat.py``.

Calls ``run_rag_pipeline`` synchronously (single ainvoke under
``run_async``), then renders the assistant message in one shot. Used when
the operator toggles "Enable streaming" off in the chat header.
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
from app.components.chat_message import render_assistant_message, render_security_blocked
from app.components.sidebar import get_current_user_context
from core.graph import run_rag_pipeline
from utils.async_helpers import run_async
from utils.logging import get_logger
from utils.query_cache import set_cached_result

logger = get_logger(__name__)


def process_query(query: str) -> None:
    """Run the RAG pipeline non-streaming and render the final response.

    Args:
        query: The user's natural language question (already validated).
    """
    start_time = time.time()
    user_context = get_current_user_context()
    thread_id = st.session_state.get("thread_id", str(uuid.uuid4()))
    prefer_cloud = st.session_state.get("inference_mode") == "cloud"

    try:
        final_state = run_async(
            run_rag_pipeline(
                query=query,
                user_context=user_context,
                thread_id=thread_id,
                prefer_cloud=prefer_cloud,
            )
        )

        latency_ms = (time.time() - start_time) * 1000

        security_passed = final_state.get("security_passed", False)
        security_message = final_state.get("security_message", "")
        generation = final_state.get("generation", "")
        citations = final_state.get("citations", [])
        confidence = final_state.get("confidence_score", 0.0)
        needs_human_review = final_state.get("needs_human_review", False)
        evaluation_notes = final_state.get("evaluation_notes", "")
        query_type = final_state.get("query_type", "unknown")

        routing_info = {
            "provider": "ollama" if st.session_state.inference_mode == "local" else "cloud",
            "model": st.session_state.selected_model,
            "forced_local": final_state.get("query_type") == "sensitive",
        }

        if not security_passed and not generation:
            blocked_text = security_message or "Your query was blocked by security policy."
            render_security_blocked(blocked_text)
            st.session_state.chat_history.append({"role": "blocked", "content": blocked_text})
            persist_message("blocked", blocked_text)
            log_audit_entry(
                action="query_blocked",
                query=query,
                details=security_message,
                latency_ms=latency_ms,
            )
        else:
            if needs_human_review:
                st.warning(
                    "⚠️ **Low Confidence Response** — Human review recommended\n\n"
                    f"{evaluation_notes}",
                    icon="⚠️",
                )
            render_assistant_message(
                response_text=generation or "No response generated.",
                citations=citations,
                confidence=confidence,
                routing_info=routing_info,
            )
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": generation or "No response generated.",
                    "citations": citations,
                    "confidence": confidence,
                    "routing_info": routing_info,
                }
            )
            persist_message(
                "assistant",
                generation or "No response generated.",
                metadata={
                    "citations": citations,
                    "confidence": confidence,
                    "routing_info": routing_info,
                },
            )
            log_audit_entry(
                action="query",
                query=query,
                details=f"confidence={confidence:.2f}, citations={len(citations)}",
                latency_ms=latency_ms,
                confidence=confidence,
            )
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

        store_evaluation_data(
            query=query,
            confidence=confidence,
            latency_ms=latency_ms,
            query_type=query_type,
            security_passed=security_passed,
            provider=final_state.get("synth_provider"),
            model=final_state.get("synth_model"),
            synth_latency_ms=final_state.get("synth_latency_ms"),
            tokens=(final_state.get("synth_usage") or {}).get("total_tokens", 0),
        )
        run_ragas_evaluation(query, generation, citations)

    except Exception as exc:
        logger.error("chat_pipeline_error", error=str(exc), query_len=len(query))
        st.error(f"⚠️ An error occurred while processing your query: {exc}")
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": f"❌ Error: {exc}",
                "citations": [],
                "confidence": 0.0,
            }
        )
