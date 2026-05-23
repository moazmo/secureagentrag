"""Chat page — conversational RAG interface with citations.

Thin view: dispatches between the streaming and non-streaming chat paths
(``app.chat_stream`` / ``app.chat_sync``) and re-renders the on-screen
transcript. Persistence, audit, Ragas, metrics, and the conversation
thread sidebar live in their own modules so this file stays a UI shell.
"""

from __future__ import annotations

import uuid

import streamlit as st

from app.chat_service import log_audit_entry, persist_message
from app.chat_stream import process_query_streaming
from app.chat_sync import process_query
from app.components.chat_message import render_assistant_message, render_security_blocked
from app.components.thread_sidebar import render_thread_sidebar
from utils.logging import correlation_id_scope, get_logger
from utils.query_cache import get_cached_result
from utils.rate_limiter import check_query_rate_limit
from utils.validation import validate_query

logger = get_logger(__name__)


def render_chat_page() -> None:
    """Render the chat page with conversational RAG interface."""
    st.title("💬 Chat")
    st.caption(
        "Ask questions about your documents. Responses include citations and confidence scores."
    )

    render_thread_sidebar()
    _render_transcript()

    use_streaming = st.toggle(
        "Enable streaming",
        value=True,
        help="Stream tokens as they are generated (all providers)",
    )

    if prompt := st.chat_input("Ask a question about your documents..."):
        with correlation_id_scope() as cid:
            logger.info("chat_query_received", correlation_id=cid, query_len=len(prompt))
            _handle_prompt(prompt, use_streaming=use_streaming)


def _render_transcript() -> None:
    """Render the in-memory chat history above the input."""
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_assistant_message(
                    response_text=msg["content"],
                    citations=msg.get("citations", []),
                    confidence=msg.get("confidence", 0.0),
                    routing_info=msg.get("routing_info"),
                )
            elif msg["role"] == "blocked":
                render_security_blocked(msg["content"])
            else:
                st.markdown(msg["content"])


def _handle_prompt(prompt: str, *, use_streaming: bool) -> None:
    """Pre-flight checks then dispatch to the right pipeline path.

    Pre-flight in order:

    1. ``utils.validation`` — empty / too-long / obvious injection.
    2. ``utils.rate_limiter`` — token-bucket per user (optional Redis).
    3. ``utils.query_cache`` — return cached result for identical (user, query).
    4. Ensure ``active_thread_id`` exists so persist_message has a target.
    5. Echo user message to transcript + persist.
    6. Hand off to streaming or sync processor.
    """
    validation = validate_query(prompt)
    if not validation.valid:
        st.warning(f"⚠️ {validation.message}", icon="⚠️")
        st.session_state.chat_history.append({"role": "blocked", "content": validation.message})
        st.stop()

    prompt = validation.sanitized_query
    user = st.session_state.current_user

    allowed, rl_meta = check_query_rate_limit(user["user_id"])
    if not allowed:
        msg = f"Rate limited: retry after {rl_meta['retry_after']}s"
        st.warning(
            f"⏳ Rate limit exceeded. Please wait {rl_meta['retry_after']} seconds "
            "before trying again.",
            icon="⏳",
        )
        st.session_state.chat_history.append({"role": "blocked", "content": msg})
        persist_message("blocked", msg)
        st.stop()

    cached_result = get_cached_result(user_id=user["user_id"], query=prompt)
    if cached_result:
        st.info("📦 Cached result (identical query)", icon="📦")
        _render_cached_result(prompt, cached_result)
        st.stop()

    if not st.session_state.get("active_thread_id"):
        st.session_state.active_thread_id = str(uuid.uuid4())

    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    persist_message("user", prompt)

    with st.chat_message("assistant"):
        if use_streaming:
            process_query_streaming(prompt)
        else:
            with st.spinner("🤔 Processing through RAG pipeline..."):
                process_query(prompt)


def _render_cached_result(query: str, cached: dict) -> None:
    """Render a cached query result in the chat interface."""
    generation = cached.get("generation", "")
    citations = cached.get("citations", [])
    confidence = cached.get("confidence", 0.0)

    render_assistant_message(
        response_text=generation,
        citations=citations,
        confidence=confidence,
        routing_info=cached.get("routing_info"),
    )
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": generation,
            "citations": citations,
            "confidence": confidence,
            "routing_info": cached.get("routing_info"),
        }
    )
    persist_message(
        "assistant",
        generation,
        metadata={
            "citations": citations,
            "confidence": confidence,
            "routing_info": cached.get("routing_info"),
            "cached": True,
        },
    )
    # Log cache hits so the Audit Log tab tells the story.
    log_audit_entry(
        action="query_cached",
        query=query,
        details=f"confidence={confidence:.2f}, citations={len(citations)} (cache hit)",
        latency_ms=0.0,
        confidence=confidence,
    )
