"""Chat page — conversational RAG interface with citations."""

from __future__ import annotations

import time
import uuid

import streamlit as st

from app.chat_service import (
    log_audit_entry,
    run_ragas_evaluation,
    store_evaluation_data,
)
from app.components.chat_message import (
    render_assistant_message,
    render_security_blocked,
)
from app.components.sidebar import get_current_user_context
from core.graph import run_rag_pipeline, run_rag_pipeline_stream
from utils.async_helpers import run_async
from utils.conversation_store import ConversationMessage, conversation_store
from utils.logging import correlation_id_scope, get_logger
from utils.query_cache import get_cached_result, set_cached_result
from utils.rate_limiter import check_query_rate_limit
from utils.validation import validate_query

logger = get_logger(__name__)


def render_chat_page() -> None:
    """Render the chat page with conversational RAG interface."""
    st.title("💬 Chat")
    st.caption(
        "Ask questions about your documents. Responses include citations and confidence scores."
    )

    # Thread management sidebar within chat page
    _render_thread_sidebar()

    # Display chat history
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

    # Streaming toggle
    use_streaming = st.toggle(
        "Enable streaming",
        value=True,
        help="Stream tokens as they are generated (all providers)",
    )

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Set correlation ID for this request
        with correlation_id_scope() as cid:
            logger.info("chat_query_received", correlation_id=cid, query_len=len(prompt))

            # Input validation
            validation = validate_query(prompt)
            if not validation.valid:
                st.warning(f"⚠️ {validation.message}", icon="⚠️")
                st.session_state.chat_history.append(
                    {
                        "role": "blocked",
                        "content": validation.message,
                    }
                )
                st.stop()

            # Use sanitized query
            prompt = validation.sanitized_query

            # Rate limit check
            user = st.session_state.current_user
            allowed, rl_meta = check_query_rate_limit(user["user_id"])
            if not allowed:
                st.warning(
                    f"⏳ Rate limit exceeded. Please wait {rl_meta['retry_after']} seconds before trying again.",
                    icon="⏳",
                )
                st.session_state.chat_history.append(
                    {
                        "role": "blocked",
                        "content": f"Rate limited: retry after {rl_meta['retry_after']}s",
                    }
                )
                _persist_message(
                    "blocked",
                    f"Rate limited: retry after {rl_meta['retry_after']}s",
                )
                st.stop()

            # Check query cache
            cached_result = get_cached_result(
                user_id=user["user_id"],
                query=prompt,
            )
            if cached_result:
                st.info("📦 Cached result (identical query)", icon="📦")
                _render_cached_result(cached_result)
                st.stop()

            # Ensure we have an active thread
            if not st.session_state.get("active_thread_id"):
                st.session_state.active_thread_id = str(uuid.uuid4())

            # Display user message immediately
            with st.chat_message("user"):
                st.markdown(prompt)

            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            # Persist user message
            _persist_message("user", prompt)

            # Process with RAG pipeline
            with st.chat_message("assistant"):
                if use_streaming:
                    _process_query_streaming(prompt)
                else:
                    with st.spinner("🤔 Processing through RAG pipeline..."):
                        _process_query(prompt)


def _render_cached_result(cached: dict) -> None:
    """Render a cached query result in the chat interface.

    Args:
        cached: The cached result dict with generation, citations, confidence.
    """
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
    _persist_message(
        "assistant",
        generation,
        metadata={
            "citations": citations,
            "confidence": confidence,
            "routing_info": cached.get("routing_info"),
            "cached": True,
        },
    )


def _render_thread_sidebar() -> None:
    """Render thread selection and management controls."""
    user = st.session_state.current_user

    with st.expander("🧵 Conversation Threads", expanded=False):
        # List existing threads
        threads = conversation_store.list_threads(
            user_id=user["user_id"],
            org_id=user["org_id"],
            limit=20,
        )

        if threads:
            thread_options = {
                t["thread_id"]: f"Thread {t['thread_id'][:8]}... ({t['message_count']} msgs)"
                for t in threads
            }
            thread_options["new"] = "+ Start New Thread"

            current = st.session_state.get("active_thread_id", "new")
            selected = st.selectbox(
                "Active Thread",
                options=list(thread_options.keys()),
                format_func=lambda x: thread_options[x],
                index=list(thread_options.keys()).index(current)
                if current in thread_options
                else 0,
                key="thread_selector",
            )

            if selected == "new":
                if st.button("Create New Thread", width="stretch"):
                    st.session_state.active_thread_id = str(uuid.uuid4())
                    st.session_state.chat_history = []
                    st.rerun()
            elif selected != st.session_state.get("active_thread_id"):
                # Load selected thread
                thread = conversation_store.load_thread(selected)
                if thread:
                    st.session_state.active_thread_id = selected
                    st.session_state.chat_history = [
                        {
                            "role": msg.role,
                            "content": msg.content,
                            "citations": msg.metadata.get("citations", []),
                            "confidence": msg.metadata.get("confidence", 0.0),
                            "routing_info": msg.metadata.get("routing_info"),
                        }
                        for msg in thread.messages
                    ]
                    st.rerun()

            # Delete thread button
            if (
                selected != "new"
                and st.button("🗑️ Delete Thread", type="secondary", width="stretch")
                and conversation_store.delete_thread(selected)
            ):
                st.session_state.active_thread_id = None
                st.session_state.chat_history = []
                st.rerun()
        else:
            st.info("No saved conversations yet.")
            if st.button("Start New Thread", width="stretch"):
                st.session_state.active_thread_id = str(uuid.uuid4())
                st.session_state.chat_history = []
                st.rerun()


def _persist_message(role: str, content: str, metadata: dict | None = None) -> None:
    """Persist a message to the conversation store.

    Args:
        role: Message role.
        content: Message content.
        metadata: Optional metadata dict.
    """
    thread_id = st.session_state.get("active_thread_id")
    if not thread_id:
        return

    user = st.session_state.current_user
    msg = ConversationMessage(
        role=role,
        content=content,
        metadata=metadata or {},
    )
    conversation_store.append_message(
        thread_id=thread_id,
        message=msg,
        user_id=user["user_id"],
        org_id=user["org_id"],
        metadata={
            "model": st.session_state.selected_model,
            "mode": st.session_state.inference_mode,
        },
    )


def _process_query_streaming(query: str) -> None:
    """Process a user query with TRUE token-by-token streaming.

    Uses ``run_rag_pipeline_stream`` to execute router, security, retriever,
    grader (and optional rewrite loop), then streams synthesis tokens
    directly from the LLM to the Streamlit UI as they arrive, then runs
    the evaluator on the collected text.

    Args:
        query: The user's natural language question.
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
    phase_labels = {
        "router": "🧭 Classifying query",
        "security": "🛡️ Security check",
        "retriever": "🔍 Retrieving documents",
        "grader": "🎯 Grading relevance",
        "rewriter": "✏️ Rewriting query (corrective loop)",
    }

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
                label = phase_labels.get(event["name"], event["name"])
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
            _persist_message("blocked", blocked_message)
            log_audit_entry(
                action="query_blocked",
                query=query,
                details=blocked_message,
                latency_ms=latency_ms,
            )
            return

        if final_state is None:
            err = "Streaming pipeline did not produce a final state."
            logger.error("chat_streaming_no_final_state", query_len=len(query))
            st.error(err)
            return

        generation = final_state.get("generation", "")
        citations = final_state.get("citations", [])
        confidence = final_state.get("confidence_score", 0.0)
        needs_human_review = final_state.get("needs_human_review", False)
        evaluation_notes = final_state.get("evaluation_notes", "")
        query_type = final_state.get("query_type", "unknown")
        security_passed = final_state.get("security_passed", False)

        # Ensure final placeholder shows the disclaimer-augmented text
        if generation and generation != "".join(collected_text):
            placeholder.markdown(generation)

        if needs_human_review:
            st.warning(
                f"⚠️ **Low Confidence Response** — Human review recommended\n\n{evaluation_notes}",
                icon="⚠️",
            )

        # Render confidence + provenance + sources inline in the streaming
        # path so all metadata is visible right after generation finishes.
        provider = final_state.get("synth_provider", "ollama")
        model = final_state.get("synth_model", st.session_state.selected_model)
        synth_latency = final_state.get("synth_latency_ms", 0.0)
        usage = final_state.get("synth_usage") or {}
        total_tokens = usage.get("total_tokens", 0)

        with citations_placeholder.container():
            if confidence >= 0.8:
                color, icon = "#22c55e", "🟢"
            elif confidence >= 0.6:
                color, icon = "#eab308", "🟡"
            else:
                color, icon = "#ef4444", "🔴"
            prov_badge = f"☁️ {provider}" if provider != "ollama" else f"🏠 {provider}"
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

        routing_info = {
            "provider": provider,
            "model": model,
            "forced_local": provider == "ollama" and st.session_state.inference_mode == "cloud",
            "latency_ms": synth_latency,
            "tokens": total_tokens,
        }

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": generation,
                "citations": citations,
                "confidence": confidence,
                "routing_info": routing_info,
            }
        )
        _persist_message(
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
        error_msg = f"An error occurred while processing your query: {exc}"
        logger.error("chat_streaming_error", error=str(exc), query_len=len(query))
        st.error(f"⚠️ {error_msg}")
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": f"❌ Error: {exc}",
                "citations": [],
                "confidence": 0.0,
            }
        )


def _process_query(query: str) -> None:
    """Process a user query through the RAG pipeline and render the response.

    Args:
        query: The user's natural language question.
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

        # Extract results from final state
        security_passed = final_state.get("security_passed", False)
        security_message = final_state.get("security_message", "")
        generation = final_state.get("generation", "")
        citations = final_state.get("citations", [])
        confidence = final_state.get("confidence_score", 0.0)
        needs_human_review = final_state.get("needs_human_review", False)
        evaluation_notes = final_state.get("evaluation_notes", "")
        query_type = final_state.get("query_type", "unknown")

        # Build routing info from session state
        routing_info = {
            "provider": "ollama" if st.session_state.inference_mode == "local" else "cloud",
            "model": st.session_state.selected_model,
            "forced_local": final_state.get("query_type") == "sensitive",
        }

        if not security_passed and not generation:
            # Security blocked
            render_security_blocked(
                security_message or "Your query was blocked by security policy."
            )
            st.session_state.chat_history.append(
                {
                    "role": "blocked",
                    "content": security_message or "Query blocked by security policy.",
                }
            )
            _persist_message(
                "blocked",
                security_message or "Query blocked by security policy.",
            )
            log_audit_entry(
                action="query_blocked",
                query=query,
                details=security_message,
                latency_ms=latency_ms,
            )
        else:
            # Successful response
            if needs_human_review:
                st.warning(
                    f"⚠️ **Low Confidence Response** — Human review recommended\n\n"
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
            _persist_message(
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

            # Cache the result for future identical queries
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

        # Store evaluation data
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

        # Run Ragas evaluation asynchronously (non-blocking)
        run_ragas_evaluation(query, generation, citations)

    except Exception as exc:
        error_msg = f"An error occurred while processing your query: {exc}"
        logger.error("chat_pipeline_error", error=str(exc), query_len=len(query))
        st.error(f"⚠️ {error_msg}")
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": f"❌ Error: {exc}",
                "citations": [],
                "confidence": 0.0,
            }
        )
