"""Chat page — conversational RAG interface with citations."""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime

import streamlit as st

from app.components.chat_message import (
    render_assistant_message,
    render_security_blocked,
)
from app.components.sidebar import get_current_user_context
from core.graph import run_rag_pipeline
from evaluation.ragas_eval import EvalSample, RagasEvaluator
from utils.async_helpers import run_async
from utils.conversation_store import ConversationMessage, conversation_store
from utils.logging import correlation_id_scope, get_logger
from utils.metrics_store import store_metric
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
            thread_options = {t["thread_id"]: f"Thread {t['thread_id'][:8]}... ({t['message_count']} msgs)" for t in threads}
            thread_options["new"] = "+ Start New Thread"

            current = st.session_state.get("active_thread_id", "new")
            selected = st.selectbox(
                "Active Thread",
                options=list(thread_options.keys()),
                format_func=lambda x: thread_options[x],
                index=list(thread_options.keys()).index(current) if current in thread_options else 0,
                key="thread_selector",
            )

            if selected == "new":
                if st.button("Create New Thread", use_container_width=True):
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
            if selected != "new" and st.button(
                "🗑️ Delete Thread", type="secondary", use_container_width=True
            ) and conversation_store.delete_thread(selected):
                st.session_state.active_thread_id = None
                st.session_state.chat_history = []
                st.rerun()
        else:
            st.info("No saved conversations yet.")
            if st.button("Start New Thread", use_container_width=True):
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


async def _stream_tokens_from_text(text: str, chunk_size: int = 4):
    """Simulate streaming by yielding chunks of text.

    Used when the graph has already synthesized the answer and we just
    want to present it with a streaming effect in the UI.

    Args:
        text: The full text to stream.
        chunk_size: Number of words per chunk.

    Yields:
        Text chunks simulating token streaming.
    """
    words = text.split(" ")
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        yield chunk + " "


def _process_query_streaming(query: str) -> None:
    """Process a user query with streaming display of the synthesized answer.

    Runs the full RAG pipeline (including synthesis), then streams the
    already-generated answer token-by-token in the UI. This avoids the
    double-synthesis problem — the graph generates the answer once, and
    we just present it with a streaming effect.

    Args:
        query: The user's natural language question.
    """
    start_time = time.time()
    user_context = get_current_user_context()
    thread_id = st.session_state.get("thread_id", str(uuid.uuid4()))

    try:
        # Run the FULL RAG pipeline (router, security, retriever, grader, synthesizer, evaluator)
        final_state = run_async(
            run_rag_pipeline(
                query=query,
                user_context=user_context,
                thread_id=thread_id,
            )
        )

        latency_ms = (time.time() - start_time) * 1000

        security_passed = final_state.get("security_passed", False)
        security_message = final_state.get("security_message", "")
        confidence = final_state.get("confidence_score", 0.0)
        needs_human_review = final_state.get("needs_human_review", False)
        evaluation_notes = final_state.get("evaluation_notes", "")
        query_type = final_state.get("query_type", "unknown")
        generation = final_state.get("generation", "")
        citations = final_state.get("citations", [])

        if not security_passed and not generation:
            render_security_blocked(
                security_message or "Your query was blocked by security policy."
            )
            st.session_state.chat_history.append(
                {
                    "role": "blocked",
                    "content": security_message or "Query blocked by security policy.",
                }
            )
            return

        if not generation:
            generation = (
                "I was unable to find relevant documents to answer your question. "
                "Please try rephrasing your query or check that the relevant "
                "documents have been ingested."
            )
            st.markdown(generation)
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": generation,
                    "citations": [],
                    "confidence": 0.0,
                    "routing_info": None,
                }
            )
            return

        if needs_human_review:
            st.warning(
                f"⚠️ **Low Confidence Response** — Human review recommended\n\n"
                f"{evaluation_notes}",
                icon="⚠️",
            )

        # Stream the ALREADY-SYNTHESIZED answer with a typing effect
        st.write_stream(
            run_async(_stream_tokens_from_text(generation, chunk_size=3))
        )

        # Display citations below the streamed response
        if citations:
            st.markdown("\n\n**Sources:**")
            for i, cite in enumerate(citations, 1):
                st.markdown(
                    f"{i}. *{cite['source_file']}* (p. {cite['page_number']}) — "
                    f"score: {cite['relevance_score']:.2f}"
                )

        routing_info = {
            "provider": "ollama",
            "model": st.session_state.selected_model,
            "forced_local": query_type == "sensitive",
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
        _log_audit_entry(
            action="query",
            query=query,
            details=f"confidence={confidence:.2f}, citations={len(citations)}, streamed=True",
            latency_ms=latency_ms,
            confidence=confidence,
        )
        _store_evaluation_data(
            query=query,
            confidence=confidence,
            latency_ms=latency_ms,
            query_type=query_type,
            security_passed=security_passed,
        )

        # Run Ragas evaluation asynchronously (non-blocking)
        _run_ragas_evaluation(query, generation, citations)

        # Cache the result for future identical queries
        set_cached_result(
            user_id=user_context.get("user_id", "unknown"),
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

    try:
        final_state = run_async(
            run_rag_pipeline(
                query=query,
                user_context=user_context,
                thread_id=thread_id,
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
            _log_audit_entry(
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
            _log_audit_entry(
                action="query",
                query=query,
                details=f"confidence={confidence:.2f}, citations={len(citations)}",
                latency_ms=latency_ms,
                confidence=confidence,
            )

            # Cache the result for future identical queries
            set_cached_result(
                user_id=user_context.get("user_id", "unknown"),
                query=query,
                result={
                    "generation": generation,
                    "citations": citations,
                    "confidence": confidence,
                    "routing_info": routing_info,
                },
            )

        # Store evaluation data
        _store_evaluation_data(
            query=query,
            confidence=confidence,
            latency_ms=latency_ms,
            query_type=query_type,
            security_passed=security_passed,
        )

        # Run Ragas evaluation asynchronously (non-blocking)
        _run_ragas_evaluation(query, generation, citations)

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


def _log_audit_entry(
    action: str,
    query: str,
    details: str,
    latency_ms: float,
    confidence: float = 0.0,
) -> None:
    """Add an entry to the session audit log.

    Args:
        action: Type of action (query, query_blocked, upload).
        query: The query text.
        details: Additional details string.
        latency_ms: Processing time in milliseconds.
        confidence: Confidence score if applicable.
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


def _run_ragas_evaluation(
    query: str,
    generation: str,
    citations: list[dict],
) -> None:
    """Run Ragas evaluation in the background (non-blocking).

    Stores results in session state for display in the evaluation dashboard.
    Gracefully handles missing ragas dependency.

    Args:
        query: The user's query.
        generation: The generated response.
        citations: List of citation dicts with chunk_text.
    """
    try:
        contexts = [c.get("chunk_text", "") for c in citations if c.get("chunk_text")]
        if not contexts:
            return

        sample = EvalSample(
            query=query,
            response=generation,
            contexts=contexts,
        )

        evaluator = RagasEvaluator()
        if not evaluator.is_available():
            logger.debug("ragas_not_available_skipping")
            return

        # Run async evaluation via run_async
        result = run_async(evaluator.evaluate_single(sample))

        # Store Ragas scores in session state
        if "ragas_scores" not in st.session_state:
            st.session_state.ragas_scores = []

        st.session_state.ragas_scores.append({
            "timestamp": datetime.now(UTC).isoformat(),
            "query": query[:80],
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "overall_score": result.overall_score,
            "latency_ms": result.latency_ms,
        })

        logger.info(
            "ragas_evaluation_completed",
            query_len=len(query),
            overall_score=result.overall_score,
        )
    except Exception as exc:
        logger.debug("ragas_evaluation_failed", error=str(exc))


def _store_evaluation_data(
    query: str,
    confidence: float,
    latency_ms: float,
    query_type: str,
    security_passed: bool,
) -> None:
    """Store query evaluation data for the evaluation dashboard.

    Persists to both in-memory session state (for immediate UI display)
    and SQLite (for long-term persistence across restarts).

    Args:
        query: The user query.
        confidence: Confidence score.
        latency_ms: Processing latency in milliseconds.
        query_type: Type of query (simple, complex, etc.).
        security_passed: Whether security check passed.
    """
    if "evaluation_data" not in st.session_state:
        st.session_state.evaluation_data = []

    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "query": query[:80],
        "confidence": confidence,
        "latency_ms": round(latency_ms, 1),
        "query_type": query_type,
        "security_passed": security_passed,
        "user": st.session_state.current_user.get("display_name", "unknown"),
        "model": st.session_state.selected_model,
        "mode": st.session_state.inference_mode,
    }

    st.session_state.evaluation_data.append(entry)

    # Persist to SQLite for long-term storage
    try:
        store_metric(
            query=query[:200],
            confidence=confidence,
            latency_ms=latency_ms,
            query_type=query_type,
            user_id=st.session_state.current_user.get("display_name", "unknown"),
            model=st.session_state.selected_model,
            security_passed=security_passed,
            metadata={"mode": st.session_state.inference_mode},
        )
    except Exception as exc:
        logger.debug("metric_persistence_failed", error=str(exc))
