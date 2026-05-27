"""LangGraph graph compilation and execution."""

from __future__ import annotations

import asyncio
import contextlib
import sys
import time
from typing import TYPE_CHECKING, Any

# psycopg's async driver does not support the Proactor event loop (Windows
# default). Switch to the Selector policy at import time so every asyncio.run
# the process spawns picks it up. No-op on POSIX. Must run before any other
# code in this project calls asyncio.run / asyncio.new_event_loop.
if sys.platform == "win32":
    with contextlib.suppress(Exception):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from config.settings import settings
from core.agents.evaluator import evaluate_response
from core.agents.faithfulness import check_faithfulness
from core.agents.guardrails import guardrails_check, guardrails_gate
from core.agents.retriever import grade_documents, retrieve_documents, should_retry
from core.agents.router import rewrite_query, route_query
from core.agents.security import check_security, security_gate
from core.agents.synthesizer import synthesize_answer
from core.state import GraphState
from utils.logging import get_logger
from utils.observability import trace_graph_execution

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from ingestion.metadata import UserContext

logger = get_logger(__name__)

# Module-level checkpointer cache
_checkpointer: MemorySaver | None = None


def _running_inside_event_loop() -> bool:
    """Return True if we are already inside an active asyncio loop.

    Async checkpointers (aiosqlite, psycopg async) bind their connection to
    the loop that opened it. Constructing one with ``asyncio.run`` while
    another loop is already running raises RuntimeError. We detect that
    condition and fall back to MemorySaver so tests / nest_asyncio harnesses
    don't fail; production startup paths create the graph from a fresh
    synchronous context and get the real persistent saver.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _try_async_postgres_saver():
    """Build an ``AsyncPostgresSaver`` bound to the current connection.

    Returns the saver on success, or ``None`` if the extras are not
    installed, we're inside a running loop, or the connection fails.
    """
    if _running_inside_event_loop():
        logger.info("postgres_checkpointer_skipped", reason="inside_running_loop")
        return None
    try:
        from langgraph.checkpoint.postgres.aio import (  # type: ignore[import-not-found]
            AsyncPostgresSaver,
        )
        from psycopg_pool import AsyncConnectionPool  # type: ignore[import-not-found]
    except ImportError:
        logger.warning(
            "postgres_checkpointer_not_available",
            hint="pip install langgraph-checkpoint-postgres 'psycopg[binary,pool]'",
        )
        return None

    async def _open() -> Any:
        pool = AsyncConnectionPool(
            settings.postgres_url,
            min_size=1,
            max_size=5,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        )
        await pool.open()
        saver = AsyncPostgresSaver(pool)
        await saver.setup()
        return saver

    # Windows event-loop policy is already pinned at module import time
    # so a fresh `asyncio.run(_open())` here gets the selector loop.

    try:
        saver = asyncio.run(_open())
        logger.info(
            "postgres_checkpointer_initialized",
            db=settings.postgres_url.rsplit("/", 1)[-1],
        )
        return saver
    except Exception as exc:
        logger.error("postgres_checkpointer_failed", error=str(exc))
        return None


def _try_async_sqlite_saver():
    """Build an ``AsyncSqliteSaver`` for local persistent checkpointing.

    Returns the saver on success or ``None`` on any failure (missing deps,
    inside a running loop, I/O error, etc.).
    """
    if _running_inside_event_loop():
        logger.info("sqlite_checkpointer_skipped", reason="inside_running_loop")
        return None
    try:
        import pathlib

        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    except ImportError:
        logger.warning(
            "sqlite_checkpointer_not_available",
            hint="pip install langgraph-checkpoint-sqlite aiosqlite",
        )
        return None

    db_path = pathlib.Path(settings.checkpoint_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    async def _open() -> Any:
        conn = await aiosqlite.connect(str(db_path), check_same_thread=False)
        saver = AsyncSqliteSaver(conn)
        await saver.setup()
        return saver

    try:
        saver = asyncio.run(_open())
        logger.info("sqlite_checkpointer_initialized", path=str(db_path))
        return saver
    except Exception as exc:
        logger.error("sqlite_checkpointer_failed", error=str(exc))
        return None


def _get_checkpointer():
    """Get or create the LangGraph checkpointer.

    Priority (when ``use_persistent_checkpointer`` is True):
      1. ``AsyncPostgresSaver`` if ``postgres_url`` is set AND the
         ``[persistence]`` extras are installed.
      2. ``AsyncSqliteSaver`` against ``settings.checkpoint_db_path``.
      3. ``MemorySaver`` (conversations lost on restart).

    Both async savers refuse to construct from within a running event loop
    to avoid cross-loop binding bugs in pytest-asyncio / nest_asyncio
    contexts; in those cases we fall back to ``MemorySaver``.

    Returns:
        Configured checkpointer instance.
    """
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer

    # Persistent checkpointing is opt-in. Default to MemorySaver so the
    # graph compiles without external deps and pytest-asyncio's per-test
    # event loops don't collide with the async saver's loop-bound state.
    if not settings.use_persistent_checkpointer:
        _checkpointer = MemorySaver()
        logger.info("memory_checkpointer_initialized", reason="persistence_opt_in_disabled")
        return _checkpointer

    if settings.postgres_url:
        saver = _try_async_postgres_saver()
        if saver is not None:
            _checkpointer = saver
            return _checkpointer

    saver = _try_async_sqlite_saver()
    if saver is not None:
        _checkpointer = saver
        return _checkpointer

    # Final fallback: in-memory (conversations lost on restart)
    _checkpointer = MemorySaver()
    logger.info("memory_checkpointer_initialized", reason="all_persistent_paths_failed")
    return _checkpointer


async def _get_async_checkpointer():
    """Async variant of ``_get_checkpointer`` — safe to call from inside a
    running event loop.

    The async ``AsyncPostgresSaver`` / ``AsyncSqliteSaver`` cannot be opened
    via ``asyncio.run()`` from within another loop. When the pipeline is
    invoked from within an already-running loop (Streamlit, FastAPI,
    user-supplied ``asyncio.run`` wrappers) we open the saver natively
    here and cache it.
    """
    global _checkpointer
    if _checkpointer is not None and not isinstance(_checkpointer, MemorySaver):
        return _checkpointer

    if not settings.use_persistent_checkpointer:
        _checkpointer = MemorySaver()
        return _checkpointer

    if settings.postgres_url:
        try:
            from langgraph.checkpoint.postgres.aio import (  # type: ignore[import-not-found]
                AsyncPostgresSaver,
            )
            from psycopg_pool import AsyncConnectionPool  # type: ignore[import-not-found]

            pool = AsyncConnectionPool(
                settings.postgres_url,
                min_size=1,
                max_size=5,
                kwargs={"autocommit": True, "prepare_threshold": 0},
                open=False,
            )
            await pool.open()
            saver = AsyncPostgresSaver(pool)
            await saver.setup()
            _checkpointer = saver
            logger.info(
                "postgres_checkpointer_initialized_async",
                db=settings.postgres_url.rsplit("/", 1)[-1],
            )
            return _checkpointer
        except ImportError:
            logger.warning(
                "postgres_checkpointer_not_available",
                hint="pip install langgraph-checkpoint-postgres 'psycopg[binary,pool]'",
            )
        except Exception as exc:
            logger.error("postgres_checkpointer_failed_async", error=str(exc))

    try:
        import pathlib

        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        db_path = pathlib.Path(settings.checkpoint_db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(str(db_path), check_same_thread=False)
        saver = AsyncSqliteSaver(conn)
        await saver.setup()
        _checkpointer = saver
        logger.info("sqlite_checkpointer_initialized_async", path=str(db_path))
        return _checkpointer
    except ImportError:
        logger.warning(
            "sqlite_checkpointer_not_available",
            hint="pip install langgraph-checkpoint-sqlite aiosqlite",
        )
    except Exception as exc:
        logger.error("sqlite_checkpointer_failed_async", error=str(exc))

    _checkpointer = MemorySaver()
    return _checkpointer


async def build_rag_graph_async() -> StateGraph:
    """Build the LangGraph workflow with an async-resolved checkpointer.

    Equivalent to :func:`build_rag_graph` but suitable for callers that are
    already inside an event loop and want a persistent (Postgres / aiosqlite)
    saver instead of the MemorySaver fallback ``build_rag_graph`` returns
    in that situation.
    """
    workflow = _compose_workflow()
    checkpointer = await _get_async_checkpointer()
    compiled = workflow.compile(checkpointer=checkpointer)
    logger.info("rag_graph_compiled_async", nodes=list(workflow.nodes.keys()))
    return compiled


def _compose_workflow() -> StateGraph:
    """Build the agent graph structure (no checkpointer attached)."""
    workflow = StateGraph(GraphState)
    workflow.add_node("router", route_query)
    workflow.add_node("guardrails", guardrails_check)
    workflow.add_node("security", check_security)
    workflow.add_node("retriever", retrieve_documents)
    workflow.add_node("grader", grade_documents)
    workflow.add_node("rewriter", rewrite_query)
    workflow.add_node("synthesizer", synthesize_answer)
    workflow.add_node("faithfulness", check_faithfulness)
    workflow.add_node("evaluator", evaluate_response)
    workflow.add_edge(START, "router")
    workflow.add_edge("router", "guardrails")
    workflow.add_conditional_edges(
        "guardrails",
        guardrails_gate,
        {"proceed": "security", "blocked": END},
    )
    workflow.add_conditional_edges(
        "security",
        security_gate,
        {"proceed": "retriever", "blocked": END},
    )
    workflow.add_edge("retriever", "grader")
    workflow.add_conditional_edges(
        "grader",
        should_retry,
        {"rewrite": "rewriter", "generate": "synthesizer"},
    )
    workflow.add_edge("rewriter", "retriever")
    # Faithfulness sits between synth and evaluator so the evaluator's
    # confidence math can read faithfulness_ratio directly. When the gate
    # is disabled the node is a no-op pass-through.
    workflow.add_edge("synthesizer", "faithfulness")
    workflow.add_edge("faithfulness", "evaluator")
    workflow.add_edge("evaluator", END)
    return workflow


def build_rag_graph() -> StateGraph:
    """Build and compile the multi-agent RAG workflow graph.

    Creates a StateGraph with the following flow:
        START -> router -> guardrails -> security -> [proceed: retriever | blocked: END]
        retriever -> grader -> [rewrite: rewriter -> retriever | generate: synthesizer]
        synthesizer -> evaluator -> END

    Uses the sync checkpointer resolver, which falls back to MemorySaver
    when called from inside a running event loop. Production async paths
    should use :func:`build_rag_graph_async` instead so the persistent
    Postgres / aiosqlite saver can be opened natively in the running loop.

    Returns:
        Compiled LangGraph StateGraph ready for invocation.
    """
    workflow = _compose_workflow()
    checkpointer = _get_checkpointer()
    compiled = workflow.compile(checkpointer=checkpointer)
    logger.info("rag_graph_compiled", nodes=list(workflow.nodes.keys()))
    return compiled


def create_initial_state(
    query: str,
    user_context: UserContext,
    prefer_cloud: bool = False,
    override_provider: str = "",
    persona_style: str = "",
) -> GraphState:
    """Create the proper initial state dict for graph invocation.

    Args:
        query: The user's natural language query.
        user_context: Authenticated user context for RBAC.
        prefer_cloud: Whether the caller is willing to route LOW/MEDIUM
            sensitivity work to cloud providers. HIGH sensitivity always
            stays local regardless.
        override_provider: Explicit provider override ("ollama" / "groq" /
            "openai" / "anthropic"). Bypasses the sensitivity routing —
            intended for admin/debug. Empty string means no override.

    Returns:
        GraphState dict ready to pass to graph.invoke() or graph.ainvoke().
    """
    return {
        "query": query,
        "user_context": user_context.model_dump(),
        "prefer_cloud": prefer_cloud,
        "override_provider": override_provider,
        "persona_style": persona_style,
        "_stream": False,
        "query_type": "",
        "rewritten_query": "",
        "query_sensitivity": "low",
        "guardrails_passed": False,
        "guardrails_reason": "",
        "security_passed": False,
        "security_message": "",
        "documents": [],
        "relevant_documents": [],
        "relevance_ratio": 0.0,
        "retry_count": 0,
        "max_retries": settings.max_retries,
        "generation": "",
        "citations": [],
        "confidence_score": 0.0,
        "synth_provider": "",
        "synth_model": "",
        "synth_usage": {},
        "synth_latency_ms": 0.0,
        "needs_human_review": False,
        "evaluation_notes": "",
        "faithfulness_ratio": 1.0,
        "faithfulness_unsupported": [],
        "audit_trail": [],
    }


def _build_timeout_state(
    query: str,
    user_context: UserContext,
    elapsed_ms: float,
    prefer_cloud: bool,
    override_provider: str,
) -> GraphState:
    """Synthesize a final-state dict for a request that hit the SLO deadline.

    Mirrors the shape of a normal final state so downstream code (UI rendering,
    cost dashboard, audit logger) treats it the same as a synthesized answer.
    """
    state = create_initial_state(
        query, user_context, prefer_cloud=prefer_cloud, override_provider=override_provider
    )
    state["generation"] = (
        "Request exceeded the configured wall-clock budget and was cancelled. "
        "Try a shorter query, disable streaming, or raise SAR_REQUEST_TIMEOUT_S."
    )
    state["citations"] = []
    state["confidence_score"] = 0.0
    state["needs_human_review"] = True
    state["evaluation_notes"] = "request_timeout"
    state["audit_trail"] = [
        {
            "node": "deadline",
            "action": "timeout",
            "elapsed_ms": elapsed_ms,
            "budget_s": settings.request_timeout_s,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    ]
    return state


async def run_rag_pipeline(
    query: str,
    user_context: UserContext,
    thread_id: str = "default",
    prefer_cloud: bool = False,
    override_provider: str = "",
    persona_style: str = "",
) -> GraphState:
    """Execute the full RAG pipeline and return the final state.

    High-level async function that builds the graph, creates initial state,
    and invokes the workflow with checkpointing enabled. Bounded by
    ``settings.request_timeout_s``: on deadline, returns a graceful timeout
    state with ``needs_human_review=True`` rather than blocking indefinitely.

    Args:
        query: The user's natural language query.
        user_context: Authenticated user context for RBAC filtering.
        thread_id: Thread identifier for checkpointing/session tracking.

    Returns:
        Final GraphState dict with generation, citations, confidence, etc.
    """
    logger.info(
        "running_rag_pipeline",
        query_len=len(query),
        user_id=user_context.user_id,
        thread_id=thread_id,
    )

    start_time = time.perf_counter()
    graph = await build_rag_graph_async()
    initial_state = create_initial_state(
        query,
        user_context,
        prefer_cloud=prefer_cloud,
        override_provider=override_provider,
        persona_style=persona_style,
    )

    config = {"configurable": {"thread_id": thread_id}}

    budget = settings.request_timeout_s
    try:
        if budget and budget > 0:
            async with asyncio.timeout(budget):
                final_state = await graph.ainvoke(initial_state, config=config)
        else:
            final_state = await graph.ainvoke(initial_state, config=config)
    except TimeoutError:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            "rag_pipeline_timeout",
            budget_s=budget,
            elapsed_ms=elapsed_ms,
            user_id=user_context.user_id,
            thread_id=thread_id,
        )
        return _build_timeout_state(
            query, user_context, elapsed_ms, prefer_cloud, override_provider
        )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Extract executed nodes from audit trail
    nodes_executed = [
        entry["node"] for entry in final_state.get("audit_trail", []) if "node" in entry
    ]

    trace_graph_execution(
        query=query,
        nodes_executed=nodes_executed,
        total_latency_ms=elapsed_ms,
        final_confidence=final_state.get("confidence_score", 0.0),
        retries=final_state.get("retry_count", 0),
    )

    logger.info(
        "rag_pipeline_completed",
        confidence_score=final_state.get("confidence_score", 0.0),
        needs_review=final_state.get("needs_human_review", False),
        generation_len=len(final_state.get("generation", "")),
        latency_ms=elapsed_ms,
    )

    return final_state


def _apply_audit(state: dict, entries: list[dict] | None) -> None:
    """Append audit entries to mutable state['audit_trail'] in place."""
    if not entries:
        return
    state.setdefault("audit_trail", []).extend(entries)


def _merge_update(state: dict, update: dict) -> None:
    """Merge a node's partial update into state.

    Mirrors LangGraph's reducer semantics: audit_trail is appended,
    every other field is overwritten.
    """
    if not update:
        return
    audit_extra = update.pop("audit_trail", None)
    state.update(update)
    if audit_extra:
        _apply_audit(state, audit_extra)


async def run_rag_pipeline_stream(
    query: str,
    user_context: UserContext,
    thread_id: str = "default",
    prefer_cloud: bool = False,
    override_provider: str = "",
    persona_style: str = "",
) -> AsyncGenerator[dict, None]:
    """Execute the full RAG pipeline with real token-by-token streaming.

    Single source of truth: runs the same compiled LangGraph workflow the
    non-streaming path uses via ``graph.astream(stream_mode=["updates",
    "custom"])``. Node updates become ``phase`` events; the synthesizer's
    ``get_stream_writer()`` calls surface as ``token`` events. Blocked
    gates and timeouts are detected from the merged state — no parallel
    hand-walked graph.

    Event types yielded:
        {"type": "phase",   "name": str, "state": dict}   — after each node
        {"type": "blocked", "message": str, "state": dict, "latency_ms": float}
        {"type": "token",   "text": str}                  — synthesis token
        {"type": "final",   "state": dict, "latency_ms": float}

    Args:
        query: Natural language query.
        user_context: Authenticated user context for RBAC.
        thread_id: Thread identifier for audit/log correlation.
        prefer_cloud: Caller opts into cloud providers for LOW/MEDIUM.
        override_provider: Admin-only provider pin.

    Yields:
        Event dicts as described above.
    """
    logger.info(
        "running_rag_pipeline_stream",
        query_len=len(query),
        user_id=user_context.user_id,
        thread_id=thread_id,
    )
    start_time = time.perf_counter()
    budget = settings.request_timeout_s

    graph = await build_rag_graph_async()
    initial_state = create_initial_state(
        query,
        user_context,
        prefer_cloud=prefer_cloud,
        override_provider=override_provider,
        persona_style=persona_style,
    )
    # Opt the synthesizer into the streaming dispatch path. The flag is
    # local to this run and is not part of the public state contract — it
    # exists so the synthesizer can deterministically choose call_llm_stream
    # over call_llm_with_decision without sniffing framework internals.
    initial_state["_stream"] = True
    config = {"configurable": {"thread_id": thread_id}}

    # Track the merged state as it grows. LangGraph's "updates" stream
    # yields one partial dict per node; we apply them locally so we can
    # detect blocked gates without waiting for the entire graph.
    state: dict = dict(initial_state)
    emitted_blocked = False

    async def _astream():
        async for chunk in graph.astream(
            initial_state, config=config, stream_mode=["updates", "custom"]
        ):
            yield chunk

    try:
        stream_ctx = asyncio.timeout(budget) if budget and budget > 0 else contextlib.nullcontext()
        async with stream_ctx:
            async for chunk in _astream():
                # LangGraph yields (mode, payload) tuples when stream_mode
                # is a list.
                if not isinstance(chunk, tuple) or len(chunk) != 2:
                    continue
                mode, payload = chunk

                if mode == "custom":
                    # Synthesizer pushes {"type": "token", "text": ...}
                    # through the writer; relay verbatim.
                    if isinstance(payload, dict):
                        yield payload
                    continue

                if mode != "updates":
                    continue

                # `updates` payload is {node_name: partial_state}. Apply
                # the partial to our local state and emit a phase event.
                if not isinstance(payload, dict):
                    continue
                for node_name, partial in payload.items():
                    if isinstance(partial, dict):
                        _merge_update(state, dict(partial))
                    yield {"type": "phase", "name": node_name, "state": dict(state)}

                    # Detect blocked gates as soon as they fire.
                    if (
                        node_name == "guardrails"
                        and state.get("guardrails_passed") is False
                        and not emitted_blocked
                    ):
                        emitted_blocked = True
                        yield {
                            "type": "blocked",
                            "message": (
                                "Blocked by guardrails: "
                                f"{state.get('guardrails_reason', 'prompt_injection')}"
                            ),
                            "state": dict(state),
                            "latency_ms": (time.perf_counter() - start_time) * 1000,
                        }
                    elif (
                        node_name == "security"
                        and state.get("security_passed") is False
                        and not emitted_blocked
                    ):
                        emitted_blocked = True
                        yield {
                            "type": "blocked",
                            "message": state.get("security_message", "Blocked by security policy."),
                            "state": dict(state),
                            "latency_ms": (time.perf_counter() - start_time) * 1000,
                        }
    except TimeoutError:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            "rag_pipeline_stream_timeout",
            budget_s=budget,
            elapsed_ms=elapsed_ms,
            user_id=user_context.user_id,
            thread_id=thread_id,
        )
        _apply_audit(
            state,
            [
                {
                    "node": "deadline",
                    "action": "timeout",
                    "elapsed_ms": elapsed_ms,
                    "budget_s": budget,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            ],
        )
        state["needs_human_review"] = True
        state["evaluation_notes"] = "request_timeout"
        yield {
            "type": "blocked",
            "message": (
                f"Request exceeded the configured wall-clock budget ({budget:.1f}s) "
                "and was cancelled."
            ),
            "state": dict(state),
            "latency_ms": elapsed_ms,
        }
        return

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    nodes_executed = [entry["node"] for entry in state.get("audit_trail", []) if "node" in entry]
    trace_graph_execution(
        query=query,
        nodes_executed=nodes_executed,
        total_latency_ms=elapsed_ms,
        final_confidence=state.get("confidence_score", 0.0),
        retries=state.get("retry_count", 0),
    )

    logger.info(
        "rag_pipeline_stream_completed",
        confidence_score=state.get("confidence_score", 0.0),
        needs_review=state.get("needs_human_review", False),
        generation_len=len(state.get("generation", "")),
        latency_ms=elapsed_ms,
    )

    yield {"type": "final", "state": dict(state), "latency_ms": elapsed_ms}
