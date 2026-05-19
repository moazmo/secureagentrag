"""LangGraph graph compilation and execution."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from config.settings import settings
from core.agents.evaluator import evaluate_response
from core.agents.guardrails import guardrails_check, guardrails_gate
from core.agents.retriever import grade_documents, retrieve_documents, should_retry
from core.agents.router import rewrite_query, route_query
from core.agents.security import check_security, security_gate
from core.agents.synthesizer import synthesize_answer, synthesize_answer_stream
from core.state import GraphState
from utils.logging import get_logger
from utils.observability import trace_graph_execution

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from ingestion.metadata import UserContext

logger = get_logger(__name__)

# Module-level checkpointer cache
_checkpointer: MemorySaver | None = None


def _get_checkpointer():
    """Get or create the LangGraph checkpointer.

    Uses PostgresSaver if ``settings.postgres_url`` is configured and
    ``langgraph-checkpoint-postgres`` is installed. Falls back to
    in-memory MemorySaver otherwise.

    Returns:
        Configured checkpointer instance.
    """
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer

    # Persistent checkpointing is opt-in. Default to MemorySaver so the
    # graph compiles without external deps and pytest-asyncio's per-test
    # event loops don't collide with aiosqlite's loop-bound connection.
    if not settings.use_persistent_checkpointer:
        _checkpointer = MemorySaver()
        logger.info("memory_checkpointer_initialized", reason="persistence_opt_in_disabled")
        return _checkpointer

    # Try Postgres first
    if settings.postgres_url:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            from psycopg import Connection

            conn = Connection.connect(settings.postgres_url)
            _checkpointer = PostgresSaver(conn)
            _checkpointer.setup()
            logger.info(
                "postgres_checkpointer_initialized",
                db=settings.postgres_url.rsplit("/", 1)[-1],
            )
            return _checkpointer
        except ImportError:
            logger.warning(
                "postgres_checkpointer_not_available",
                hint="pip install langgraph-checkpoint-postgres psycopg",
            )
        except Exception as exc:
            logger.error("postgres_checkpointer_failed", error=str(exc))

    # Try SQLite for persistent local checkpointing (no external DB needed).
    # AsyncSqliteSaver wraps an aiosqlite.Connection bound to the event loop
    # that opens it — so we only construct it from a fresh sync context
    # (application startup). If we are already inside a running loop (tests,
    # nest_asyncio contexts), we fall back to MemorySaver to avoid cross-loop
    # binding bugs; production code paths build the graph at startup before
    # any event loop is created.
    try:
        import pathlib

        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        try:
            asyncio.get_running_loop()
            inside_loop = True
        except RuntimeError:
            inside_loop = False

        if inside_loop:
            raise RuntimeError(
                "graph compiled inside a running event loop — using in-memory "
                "checkpointer to avoid cross-loop SQLite binding"
            )

        db_path = pathlib.Path(settings.checkpoint_db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        async def _open_async_saver() -> AsyncSqliteSaver:
            conn = await aiosqlite.connect(str(db_path), check_same_thread=False)
            saver = AsyncSqliteSaver(conn)
            await saver.setup()
            return saver

        _checkpointer = asyncio.run(_open_async_saver())
        logger.info("sqlite_checkpointer_initialized", path=str(db_path))
        return _checkpointer
    except ImportError:
        logger.warning(
            "sqlite_checkpointer_not_available",
            hint="pip install langgraph-checkpoint-sqlite aiosqlite",
        )
    except RuntimeError as exc:
        logger.info("sqlite_checkpointer_skipped", reason=str(exc))
    except Exception as exc:
        logger.error("sqlite_checkpointer_failed", error=str(exc))

    # Final fallback: in-memory (conversations lost on restart)
    _checkpointer = MemorySaver()
    logger.info("memory_checkpointer_initialized")
    return _checkpointer


def build_rag_graph() -> StateGraph:
    """Build and compile the multi-agent RAG workflow graph.

    Creates a StateGraph with the following flow:
        START -> router -> security -> [proceed: retriever | blocked: END]
        retriever -> grader -> [rewrite: rewriter -> retriever | generate: synthesizer]
        synthesizer -> evaluator -> END

    Returns:
        Compiled LangGraph StateGraph ready for invocation.
    """
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("router", route_query)
    workflow.add_node("guardrails", guardrails_check)
    workflow.add_node("security", check_security)
    workflow.add_node("retriever", retrieve_documents)
    workflow.add_node("grader", grade_documents)
    workflow.add_node("rewriter", rewrite_query)
    workflow.add_node("synthesizer", synthesize_answer)
    workflow.add_node("evaluator", evaluate_response)

    # Note: All graph nodes are async functions. LangGraph natively supports
    # async node functions — when graph.ainvoke() is called, all nodes execute
    # within the same async event loop without nested loop hacks.

    # Add edges
    workflow.add_edge(START, "router")
    workflow.add_edge("router", "guardrails")

    # Guardrails conditional edge — block prompt-injection before any work
    workflow.add_conditional_edges(
        "guardrails",
        guardrails_gate,
        {
            "proceed": "security",
            "blocked": END,
        },
    )

    # Security conditional edge
    workflow.add_conditional_edges(
        "security",
        security_gate,
        {
            "proceed": "retriever",
            "blocked": END,
        },
    )

    workflow.add_edge("retriever", "grader")

    # Corrective RAG conditional edge
    workflow.add_conditional_edges(
        "grader",
        should_retry,
        {
            "rewrite": "rewriter",
            "generate": "synthesizer",
        },
    )

    # Rewriter loops back to retriever
    workflow.add_edge("rewriter", "retriever")

    workflow.add_edge("synthesizer", "evaluator")
    workflow.add_edge("evaluator", END)

    # Compile with persistent checkpointer (Postgres or Memory)
    checkpointer = _get_checkpointer()
    compiled = workflow.compile(checkpointer=checkpointer)

    logger.info("rag_graph_compiled", nodes=list(workflow.nodes.keys()))

    return compiled


def create_initial_state(
    query: str,
    user_context: UserContext,
    prefer_cloud: bool = False,
    override_provider: str = "",
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
        "audit_trail": [],
    }


async def run_rag_pipeline(
    query: str,
    user_context: UserContext,
    thread_id: str = "default",
    prefer_cloud: bool = False,
    override_provider: str = "",
) -> GraphState:
    """Execute the full RAG pipeline and return the final state.

    High-level async function that builds the graph, creates initial state,
    and invokes the workflow with checkpointing enabled.

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
    graph = build_rag_graph()
    initial_state = create_initial_state(
        query, user_context, prefer_cloud=prefer_cloud, override_provider=override_provider
    )

    config = {"configurable": {"thread_id": thread_id}}

    final_state = await graph.ainvoke(initial_state, config=config)

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
) -> AsyncGenerator[dict, None]:
    """Execute the full RAG pipeline with real token-by-token streaming of the
    synthesized answer.

    Runs all non-synthesis nodes (router, security, retriever, grader,
    optional rewrite loop), then streams synthesizer tokens to the caller,
    then runs the evaluator on the collected text.

    Event types yielded:
        {"type": "phase", "name": str, "state": dict}    -- after each non-synth node
        {"type": "blocked", "message": str, "state": dict}
        {"type": "token", "text": str}                    -- synthesis token
        {"type": "final", "state": dict, "latency_ms": float}

    Args:
        query: Natural language query.
        user_context: Authenticated user context for RBAC.
        thread_id: Thread identifier for audit/log correlation.

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

    state: dict = create_initial_state(
        query, user_context, prefer_cloud=prefer_cloud, override_provider=override_provider
    )

    # 1. Router
    _merge_update(state, await route_query(state))
    yield {"type": "phase", "name": "router", "state": dict(state)}

    # 2. Guardrails (prompt-injection)
    _merge_update(state, await guardrails_check(state))
    yield {"type": "phase", "name": "guardrails", "state": dict(state)}

    if guardrails_gate(state) == "blocked":
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        yield {
            "type": "blocked",
            "message": (
                f"Blocked by guardrails: {state.get('guardrails_reason', 'prompt_injection')}"
            ),
            "state": dict(state),
            "latency_ms": elapsed_ms,
        }
        return

    # 3. Security
    _merge_update(state, await check_security(state))
    yield {"type": "phase", "name": "security", "state": dict(state)}

    if security_gate(state) == "blocked":
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        yield {
            "type": "blocked",
            "message": state.get("security_message", "Blocked by security policy."),
            "state": dict(state),
            "latency_ms": elapsed_ms,
        }
        return

    # 3. Retrieve + grade + (optional rewrite loop)
    while True:
        _merge_update(state, await retrieve_documents(state))
        yield {"type": "phase", "name": "retriever", "state": dict(state)}

        _merge_update(state, await grade_documents(state))
        yield {"type": "phase", "name": "grader", "state": dict(state)}

        if should_retry(state) == "generate":
            break

        _merge_update(state, await rewrite_query(state))
        yield {"type": "phase", "name": "rewriter", "state": dict(state)}

    # 4. Streaming synthesis
    final_synth_event: dict | None = None
    async for event in synthesize_answer_stream(state):
        if event["type"] == "token":
            yield {"type": "token", "text": event["text"]}
        elif event["type"] == "final":
            final_synth_event = event

    if final_synth_event:
        state["generation"] = final_synth_event["generation"]
        state["citations"] = final_synth_event["citations"]
        state["confidence_score"] = final_synth_event["confidence_score"]
        state["synth_provider"] = final_synth_event.get("synth_provider", "")
        state["synth_model"] = final_synth_event.get("synth_model", "")
        state["synth_usage"] = final_synth_event.get("synth_usage", {})
        state["synth_latency_ms"] = final_synth_event.get("synth_latency_ms", 0.0)
        _apply_audit(state, [final_synth_event["audit_entry"]])

    # 5. Evaluator (runs on collected text, not streamed)
    _merge_update(state, await evaluate_response(state))

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
