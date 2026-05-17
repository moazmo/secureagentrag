"""LangGraph graph compilation and execution."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from config.settings import settings
from core.agents.evaluator import evaluate_response
from core.agents.retriever import grade_documents, retrieve_documents, should_retry
from core.agents.router import rewrite_query, route_query
from core.agents.security import check_security, security_gate
from core.agents.synthesizer import synthesize_answer
from core.state import GraphState
from utils.logging import get_logger
from utils.observability import trace_graph_execution

if TYPE_CHECKING:
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

    # Try SQLite for persistent local checkpointing (no external DB needed)
    try:
        import pathlib

        from langgraph.checkpoint.sqlite import SqliteSaver

        db_path = pathlib.Path("data/checkpoints.sqlite")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _checkpointer = SqliteSaver.from_conn_string(str(db_path))
        logger.info("sqlite_checkpointer_initialized", path=str(db_path))
        return _checkpointer
    except ImportError:
        logger.warning(
            "sqlite_checkpointer_not_available",
            hint="pip install langgraph-checkpoint-sqlite",
        )
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
    workflow.add_edge("router", "security")

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


def create_initial_state(query: str, user_context: UserContext) -> GraphState:
    """Create the proper initial state dict for graph invocation.

    Serializes UserContext to dict and initializes all required state fields.

    Args:
        query: The user's natural language query.
        user_context: Authenticated user context for RBAC.

    Returns:
        GraphState dict ready to pass to graph.invoke() or graph.ainvoke().
    """
    return {
        "query": query,
        "user_context": user_context.model_dump(),
        "query_type": "",
        "rewritten_query": "",
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
        "needs_human_review": False,
        "evaluation_notes": "",
        "audit_trail": [],
    }


async def run_rag_pipeline(
    query: str,
    user_context: UserContext,
    thread_id: str = "default",
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
    initial_state = create_initial_state(query, user_context)

    config = {"configurable": {"thread_id": thread_id}}

    final_state = await graph.ainvoke(initial_state, config=config)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Extract executed nodes from audit trail
    nodes_executed = [
        entry["node"]
        for entry in final_state.get("audit_trail", [])
        if "node" in entry
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
