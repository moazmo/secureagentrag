"""Observability setup using Arize Phoenix for LLM tracing.

Provides OpenTelemetry-compatible distributed tracing for LLM calls,
retrieval operations, and LangGraph execution. Gracefully degrades
when Phoenix is not installed or configured.

Usage:
    Call setup_tracing() once at application startup (e.g., in app/main.py).
    All trace_* functions will automatically emit spans when tracing is enabled.
"""

from __future__ import annotations

from config.settings import settings
from utils.logging import get_logger

_log = get_logger(__name__)

# Module-level state
_tracer = None
_phoenix_configured = False
_phoenix_project_name: str = settings.app_name


def setup_tracing() -> bool:
    """Initialize Phoenix tracing if ``settings.phoenix_endpoint`` is set.

    This function is safe to call unconditionally at startup — it will
    log a message and return immediately if Phoenix is not configured.
    Tracing failures never crash the application.

    Returns:
        True if tracing was successfully enabled, False otherwise.
    """
    global _tracer, _phoenix_configured, _phoenix_project_name

    if not settings.phoenix_endpoint:
        _log.info("phoenix_tracing_disabled", reason="No phoenix_endpoint configured")
        return False

    try:
        from phoenix.otel import register

        tracer_provider = register(
            project_name=settings.app_name,
            endpoint=settings.phoenix_endpoint,
        )

        # Attempt to instrument LLM and retrieval calls
        _instrument_providers()

        _phoenix_configured = True
        _phoenix_project_name = settings.app_name
        _log.info(
            "phoenix_tracing_enabled",
            endpoint=settings.phoenix_endpoint,
            project=settings.app_name,
            tracer_provider=str(tracer_provider),
        )
        return True
    except ImportError:
        _log.warning(
            "phoenix_import_failed",
            msg=(
                "arize-phoenix not installed; tracing unavailable. "
                "Install with: pip install 'arize-phoenix-otel'"
            ),
        )
        return False
    except Exception as exc:
        _log.error(
            "phoenix_tracing_init_error",
            error=str(exc),
            endpoint=settings.phoenix_endpoint,
        )
        return False


def _instrument_providers() -> None:
    """Instrument LLM and retrieval providers with OpenTelemetry.

    Attempts to auto-instrument supported providers. Failures are
    logged but never raised — partial instrumentation is acceptable.
    """
    # Instrument LangChain/LangGraph if available
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor

        LangChainInstrumentor().instrument()
        _log.info("instrumented_langchain")
    except ImportError:
        _log.debug(
            "langchain_instrumentation_skipped",
            reason="openinference-instrumentation-langchain not installed",
        )
    except Exception as exc:
        _log.debug("langchain_instrumentation_error", reason=str(exc))

    # Instrument OpenAI-compatible calls if available
    try:
        from openinference.instrumentation.openai import OpenAIInstrumentor

        OpenAIInstrumentor().instrument()
        _log.info("instrumented_openai")
    except ImportError:
        _log.debug(
            "openai_instrumentation_skipped",
            reason="openinference-instrumentation-openai not installed",
        )
    except Exception as exc:
        _log.debug("openai_instrumentation_error", reason=str(exc))


def trace_llm_call(
    provider: str,
    model: str,
    prompt: str,
    response: str,
    latency_ms: float,
    tokens: dict[str, int] | None = None,
) -> None:
    """Record a manual trace span for an LLM call.

    Can be used as an explicit trace point when auto-instrumentation
    is unavailable or for custom tracking.

    Args:
        provider: LLM provider name (e.g., "ollama", "groq").
        model: Model identifier used for generation.
        prompt: The input prompt text.
        response: The generated response text.
        latency_ms: Response latency in milliseconds.
        tokens: Optional token usage dict with keys like
            "prompt_tokens", "completion_tokens", "total_tokens".
    """
    if not _phoenix_configured:
        return

    try:
        from opentelemetry import trace

        tracer = trace.get_tracer("secureagentrag.llm")
        with tracer.start_as_current_span("llm_call") as span:
            span.set_attribute("llm.provider", provider)
            span.set_attribute("llm.model", model)
            span.set_attribute("llm.prompt_length", len(prompt))
            span.set_attribute("llm.response_length", len(response))
            span.set_attribute("llm.latency_ms", latency_ms)
            if tokens:
                for key, value in tokens.items():
                    span.set_attribute(f"llm.tokens.{key}", value)
    except Exception as exc:
        _log.debug("trace_llm_call_failed", error=str(exc))


def trace_retrieval(
    query: str,
    num_results: int,
    latency_ms: float,
    method: str = "hybrid",
) -> None:
    """Record a manual trace span for a retrieval operation.

    Args:
        query: The search query string.
        num_results: Number of results returned.
        latency_ms: Retrieval latency in milliseconds.
        method: Retrieval method used ("hybrid", "dense", "bm25").
    """
    if not _phoenix_configured:
        return

    try:
        from opentelemetry import trace

        tracer = trace.get_tracer("secureagentrag.retrieval")
        with tracer.start_as_current_span("retrieval") as span:
            span.set_attribute("retrieval.query_length", len(query))
            span.set_attribute("retrieval.num_results", num_results)
            span.set_attribute("retrieval.latency_ms", latency_ms)
            span.set_attribute("retrieval.method", method)
    except Exception as exc:
        _log.debug("trace_retrieval_failed", error=str(exc))


def trace_graph_execution(
    query: str,
    nodes_executed: list[str],
    total_latency_ms: float,
    final_confidence: float,
    retries: int = 0,
) -> None:
    """Record a manual trace span for LangGraph pipeline execution.

    Args:
        query: The original user query.
        nodes_executed: List of graph node names that were executed.
        total_latency_ms: Total pipeline execution time in milliseconds.
        final_confidence: Final confidence score of the generated answer.
        retries: Number of corrective retrieval retries performed.
    """
    if not _phoenix_configured:
        return

    try:
        from opentelemetry import trace

        tracer = trace.get_tracer("secureagentrag.graph")
        with tracer.start_as_current_span("graph_execution") as span:
            span.set_attribute("graph.query_length", len(query))
            span.set_attribute("graph.nodes_executed", ",".join(nodes_executed))
            span.set_attribute("graph.total_latency_ms", total_latency_ms)
            span.set_attribute("graph.confidence", final_confidence)
            span.set_attribute("graph.retries", retries)
    except Exception as exc:
        _log.debug("trace_graph_execution_failed", error=str(exc))


def get_trace_url() -> str | None:
    """Return the Phoenix dashboard URL if tracing is configured.

    Returns:
        Phoenix UI URL string, or None if Phoenix is not configured.
    """
    if not _phoenix_configured or not settings.phoenix_endpoint:
        return None

    # Phoenix UI typically runs on the same host
    endpoint = settings.phoenix_endpoint.rstrip("/")
    # Replace gRPC/collector port with UI port if needed
    if ":4317" in endpoint:
        return endpoint.replace(":4317", ":6006")
    if ":6006" in endpoint:
        return endpoint
    return endpoint


def is_tracing_enabled() -> bool:
    """Check if Phoenix tracing is currently active.

    Returns:
        True if tracing was successfully configured, False otherwise.
    """
    return _phoenix_configured
