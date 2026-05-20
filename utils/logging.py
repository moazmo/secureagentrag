"""Structured logging configuration using structlog.

Provides JSON output for production deployments and pretty console output
for local development, controlled by ``settings.debug``.

Includes correlation ID support for distributed request tracing across
all services and components.
"""

from __future__ import annotations

import contextlib
import io
import logging
import sys
import uuid
from contextlib import contextmanager
from contextvars import ContextVar

import structlog

from config.settings import settings

# Switch stdout/stderr to UTF-8 with replacement on import so any logger
# (even before setup_logging() runs) survives Arabic / CJK / emoji content.
# Without this, Windows cp1252 stdout aborts log.emit and bubbles a
# "'charmap' codec can't encode" error into the calling function, which
# we previously saw crashing retrieve_documents for Arabic queries.
for _stream_name in ("stdout", "stderr"):
    _s = getattr(sys, _stream_name, None)
    if _s is not None and hasattr(_s, "reconfigure"):
        with contextlib.suppress(Exception):
            _s.reconfigure(encoding="utf-8", errors="replace")


def _utf8_stream(stream):
    """Wrap a stream in a UTF-8 writer that replaces (not raises) on
    unencodable chars.

    On Windows the default stdout codec is cp1252, so Arabic / CJK / emoji
    in log payloads crashes ``StreamHandler.emit`` mid-write — which we saw
    surface as ``"'charmap' codec can't encode characters"`` errors that
    aborted retrieve_documents whenever the query contained non-Latin text.
    """
    try:
        return io.TextIOWrapper(
            stream.buffer,
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
            write_through=True,
        )
    except Exception:
        return stream


# Context variable for the current correlation ID
_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def get_correlation_id() -> str | None:
    """Get the current correlation ID for this request context.

    Returns:
        The current correlation ID string, or None if not set.
    """
    return _correlation_id.get()


def set_correlation_id(cid: str | None = None) -> str:
    """Set (or generate) a correlation ID for the current context.

    Args:
        cid: Optional correlation ID. If not provided, a UUID is generated.

    Returns:
        The correlation ID that was set.
    """
    new_cid = cid or str(uuid.uuid4())[:16]
    _correlation_id.set(new_cid)
    return new_cid


@contextmanager
def correlation_id_scope(cid: str | None = None):
    """Context manager for correlation ID scoping.

    Automatically sets and cleans up the correlation ID.

    Args:
        cid: Optional correlation ID. Auto-generated if not provided.

    Yields:
        The correlation ID string.
    """
    new_cid = set_correlation_id(cid)
    try:
        yield new_cid
    finally:
        _correlation_id.set(None)


def _add_correlation_id(logger, method_name, event_dict):
    """Structlog processor that injects the correlation ID into log events."""
    cid = _correlation_id.get()
    if cid:
        event_dict["correlation_id"] = cid
    return event_dict


def setup_logging() -> None:
    """Initialize structlog and stdlib logging with environment-appropriate renderers.

    Call this once at application startup (e.g., in ``app/main.py``).
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        _add_correlation_id,
    ]

    if settings.debug:
        # Human-readable colored output for development
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer(
            colors=sys.stderr.isatty()
        )
    else:
        # Structured JSON for production (easy to ingest into log aggregators)
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(_utf8_stream(sys.stdout))
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(settings.log_level.upper())

    # Pin chatty third-party loggers to WARNING. httpx in particular emits a
    # request-level INFO line for every HTTP call, which under Streamlit's
    # thread model races with the script-runner closing stdout and surfaces
    # as cosmetic 'ValueError: I/O operation on closed file' tracebacks.
    for noisy in ("httpx", "httpcore", "openai", "anthropic", "groq", "qdrant_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a named structlog logger instance.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A bound structlog logger.
    """
    return structlog.get_logger(name)


# Initialise logging at import time. Modules instantiated at *import time*
# (e.g. ``utils.conversation_store.conversation_store``, ``utils.audit.audit_logger``)
# call ``get_logger(...).info(...)`` before any application entry point gets
# a chance to call setup_logging(). Without this bootstrap structlog's
# default ``PrintLoggerFactory`` writes raw bytes to sys.stdout, which
# Streamlit's stdout capture treats as ``OSError [Errno 22]`` on Windows.
# Running setup_logging() unconditionally here is idempotent — the second
# call from app/main.py is a harmless re-configure.
with contextlib.suppress(Exception):
    setup_logging()
