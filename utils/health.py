"""Health check utilities for service dependency monitoring.

Provides async health checks for Qdrant, Ollama, PostgreSQL, and Redis
with configurable timeouts, status aggregation, and result caching
to avoid repeated expensive checks within short time windows.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx

from config.settings import settings
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class HealthStatus:
    """Health status for a single service."""

    name: str
    healthy: bool
    latency_ms: float = 0.0
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthReport:
    """Aggregated health report for all services."""

    overall_healthy: bool
    services: list[HealthStatus]
    checked_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert report to a JSON-serializable dict."""
        return {
            "overall_healthy": self.overall_healthy,
            "checked_at": self.checked_at,
            "services": [
                {
                    "name": s.name,
                    "healthy": s.healthy,
                    "latency_ms": round(s.latency_ms, 2),
                    "message": s.message,
                    "metadata": s.metadata,
                }
                for s in self.services
            ],
        }


# Health check cache (defined after dataclasses to avoid forward reference issues)
_health_cache: dict[str, tuple[HealthStatus, float]] = {}
_health_cache_ttl_seconds: float = 15.0  # Cache health results for 15s


def _get_cached_status(name: str) -> HealthStatus | None:
    """Return cached health status if still valid."""
    if name in _health_cache:
        status, cached_at = _health_cache[name]
        if time.time() - cached_at < _health_cache_ttl_seconds:
            return status
    return None


def _set_cached_status(status: HealthStatus) -> None:
    """Cache a health status result."""
    _health_cache[status.name] = (status, time.time())


async def _check_qdrant(timeout: float = 5.0) -> HealthStatus:
    """Check Qdrant vector store connectivity."""
    cached = _get_cached_status("qdrant")
    if cached:
        return cached

    start = time.perf_counter()
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=int(timeout),
        )
        collections = client.get_collections()
        latency_ms = (time.perf_counter() - start) * 1000
        status = HealthStatus(
            name="qdrant",
            healthy=True,
            latency_ms=latency_ms,
            message=f"Connected. {len(collections.collections)} collections.",
        )
        _set_cached_status(status)
        return status
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        status = HealthStatus(
            name="qdrant",
            healthy=False,
            latency_ms=latency_ms,
            message=f"Connection failed: {exc!s}",
        )
        _set_cached_status(status)
        return status


async def _check_ollama(timeout: float = 5.0) -> HealthStatus:
    """Check Ollama local inference service."""
    cached = _get_cached_status("ollama")
    if cached:
        return cached

    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{settings.ollama_url}/api/tags")
            latency_ms = (time.perf_counter() - start) * 1000
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                status = HealthStatus(
                    name="ollama",
                    healthy=True,
                    latency_ms=latency_ms,
                    message=f"Running. {len(models)} models available.",
                    metadata={"model_count": len(models)},
                )
                _set_cached_status(status)
                return status
            status = HealthStatus(
                name="ollama",
                healthy=False,
                latency_ms=latency_ms,
                message=f"HTTP {response.status_code}",
            )
            _set_cached_status(status)
            return status
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        status = HealthStatus(
            name="ollama",
            healthy=False,
            latency_ms=latency_ms,
            message=f"Connection failed: {exc!s}",
        )
        _set_cached_status(status)
        return status


async def _check_postgres(timeout: float = 5.0) -> HealthStatus:
    """Check PostgreSQL connectivity."""
    cached = _get_cached_status("postgres")
    if cached:
        return cached

    start = time.perf_counter()
    if not settings.postgres_url:
        status = HealthStatus(
            name="postgres",
            healthy=True,
            latency_ms=0.0,
            message="Not configured (using in-memory checkpoints).",
        )
        _set_cached_status(status)
        return status

    try:
        import psycopg

        with (
            psycopg.connect(settings.postgres_url, connect_timeout=int(timeout)) as conn,
            conn.cursor() as cur,
        ):
            cur.execute("SELECT 1")
            cur.fetchone()
        latency_ms = (time.perf_counter() - start) * 1000
        status = HealthStatus(
            name="postgres",
            healthy=True,
            latency_ms=latency_ms,
            message="Connected and responding.",
        )
        _set_cached_status(status)
        return status
    except ImportError:
        status = HealthStatus(
            name="postgres",
            healthy=True,
            latency_ms=0.0,
            message="psycopg not installed (checkpoints use MemorySaver).",
        )
        _set_cached_status(status)
        return status
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        status = HealthStatus(
            name="postgres",
            healthy=False,
            latency_ms=latency_ms,
            message=f"Connection failed: {exc!s}",
        )
        _set_cached_status(status)
        return status


async def _check_redis(timeout: float = 5.0) -> HealthStatus:
    """Check Redis connectivity."""
    cached = _get_cached_status("redis")
    if cached:
        return cached

    start = time.perf_counter()
    if not settings.redis_url:
        status = HealthStatus(
            name="redis",
            healthy=True,
            latency_ms=0.0,
            message="Not configured (using in-memory rate limiting).",
        )
        _set_cached_status(status)
        return status

    try:
        import redis

        r = redis.from_url(settings.redis_url, socket_connect_timeout=int(timeout))
        r.ping()
        latency_ms = (time.perf_counter() - start) * 1000
        info = r.info("server")
        status = HealthStatus(
            name="redis",
            healthy=True,
            latency_ms=latency_ms,
            message=f"Connected. Redis v{info.get('redis_version', 'unknown')}.",
            metadata={"redis_version": info.get("redis_version", "unknown")},
        )
        _set_cached_status(status)
        return status
    except ImportError:
        status = HealthStatus(
            name="redis",
            healthy=True,
            latency_ms=0.0,
            message="redis-py not installed (using in-memory fallbacks).",
        )
        _set_cached_status(status)
        return status
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        status = HealthStatus(
            name="redis",
            healthy=False,
            latency_ms=latency_ms,
            message=f"Connection failed: {exc!s}",
        )
        _set_cached_status(status)
        return status


async def run_health_checks(
    include_qdrant: bool = True,
    include_ollama: bool = True,
    include_postgres: bool = True,
    include_redis: bool = True,
    timeout: float = 5.0,
) -> HealthReport:
    """Run all configured health checks in parallel.

    Args:
        include_qdrant: Whether to check Qdrant.
        include_ollama: Whether to check Ollama.
        include_postgres: Whether to check PostgreSQL.
        include_redis: Whether to check Redis.
        timeout: Per-check timeout in seconds.

    Returns:
        Aggregated HealthReport with all service statuses.
    """
    checks = []
    if include_qdrant:
        checks.append(_check_qdrant(timeout))
    if include_ollama:
        checks.append(_check_ollama(timeout))
    if include_postgres:
        checks.append(_check_postgres(timeout))
    if include_redis:
        checks.append(_check_redis(timeout))

    services = await asyncio.gather(*checks, return_exceptions=True)

    # Handle any exceptions that weren't caught inside individual checks
    processed: list[HealthStatus] = []
    for svc in services:
        if isinstance(svc, Exception):
            processed.append(
                HealthStatus(
                    name="unknown",
                    healthy=False,
                    message=f"Check crashed: {svc!s}",
                )
            )
        else:
            processed.append(svc)

    overall = all(s.healthy for s in processed)
    return HealthReport(overall_healthy=overall, services=processed)
