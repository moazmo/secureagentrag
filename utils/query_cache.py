"""Query result caching with Redis fallback to in-memory.

Caches RAG pipeline results to avoid redundant LLM calls and retrieval
for identical queries from the same user. Uses Redis when available for
distributed caching across multiple app instances.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from config.settings import settings
from utils.logging import get_logger
from utils.pii import redact_dict

logger = get_logger(__name__)

# In-memory fallback cache
_memory_cache: dict[str, tuple[dict[str, Any], float]] = {}
_memory_cache_ttl_seconds: float = 300.0  # 5 minutes default

# Redis singleton
_redis_client = None

# Cache metrics counters
_cache_hits: int = 0
_cache_misses: int = 0


def get_cache_metrics() -> dict[str, int]:
    """Return cache hit/miss counters."""
    total = _cache_hits + _cache_misses
    return {
        "hits": _cache_hits,
        "misses": _cache_misses,
        "total": total,
        "hit_rate": round(_cache_hits / total, 4) if total > 0 else 0.0,
    }


def reset_cache_metrics() -> None:
    """Reset cache hit/miss counters."""
    global _cache_hits, _cache_misses
    _cache_hits = 0
    _cache_misses = 0


def _get_redis_client():
    """Lazy-initialize Redis client for query caching.

    Returns:
        Redis client instance or None if unavailable.
    """
    global _redis_client
    if _redis_client is not None:
        return _redis_client

    if not settings.redis_url:
        return None

    try:
        import redis

        _redis_client = redis.from_url(settings.redis_url, decode_responses=True)
        _redis_client.ping()
        logger.info("query_cache_redis_connected")
        return _redis_client
    except ImportError:
        logger.debug("redis_not_installed_for_query_cache")
    except Exception as exc:
        logger.warning("query_cache_redis_connection_failed", error=str(exc))

    _redis_client = False  # Mark as unavailable
    return None


def _user_prefix(user_id: str) -> str:
    """Stable per-user key prefix so ``invalidate_user_cache`` can scan by user."""
    return hashlib.sha256(user_id.encode()).hexdigest()[:12]


def _build_cache_key(user_id: str, query: str, context_hash: str = "") -> str:
    """Build a deterministic cache key from user + query.

    The key is ``<user_prefix><body_hash>`` so a single user's entries share a
    common prefix — that is what makes ``invalidate_user_cache`` work (a hash of
    one string is never a prefix of a hash of a different string, so the old
    ``startswith(sha256(user_id))`` scan silently matched nothing).

    Args:
        user_id: The user's identifier.
        query: The query text.
        context_hash: Optional hash of additional context (model, filters, etc.).

    Returns:
        A hash string suitable for use as a cache key.
    """
    body = f"{query.lower().strip()}:{context_hash}"
    body_hash = hashlib.sha256(body.encode()).hexdigest()[:20]
    return f"{_user_prefix(user_id)}{body_hash}"


def get_cached_result(
    user_id: str,
    query: str,
    context_hash: str = "",
    ttl_seconds: float | None = None,
) -> dict[str, Any] | None:
    """Retrieve a cached query result if available and not expired.

    Args:
        user_id: The user's identifier.
        query: The query text.
        context_hash: Optional hash of additional context.
        ttl_seconds: Cache TTL. Defaults to settings or 300s.

    Returns:
        Cached result dict, or None if not found or expired.
    """
    cache_key = _build_cache_key(user_id, query, context_hash)
    _ = ttl_seconds or _memory_cache_ttl_seconds

    global _cache_hits, _cache_misses

    # Try Redis first
    redis_client = _get_redis_client()
    if redis_client:
        try:
            cached = redis_client.get(f"rag:query:{cache_key}")
            if cached:
                result = json.loads(cached)
                _cache_hits += 1
                logger.info("query_cache_hit", source="redis", user_id=user_id)
                return result
        except Exception as exc:
            logger.debug("query_cache_redis_read_failed", error=str(exc))

    # Fallback to in-memory
    if cache_key in _memory_cache:
        result, expiry = _memory_cache[cache_key]
        if time.time() < expiry:
            _cache_hits += 1
            logger.info("query_cache_hit", source="memory", user_id=user_id)
            return result
        # Expired — clean up
        del _memory_cache[cache_key]

    _cache_misses += 1
    return None


def set_cached_result(
    user_id: str,
    query: str,
    result: dict[str, Any],
    context_hash: str = "",
    ttl_seconds: float | None = None,
) -> None:
    """Store a query result in the cache.

    Args:
        user_id: The user's identifier.
        query: The query text.
        result: The result dict to cache.
        context_hash: Optional hash of additional context.
        ttl_seconds: Cache TTL. Defaults to settings or 300s.
    """
    cache_key = _build_cache_key(user_id, query, context_hash)
    ttl = ttl_seconds or _memory_cache_ttl_seconds

    # Serialize result (exclude non-serializable fields) + redact PII before
    # persistence so disk/Redis never sees emails, phones, card numbers, etc.
    serializable_result = redact_dict(_make_serializable(result))

    # Try Redis first
    redis_client = _get_redis_client()
    if redis_client:
        try:
            redis_client.setex(
                f"rag:query:{cache_key}",
                int(ttl),
                json.dumps(serializable_result),
            )
            logger.info("query_cache_stored", source="redis", user_id=user_id)
            return
        except Exception as exc:
            logger.debug("query_cache_redis_write_failed", error=str(exc))

    # Fallback to in-memory
    _memory_cache[cache_key] = (serializable_result, time.time() + ttl)
    logger.info("query_cache_stored", source="memory", user_id=user_id)

    # Prune memory cache if too large
    if len(_memory_cache) > 1000:
        _prune_memory_cache()


def _make_serializable(obj: Any) -> Any:
    """Convert an object to a JSON-serializable form.

    Args:
        obj: Object to serialize.

    Returns:
        JSON-serializable representation.
    """
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)


def _prune_memory_cache() -> None:
    """Remove expired entries from the in-memory cache."""
    now = time.time()
    expired_keys = [k for k, (_, expiry) in _memory_cache.items() if expiry < now]
    for k in expired_keys:
        del _memory_cache[k]

    # If still too large, remove oldest
    if len(_memory_cache) > 1000:
        sorted_items = sorted(_memory_cache.items(), key=lambda x: x[1][1])
        for k, _ in sorted_items[:100]:
            del _memory_cache[k]


def invalidate_user_cache(user_id: str) -> int:
    """Invalidate all cached queries for a specific user.

    Args:
        user_id: The user's identifier.

    Returns:
        Number of entries invalidated.
    """
    count = 0

    # In-memory — keys are namespaced ``<user_prefix><body_hash>`` so a single
    # user's entries share this prefix (see _build_cache_key).
    prefix = _user_prefix(user_id)
    keys_to_remove = [k for k in _memory_cache if k.startswith(prefix)]
    for k in keys_to_remove:
        del _memory_cache[k]
        count += 1

    # Redis — scan for user-specific keys
    redis_client = _get_redis_client()
    if redis_client:
        try:
            pattern = "rag:query:*"
            for key in redis_client.scan_iter(match=pattern, count=100):
                # Best-effort: we can't easily decode the key back to user_id
                # So we just clear all query cache entries
                redis_client.delete(key)
                count += 1
        except Exception as exc:
            logger.debug("query_cache_redis_invalidate_failed", error=str(exc))

    logger.info("query_cache_invalidated", user_id=user_id, count=count)
    return count


def clear_all_cache() -> int:
    """Clear all query caches (memory + Redis).

    Returns:
        Number of entries cleared.
    """
    count = len(_memory_cache)
    _memory_cache.clear()

    redis_client = _get_redis_client()
    if redis_client:
        try:
            pattern = "rag:query:*"
            for key in redis_client.scan_iter(match=pattern, count=100):
                redis_client.delete(key)
                count += 1
        except Exception as exc:
            logger.debug("query_cache_redis_clear_failed", error=str(exc))

    logger.info("query_cache_cleared_all", count=count)
    return count
