"""Token-bucket rate limiter for API request throttling.

Provides per-user and per-endpoint rate limiting to prevent abuse and
ensure fair resource allocation. Uses an in-memory token bucket algorithm
with optional Redis backend for distributed deployments.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for a rate limit bucket.

    Attributes:
        requests_per_minute: Maximum requests allowed per minute.
        burst_size: Maximum burst capacity (bucket size).
        cooldown_seconds: Seconds to wait after being rate limited.
    """

    requests_per_minute: int = 60
    burst_size: int = 10
    cooldown_seconds: float = 1.0


@dataclass
class TokenBucket:
    """In-memory token bucket for rate limiting.

    Attributes:
        tokens: Current available tokens.
        last_update: Timestamp of last token refill.
        config: Rate limit configuration.
        blocked_until: Timestamp when the bucket is unblocked.
    """

    tokens: float = field(default=0.0)
    last_update: float = field(default_factory=time.time)
    config: RateLimitConfig = field(default_factory=RateLimitConfig)
    blocked_until: float = field(default=0.0)

    def _refill(self) -> None:
        """Refill tokens based on elapsed time since last update."""
        now = time.time()
        elapsed = now - self.last_update
        # Refill rate: tokens per second
        refill_rate = self.config.requests_per_minute / 60.0
        self.tokens = min(self.config.burst_size, self.tokens + elapsed * refill_rate)
        self.last_update = now

    def consume(self, tokens: float = 1.0) -> tuple[bool, dict[str, Any]]:
        """Attempt to consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume (default 1 per request).

        Returns:
            Tuple of (allowed, metadata) where metadata contains
            remaining tokens, retry_after, etc.
        """
        now = time.time()

        # Check if currently blocked
        if now < self.blocked_until:
            retry_after = int(self.blocked_until - now) + 1
            return False, {
                "allowed": False,
                "remaining": 0,
                "retry_after": retry_after,
                "reason": "cooldown_active",
            }

        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            remaining = int(self.tokens)
            return True, {
                "allowed": True,
                "remaining": remaining,
                "retry_after": 0,
                "reason": None,
            }

        # Rate limit exceeded — enter cooldown
        self.blocked_until = now + self.config.cooldown_seconds
        retry_after = int(self.config.cooldown_seconds) + 1
        return False, {
            "allowed": False,
            "remaining": 0,
            "retry_after": retry_after,
            "reason": "rate_limit_exceeded",
        }


class RateLimiter:
    """Multi-key rate limiter with per-user and per-endpoint tracking.

    Uses in-memory token buckets. For distributed deployments, wrap
    with a Redis-backed implementation.

    Args:
        default_config: Default rate limit configuration.
    """

    def __init__(self, default_config: RateLimitConfig | None = None) -> None:
        """Initialize the rate limiter.

        Args:
            default_config: Default configuration for new buckets.
        """
        self._default_config = default_config or RateLimitConfig()
        self._buckets: dict[str, TokenBucket] = {}

    def _get_bucket(self, key: str, config: RateLimitConfig | None = None) -> TokenBucket:
        """Get or create a token bucket for the given key.

        Args:
            key: Unique identifier for the bucket (e.g., user_id + endpoint).
            config: Optional custom configuration.

        Returns:
            The token bucket for the key.
        """
        if key not in self._buckets:
            self._buckets[key] = TokenBucket(
                tokens=config.burst_size if config else self._default_config.burst_size,
                config=config or self._default_config,
            )
        return self._buckets[key]

    def check_rate_limit(
        self,
        key: str,
        tokens: float = 1.0,
        config: RateLimitConfig | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Check if a request is within the rate limit.

        Args:
            key: Rate limit bucket key (e.g., "user_123:query").
            tokens: Tokens to consume.
            config: Optional custom config for this key.

        Returns:
            Tuple of (allowed, metadata).
        """
        bucket = self._get_bucket(key, config)
        allowed, metadata = bucket.consume(tokens)

        if not allowed:
            logger.warning(
                "rate_limit_exceeded",
                key=key,
                retry_after=metadata["retry_after"],
                reason=metadata["reason"],
            )
        else:
            logger.debug(
                "rate_limit_allowed",
                key=key,
                remaining=metadata["remaining"],
            )

        return allowed, metadata

    def is_allowed(self, key: str, tokens: float = 1.0) -> bool:
        """Simple check — returns True if request is allowed.

        Args:
            key: Rate limit bucket key.
            tokens: Tokens to consume.

        Returns:
            True if within rate limit, False otherwise.
        """
        allowed, _ = self.check_rate_limit(key, tokens)
        return allowed

    def get_status(self, key: str) -> dict[str, Any]:
        """Get current rate limit status for a key.

        Args:
            key: Rate limit bucket key.

        Returns:
            Dict with remaining tokens, reset time, etc.
        """
        bucket = self._buckets.get(key)
        if not bucket:
            return {
                "remaining": self._default_config.burst_size,
                "limit": self._default_config.requests_per_minute,
                "reset": 0,
            }

        bucket._refill()
        return {
            "remaining": int(bucket.tokens),
            "limit": bucket.config.requests_per_minute,
            "reset": int(max(0, bucket.blocked_until - time.time())),
        }

    def reset(self, key: str) -> None:
        """Reset a specific rate limit bucket.

        Args:
            key: Bucket key to reset.
        """
        if key in self._buckets:
            del self._buckets[key]
            logger.info("rate_limit_reset", key=key)


class OwnerKeyHourThrottle:
    """Per-IP hourly throttle for the BYOK owner-key fallback.

    Distinct from the request-level :class:`RateLimiter` because the BYOK
    semantics are different:

    - Visitors who bring their own LLM key (``ByokCreds.has_user_key()``)
      bypass this throttle entirely — they are paying for their own tokens.
    - Visitors who do NOT bring a key fall back to the platform owner's
      Groq key. This throttle exists to stop a single recruiter or curious
      visitor from burning the free-tier 30 RPM / 14,400 RPD budget.

    Bucket window is rolling one hour from the first allowed request in
    the window. Sliding-window precision is not needed — three requests an
    hour is already conservative. We keep timestamps in a tiny list per IP
    and prune entries older than 3600 seconds on each check.
    """

    __slots__ = ("_buckets", "_quota_per_hour")

    def __init__(self, quota_per_hour: int) -> None:
        if quota_per_hour < 0:
            raise ValueError("quota_per_hour must be non-negative")
        self._quota_per_hour = quota_per_hour
        self._buckets: dict[str, list[float]] = {}

    def allow(self, ip: str, *, now: float | None = None) -> tuple[bool, dict[str, Any]]:
        """Return whether ``ip`` may consume one owner-key request.

        Args:
            ip: Client IP address (use ``"anon"`` when unavailable so the
                fallback path still throttles instead of leaking quota).
            now: Optional monotonic clock override for tests.

        Returns:
            ``(allowed, meta)`` where ``meta`` carries ``remaining`` and
            ``retry_after`` seconds, ready for an HTTP 429 response.
        """
        t = now if now is not None else time.monotonic()
        # Prune entries older than 1h, then count.
        bucket = [ts for ts in self._buckets.get(ip, []) if t - ts < 3600.0]
        if len(bucket) >= self._quota_per_hour:
            # ``retry_after`` defaults to a full window when quota_per_hour=0
            # (kill switch) — there is no "oldest entry" to expire.
            retry_after = max(1, int(3600.0 - (t - bucket[0])) + 1) if bucket else 3600
            self._buckets[ip] = bucket  # write pruned list back
            return False, {
                "allowed": False,
                "remaining": 0,
                "retry_after": retry_after,
                "reason": "owner_key_hourly_quota_exhausted",
            }
        bucket.append(t)
        self._buckets[ip] = bucket
        return True, {
            "allowed": True,
            "remaining": self._quota_per_hour - len(bucket),
            "retry_after": 0,
            "reason": None,
        }

    def reset(self, ip: str) -> None:
        """Drop all timestamps for ``ip`` (test/cleanup helper)."""
        self._buckets.pop(ip, None)

    def reset_all(self) -> None:
        """Drop every bucket — used between test cases to avoid leakage."""
        self._buckets.clear()


# Module-level singleton — lazy-initialised from settings on first use so
# unit tests that monkey-patch SAR_BYOK_OWNER_QUOTA see the right value.
_owner_key_throttle: OwnerKeyHourThrottle | None = None


def get_owner_key_throttle() -> OwnerKeyHourThrottle:
    """Return the process-wide owner-key throttle, creating it lazily.

    Reads ``settings.byok_owner_key_quota_per_hour`` at first call. Tests
    that need a different quota value should call :func:`reset_owner_key_throttle`
    after the monkey-patch.
    """
    global _owner_key_throttle
    if _owner_key_throttle is None:
        from config.settings import settings  # local import to avoid cycle

        _owner_key_throttle = OwnerKeyHourThrottle(
            quota_per_hour=settings.byok_owner_key_quota_per_hour,
        )
    return _owner_key_throttle


def reset_owner_key_throttle() -> None:
    """Force the next :func:`get_owner_key_throttle` call to rebuild from settings.

    Test-only hook; production code never calls this.
    """
    global _owner_key_throttle
    _owner_key_throttle = None


class RedisRateLimiter:
    """Distributed rate limiter backed by Redis.

    Uses Redis sorted sets with sliding window algorithm for accurate
    per-user rate limiting across multiple application instances.

    Args:
        redis_url: Redis connection URL.
        default_config: Default rate limit configuration.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        default_config: RateLimitConfig | None = None,
    ) -> None:
        """Initialize the Redis rate limiter.

        Args:
            redis_url: Redis connection URL. Falls back to settings.
            default_config: Default configuration for new keys.
        """
        import redis

        from config.settings import settings

        self._redis = redis.from_url(redis_url or settings.redis_url)
        self._default_config = default_config or RateLimitConfig()

    def check_rate_limit(
        self,
        key: str,
        tokens: float = 1.0,
        config: RateLimitConfig | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Check if a request is within the rate limit using Redis.

        Uses a sliding window algorithm based on Redis sorted sets.

        Args:
            key: Rate limit bucket key.
            tokens: Tokens to consume.
            config: Optional custom config.

        Returns:
            Tuple of (allowed, metadata).
        """
        cfg = config or self._default_config
        now = time.time()
        window_start = now - 60.0  # 1-minute window
        redis_key = f"ratelimit:{key}"

        # Remove old entries outside the window
        self._redis.zremrangebyscore(redis_key, 0, window_start)

        # Count current requests in window
        current_count = self._redis.zcard(redis_key)

        # Check burst limit
        if current_count >= cfg.burst_size:
            retry_after = int(cfg.cooldown_seconds) + 1
            return False, {
                "allowed": False,
                "remaining": 0,
                "retry_after": retry_after,
                "reason": "rate_limit_exceeded",
            }

        # Check per-minute rate
        rpm_limit = cfg.requests_per_minute
        if current_count >= rpm_limit:
            retry_after = int(60 - (now % 60)) + 1
            return False, {
                "allowed": False,
                "remaining": 0,
                "retry_after": retry_after,
                "reason": "rate_limit_exceeded",
            }

        # Record this request
        self._redis.zadd(redis_key, {str(now): now})
        # Set expiry on the key
        self._redis.expire(redis_key, 120)

        remaining = min(cfg.burst_size, rpm_limit) - current_count - 1
        return True, {
            "allowed": True,
            "remaining": max(0, remaining),
            "retry_after": 0,
            "reason": None,
        }

    def is_allowed(self, key: str, tokens: float = 1.0) -> bool:
        """Simple check — returns True if request is allowed.

        Args:
            key: Rate limit bucket key.
            tokens: Tokens to consume.

        Returns:
            True if within rate limit, False otherwise.
        """
        allowed, _ = self.check_rate_limit(key, tokens)
        return allowed

    def get_status(self, key: str) -> dict[str, Any]:
        """Get current rate limit status for a key.

        Args:
            key: Rate limit bucket key.

        Returns:
            Dict with remaining tokens, limit, and reset time.
        """
        redis_key = f"ratelimit:{key}"
        now = time.time()
        window_start = now - 60.0
        self._redis.zremrangebyscore(redis_key, 0, window_start)
        current_count = self._redis.zcard(redis_key)
        remaining = max(0, self._default_config.burst_size - current_count)
        return {
            "remaining": remaining,
            "limit": self._default_config.requests_per_minute,
            "reset": int(60 - (now % 60)),
        }

    def reset(self, key: str) -> None:
        """Reset a specific rate limit bucket.

        Args:
            key: Bucket key to reset.
        """
        self._redis.delete(f"ratelimit:{key}")
        logger.info("redis_rate_limit_reset", key=key)


def _get_rate_limiter() -> RateLimiter | RedisRateLimiter:
    """Get the appropriate rate limiter based on configuration.

    Returns:
        RateLimiter (in-memory) or RedisRateLimiter (distributed).
    """
    from config.settings import settings

    if settings.use_redis_rate_limiter:
        try:
            return RedisRateLimiter()
        except Exception as exc:
            logger.warning("redis_rate_limiter_failed", error=str(exc), fallback="memory")
    return RateLimiter(default_config=RATE_LIMIT_PROFILES["default"])


# Pre-configured rate limit profiles
RATE_LIMIT_PROFILES: dict[str, RateLimitConfig] = {
    "default": RateLimitConfig(requests_per_minute=60, burst_size=10),
    "strict": RateLimitConfig(requests_per_minute=10, burst_size=3, cooldown_seconds=5.0),
    "generous": RateLimitConfig(requests_per_minute=300, burst_size=50),
    "upload": RateLimitConfig(requests_per_minute=5, burst_size=2, cooldown_seconds=10.0),
    "query": RateLimitConfig(requests_per_minute=30, burst_size=5, cooldown_seconds=2.0),
}

# Module-level singleton (lazy initialization)
_rate_limiter_instance: RateLimiter | RedisRateLimiter | None = None


def _get_limiter() -> RateLimiter | RedisRateLimiter:
    """Get the singleton rate limiter instance.

    Returns:
        The configured rate limiter.
    """
    global _rate_limiter_instance
    if _rate_limiter_instance is None:
        _rate_limiter_instance = _get_rate_limiter()
    return _rate_limiter_instance


def check_query_rate_limit(user_id: str) -> tuple[bool, dict[str, Any]]:
    """Check rate limit for a user query.

    Args:
        user_id: The user making the query.

    Returns:
        Tuple of (allowed, metadata).
    """
    key = f"{user_id}:query"
    return _get_limiter().check_rate_limit(key, config=RATE_LIMIT_PROFILES["query"])


def check_upload_rate_limit(user_id: str) -> tuple[bool, dict[str, Any]]:
    """Check rate limit for a document upload.

    Args:
        user_id: The user uploading.

    Returns:
        Tuple of (allowed, metadata).
    """
    key = f"{user_id}:upload"
    return _get_limiter().check_rate_limit(key, config=RATE_LIMIT_PROFILES["upload"])
