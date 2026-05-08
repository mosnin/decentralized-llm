"""Per-key token-bucket rate limiter (stdlib only)."""

import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class RateLimitConfig:
    requests_per_minute: int = 60
    burst_size: int = 10  # Max tokens in bucket


@dataclass
class BucketState:
    tokens: float
    last_refill: float = field(default_factory=time.monotonic)


class RateLimiter:
    """Per-key token-bucket rate limiter."""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._buckets: dict[str, BucketState] = {}
        self._lock = asyncio.Lock()

    async def is_allowed(self, key: str) -> bool:
        """Return True if request is allowed; False if rate-limited."""
        async with self._lock:
            if key not in self._buckets:
                self._buckets[key] = BucketState(tokens=self.config.burst_size)

            bucket = self._buckets[key]
            self._refill(bucket)

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True
            return False

    def _refill(self, bucket: BucketState) -> None:
        """Add tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - bucket.last_refill
        refill = elapsed * (self.config.requests_per_minute / 60.0)
        bucket.tokens = min(self.config.burst_size, bucket.tokens + refill)
        bucket.last_refill = now

    def reset(self, key: str) -> None:
        """Clear bucket for a key (for testing)."""
        self._buckets.pop(key, None)
