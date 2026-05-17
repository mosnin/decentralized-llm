"""Tests for the token-bucket rate limiter."""

import asyncio
import time

import pytest

from node.rate_limiter import RateLimitConfig, RateLimiter


@pytest.fixture
def limiter():
    config = RateLimitConfig(requests_per_minute=60, burst_size=5)
    return RateLimiter(config)


class TestRateLimiter:
    def test_allows_requests_within_limit(self, limiter):
        """N < burst_size requests should all be allowed."""

        async def _run():
            results = [await limiter.is_allowed("user1") for _ in range(5)]
            return results

        results = asyncio.run(_run())
        assert all(results), "All requests within burst_size should be allowed"

    def test_blocks_when_bucket_empty(self, limiter):
        """Exhaust the bucket, next request should be blocked."""

        async def _run():
            # Drain all 5 tokens
            for _ in range(5):
                await limiter.is_allowed("user1")
            # Next one should be blocked
            return await limiter.is_allowed("user1")

        result = asyncio.run(_run())
        assert result is False, "Request after bucket exhausted should be blocked"

    def test_refills_over_time(self):
        """After tokens consumed, time passes, refill allows again."""
        # Use a high RPM config so a small sleep refills enough
        config = RateLimitConfig(requests_per_minute=600, burst_size=2)
        limiter = RateLimiter(config)

        async def _run():
            # Drain the bucket
            await limiter.is_allowed("user1")
            await limiter.is_allowed("user1")
            blocked = await limiter.is_allowed("user1")
            assert blocked is False

            # Manually manipulate last_refill to simulate time passing
            bucket = limiter._buckets["user1"]
            bucket.last_refill = time.monotonic() - 1.0  # 1 second ago → 10 tokens at 600 RPM

            # Should be allowed now
            return await limiter.is_allowed("user1")

        result = asyncio.run(_run())
        assert result is True, "Request should be allowed after refill"

    def test_different_keys_independent(self, limiter):
        """Exhausting key 'a' must not affect key 'b'."""

        async def _run():
            # Exhaust key "a"
            for _ in range(5):
                await limiter.is_allowed("a")
            blocked_a = await limiter.is_allowed("a")
            # Key "b" should still be fine
            allowed_b = await limiter.is_allowed("b")
            return blocked_a, allowed_b

        blocked_a, allowed_b = asyncio.run(_run())
        assert blocked_a is False
        assert allowed_b is True

    def test_reset_clears_bucket(self, limiter):
        """reset() should restore full capacity for a key."""

        async def _run():
            # Drain key
            for _ in range(5):
                await limiter.is_allowed("user1")
            assert await limiter.is_allowed("user1") is False
            limiter.reset("user1")
            return await limiter.is_allowed("user1")

        result = asyncio.run(_run())
        assert result is True, "Request after reset should be allowed"

    def test_burst_size_capped(self):
        """Bucket tokens should never exceed burst_size."""
        config = RateLimitConfig(requests_per_minute=60, burst_size=5)
        limiter = RateLimiter(config)

        async def _run():
            # Create a bucket entry
            await limiter.is_allowed("user1")
            bucket = limiter._buckets["user1"]
            # Push last_refill far into the past to simulate long idle time
            bucket.last_refill = time.monotonic() - 1000.0
            bucket.tokens = 0.0
            # Trigger a refill via is_allowed
            await limiter.is_allowed("user1")
            return bucket.tokens

        tokens = asyncio.run(_run())
        # After consuming one token the bucket should be at burst_size - 1
        assert tokens <= config.burst_size, "Tokens must not exceed burst_size"

    def test_reset_on_unknown_key_is_noop(self, limiter):
        """reset() on a key that doesn't exist should not raise."""
        limiter.reset("nonexistent")  # Should not raise
