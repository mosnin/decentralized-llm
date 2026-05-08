"""Tests for exponential backoff retry utilities."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from client.python.retry import RetryableError, RetryConfig, with_retry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _make_coro(value):
    """Return a coroutine factory that succeeds immediately."""
    return value


def _succeeding_factory(value="ok"):
    """Return a coro_factory that always returns value."""
    return lambda: asyncio.coroutine(lambda: value)()


def _factory_from_side_effects(side_effects):
    """
    Return a coro_factory whose successive calls raise/return side_effects items.
    Each item is either an Exception instance (to raise) or a plain value (to return).
    """
    it = iter(side_effects)

    async def _coro():
        item = next(it)
        if isinstance(item, BaseException):
            raise item
        return item

    return _coro


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRetryConfigDefaults:
    def test_retry_config_defaults(self):
        cfg = RetryConfig()
        assert cfg.max_attempts == 3
        assert cfg.base_delay_s == 1.0
        assert cfg.max_delay_s == 30.0
        assert cfg.exponential_base == 2.0
        assert cfg.jitter is True


class TestWithRetry:
    async def test_succeeds_on_first_attempt(self):
        """Should return immediately without any sleep when first call succeeds."""
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await with_retry(lambda: _make_coro("hello"), config=RetryConfig())
        assert result == "hello"
        mock_sleep.assert_not_called()

    async def test_retries_on_transient_error(self):
        """Should retry after a retryable exception and return on second call."""
        calls = _factory_from_side_effects([TimeoutError("boom"), "success"])

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await with_retry(
                calls,
                config=RetryConfig(max_attempts=3, base_delay_s=0.0, jitter=False),
                retryable_exceptions=(TimeoutError,),
            )
        assert result == "success"

    async def test_raises_after_max_attempts(self):
        """Should raise the last exception once all attempts are exhausted."""
        calls = _factory_from_side_effects([OSError("err1"), OSError("err2"), OSError("err3")])

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(OSError, match="err3"):
                await with_retry(
                    calls,
                    config=RetryConfig(max_attempts=3, base_delay_s=0.0, jitter=False),
                    retryable_exceptions=(OSError,),
                )

    async def test_delay_increases_exponentially(self):
        """Sleep durations should follow base * exponential_base**(attempt-1)."""
        calls = _factory_from_side_effects([TimeoutError(), TimeoutError(), TimeoutError(), "ok"])

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await with_retry(
                calls,
                config=RetryConfig(
                    max_attempts=4,
                    base_delay_s=1.0,
                    max_delay_s=100.0,
                    exponential_base=2.0,
                    jitter=False,
                ),
                retryable_exceptions=(TimeoutError,),
            )

        assert result == "ok"
        sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]
        # attempt 1 → delay 1.0*2^0=1.0, attempt 2 → 1.0*2^1=2.0, attempt 3 → 1.0*2^2=4.0
        assert sleep_calls == pytest.approx([1.0, 2.0, 4.0])

    async def test_jitter_applied(self):
        """With jitter=True, the actual sleep should differ from the base delay."""
        errors = [TimeoutError()] * 2 + ["ok"]
        calls = _factory_from_side_effects(errors)

        sleep_values: list[float] = []

        async def capture_sleep(delay):
            sleep_values.append(delay)

        with patch("asyncio.sleep", side_effect=capture_sleep):
            await with_retry(
                calls,
                config=RetryConfig(
                    max_attempts=3,
                    base_delay_s=10.0,
                    max_delay_s=100.0,
                    exponential_base=2.0,
                    jitter=True,
                ),
                retryable_exceptions=(TimeoutError,),
            )

        # With jitter the delays should not be exact powers but within ±20%
        assert len(sleep_values) == 2
        assert sleep_values[0] != pytest.approx(10.0, abs=0)  # unlikely to be exactly 10
        assert 8.0 <= sleep_values[0] <= 12.0  # base_delay * (1 ± 0.2)
        assert 16.0 <= sleep_values[1] <= 24.0  # base_delay * 2 * (1 ± 0.2)

    async def test_no_jitter_when_disabled(self):
        """With jitter=False, sleep values should be exactly the computed delays."""
        calls = _factory_from_side_effects([TimeoutError(), TimeoutError(), "ok"])

        sleep_values: list[float] = []

        async def capture_sleep(delay):
            sleep_values.append(delay)

        with patch("asyncio.sleep", side_effect=capture_sleep):
            await with_retry(
                calls,
                config=RetryConfig(
                    max_attempts=3,
                    base_delay_s=2.0,
                    max_delay_s=100.0,
                    exponential_base=3.0,
                    jitter=False,
                ),
                retryable_exceptions=(TimeoutError,),
            )

        assert sleep_values == pytest.approx([2.0, 6.0])  # 2*3^0, 2*3^1

    async def test_non_retryable_exception_not_retried(self):
        """A ValueError (not in retryable_exceptions) should propagate immediately."""
        call_count = 0

        async def _coro():
            nonlocal call_count
            call_count += 1
            raise ValueError("not retryable")

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(ValueError, match="not retryable"):
                await with_retry(
                    _coro,
                    config=RetryConfig(max_attempts=5),
                    retryable_exceptions=(TimeoutError, OSError),
                )

        # Only one call should have been made (no retry)
        assert call_count == 1
        mock_sleep.assert_not_called()

    async def test_retryable_error_wrapper(self):
        """RetryableError cause should be retried and re-raised on exhaustion."""

        async def _coro():
            try:
                raise OSError("inner")
            except OSError as exc:
                raise RetryableError("transient") from exc

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(OSError, match="inner"):
                await with_retry(
                    _coro,
                    config=RetryConfig(max_attempts=2, base_delay_s=0.0, jitter=False),
                    # OSError NOT listed — but RetryableError handles it
                    retryable_exceptions=(TimeoutError,),
                )

    async def test_max_delay_capped(self):
        """Delay should never exceed max_delay_s."""
        calls = _factory_from_side_effects([TimeoutError(), TimeoutError(), TimeoutError(), "ok"])

        sleep_values: list[float] = []

        async def capture_sleep(delay):
            sleep_values.append(delay)

        with patch("asyncio.sleep", side_effect=capture_sleep):
            await with_retry(
                calls,
                config=RetryConfig(
                    max_attempts=4,
                    base_delay_s=10.0,
                    max_delay_s=15.0,
                    exponential_base=2.0,
                    jitter=False,
                ),
                retryable_exceptions=(TimeoutError,),
            )

        # 10.0, min(20.0, 15.0)=15.0, min(40.0, 15.0)=15.0
        assert sleep_values == pytest.approx([10.0, 15.0, 15.0])
