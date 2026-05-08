"""Tests for the async circuit breaker."""

import asyncio
import time

import pytest

from node.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _succeed(value=42):
    return value


async def _fail(exc=None):
    raise (exc or ValueError("boom"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_closed_state_allows_calls():
    """Successful calls go through when circuit is CLOSED."""
    cb = CircuitBreaker(name="test")
    result = asyncio.run(cb.call(lambda: _succeed(99)))
    assert result == 99
    assert cb.state == CircuitState.CLOSED


def test_opens_after_failure_threshold():
    """N failures cause the circuit to transition to OPEN."""
    config = CircuitBreakerConfig(failure_threshold=3)
    cb = CircuitBreaker(config, name="test")

    for _ in range(3):
        with pytest.raises(ValueError):
            asyncio.run(cb.call(lambda: _fail()))

    assert cb._state == CircuitState.OPEN


def test_open_raises_circuit_open_error():
    """When circuit is OPEN, calls raise CircuitOpenError without invoking the factory."""
    config = CircuitBreakerConfig(failure_threshold=2)
    cb = CircuitBreaker(config, name="test")

    for _ in range(2):
        with pytest.raises(ValueError):
            asyncio.run(cb.call(lambda: _fail()))

    called = []

    async def _probe():
        called.append(True)
        return 1

    with pytest.raises(CircuitOpenError):
        asyncio.run(cb.call(_probe))

    assert called == [], "factory must not be called when circuit is OPEN"


def test_transitions_to_half_open():
    """After timeout elapses, OPEN circuit transitions to HALF_OPEN."""
    config = CircuitBreakerConfig(failure_threshold=1, timeout_seconds=0.05)
    cb = CircuitBreaker(config, name="test")

    with pytest.raises(ValueError):
        asyncio.run(cb.call(lambda: _fail()))

    assert cb._state == CircuitState.OPEN

    time.sleep(0.1)  # wait for timeout

    assert cb.state == CircuitState.HALF_OPEN


def test_half_open_success_closes():
    """Enough successes in HALF_OPEN close the circuit."""
    config = CircuitBreakerConfig(failure_threshold=1, success_threshold=2, timeout_seconds=0.05)
    cb = CircuitBreaker(config, name="test")

    with pytest.raises(ValueError):
        asyncio.run(cb.call(lambda: _fail()))

    time.sleep(0.1)
    assert cb.state == CircuitState.HALF_OPEN

    asyncio.run(cb.call(lambda: _succeed()))
    assert cb._state == CircuitState.HALF_OPEN  # still need one more

    asyncio.run(cb.call(lambda: _succeed()))
    assert cb._state == CircuitState.CLOSED


def test_half_open_failure_reopens():
    """A failure in HALF_OPEN sends the circuit back to OPEN."""
    config = CircuitBreakerConfig(failure_threshold=1, timeout_seconds=0.05)
    cb = CircuitBreaker(config, name="test")

    with pytest.raises(ValueError):
        asyncio.run(cb.call(lambda: _fail()))

    time.sleep(0.1)
    assert cb.state == CircuitState.HALF_OPEN

    with pytest.raises(ValueError):
        asyncio.run(cb.call(lambda: _fail()))

    assert cb._state == CircuitState.OPEN


def test_non_matching_exception_not_counted():
    """An exception not in config.exceptions does not count as a failure."""

    class SpecialError(Exception):
        pass

    class OtherError(Exception):
        pass

    config = CircuitBreakerConfig(failure_threshold=2, exceptions=(SpecialError,))
    cb = CircuitBreaker(config, name="test")

    # OtherError should NOT increment failure count
    with pytest.raises(OtherError):
        asyncio.run(cb.call(lambda: _fail(OtherError("not tracked"))))

    assert cb._failure_count == 0
    assert cb._state == CircuitState.CLOSED

    # SpecialError SHOULD increment failure count
    with pytest.raises(SpecialError):
        asyncio.run(cb.call(lambda: _fail(SpecialError("tracked"))))

    assert cb._failure_count == 1


def test_reset_closes_circuit():
    """reset() transitions the circuit to CLOSED regardless of current state."""
    config = CircuitBreakerConfig(failure_threshold=1)
    cb = CircuitBreaker(config, name="test")

    with pytest.raises(ValueError):
        asyncio.run(cb.call(lambda: _fail()))

    assert cb._state == CircuitState.OPEN

    cb.reset()

    assert cb._state == CircuitState.CLOSED
    assert cb._failure_count == 0
    assert cb._success_count == 0


def test_stats_returns_dict():
    """stats() returns a dict with the required keys."""
    cb = CircuitBreaker(name="my-service")
    s = cb.stats()

    assert isinstance(s, dict)
    assert s["name"] == "my-service"
    assert s["state"] == CircuitState.CLOSED.value
    assert "failure_count" in s
    assert "success_count" in s


def test_success_resets_failure_count():
    """A success in CLOSED state resets the failure count."""
    config = CircuitBreakerConfig(failure_threshold=5)
    cb = CircuitBreaker(config, name="test")

    # Accumulate some failures (below threshold)
    for _ in range(3):
        with pytest.raises(ValueError):
            asyncio.run(cb.call(lambda: _fail()))

    assert cb._failure_count == 3

    # One success should reset the failure count
    asyncio.run(cb.call(lambda: _succeed()))
    assert cb._failure_count == 0
    assert cb._state == CircuitState.CLOSED
