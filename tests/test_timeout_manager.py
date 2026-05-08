"""Tests for node.timeout_manager."""

import asyncio
import time

import pytest

from node.timeout_manager import JobTimeoutError, TimeoutManager


@pytest.fixture
def tm():
    return TimeoutManager()


# ---------------------------------------------------------------------------
# run_with_deadline
# ---------------------------------------------------------------------------


async def test_run_with_deadline_completes_before_deadline(tm):
    """A fast coroutine should complete normally when deadline is in the future."""

    async def fast():
        return "done"

    deadline = time.time() + 10.0
    result = await tm.run_with_deadline(fast(), deadline)
    assert result == "done"


async def test_run_with_deadline_raises_on_expiry(tm):
    """A slow coroutine should raise JobTimeoutError when deadline is in the past."""

    async def slow():
        await asyncio.sleep(5.0)
        return "never"

    # Deadline already expired
    deadline = time.time() - 1.0
    with pytest.raises(JobTimeoutError):
        await tm.run_with_deadline(slow(), deadline)


async def test_run_with_deadline_raises_when_coro_too_slow(tm):
    """A coroutine that runs longer than the remaining deadline should time out."""

    async def slow():
        await asyncio.sleep(10.0)
        return "never"

    deadline = time.time() + 0.05  # only 50 ms remaining
    with pytest.raises(JobTimeoutError):
        await tm.run_with_deadline(slow(), deadline)


# ---------------------------------------------------------------------------
# run_with_timeout
# ---------------------------------------------------------------------------


async def test_run_with_timeout_completes(tm):
    """A fast coroutine should complete within a generous timeout."""

    async def fast():
        return 42

    result = await tm.run_with_timeout(fast(), timeout_seconds=10.0)
    assert result == 42


async def test_run_with_timeout_raises(tm):
    """A coroutine that exceeds the timeout_seconds should raise JobTimeoutError."""

    async def slow():
        await asyncio.sleep(10.0)

    with pytest.raises(JobTimeoutError):
        await tm.run_with_timeout(slow(), timeout_seconds=0.05)


# ---------------------------------------------------------------------------
# seconds_remaining
# ---------------------------------------------------------------------------


def test_seconds_remaining_positive():
    """Should return a positive value when the deadline is in the future."""
    deadline = time.time() + 100.0
    remaining = TimeoutManager.seconds_remaining(deadline)
    assert remaining > 0.0
    assert remaining <= 100.0


def test_seconds_remaining_zero_when_past():
    """Should return 0.0 when the deadline has already passed."""
    past_deadline = time.time() - 5.0
    remaining = TimeoutManager.seconds_remaining(past_deadline)
    assert remaining == 0.0


# ---------------------------------------------------------------------------
# is_expired
# ---------------------------------------------------------------------------


def test_is_expired_false_when_future():
    """Should return False when the deadline is still in the future."""
    deadline = time.time() + 100.0
    assert TimeoutManager.is_expired(deadline) is False


def test_is_expired_true_when_past():
    """Should return True when the deadline has already passed."""
    past_deadline = time.time() - 1.0
    assert TimeoutManager.is_expired(past_deadline) is True
