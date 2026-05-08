"""Tests for BackoffSemaphore."""

import asyncio

from node.backoff_semaphore import BackoffConfig, BackoffSemaphore

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_acquire_and_release():
    """Acquire then release: value returns to the initial count."""

    async def _run():
        sem = BackoffSemaphore(2)
        assert sem.value == 2
        await sem.acquire()
        assert sem.value == 1
        await sem.release()
        assert sem.value == 2

    asyncio.run(_run())


def test_context_manager():
    """async with BackoffSemaphore works and releases on exit."""

    async def _run():
        sem = BackoffSemaphore(1)
        async with sem:
            assert sem.value == 0
        assert sem.value == 1

    asyncio.run(_run())


def test_contention_eventually_resolves():
    """Two coroutines competing for 1 slot both complete successfully."""
    results = []

    async def _worker(sem: BackoffSemaphore, tag: str) -> None:
        async with sem:
            results.append(tag)
            await asyncio.sleep(0)  # yield to let other task contend

    async def _run():
        sem = BackoffSemaphore(1, config=BackoffConfig(initial_delay_s=0.001, max_attempts=20))
        await asyncio.gather(_worker(sem, "a"), _worker(sem, "b"))

    asyncio.run(_run())
    assert sorted(results) == ["a", "b"]


def test_config_defaults():
    """BackoffConfig() has the documented default values."""
    cfg = BackoffConfig()
    assert cfg.initial_delay_s == 0.01
    assert cfg.max_delay_s == 5.0
    assert cfg.multiplier == 2.0
    assert cfg.max_attempts == 10


def test_value_decrements_on_acquire():
    """value property decreases by 1 after each acquire."""

    async def _run():
        sem = BackoffSemaphore(3)
        assert sem.value == 3
        await sem.acquire()
        assert sem.value == 2
        await sem.acquire()
        assert sem.value == 1
        await sem.acquire()
        assert sem.value == 0
        # Release all
        await sem.release()
        await sem.release()
        await sem.release()
        assert sem.value == 3

    asyncio.run(_run())
