"""
Tests for node/resource_manager.py — consolidated resource management.

Covers BackoffSemaphore, TimeoutEstimator, TimeoutManager, JobCleaner,
the _exponential_backoff helper, and backwards-compat shim imports.
"""

from __future__ import annotations

import asyncio
import sys
import time
import types
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# sys.modules stubs (needed because JobCleaner tests pull in node.blockchain)
# ---------------------------------------------------------------------------


def _install_stubs():
    def _stub(name):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    for mod in (
        "solana",
        "solana.rpc",
        "solana.rpc.async_api",
        "solders",
        "solders.keypair",
        "solders.pubkey",
        "anchorpy",
    ):
        _stub(mod)

    _stub("hivemind")
    _stub("hivemind.moe")
    _stub("hivemind.moe.server")
    _stub("hivemind.moe.server.layers")

    if "torch" not in sys.modules:
        torch_mod = types.ModuleType("torch")

        class _FakeTensor:
            pass

        torch_mod.Tensor = _FakeTensor
        nn_mod = types.ModuleType("torch.nn")

        class _FakeModule:
            def __init__(self, *args, **kwargs):
                pass

        nn_mod.Module = _FakeModule
        torch_mod.nn = nn_mod
        sys.modules["torch"] = torch_mod
        sys.modules["torch.nn"] = nn_mod
    else:
        _stub("torch.nn")

    for mod in (
        "transformers",
        "accelerate",
        "bitsandbytes",
        "peft",
        "lighthouseweb3",
    ):
        _stub(mod)


_install_stubs()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_job(job_id: int, deadline: float):
    from node.blockchain import OpenJob

    return OpenJob(
        job_id=job_id,
        client="client_pubkey",
        model_id=b"\x00" * 32,
        prompt_hash=b"\x00" * 32,
        prompt_cid="bafybeiabc",
        max_tokens=100,
        payment_amount=50,
        deadline=deadline,
        job_pda="job_pda_pubkey",
    )


def _make_node_with_jobs(jobs: dict):
    node = MagicMock()
    node._active_jobs = dict(jobs)
    return node


# ---------------------------------------------------------------------------
# BackoffSemaphore
# ---------------------------------------------------------------------------


def test_backoff_semaphore_acquire_release():
    """Acquire then release: value returns to the initial count."""

    async def _run():
        from node.resource_manager import BackoffSemaphore

        sem = BackoffSemaphore(2)
        assert sem.value == 2
        await sem.acquire()
        assert sem.value == 1
        await sem.release()
        assert sem.value == 2

    asyncio.run(_run())


def test_backoff_semaphore_blocks_at_capacity():
    """Two coroutines competing for 1 slot both complete; only 1 runs at a time."""
    results: list[str] = []

    async def _worker(sem, tag: str) -> None:
        from node.resource_manager import BackoffSemaphore  # noqa: F401 — type hint only

        async with sem:
            results.append(tag)
            await asyncio.sleep(0)

    async def _run():
        from node.resource_manager import BackoffConfig, BackoffSemaphore

        sem = BackoffSemaphore(1, config=BackoffConfig(initial_delay_s=0.001, max_attempts=20))
        await asyncio.gather(_worker(sem, "a"), _worker(sem, "b"))

    asyncio.run(_run())
    assert sorted(results) == ["a", "b"]


# ---------------------------------------------------------------------------
# TimeoutEstimator
# ---------------------------------------------------------------------------


def test_timeout_estimator_estimates_based_on_history():
    """After enough samples the estimate should reflect observed latencies."""
    from node.resource_manager import TimeoutEstimator

    est = TimeoutEstimator(min_timeout_ms=0.0, max_timeout_ms=float("inf"), safety_multiplier=1.0)
    for _ in range(10):
        est.record("model_a", latency_ms=200.0)
    result = est.estimate("model_a")
    # p95 of 10 identical 200ms samples = 200ms; multiplier = 1.0 → 200ms
    assert result == pytest.approx(200.0)


def test_timeout_estimator_clamps_to_min_max():
    """Estimates are clamped to [min_timeout_ms, max_timeout_ms]."""
    from node.resource_manager import TimeoutEstimator

    est_min = TimeoutEstimator(min_timeout_ms=1_000.0, safety_multiplier=1.5)
    for _ in range(10):
        est_min.record("fast_model", latency_ms=0.1)
    assert est_min.estimate("fast_model") >= 1_000.0

    est_max = TimeoutEstimator(max_timeout_ms=300_000.0, safety_multiplier=1.5)
    for _ in range(10):
        est_max.record("slow_model", latency_ms=500_000.0)
    assert est_max.estimate("slow_model") <= 300_000.0


# ---------------------------------------------------------------------------
# TimeoutManager
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_manager_raises_on_timeout():
    """A coroutine that exceeds its timeout raises JobTimeoutError."""
    from node.resource_manager import JobTimeoutError, TimeoutManager

    tm = TimeoutManager()

    async def slow():
        await asyncio.sleep(10.0)

    with pytest.raises(JobTimeoutError):
        await tm.run_with_timeout(slow(), timeout_seconds=0.05)


@pytest.mark.asyncio
async def test_timeout_manager_completes_within_limit():
    """A fast coroutine completes normally within a generous timeout."""
    from node.resource_manager import TimeoutManager

    tm = TimeoutManager()

    async def fast():
        return "ok"

    result = await tm.run_with_timeout(fast(), timeout_seconds=10.0)
    assert result == "ok"


# ---------------------------------------------------------------------------
# JobCleaner
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_job_cleaner_removes_expired_jobs():
    """Jobs well past deadline + grace period must be removed."""
    from node.resource_manager import JobCleaner

    past = time.time() - 200
    job = _make_job(job_id=1, deadline=past)
    node = _make_node_with_jobs({1: job})

    cleaner = JobCleaner(node, scan_interval_s=30, grace_period_s=60)
    count = await cleaner.scan_once()

    assert count == 1
    assert 1 not in node._active_jobs


@pytest.mark.asyncio
async def test_job_cleaner_keeps_active_jobs():
    """Jobs whose deadline is in the future must NOT be removed."""
    from node.resource_manager import JobCleaner

    future = time.time() + 300
    job = _make_job(job_id=2, deadline=future)
    node = _make_node_with_jobs({2: job})

    cleaner = JobCleaner(node, scan_interval_s=30, grace_period_s=60)
    count = await cleaner.scan_once()

    assert count == 0
    assert 2 in node._active_jobs


# ---------------------------------------------------------------------------
# _exponential_backoff helper
# ---------------------------------------------------------------------------


def test_exponential_backoff_increases_with_attempts():
    """Delay should grow strictly with attempt number (before cap)."""
    from node.resource_manager import _exponential_backoff

    delays = [_exponential_backoff(i, base=1.0, cap=1000.0) for i in range(5)]
    for i in range(len(delays) - 1):
        assert delays[i] < delays[i + 1], (
            f"delay[{i}]={delays[i]} not < delay[{i + 1}]={delays[i + 1]}"
        )


def test_exponential_backoff_capped_at_maximum():
    """Delay must never exceed the cap regardless of attempt number."""
    from node.resource_manager import _exponential_backoff

    cap = 60.0
    for attempt in range(20):
        assert _exponential_backoff(attempt, base=1.0, cap=cap) <= cap


# ---------------------------------------------------------------------------
# Backwards-compat shim imports
# ---------------------------------------------------------------------------


def test_backwards_compat_imports_still_work():
    """Importing from the original module paths must not raise ImportError."""
    from node.backoff_semaphore import BackoffConfig, BackoffSemaphore  # noqa: F401
    from node.job_cleaner import JobCleaner  # noqa: F401

    # Verify the symbols are the same objects as those in resource_manager
    from node.resource_manager import BackoffConfig as RM_BackoffConfig
    from node.resource_manager import BackoffSemaphore as RM_BackoffSemaphore
    from node.resource_manager import JobCleaner as RM_JobCleaner
    from node.resource_manager import JobTimeoutError as RM_JobTimeoutError
    from node.resource_manager import LatencySample as RM_LatencySample
    from node.resource_manager import TimeoutEstimator as RM_TimeoutEstimator
    from node.resource_manager import TimeoutManager as RM_TimeoutManager
    from node.timeout_estimator import LatencySample, TimeoutEstimator  # noqa: F401
    from node.timeout_manager import JobTimeoutError, TimeoutManager  # noqa: F401

    assert BackoffSemaphore is RM_BackoffSemaphore
    assert BackoffConfig is RM_BackoffConfig
    assert TimeoutEstimator is RM_TimeoutEstimator
    assert LatencySample is RM_LatencySample
    assert TimeoutManager is RM_TimeoutManager
    assert JobTimeoutError is RM_JobTimeoutError
    assert JobCleaner is RM_JobCleaner
