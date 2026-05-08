"""
Tests for node/job_cleaner.py — stale job cleanup.

All heavy dependencies are stubbed via sys.modules so the suite runs without
a real blockchain, torch, or IPFS installation.
"""

import asyncio
import sys
import time
import types
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# sys.modules stubs — installed once before any node imports
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
    """Return an OpenJob-like object with a specific deadline."""
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
    """Return a minimal fake Node whose _active_jobs is pre-populated."""
    node = MagicMock()
    node._active_jobs = dict(jobs)
    return node


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestScanCleansExpiredJob:
    """A job well past its deadline + grace period must be removed."""

    @pytest.mark.asyncio
    async def test_scan_cleans_expired_job(self):
        from node.job_cleaner import JobCleaner

        # Deadline far in the past — clearly stale.
        past = time.time() - 200
        job = _make_job(job_id=1, deadline=past)
        node = _make_node_with_jobs({1: job})

        cleaner = JobCleaner(node, scan_interval_s=30, grace_period_s=60)
        await cleaner.scan_once()

        assert 1 not in node._active_jobs, "Expired job should have been removed"


class TestScanSkipsActiveJob:
    """A job whose deadline is in the future must NOT be removed."""

    @pytest.mark.asyncio
    async def test_scan_skips_active_job(self):
        from node.job_cleaner import JobCleaner

        future = time.time() + 300
        job = _make_job(job_id=2, deadline=future)
        node = _make_node_with_jobs({2: job})

        cleaner = JobCleaner(node, scan_interval_s=30, grace_period_s=60)
        await cleaner.scan_once()

        assert 2 in node._active_jobs, "Active job should not have been removed"


class TestScanReturnsCount:
    """scan_once() must return the number of jobs cleaned."""

    @pytest.mark.asyncio
    async def test_scan_returns_count(self):
        from node.job_cleaner import JobCleaner

        past = time.time() - 200
        future = time.time() + 300
        jobs = {
            1: _make_job(job_id=1, deadline=past),
            2: _make_job(job_id=2, deadline=future),
        }
        node = _make_node_with_jobs(jobs)

        cleaner = JobCleaner(node, scan_interval_s=30, grace_period_s=60)
        count = await cleaner.scan_once()

        assert count == 1, f"Expected 1 cleaned, got {count}"


class TestStatsUpdatedAfterScan:
    """After scan_once(), stats must reflect the latest scan."""

    @pytest.mark.asyncio
    async def test_stats_updated_after_scan(self):
        from node.job_cleaner import JobCleaner

        past = time.time() - 200
        job = _make_job(job_id=3, deadline=past)
        node = _make_node_with_jobs({3: job})

        cleaner = JobCleaner(node, scan_interval_s=30, grace_period_s=60)

        assert cleaner.stats["total_cleaned"] == 0
        assert cleaner.stats["last_scan_time"] == 0.0

        before = time.time()
        await cleaner.scan_once()
        after = time.time()

        assert cleaner.stats["total_cleaned"] == 1
        assert before <= cleaner.stats["last_scan_time"] <= after


class TestGracePeriodRespected:
    """A job past its deadline but within the grace window must NOT be cleaned."""

    @pytest.mark.asyncio
    async def test_grace_period_respected(self):
        from node.job_cleaner import JobCleaner

        # Deadline just 10 seconds ago — still within 60-second grace period.
        slightly_past = time.time() - 10
        job = _make_job(job_id=4, deadline=slightly_past)
        node = _make_node_with_jobs({4: job})

        cleaner = JobCleaner(node, scan_interval_s=30, grace_period_s=60)
        count = await cleaner.scan_once()

        assert count == 0, "Job within grace period should not be cleaned"
        assert 4 in node._active_jobs, "Job within grace period should still be present"


class TestRunLoopsUntilCancelled:
    """run() should keep calling scan_once() until the task is cancelled."""

    @pytest.mark.asyncio
    async def test_run_loops_until_cancelled(self):
        from node.job_cleaner import JobCleaner

        node = _make_node_with_jobs({})
        cleaner = JobCleaner(node, scan_interval_s=0.01)

        scan_call_count = 0
        original_scan = cleaner.scan_once

        async def counting_scan():
            nonlocal scan_call_count
            scan_call_count += 1
            return await original_scan()

        cleaner.scan_once = counting_scan

        task = asyncio.create_task(cleaner.run())
        # Allow a couple of iterations.
        await asyncio.sleep(0.05)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert scan_call_count >= 2, (
            f"Expected at least 2 scan calls before cancel, got {scan_call_count}"
        )


class TestMultipleExpiredJobsAllCleaned:
    """All stale jobs must be removed in a single scan pass."""

    @pytest.mark.asyncio
    async def test_multiple_expired_jobs_all_cleaned(self):
        from node.job_cleaner import JobCleaner

        past = time.time() - 200
        jobs = {
            10: _make_job(job_id=10, deadline=past),
            11: _make_job(job_id=11, deadline=past),
            12: _make_job(job_id=12, deadline=past),
        }
        node = _make_node_with_jobs(jobs)

        cleaner = JobCleaner(node, scan_interval_s=30, grace_period_s=60)
        count = await cleaner.scan_once()

        assert count == 3, f"Expected 3 cleaned, got {count}"
        assert len(node._active_jobs) == 0, "All stale jobs should have been removed"
        assert cleaner.stats["total_cleaned"] == 3
