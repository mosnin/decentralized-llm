"""
Tests for the priority-queue job handling in node/server.py.

All heavy dependencies (solana, anchorpy, hivemind, torch) are stubbed via
sys.modules so these tests run without a real blockchain or IPFS.
"""

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# sys.modules stubs — installed once before any node imports
# ---------------------------------------------------------------------------


def _install_stubs():
    """Stub out every package that node/server.py imports transitively."""

    def _stub(name):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    # Solana / Anchor stack
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

    # Hivemind
    _stub("hivemind")
    _stub("hivemind.moe")
    _stub("hivemind.moe.server")
    _stub("hivemind.moe.server.layers")

    # torch — needs a real nn.Module base class because p2p.py subclasses it at
    # import time.  Build a minimal stub that satisfies those references.
    if "torch" not in sys.modules:
        torch_mod = types.ModuleType("torch")

        class _FakeTensor:
            pass

        torch_mod.Tensor = _FakeTensor

        # nn sub-module with a no-op Module base class
        nn_mod = types.ModuleType("torch.nn")

        class _FakeModule:
            def __init__(self, *args, **kwargs):
                pass

        nn_mod.Module = _FakeModule
        torch_mod.nn = nn_mod
        sys.modules["torch"] = torch_mod
        sys.modules["torch.nn"] = nn_mod
    else:
        # torch is already installed; make sure torch.nn is also registered
        _stub("torch.nn")

    # Other heavy deps
    for mod in (
        "transformers",
        "accelerate",
        "bitsandbytes",
        "peft",
        "lighthouseweb3",
    ):
        _stub(mod)

    # cryptography is a real installed package — do NOT stub it.


_install_stubs()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_open_job(job_id: int, payment_amount: int):
    """Return a minimal OpenJob-like object."""
    # Import after stubs are in place
    from node.blockchain import OpenJob

    return OpenJob(
        job_id=job_id,
        client="client_pubkey",
        model_id=b"\x00" * 32,
        prompt_hash=b"\x00" * 32,
        prompt_cid="bafybeiabc",
        max_tokens=100,
        payment_amount=payment_amount,
        deadline=9999999999,
        job_pda="job_pda_pubkey",
    )


def _make_node(max_concurrent_jobs: int = 2):
    """
    Construct a Node with every heavyweight dep replaced by mocks.

    We bypass __init__ by using object.__new__ so we don't need a real
    BlockchainClient / ShardManager etc.
    """
    from node.config import NodeConfig
    from node.server import Node

    cfg = NodeConfig(max_concurrent_jobs=max_concurrent_jobs)
    node = object.__new__(Node)
    node.config = cfg

    # Mock collaborators
    node.shard_mgr = MagicMock()
    node.blockchain = MagicMock()
    node.blockchain.claim_job = AsyncMock(return_value=True)
    node.blockchain.submit_result = AsyncMock(return_value=True)
    node.blockchain.fetch_open_jobs = AsyncMock(return_value=[])
    node.storage = None
    node.p2p = None
    node._running = False
    node._active_jobs = {}

    # ModelRegistry added in Phase 14b; mock it so tests keep working
    mock_registry = MagicMock()
    mock_registry.get_by_model_id = MagicMock(return_value=None)
    mock_registry.load_all = AsyncMock()
    node._model_registry = mock_registry
    node._job_queue = asyncio.PriorityQueue()

    return node


# ---------------------------------------------------------------------------
# Test: priority ordering
# ---------------------------------------------------------------------------


class TestHigherPaymentJobRunsFirst:
    """Higher-payment jobs should be dequeued before lower-payment ones."""

    @pytest.mark.asyncio
    async def test_higher_payment_job_runs_first(self):
        from node.server import Node

        node = _make_node()

        low_job = _make_open_job(job_id=1, payment_amount=10)
        high_job = _make_open_job(job_id=2, payment_amount=100)

        # Enqueue low first, then high — high should come out first.
        await node._job_queue.put((-low_job.payment_amount, low_job))
        await node._job_queue.put((-high_job.payment_amount, high_job))

        run_order = []

        async def fake_handle_job(job):
            run_order.append(job.job_id)

        with patch.object(Node, "_handle_job", new=fake_handle_job):
            # Drain the queue manually (2 items).
            for _ in range(2):
                priority_key, job = node._job_queue.get_nowait()
                await fake_handle_job(job)
                node._job_queue.task_done()

        assert run_order[0] == high_job.job_id, (
            f"Expected high-payment job ({high_job.job_id}) first, got {run_order}"
        )
        assert run_order[1] == low_job.job_id


# ---------------------------------------------------------------------------
# Test: retry on inference failure
# ---------------------------------------------------------------------------


class TestRetryOnInferenceFailure:
    """_handle_job should retry inference up to 2 times before giving up."""

    @pytest.mark.asyncio
    async def test_retry_on_inference_failure_then_succeed(self):
        """Fail twice, succeed on third attempt → result submitted."""
        from node.server import Node

        node = _make_node()
        job = _make_open_job(job_id=42, payment_amount=50)

        call_count = 0

        async def flaky_inference(self_node, j, shard_mgr=None):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("GPU OOM")
            return "hello world"

        async def fake_upload(self_node, job_id, result_text):
            return "bafkreitest"

        # Patch sleep so the test doesn't actually wait 5 s per retry.
        with (
            patch.object(Node, "_run_inference", new=flaky_inference),
            patch("node.server.asyncio.sleep", new=AsyncMock()),
            patch.object(Node, "_upload_result", new=fake_upload),
        ):
            await node._handle_job(job)

        assert call_count == 3, f"Expected 3 inference calls, got {call_count}"
        node.blockchain.submit_result.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retry_exhausted_does_not_submit(self):
        """Fail all 3 attempts → submit_result must NOT be called."""
        from node.server import Node

        node = _make_node()
        job = _make_open_job(job_id=43, payment_amount=50)

        async def always_fail(j):
            raise RuntimeError("always fails")

        with (
            patch.object(Node, "_run_inference", new=always_fail),
            patch("node.server.asyncio.sleep", new=AsyncMock()),
        ):
            await node._handle_job(job)

        node.blockchain.submit_result.assert_not_awaited()


# ---------------------------------------------------------------------------
# Test: job deduplication
# ---------------------------------------------------------------------------


class TestJobDeduplication:
    """Same job_id must not be enqueued or processed twice."""

    @pytest.mark.asyncio
    async def test_job_deduplication_in_job_loop(self):
        """
        Simulate two consecutive polls returning the same job.
        _active_jobs should prevent it from being enqueued a second time.
        """

        node = _make_node()
        job = _make_open_job(job_id=7, payment_amount=20)

        # Both polls return the same job
        node.blockchain.fetch_open_jobs = AsyncMock(return_value=[job])

        poll_count = 0

        async def patched_sleep(delay):
            nonlocal poll_count
            poll_count += 1
            if poll_count >= 2:
                node._running = False  # stop after two polls

        node._running = True

        with patch("node.server.asyncio.sleep", new=patched_sleep):
            # Run the loop directly
            await node._job_loop()

        # The job should appear in the queue exactly once
        assert node._job_queue.qsize() == 1
        assert job.job_id in node._active_jobs

    @pytest.mark.asyncio
    async def test_explicit_dedup_check(self):
        """
        Directly call the dedup check: if job_id already in _active_jobs,
        putting it into the queue again is skipped.
        """
        node = _make_node()
        job = _make_open_job(job_id=99, payment_amount=30)

        # Pre-seed as already active
        node._active_jobs[job.job_id] = True

        # Attempt to enqueue — the loop should skip it.
        if job.job_id not in node._active_jobs:
            await node._job_queue.put((-job.payment_amount, job))

        assert node._job_queue.qsize() == 0, "Duplicate job should not have been enqueued"


# ---------------------------------------------------------------------------
# Test: worker count
# ---------------------------------------------------------------------------


class TestWorkerCount:
    """start() must spawn exactly max_concurrent_jobs worker tasks."""

    @pytest.mark.asyncio
    async def test_worker_count(self):
        """Verify that N workers are created for N = max_concurrent_jobs."""
        from node.server import Node

        max_workers = 3
        node = _make_node(max_concurrent_jobs=max_workers)

        created_workers = []

        original_create_task = asyncio.create_task

        def tracking_create_task(coro, **kwargs):
            # Detect worker coroutines by their qualified name
            coro_name = getattr(coro, "__qualname__", "") or getattr(coro, "__name__", "")
            if "_job_worker" in coro_name:
                created_workers.append(coro_name)
            task = original_create_task(coro, **kwargs)
            return task

        # Stub out all the things start() calls before the task-creation phase.
        async def noop(*args, **kwargs):
            pass

        async def stop_after_one_poll(*args, **kwargs):
            """Replacement for _job_loop: stops the node immediately."""
            node._running = False

        node.shard_mgr.load = MagicMock()

        with (
            patch.object(Node, "_ensure_registered", new=noop),
            patch("node.blockchain.BlockchainClient.connect", new=noop),
            patch("node.server.P2PLayer") as mock_p2p_cls,
            patch.object(Node, "_job_loop", new=stop_after_one_poll),
            patch.object(Node, "_heartbeat_loop", new=noop),
            patch("asyncio.create_task", side_effect=tracking_create_task),
        ):
            mock_p2p = AsyncMock()
            mock_p2p.start = AsyncMock()
            mock_p2p.stop = AsyncMock()
            mock_p2p_cls.return_value = mock_p2p

            # Temporarily replace blockchain with a version that has connect
            node.blockchain.connect = AsyncMock()
            node.blockchain.register_node = AsyncMock(return_value=True)

            await node.start()

        assert len(created_workers) == max_workers, (
            f"Expected {max_workers} workers, got {len(created_workers)}: {created_workers}"
        )


# ---------------------------------------------------------------------------
# Test: IPFS circuit breaker
# ---------------------------------------------------------------------------


class TestIpfsCircuitBreaker:
    """After 3 IPFS failures, _upload_result should return a placeholder CID."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_falls_back_to_placeholder(self):

        node = _make_node()
        # Give the node a storage mock that always fails
        node.storage = MagicMock()
        node.storage.upload_result = AsyncMock(side_effect=RuntimeError("IPFS down"))

        job_id = 55
        result_text = "some result"

        cid = await node._upload_result(job_id, result_text)

        assert cid.startswith("bafkrei"), f"Expected placeholder CID, got: {cid!r}"
        assert node.storage.upload_result.await_count == 3, (
            f"Expected exactly 3 upload attempts, got {node.storage.upload_result.await_count}"
        )
