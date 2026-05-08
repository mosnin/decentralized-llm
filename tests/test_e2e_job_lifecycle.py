"""
End-to-end integration tests for the full job lifecycle.

All external dependencies (Solana RPC, IPFS / Lighthouse, GPU inference) are
mocked so the suite runs without hardware, network, or installed Solana packages.

Flow under test
---------------
Client posts job → node claims → inference runs → result uploaded to IPFS
→ result hash submitted on-chain → client receives text

The _make_node / _make_open_job helpers mirror the patterns in test_job_queue.py
so both suites share the same bypass-__init__ approach.
"""

import asyncio
import hashlib
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# sys.modules stubs — installed once before any node imports
# ---------------------------------------------------------------------------


def _install_stubs() -> None:
    """Stub every package that node/server.py imports transitively."""

    def _stub(name: str) -> None:
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
        "hivemind",
        "hivemind.moe",
        "hivemind.moe.server",
        "hivemind.moe.server.layers",
    ):
        _stub(mod)

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
# Shared helpers
# ---------------------------------------------------------------------------

_RESULT_TEXT = "The answer is 42."
_RESULT_BYTES = _RESULT_TEXT.encode()
_RESULT_HASH = hashlib.sha256(_RESULT_BYTES).digest()
_RESULT_CID = "bafkreiresultcid1234567890abcdef"


def _make_prompt_blob(prompt: str = "What is the answer?") -> bytes:
    return prompt.encode()


def _make_open_job(
    job_id: int = 1,
    payment_amount: int = 100,
    deadline: float = 9_999_999_999.0,
    prompt_text: str = "What is the answer?",
    max_tokens: int = 512,
) -> object:
    from node.blockchain import OpenJob

    blob = _make_prompt_blob(prompt_text)
    prompt_hash = hashlib.sha256(blob).digest()

    return OpenJob(
        job_id=job_id,
        client="client_pubkey",
        model_id=b"\x00" * 32,
        prompt_hash=prompt_hash,
        prompt_cid="bafybeipromptcid123",
        max_tokens=max_tokens,
        payment_amount=payment_amount,
        deadline=deadline,
        job_pda="job_pda_pubkey",
    )


def _make_node():
    """
    Build a Node with all heavyweight deps replaced by mocks.

    Bypasses __init__ via object.__new__ (same pattern as test_job_queue.py).
    """
    from node.config import NodeConfig
    from node.server import Node

    cfg = NodeConfig(max_concurrent_jobs=1)
    node = object.__new__(Node)
    node.config = cfg

    node.shard_mgr = MagicMock()
    node.blockchain = MagicMock()
    node.blockchain.claim_job = AsyncMock(return_value=True)
    node.blockchain.submit_result = AsyncMock(return_value=True)
    node.blockchain.fetch_open_jobs = AsyncMock(return_value=[])
    node.storage = None
    node.p2p = None
    node._running = False
    node._active_jobs = {}

    mock_registry = MagicMock()
    mock_registry.get_by_model_id = MagicMock(return_value=None)
    mock_registry.load_all = AsyncMock()
    node._model_registry = mock_registry

    node._job_queue = asyncio.PriorityQueue()

    return node


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHappyPathJobCompletes:
    """
    Client posts job → node claims → inference runs → result uploaded to IPFS
    → result hash submitted on-chain → client receives text.
    """

    @pytest.mark.asyncio
    async def test_happy_path_job_completes(self):
        from node.server import Node

        node = _make_node()
        job = _make_open_job()

        prompt_blob = _make_prompt_blob()

        async def fake_run_inference(self_node, j, shard_mgr=None):
            return _RESULT_TEXT

        async def fake_upload(self_node, job_id, result_text):
            return _RESULT_CID

        # Provide a storage client so _fetch_prompt can be called through
        # _run_inference (we mock _run_inference entirely, but set storage so
        # the node doesn't raise on None checks elsewhere).
        node.storage = MagicMock()
        node.storage.download = AsyncMock(return_value=prompt_blob)

        with (
            patch.object(Node, "_run_inference", new=fake_run_inference),
            patch.object(Node, "_upload_result", new=fake_upload),
        ):
            await node._handle_job(job)

        # The result must have been submitted on-chain exactly once.
        node.blockchain.submit_result.assert_awaited_once()
        _, submit_kwargs = node.blockchain.submit_result.call_args
        # Positional args: (job, result_bytes, result_cid)
        call_args = node.blockchain.submit_result.call_args[0]
        assert call_args[0] is job
        assert call_args[2] == _RESULT_CID


class TestPromptHashVerifiedBeforeInference:
    """
    If IPFS returns a blob whose SHA-256 ≠ prompt_hash, the node must NOT
    call inference.
    """

    @pytest.mark.asyncio
    async def test_prompt_hash_verified_before_inference(self):
        from node.server import Node

        node = _make_node()
        job = _make_open_job(prompt_text="legitimate prompt")

        # Return a blob that does NOT match job.prompt_hash so that
        # verify_prompt_hash inside _fetch_prompt raises IntegrityError.
        tampered_blob = b"completely different data"
        node.storage = MagicMock()
        node.storage.download = AsyncMock(return_value=tampered_blob)

        # _run_inference first calls _fetch_prompt, which will raise.
        # We wire _run_inference to forward to the real _fetch_prompt so
        # the IntegrityError propagates back to _handle_job's catch block.
        async def fetch_then_infer(self_node, j, shard_mgr=None):
            await Node._fetch_prompt(self_node, j)  # raises IntegrityError on tampered blob
            return _RESULT_TEXT  # never reached

        with patch.object(Node, "_run_inference", new=fetch_then_infer):
            await node._handle_job(job)

        # IntegrityError must cause _handle_job to abort — no submission.
        node.blockchain.submit_result.assert_not_awaited()


class TestJobTimeoutTriggersBeforeInference:
    """
    If the deadline has already expired when _handle_job is called, the node
    should skip the job entirely (no claim, no inference, no submission).
    """

    @pytest.mark.asyncio
    async def test_job_timeout_triggers_before_inference(self):
        from node.server import Node

        node = _make_node()
        # deadline=1 is safely in the past
        job = _make_open_job(deadline=1.0)

        inference_called = False

        async def spy_inference(self_node, j, shard_mgr=None):
            nonlocal inference_called
            inference_called = True
            return _RESULT_TEXT

        with patch.object(Node, "_run_inference", new=spy_inference):
            await node._handle_job(job)

        assert not inference_called, "inference must NOT run for an expired job"
        node.blockchain.claim_job.assert_not_awaited()
        node.blockchain.submit_result.assert_not_awaited()


class TestNodeRetriesOnInferenceFailure:
    """
    First inference call raises RuntimeError; second succeeds.
    submit_result must be called exactly once with the successful result.
    """

    @pytest.mark.asyncio
    async def test_node_retries_on_inference_failure(self):
        from node.server import Node

        node = _make_node()
        job = _make_open_job()

        call_count = 0

        async def flaky_inference(self_node, j, shard_mgr=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient GPU error")
            return _RESULT_TEXT

        async def fake_upload(self_node, job_id, result_text):
            return _RESULT_CID

        with (
            patch.object(Node, "_run_inference", new=flaky_inference),
            patch.object(Node, "_upload_result", new=fake_upload),
            patch("node.server.asyncio.sleep", new=AsyncMock()),
        ):
            await node._handle_job(job)

        assert call_count == 2, f"Expected 2 inference calls (1 fail + 1 success), got {call_count}"
        node.blockchain.submit_result.assert_awaited_once()


class TestIntegrityErrorIncrementsFailedCounter:
    """
    A tampered prompt triggers IntegrityError → jobs_failed_total must be
    incremented and submit_result must NOT be called.
    """

    @pytest.mark.asyncio
    async def test_integrity_error_increments_failed_counter(self):
        import node.metrics as metrics_mod
        from node.integrity import IntegrityError
        from node.server import Node

        node = _make_node()
        job = _make_open_job()

        mock_jobs_failed = MagicMock()
        mock_active_jobs = MagicMock()

        async def raise_integrity(self_node, j, shard_mgr=None):
            raise IntegrityError("tampered blob")

        # Temporarily replace metric objects so the conditional `from .metrics import`
        # inside _handle_job picks up our mocks.
        orig_failed = getattr(metrics_mod, "jobs_failed_total", None)
        orig_active = getattr(metrics_mod, "active_jobs", None)
        orig_claimed = getattr(metrics_mod, "jobs_claimed_total", None)
        metrics_mod.jobs_failed_total = mock_jobs_failed
        metrics_mod.active_jobs = mock_active_jobs
        metrics_mod.jobs_claimed_total = MagicMock()

        try:
            with (
                patch.object(Node, "_run_inference", new=raise_integrity),
                patch("node.server.METRICS_AVAILABLE", True),
            ):
                await node._handle_job(job)
        finally:
            metrics_mod.jobs_failed_total = orig_failed
            metrics_mod.active_jobs = orig_active
            metrics_mod.jobs_claimed_total = orig_claimed

        mock_jobs_failed.inc.assert_called()
        node.blockchain.submit_result.assert_not_awaited()


class TestResultVerifiedBeforeSubmission:
    """
    If the result hash doesn't match what ResultVerifier expects, submit_result
    must NOT be called.
    """

    @pytest.mark.asyncio
    async def test_result_verified_before_submission(self):
        from node.server import Node

        node = _make_node()
        job = _make_open_job()

        async def good_inference(self_node, j, shard_mgr=None):
            return _RESULT_TEXT

        async def good_upload(self_node, job_id, result_text):
            return _RESULT_CID

        # Make ResultVerifier.verify_result return invalid so submission is skipped.
        # Disable metrics to avoid side-effects from prometheus_client being installed.
        with (
            patch.object(Node, "_run_inference", new=good_inference),
            patch.object(Node, "_upload_result", new=good_upload),
            patch(
                "node.server.ResultVerifier.verify_result", return_value=(False, "hash mismatch")
            ),
            patch("node.server.METRICS_AVAILABLE", False),
        ):
            await node._handle_job(job)

        node.blockchain.submit_result.assert_not_awaited()


class TestMetricsRecordedOnSuccess:
    """
    On a successful job, jobs_completed_total.inc() and
    inference_latency_seconds.observe() must both be called.
    """

    @pytest.mark.asyncio
    async def test_metrics_recorded_on_success(self):
        from node.server import Node

        node = _make_node()
        job = _make_open_job()

        mock_completed = MagicMock()
        mock_latency = MagicMock()
        mock_active = MagicMock()

        async def good_inference(self_node, j, shard_mgr=None):
            return _RESULT_TEXT

        async def good_upload(self_node, job_id, result_text):
            return _RESULT_CID

        import node.metrics as metrics_mod

        orig_completed = getattr(metrics_mod, "jobs_completed_total", None)
        orig_latency = getattr(metrics_mod, "inference_latency_seconds", None)
        orig_active = getattr(metrics_mod, "active_jobs", None)
        orig_claimed = getattr(metrics_mod, "jobs_claimed_total", None)

        metrics_mod.jobs_completed_total = mock_completed
        metrics_mod.inference_latency_seconds = mock_latency
        metrics_mod.active_jobs = mock_active
        metrics_mod.jobs_claimed_total = MagicMock()

        try:
            with (
                patch.object(Node, "_run_inference", new=good_inference),
                patch.object(Node, "_upload_result", new=good_upload),
                patch("node.server.METRICS_AVAILABLE", True),
            ):
                await node._handle_job(job)
        finally:
            metrics_mod.jobs_completed_total = orig_completed
            metrics_mod.inference_latency_seconds = orig_latency
            metrics_mod.active_jobs = orig_active
            metrics_mod.jobs_claimed_total = orig_claimed

        mock_completed.inc.assert_called()
        mock_latency.observe.assert_called()


class TestModelRegistryUsedForShardSelection:
    """
    _model_registry.get_by_model_id() must be called during _handle_job so that
    the correct ShardManager is selected for the job's model.
    """

    @pytest.mark.asyncio
    async def test_model_registry_used_for_shard_selection(self):
        from node.server import Node

        node = _make_node()
        job = _make_open_job()

        # Return a specific (mock) shard_mgr so we can verify it was used.
        custom_shard_mgr = MagicMock(name="custom_shard_mgr")
        node._model_registry.get_by_model_id = MagicMock(return_value=custom_shard_mgr)

        received_shard_mgr = None

        async def capture_shard_mgr(self_node, j, shard_mgr=None):
            nonlocal received_shard_mgr
            received_shard_mgr = shard_mgr
            return _RESULT_TEXT

        async def fake_upload(self_node, job_id, result_text):
            return _RESULT_CID

        # Disable metrics to avoid side-effects from prometheus_client being installed.
        with (
            patch.object(Node, "_run_inference", new=capture_shard_mgr),
            patch.object(Node, "_upload_result", new=fake_upload),
            patch("node.server.METRICS_AVAILABLE", False),
        ):
            await node._handle_job(job)

        node._model_registry.get_by_model_id.assert_called_once_with(job.model_id)
        assert received_shard_mgr is custom_shard_mgr, (
            "Expected the registry-provided shard_mgr to be passed to _run_inference"
        )
