"""
Tests for the activation-stream wiring in node/server.py.

Covers _push_activations, _receive_activations, and the shard-0 / final-shard
paths in _run_inference.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

torch = pytest.importorskip("torch")

from node.config import NodeConfig  # noqa: E402
from node.server import ActivationTransportError, Node  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(shard_index: int = 0, num_shards: int = 2) -> Node:
    """Return a Node with a minimal config — no real blockchain/p2p connections."""
    cfg = NodeConfig()
    cfg.shard_index = shard_index
    cfg.num_shards = num_shards
    cfg.listen_port = 7070
    cfg.listen_host = "127.0.0.1"
    node = Node.__new__(Node)
    node.config = cfg
    node.shard_mgr = MagicMock()
    node._model_registry = MagicMock()
    node.blockchain = MagicMock()
    node.storage = None
    node.p2p = None
    node._running = False
    node._active_jobs = {}
    node._job_queue = asyncio.PriorityQueue()
    node._activation_receiver = None

    # Attach a real circuit breaker so _run_inference works
    from node.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

    node._inference_cb = CircuitBreaker(
        config=CircuitBreakerConfig(failure_threshold=5, timeout_seconds=60.0),
        name="test",
    )
    return node


def _make_job(**kwargs) -> SimpleNamespace:
    defaults = {
        "job_id": 42,
        "model_id": b"\x00" * 32,
        "prompt_cid": "bafkreitest",
        "prompt_hash": b"\x00" * 32,
        "max_tokens": 4,
        "payment_amount": 100,
        "deadline": 9_999_999_999,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _rand_hidden(shape=(1, 4, 8)) -> torch.Tensor:
    return torch.randn(*shape)


# ---------------------------------------------------------------------------
# _push_activations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_push_activations_calls_sender():
    """ActivationSender.send is called with the correct job_id and tensor."""
    node = _make_node()
    hidden = _rand_hidden()
    sent_args = {}

    async def fake_send(job_id, tensor):
        sent_args["job_id"] = job_id
        sent_args["tensor"] = tensor

    mock_sender_instance = MagicMock()
    mock_sender_instance.send = fake_send
    mock_sender_cls = MagicMock(return_value=mock_sender_instance)

    with patch("node.server.ActivationSender", mock_sender_cls):
        await node._push_activations(42, hidden, "127.0.0.1:9000")

    mock_sender_cls.assert_called_once_with(host="127.0.0.1", port=9000)
    assert sent_args["job_id"] == 42
    assert torch.equal(sent_args["tensor"], hidden)


@pytest.mark.asyncio
async def test_push_activations_retries_on_failure():
    """First call to send raises, second call succeeds — no error raised."""
    node = _make_node()
    hidden = _rand_hidden()
    call_count = 0

    async def flaky_send(job_id, tensor):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionRefusedError("refused")

    mock_sender_instance = MagicMock()
    mock_sender_instance.send = flaky_send
    mock_sender_cls = MagicMock(return_value=mock_sender_instance)

    with patch("node.server.ActivationSender", mock_sender_cls):
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await node._push_activations(42, hidden, "127.0.0.1:9000")

    assert call_count == 2


@pytest.mark.asyncio
async def test_push_activations_raises_after_exhausting_retries():
    """All retries fail — ActivationTransportError is raised."""
    node = _make_node()
    hidden = _rand_hidden()

    async def always_fail(job_id, tensor):
        raise ConnectionRefusedError("refused")

    mock_sender_instance = MagicMock()
    mock_sender_instance.send = always_fail
    mock_sender_cls = MagicMock(return_value=mock_sender_instance)

    with patch("node.server.ActivationSender", mock_sender_cls):
        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(ActivationTransportError):
                await node._push_activations(42, hidden, "127.0.0.1:9000")


# ---------------------------------------------------------------------------
# _receive_activations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_receive_activations_returns_tensor():
    """ActivationReceiver.receive is awaited and its tensor is returned."""
    node = _make_node()
    expected = _rand_hidden()

    mock_receiver = MagicMock()
    mock_receiver.receive = AsyncMock(return_value=expected)
    node._activation_receiver = mock_receiver

    result = await node._receive_activations(job_id=42, timeout_s=5.0)

    mock_receiver.receive.assert_awaited_once_with(42, timeout=5.0)
    assert torch.equal(result, expected)


@pytest.mark.asyncio
async def test_receive_activations_timeout_raises():
    """When ActivationReceiver.receive raises TimeoutError, asyncio.TimeoutError propagates."""
    node = _make_node()

    mock_receiver = MagicMock()
    mock_receiver.receive = AsyncMock(side_effect=TimeoutError("timed out"))
    node._activation_receiver = mock_receiver

    with pytest.raises(asyncio.TimeoutError):
        await node._receive_activations(job_id=99, timeout_s=0.01)


# ---------------------------------------------------------------------------
# _run_inference
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_inference_shard_0_pushes_to_next_peer():
    """
    Shard 0: tokenize → embed → forward → _push_activations → _await_result.

    _push_activations must be called, and the result from _await_result is
    returned.
    """
    node = _make_node(shard_index=0, num_shards=2)
    job = _make_job()

    hidden = _rand_hidden()

    # Shard manager stubs
    mgr = MagicMock()
    mgr.tokenizer = MagicMock(return_value={"input_ids": torch.zeros(1, 3, dtype=torch.long)})
    mgr.model = MagicMock()
    mgr.model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))
    mgr.embed = MagicMock(return_value=hidden)
    mgr.forward = MagicMock(return_value=hidden)

    push_calls = []

    async def fake_push(job_id, tensor, endpoint, **kwargs):
        push_calls.append((job_id, endpoint))

    node._fetch_prompt = AsyncMock(return_value="hello")
    node._push_activations = fake_push
    node._await_result = AsyncMock(return_value="world")
    node._resolve_next_peer_endpoint = AsyncMock(return_value="127.0.0.1:9001")

    result = await node._run_inference(job, shard_mgr=mgr)

    assert result == "world"
    assert len(push_calls) == 1
    assert push_calls[0][0] == job.job_id


@pytest.mark.asyncio
async def test_run_inference_final_shard_receives_and_decodes():
    """
    Final shard: _receive_activations → forward → _generate_with → _publish_result.

    The decoded string is returned directly.
    """
    node = _make_node(shard_index=1, num_shards=2)
    job = _make_job()

    hidden = _rand_hidden()

    # Shard manager stubs
    mgr = MagicMock()
    mgr.forward = MagicMock(return_value=hidden)
    mgr.decode = MagicMock(return_value=torch.zeros(1, 1, 100))
    mgr.tokenizer = MagicMock()
    mgr.tokenizer.eos_token_id = 0
    mgr.tokenizer.decode = MagicMock(return_value="hello world")

    node._activation_receiver = MagicMock()
    node._activation_receiver.receive = AsyncMock(return_value=hidden)
    node._publish_result = AsyncMock()

    result = await node._run_inference(job, shard_mgr=mgr)

    node._activation_receiver.receive.assert_awaited_once()
    node._publish_result.assert_awaited_once_with(job.job_id, "hello world")
    assert result == "hello world"
