"""
Tests for resilience features in node/p2p.py.

All tests avoid importing hivemind or torch by stubbing those packages in
sys.modules before the first import of node.p2p.
"""

from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Minimal stubs so node/p2p.py loads without hivemind / torch installed
# ---------------------------------------------------------------------------


def _install_stubs():
    # --- torch stub ---
    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")
        torch_stub.Tensor = object  # type: ignore[attr-defined]
        nn_stub = types.ModuleType("torch.nn")

        class _Module:
            def __init__(self):
                pass

        nn_stub.Module = _Module  # type: ignore[attr-defined]
        torch_stub.nn = nn_stub  # type: ignore[attr-defined]
        sys.modules["torch"] = torch_stub
        sys.modules["torch.nn"] = nn_stub
    else:
        torch_stub = sys.modules["torch"]
        if not hasattr(torch_stub, "Tensor"):
            torch_stub.Tensor = object  # type: ignore[attr-defined]
        if not hasattr(torch_stub, "nn"):
            nn_stub = types.ModuleType("torch.nn")

            class _Module:
                def __init__(self):
                    pass

            nn_stub.Module = _Module  # type: ignore[attr-defined]
            torch_stub.nn = nn_stub  # type: ignore[attr-defined]
            sys.modules["torch.nn"] = nn_stub

    # --- hivemind stub ---
    if "hivemind" not in sys.modules:
        hm_stub = types.ModuleType("hivemind")

        class _RemoteExpert:
            def __init__(self, uid=""):
                self.uid = uid

            def __call__(self, *args, **kwargs):
                return args[0] if args else None

        hm_stub.RemoteExpert = _RemoteExpert  # type: ignore[attr-defined]
        hm_stub.DHT = MagicMock  # type: ignore[attr-defined]
        hm_stub.Server = MagicMock  # type: ignore[attr-defined]
        hm_stub.get_dht_time = lambda: 0.0  # type: ignore[attr-defined]
        sys.modules["hivemind"] = hm_stub


_install_stubs()

# Now it is safe to import the module under test
from node.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState  # noqa: E402
from node.p2p import (  # noqa: E402
    AllPeersFailedError,
    MultiPeerShardClient,
    RemoteShardClient,
    RemoteShardError,
    health_check_peer,
)
from node.peer_manager import PeerManager  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_TENSOR = object()  # stand-in for a torch.Tensor in tests


def _make_client(
    endpoint: str = "peer1:50051",
    timeout_s: float = 30.0,
    max_retries: int = 3,
    circuit_breaker: CircuitBreaker | None = None,
    peer_manager: PeerManager | None = None,
) -> RemoteShardClient:
    return RemoteShardClient(
        endpoint=endpoint,
        timeout_s=timeout_s,
        max_retries=max_retries,
        circuit_breaker=circuit_breaker,
        peer_manager=peer_manager,
    )


# ---------------------------------------------------------------------------
# 1. test_remote_shard_retries_on_timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remote_shard_retries_on_timeout():
    """
    When _call_expert raises TimeoutError every attempt, forward() should
    retry max_retries times and eventually raise RemoteShardError.
    """
    client = _make_client(max_retries=3, timeout_s=1.0)

    with (
        patch.object(client, "_call_expert", side_effect=asyncio.TimeoutError),
        patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        with pytest.raises(RemoteShardError):
            await client.forward(FAKE_TENSOR)

        # Slept between attempts (max_retries - 1 times)
        assert mock_sleep.call_count == 2


# ---------------------------------------------------------------------------
# 2. test_remote_shard_opens_circuit_after_exhausted_retries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remote_shard_opens_circuit_after_exhausted_retries():
    """
    After all retries are exhausted, the circuit breaker must be OPEN.
    """
    cb = CircuitBreaker(config=CircuitBreakerConfig(failure_threshold=1), name="test-peer")
    client = _make_client(max_retries=3, circuit_breaker=cb)

    with (
        patch.object(client, "_call_expert", side_effect=RuntimeError("dead")),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        with pytest.raises(RemoteShardError):
            await client.forward(FAKE_TENSOR)

    assert cb.state == CircuitState.OPEN


# ---------------------------------------------------------------------------
# 3. test_remote_shard_succeeds_on_second_attempt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remote_shard_succeeds_on_second_attempt():
    """
    If the first call fails but the second succeeds, forward() should return
    the successful result without raising.
    """
    results = [RuntimeError("transient"), FAKE_TENSOR]
    call_count = 0

    async def _fake_call_expert(_hidden_states):
        nonlocal call_count
        val = results[call_count]
        call_count += 1
        if isinstance(val, Exception):
            raise val
        return val

    client = _make_client(max_retries=3)
    with (
        patch.object(client, "_call_expert", side_effect=_fake_call_expert),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        result = await client.forward(FAKE_TENSOR)

    assert result is FAKE_TENSOR
    assert call_count == 2


# ---------------------------------------------------------------------------
# 4. test_multi_peer_falls_back_to_second_peer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_peer_falls_back_to_second_peer():
    """
    When the first peer fails, MultiPeerShardClient should transparently
    fall back to the second peer and return its result.
    """
    mp = MultiPeerShardClient(
        peers=["peer1:1", "peer2:2"],
        shard_index=0,
        model_id="test-model",
        timeout_s=5.0,
    )

    peer1_client = mp._clients["peer1:1"]
    peer2_client = mp._clients["peer2:2"]

    async def _fail(_hs):
        raise RemoteShardError("peer1 dead")

    async def _succeed(_hs):
        return FAKE_TENSOR

    with (
        patch.object(peer1_client, "forward", side_effect=_fail),
        patch.object(peer2_client, "forward", side_effect=_succeed),
    ):
        result = await mp.forward(FAKE_TENSOR)

    assert result is FAKE_TENSOR


# ---------------------------------------------------------------------------
# 5. test_multi_peer_raises_when_all_peers_fail
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_peer_raises_when_all_peers_fail():
    """
    AllPeersFailedError must be raised when every peer in the pool fails.
    """
    mp = MultiPeerShardClient(
        peers=["peer1:1", "peer2:2"],
        shard_index=0,
        model_id="test-model",
    )

    async def _fail(_hs):
        raise RemoteShardError("dead")

    for client in mp._clients.values():
        patch.object(client, "forward", side_effect=_fail).start()

    try:
        with pytest.raises(AllPeersFailedError):
            await mp.forward(FAKE_TENSOR)
    finally:
        patch.stopall()


# ---------------------------------------------------------------------------
# 6. test_multi_peer_skips_open_circuit_breaker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_peer_skips_open_circuit_breaker():
    """
    Peers with an OPEN circuit breaker are skipped and the next peer is tried.
    """
    mp = MultiPeerShardClient(
        peers=["bad:1", "good:2"],
        shard_index=1,
        model_id="test-model",
    )

    # Force bad:1's breaker OPEN
    bad_cb = mp._circuit_breakers["bad:1"]
    for _ in range(bad_cb.config.failure_threshold):
        bad_cb._on_failure()
    assert bad_cb.state == CircuitState.OPEN

    good_client = mp._clients["good:2"]

    async def _succeed(_hs):
        return FAKE_TENSOR

    with patch.object(good_client, "forward", side_effect=_succeed):
        result = await mp.forward(FAKE_TENSOR)

    assert result is FAKE_TENSOR


# ---------------------------------------------------------------------------
# 7. test_get_healthy_peers_excludes_open_breakers
# ---------------------------------------------------------------------------


def test_get_healthy_peers_excludes_open_breakers():
    """
    get_healthy_peers() must return only peers whose circuit breaker is CLOSED.
    """
    mp = MultiPeerShardClient(
        peers=["a:1", "b:2", "c:3"],
        shard_index=0,
        model_id="test-model",
    )

    # Open circuit breaker for "a:1"
    cb_a = mp._circuit_breakers["a:1"]
    for _ in range(cb_a.config.failure_threshold):
        cb_a._on_failure()

    healthy = mp.get_healthy_peers()
    assert "a:1" not in healthy
    assert "b:2" in healthy
    assert "c:3" in healthy


# ---------------------------------------------------------------------------
# 8. test_health_check_returns_true_for_alive_peer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check_returns_true_for_alive_peer():
    """
    health_check_peer should return True when the TCP connection succeeds.
    """
    mock_writer = MagicMock()
    mock_writer.close = MagicMock()
    mock_writer.wait_closed = AsyncMock()

    with patch(
        "asyncio.open_connection",
        new_callable=AsyncMock,
        return_value=(MagicMock(), mock_writer),
    ):
        result = await health_check_peer("127.0.0.1:9999", timeout_s=1.0)

    assert result is True


# ---------------------------------------------------------------------------
# 9. test_health_check_returns_false_for_dead_peer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check_returns_false_for_dead_peer():
    """
    health_check_peer should return False when the connection is refused /
    times out.
    """
    with patch(
        "asyncio.open_connection",
        new_callable=AsyncMock,
        side_effect=ConnectionRefusedError("refused"),
    ):
        result = await health_check_peer("dead-host:9999", timeout_s=1.0)

    assert result is False


# ---------------------------------------------------------------------------
# 10. test_exponential_backoff_timing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exponential_backoff_timing():
    """
    Verify that asyncio.sleep is called with the correct exponential back-off
    delays: 0.5 s, 1.0 s for a 3-attempt client (2 sleeps between 3 tries).
    """
    client = _make_client(max_retries=3, timeout_s=1.0)
    sleep_calls: list[float] = []

    async def _fake_sleep(delay: float):
        sleep_calls.append(delay)

    with (
        patch.object(client, "_call_expert", side_effect=RuntimeError("err")),
        patch("asyncio.sleep", side_effect=_fake_sleep),
    ):
        with pytest.raises(RemoteShardError):
            await client.forward(FAKE_TENSOR)

    # Two sleeps: after attempt 1 → 0.5 s, after attempt 2 → 1.0 s
    assert sleep_calls == [0.5, 1.0]
