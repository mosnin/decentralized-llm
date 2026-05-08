"""
Tests for node/rpc_pool.py — connection pool for Solana RPC clients.

All tests run without the real Solana packages installed; AsyncClient is
replaced with a lightweight asyncio-aware MagicMock.
"""

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub out Solana so the module imports cleanly in CI
# ---------------------------------------------------------------------------


def _install_stubs() -> None:
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
        "transformers",
        "accelerate",
        "bitsandbytes",
        "peft",
        "lighthouseweb3",
    ):
        _stub(mod)

    if "torch" not in sys.modules:
        torch_mod = types.ModuleType("torch")

        class _FakeTensor:
            pass

        torch_mod.Tensor = _FakeTensor  # type: ignore[attr-defined]
        nn_mod = types.ModuleType("torch.nn")

        class _FakeModule:
            def __init__(self, *args, **kwargs):
                pass

        nn_mod.Module = _FakeModule  # type: ignore[attr-defined]
        torch_mod.nn = nn_mod  # type: ignore[attr-defined]
        sys.modules["torch"] = torch_mod
        sys.modules["torch.nn"] = nn_mod
    else:
        _stub("torch.nn")


_install_stubs()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_client() -> MagicMock:
    """Return an asyncio-compatible mock that stands in for AsyncClient."""
    client = MagicMock()
    client.close = AsyncMock()
    return client


def _patch_make_client(pool_size: int):
    """
    Return a context manager that replaces _make_async_client in rpc_pool with
    a factory producing fresh mock clients.
    """
    mocks = [_make_mock_client() for _ in range(pool_size)]
    call_count = [-1]

    def factory(_url: str):
        call_count[0] += 1
        return mocks[call_count[0]]

    return patch("node.rpc_pool._make_async_client", side_effect=factory), mocks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPoolStartsCorrectSize:
    """pool.start() must create exactly `size` clients."""

    async def test_pool_starts_correct_size(self):
        from node.rpc_pool import RpcPool

        size = 4
        patcher, mocks = _patch_make_client(size)
        with patcher:
            pool = RpcPool("http://localhost:8899", size=size)
            await pool.start()
            assert pool.size == size
            assert pool.available == size
            assert len(pool._clients) == size
            await pool.close()


class TestAcquireYieldsClient:
    """acquire() must yield one of the pool's AsyncClient instances."""

    async def test_acquire_yields_client(self):
        from node.rpc_pool import RpcPool

        size = 2
        patcher, mocks = _patch_make_client(size)
        with patcher:
            pool = RpcPool("http://localhost:8899", size=size)
            await pool.start()
            async with pool.acquire() as client:
                assert client in pool._clients
            await pool.close()


class TestAcquireReturnsClientToPool:
    """After the acquire() block exits, the client must be back in the pool."""

    async def test_acquire_returns_client_to_pool(self):
        from node.rpc_pool import RpcPool

        patcher, mocks = _patch_make_client(1)
        with patcher:
            pool = RpcPool("http://localhost:8899", size=1)
            await pool.start()

            assert pool.available == 1
            async with pool.acquire() as _client:
                assert pool.available == 0
            # After the block the client must be back
            assert pool.available == 1
            await pool.close()


class TestConcurrentAcquireUpToPoolSize:
    """Multiple concurrent acquires up to pool size should all succeed."""

    async def test_concurrent_acquire_up_to_pool_size(self):
        from node.rpc_pool import RpcPool

        size = 3
        patcher, mocks = _patch_make_client(size)
        with patcher:
            pool = RpcPool("http://localhost:8899", size=size)
            await pool.start()

            acquired: list = []
            barriers: list[asyncio.Event] = [asyncio.Event() for _ in range(size)]
            release = asyncio.Event()

            async def hold(idx: int) -> None:
                async with pool.acquire(timeout_s=2.0) as client:
                    acquired.append(client)
                    barriers[idx].set()
                    await release.wait()

            tasks = [asyncio.create_task(hold(i)) for i in range(size)]

            # Wait until all tasks have acquired a client
            await asyncio.gather(*(b.wait() for b in barriers))
            assert pool.available == 0
            assert len(acquired) == size
            # All acquired clients are distinct
            assert len(set(id(c) for c in acquired)) == size

            release.set()
            await asyncio.gather(*tasks)
            await pool.close()


class TestAcquireWaitsWhenPoolEmpty:
    """acquire() blocks when no clients are available and resumes when one is freed."""

    async def test_acquire_waits_when_pool_empty(self):
        from node.rpc_pool import RpcPool

        patcher, mocks = _patch_make_client(1)
        with patcher:
            pool = RpcPool("http://localhost:8899", size=1)
            await pool.start()

            holder_ready = asyncio.Event()
            waiter_got_client = asyncio.Event()
            release_holder = asyncio.Event()

            async def hold_client():
                async with pool.acquire(timeout_s=2.0):
                    holder_ready.set()
                    await release_holder.wait()

            async def wait_for_client():
                await holder_ready.wait()
                # The pool is now empty — this should block until release
                async with pool.acquire(timeout_s=5.0) as client:
                    waiter_got_client.set()
                    assert client is not None

            holder_task = asyncio.create_task(hold_client())
            waiter_task = asyncio.create_task(wait_for_client())

            await holder_ready.wait()
            # Waiter must not have gotten through yet
            assert not waiter_got_client.is_set()

            release_holder.set()
            await asyncio.gather(holder_task, waiter_task)

            assert waiter_got_client.is_set()
            await pool.close()


class TestAcquireTimeoutRaises:
    """acquire() must raise asyncio.TimeoutError when timeout elapses with no client."""

    async def test_acquire_timeout_raises(self):
        from node.rpc_pool import RpcPool

        patcher, mocks = _patch_make_client(1)
        with patcher:
            pool = RpcPool("http://localhost:8899", size=1)
            await pool.start()

            holder_ready = asyncio.Event()
            release = asyncio.Event()

            async def hold_forever():
                async with pool.acquire(timeout_s=5.0):
                    holder_ready.set()
                    await release.wait()

            holder_task = asyncio.create_task(hold_forever())
            await holder_ready.wait()

            with pytest.raises(asyncio.TimeoutError):
                async with pool.acquire(timeout_s=0.05):
                    pass

            release.set()
            await holder_task
            await pool.close()


class TestCloseReleasesAll:
    """close() must call .close() on every client in the pool."""

    async def test_close_releases_all(self):
        from node.rpc_pool import RpcPool

        size = 3
        patcher, mocks = _patch_make_client(size)
        with patcher:
            pool = RpcPool("http://localhost:8899", size=size)
            await pool.start()
            await pool.close()

        for mock_client in mocks:
            mock_client.close.assert_called_once()

        assert len(pool._clients) == 0
