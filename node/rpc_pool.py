"""Connection pool for Solana RPC AsyncClient instances."""

import asyncio
import logging
from contextlib import asynccontextmanager

try:
    from solana.rpc.async_api import AsyncClient

    _SOLANA_AVAILABLE = True
except ImportError:  # pragma: no cover
    AsyncClient = None  # type: ignore[assignment,misc]
    _SOLANA_AVAILABLE = False

logger = logging.getLogger(__name__)


def _make_async_client(rpc_url: str):
    """Return an AsyncClient, or a stub when Solana is not installed."""
    if _SOLANA_AVAILABLE:
        return AsyncClient(rpc_url)
    # Provide a minimal stub so the module is importable without Solana.
    from unittest.mock import MagicMock

    stub = MagicMock()
    stub.close = asyncio.coroutine(lambda: None)  # type: ignore[attr-defined]
    return stub


class RpcPool:
    """
    Pool of AsyncClient instances for concurrent RPC calls.

    Usage::

        pool = RpcPool(rpc_url, size=3)
        await pool.start()
        async with pool.acquire() as client:
            result = await client.get_account_info(pubkey)
        await pool.close()

    If all clients are in use, :meth:`acquire` waits (with optional
    ``timeout_s``).  When the timeout elapses before a client becomes
    available an :exc:`asyncio.TimeoutError` is raised.
    """

    def __init__(self, rpc_url: str, size: int = 3) -> None:
        if size < 1:
            raise ValueError("Pool size must be at least 1")
        self._rpc_url = rpc_url
        self._size = size
        self._clients: list = []
        self._queue: asyncio.Queue = asyncio.Queue()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Configured pool size."""
        return self._size

    @property
    def available(self) -> int:
        """Number of clients currently available for acquisition."""
        return self._queue.qsize()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Create ``size`` AsyncClient instances and put them in the pool."""
        if self._clients:
            raise RuntimeError("Pool already started")
        for _ in range(self._size):
            client = _make_async_client(self._rpc_url)
            self._clients.append(client)
            await self._queue.put(client)
        logger.debug("RpcPool started with %d clients (%s)", self._size, self._rpc_url)

    async def close(self) -> None:
        """Close all AsyncClient instances in the pool."""
        for client in self._clients:
            try:
                await client.close()
            except Exception:
                logger.exception("Error closing RPC client")
        self._clients.clear()
        # Drain the queue so any waiters unblock with an empty pool.
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        logger.debug("RpcPool closed")

    # ------------------------------------------------------------------
    # Acquire
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def acquire(self, timeout_s: float = 10.0):
        """
        Async context manager that yields an :class:`AsyncClient`.

        The client is returned to the pool when the ``async with`` block exits,
        even if an exception is raised.

        Raises :exc:`asyncio.TimeoutError` if no client becomes available
        within *timeout_s* seconds.
        """
        client = await asyncio.wait_for(self._queue.get(), timeout=timeout_s)
        try:
            yield client
        finally:
            await self._queue.put(client)
