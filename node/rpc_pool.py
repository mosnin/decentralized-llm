"""Connection pool for Solana RPC AsyncClient instances."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

try:
    from solana.rpc.async_api import AsyncClient

    _SOLANA_AVAILABLE = True
except ImportError:  # pragma: no cover
    AsyncClient = None  # type: ignore[assignment,misc]
    _SOLANA_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class HealthCheckConfig:
    """Configuration for connection health checking."""

    check_interval_s: float = 30.0
    timeout_s: float = 5.0
    max_failures_before_replace: int = 3


@dataclass
class ConnectionStats:
    """Track per-connection metrics."""

    requests: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    last_used: float = field(default_factory=time.monotonic)
    created_at: float = field(default_factory=time.monotonic)


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

    def __init__(
        self,
        rpc_url: str,
        size: int = 3,
        health_config: HealthCheckConfig | None = None,
    ) -> None:
        if size < 1:
            raise ValueError("Pool size must be at least 1")
        self._rpc_url = rpc_url
        self._size = size
        self._clients: list = []
        self._queue: asyncio.Queue = asyncio.Queue()
        self._health_config = health_config or HealthCheckConfig()
        # Map from id(client) → ConnectionStats
        self._stats: dict[int, ConnectionStats] = {}

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
            self._stats[id(client)] = ConnectionStats()
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
        self._stats.clear()
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
        stats = self._stats.get(id(client))
        if stats is not None:
            stats.requests += 1
            stats.last_used = time.monotonic()
        try:
            yield client
        finally:
            await self._queue.put(client)

    # ------------------------------------------------------------------
    # Health / stats
    # ------------------------------------------------------------------

    def mark_failed(self, client) -> None:
        """Increment failure counters for *client*."""
        stats = self._stats.get(id(client))
        if stats is not None:
            stats.failures += 1
            stats.consecutive_failures += 1
            logger.debug(
                "RpcPool: client %d marked failed (consecutive=%d)",
                id(client),
                stats.consecutive_failures,
            )

    def mark_success(self, client) -> None:
        """Reset consecutive failure counter for *client* on a successful call."""
        stats = self._stats.get(id(client))
        if stats is not None:
            stats.consecutive_failures = 0

    def connection_stats(self) -> dict[int, dict]:
        """Return a snapshot of health data keyed by connection id."""
        return {
            cid: {
                "requests": s.requests,
                "failures": s.failures,
                "consecutive_failures": s.consecutive_failures,
                "last_used": s.last_used,
                "created_at": s.created_at,
            }
            for cid, s in self._stats.items()
        }

    def healthy_count(self) -> int:
        """Return the number of connections whose consecutive failures are below threshold."""
        threshold = self._health_config.max_failures_before_replace
        return sum(1 for s in self._stats.values() if s.consecutive_failures < threshold)
