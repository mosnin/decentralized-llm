"""Semaphore with exponential backoff on contention (stdlib only)."""

import asyncio
from dataclasses import dataclass


@dataclass
class BackoffConfig:
    """Tuning parameters for exponential backoff."""

    initial_delay_s: float = 0.01
    max_delay_s: float = 5.0
    multiplier: float = 2.0
    max_attempts: int = 10


class BackoffSemaphore:
    """
    asyncio.Semaphore wrapper that retries with exponential backoff
    when the semaphore is immediately full (non-blocking acquire fails).

    Falls back to blocking acquire if backoff exhausted.

    Usage::

        sem = BackoffSemaphore(3)
        async with sem:
            ...  # at most 3 concurrent callers

    Or explicitly::

        await sem.acquire()
        try:
            ...
        finally:
            await sem.release()
    """

    def __init__(self, value: int, config: BackoffConfig | None = None) -> None:
        self._sem = asyncio.Semaphore(value)
        self.config = config or BackoffConfig()
        self._value = value

    async def acquire(self) -> None:
        """Acquire with exponential backoff on contention."""
        delay = self.config.initial_delay_s
        for _ in range(self.config.max_attempts):
            if self._sem._value > 0:  # fast path: slot likely available
                await self._sem.acquire()
                return
            await asyncio.sleep(delay)
            delay = min(delay * self.config.multiplier, self.config.max_delay_s)
        # Fallback: blocking acquire (will wait however long it takes)
        await self._sem.acquire()

    async def release(self) -> None:
        """Release the semaphore."""
        self._sem.release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *args):
        await self.release()

    @property
    def value(self) -> int:
        """Current number of available slots."""
        return self._sem._value
