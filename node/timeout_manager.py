"""
Deadline-based job timeout management.

Wraps coroutines with asyncio cancellation tied to a Unix-timestamp deadline
so that jobs exceeding their on-chain deadline are abandoned cleanly.
"""

import asyncio
import time


class JobTimeoutError(Exception):
    """Raised when a job exceeds its deadline."""


class TimeoutManager:
    """Wraps coroutines with deadline-based cancellation."""

    async def run_with_deadline(self, coro, deadline: float):
        """
        Run *coro* but cancel and raise JobTimeoutError if the deadline has
        already passed or passes before the coroutine completes.
        """
        remaining = max(0.0, deadline - time.time())
        try:
            return await asyncio.wait_for(coro, timeout=remaining)
        except TimeoutError as exc:
            raise JobTimeoutError(
                f"Job exceeded deadline (deadline={deadline:.3f}, now={time.time():.3f})"
            ) from exc

    async def run_with_timeout(self, coro, timeout_seconds: float):
        """Convenience wrapper: deadline = time.time() + timeout_seconds."""
        deadline = time.time() + timeout_seconds
        return await self.run_with_deadline(coro, deadline)

    @staticmethod
    def seconds_remaining(deadline: float) -> float:
        """Return how many seconds remain until *deadline*, or 0.0 if expired."""
        return max(0.0, deadline - time.time())

    @staticmethod
    def is_expired(deadline: float) -> bool:
        """Return True if *deadline* has already passed."""
        return time.time() >= deadline
