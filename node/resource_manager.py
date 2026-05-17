"""
Consolidated resource management for decentralised LLM nodes.

Replaces four separate modules that handled overlapping concerns:
  - backoff_semaphore.py  → BackoffSemaphore / BackoffConfig
  - timeout_estimator.py  → TimeoutEstimator / LatencySample
  - timeout_manager.py    → TimeoutManager / JobTimeoutError
  - job_cleaner.py        → JobCleaner

All four public APIs are preserved verbatim so that existing imports
(from the thin shim files) continue to work without changes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _exponential_backoff(attempt: int, base: float = 1.0, cap: float = 60.0) -> float:
    """Return the delay (seconds) for a given *attempt* number (0-indexed).

    Delay = min(base * 2**attempt, cap).

    >>> _exponential_backoff(0, base=1.0, cap=60.0)
    1.0
    >>> _exponential_backoff(3, base=1.0, cap=60.0)
    8.0
    >>> _exponential_backoff(10, base=1.0, cap=60.0)
    60.0
    """
    return min(base * (2**attempt), cap)


# ---------------------------------------------------------------------------
# BackoffSemaphore
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# TimeoutEstimator
# ---------------------------------------------------------------------------


@dataclass
class LatencySample:
    latency_ms: float
    model_name: str
    token_count: int
    timestamp: float


class TimeoutEstimator:
    """
    Estimates appropriate timeout values using exponential moving averages
    and percentile-based safety margins.

    Timeout = p95_latency * safety_multiplier, clamped to [min_timeout_ms, max_timeout_ms].
    Falls back to default_timeout_ms when insufficient data.
    """

    def __init__(
        self,
        min_timeout_ms: float = 1_000.0,
        max_timeout_ms: float = 300_000.0,
        default_timeout_ms: float = 30_000.0,
        safety_multiplier: float = 1.5,
        window_size: int = 100,
    ):
        self.min_timeout_ms = min_timeout_ms
        self.max_timeout_ms = max_timeout_ms
        self.default_timeout_ms = default_timeout_ms
        self.safety_multiplier = safety_multiplier
        self.window_size = window_size
        self._samples: dict[str, deque[LatencySample]] = {}

    def record(self, model_name: str, latency_ms: float, token_count: int = 0) -> None:
        """Record a latency observation for a model."""
        if model_name not in self._samples:
            self._samples[model_name] = deque(maxlen=self.window_size)
        self._samples[model_name].append(
            LatencySample(
                latency_ms=latency_ms,
                model_name=model_name,
                token_count=token_count,
                timestamp=time.time(),
            )
        )

    def _percentile(self, values: list[float], p: float) -> float:
        """Compute p-th percentile (0-100) of a sorted or unsorted list."""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = max(0, int(len(sorted_vals) * p / 100) - 1)
        return sorted_vals[idx]

    def estimate(self, model_name: str, token_count: int = 0) -> float:
        """
        Return estimated timeout in milliseconds for a job.

        If token_count > 0 and we have enough samples, scale by tokens-per-ms rate.
        Falls back to default if fewer than 5 samples available.
        """
        samples = self._samples.get(model_name)
        if not samples or len(samples) < 5:
            return self.default_timeout_ms

        latencies = [s.latency_ms for s in samples]
        p95 = self._percentile(latencies, 95)
        raw = p95 * self.safety_multiplier

        # Scale by token count if we have token data
        if token_count > 0:
            token_samples = [s for s in samples if s.token_count > 0]
            if len(token_samples) >= 5:
                avg_ms_per_token = sum(s.latency_ms / s.token_count for s in token_samples) / len(
                    token_samples
                )
                raw = max(raw, avg_ms_per_token * token_count * self.safety_multiplier)

        return max(self.min_timeout_ms, min(self.max_timeout_ms, raw))

    def reset(self, model_name: str | None = None) -> None:
        """Clear samples for a model, or all models if None."""
        if model_name is None:
            self._samples.clear()
        else:
            self._samples.pop(model_name, None)

    def sample_count(self, model_name: str) -> int:
        samples = self._samples.get(model_name)
        return len(samples) if samples else 0


# ---------------------------------------------------------------------------
# TimeoutManager
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# JobCleaner
# ---------------------------------------------------------------------------


class JobCleaner:
    """
    Runs as a background task alongside the main job loop.

    Responsibilities:
    1. Scan self._active_jobs on the Node every `scan_interval_s` seconds
    2. For any job whose deadline has passed, remove it from _active_jobs
    3. Log a warning for each cleaned-up job
    4. Track stats: total_cleaned (int), last_scan_time (float)

    Usage:
        cleaner = JobCleaner(node, scan_interval_s=30)
        asyncio.create_task(cleaner.run())
    """

    def __init__(self, node, scan_interval_s: float = 30.0, grace_period_s: float = 60.0):
        self._node = node
        self.scan_interval_s = scan_interval_s
        self.grace_period_s = grace_period_s
        self.total_cleaned: int = 0
        self.last_scan_time: float = 0.0

    async def run(self) -> None:
        """Loop: scan every scan_interval_s until cancelled."""
        while True:
            await self.scan_once()
            await asyncio.sleep(self.scan_interval_s)

    async def scan_once(self) -> int:
        """
        Single scan pass. Returns number of jobs cleaned up.
        A job is stale if: time.time() > job.deadline + grace_period_s (default 60)
        """
        now = time.time()
        stale_job_ids = []

        for job_id, job in list(self._node._active_jobs.items()):
            # job may be stored as True (plain dedup marker) or as an OpenJob.
            # Only clean up entries that carry deadline information.
            deadline = getattr(job, "deadline", None)
            if deadline is not None and now > deadline + self.grace_period_s:
                stale_job_ids.append(job_id)

        for job_id in stale_job_ids:
            self._node._active_jobs.pop(job_id, None)
            logger.warning(
                "JobCleaner: removed stale job %d from active jobs (deadline exceeded)",
                job_id,
            )

        cleaned = len(stale_job_ids)
        self.total_cleaned += cleaned
        self.last_scan_time = now
        return cleaned

    @property
    def stats(self) -> dict:
        return {"total_cleaned": self.total_cleaned, "last_scan_time": self.last_scan_time}
