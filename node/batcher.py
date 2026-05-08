"""
Dynamic request batcher for homogeneous inference jobs.

Accumulates inference requests arriving within a configurable time window and
dispatches them as a batch, grouping by model name so each batch is homogeneous.
"""

import asyncio
import time
from dataclasses import dataclass


@dataclass
class InferenceBatch:
    """Groups multiple jobs with the same model for batched inference."""

    job_ids: list[int]
    prompts: list[str]
    max_tokens: list[int]
    created_at: float  # epoch seconds — used for max_wait_ms enforcement


class DynamicBatcher:
    """
    Accumulates jobs arriving within a time window and dispatches them as a batch.

    Strategy:
    - Collect jobs for up to ``max_wait_ms`` (default 50 ms) OR until
      ``max_batch_size`` is reached, whichever comes first.
    - Jobs are grouped by ``model_name`` so every dispatched batch is
      homogeneous (all prompts target the same model).
    - ``add_job`` is safe to call from multiple concurrent coroutines.
    - ``next_batch`` blocks until a batch is available or ``timeout_s`` elapses.

    Metrics:
    - ``batch_size_histogram``: dict mapping batch-size → occurrence count.
    """

    def __init__(self, max_batch_size: int = 8, max_wait_ms: float = 50.0) -> None:
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

        # Per-model accumulation buffers: model_name → list of (job_id, prompt, max_tokens)
        self._buffers: dict[str, list[tuple[int, str, int]]] = {}
        # Per-model "first-job arrival" timestamps for max_wait_ms enforcement
        self._buffer_created_at: dict[str, float] = {}
        # Per-model events — signalled whenever a buffer might be ready
        self._events: dict[str, asyncio.Event] = {}
        # Shared lock protecting all mutable state
        self._lock = asyncio.Lock()
        # Metrics
        self.batch_size_histogram: dict[int, int] = {}

    # ──────────────────────────── public API ─────────────────────────────────

    async def add_job(
        self,
        job_id: int,
        prompt: str,
        max_tokens: int,
        model_name: str,
    ) -> None:
        """Append a job to the per-model accumulation buffer."""
        async with self._lock:
            if model_name not in self._buffers:
                self._buffers[model_name] = []
                self._buffer_created_at[model_name] = time.monotonic()
                self._events[model_name] = asyncio.Event()

            self._buffers[model_name].append((job_id, prompt, max_tokens))

            # Signal waiters so they can re-evaluate readiness.
            self._events[model_name].set()

    async def next_batch(
        self,
        model_name: str,
        timeout_s: float = 1.0,
    ) -> InferenceBatch | None:
        """
        Wait until a batch is ready for *model_name* and return it.

        A batch is considered ready when either:
        - ``max_batch_size`` jobs have accumulated, or
        - ``max_wait_ms`` has elapsed since the first job was added to the buffer.

        Returns ``None`` if no jobs arrive within ``timeout_s``.
        """
        deadline = time.monotonic() + timeout_s

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # Final chance: drain anything that arrived right on the deadline.
                batch = await self._try_drain(model_name, force=True)
                return batch  # may be None if buffer was empty

            async with self._lock:
                # Ensure data structures exist even if no job has arrived yet.
                if model_name not in self._buffers:
                    self._buffers[model_name] = []
                    self._buffer_created_at[model_name] = time.monotonic()
                    self._events[model_name] = asyncio.Event()

                buf = self._buffers[model_name]
                created_at = self._buffer_created_at[model_name]
                elapsed_ms = (time.monotonic() - created_at) * 1000.0

                if buf and len(buf) >= self.max_batch_size:
                    # Batch is full — dispatch immediately.
                    return self._drain(model_name)

                if buf and elapsed_ms >= self.max_wait_ms:
                    # Time window expired — dispatch whatever we have.
                    return self._drain(model_name)

                # Not ready yet — compute how long to wait before re-checking.
                event = self._events[model_name]
                event.clear()

                if buf:
                    wait_until_timeout = self.max_wait_ms / 1000.0 - (elapsed_ms / 1000.0)
                    wait_s = min(wait_until_timeout, remaining)
                else:
                    wait_s = remaining

            # Release the lock while waiting for either a new job or a timer.
            if wait_s > 0:
                try:
                    await asyncio.wait_for(event.wait(), timeout=wait_s)
                except TimeoutError:
                    pass

    # ──────────────────────────── internals ──────────────────────────────────

    def _drain(self, model_name: str) -> InferenceBatch:
        """
        Remove and return all accumulated jobs for *model_name* as a batch.

        Must be called with ``self._lock`` held.
        """
        items = self._buffers.pop(model_name, [])
        self._buffer_created_at.pop(model_name, None)
        self._events.pop(model_name, None)

        job_ids = [item[0] for item in items]
        prompts = [item[1] for item in items]
        max_tokens = [item[2] for item in items]
        created_at = time.monotonic()

        batch = InferenceBatch(
            job_ids=job_ids,
            prompts=prompts,
            max_tokens=max_tokens,
            created_at=created_at,
        )

        size = len(job_ids)
        self.batch_size_histogram[size] = self.batch_size_histogram.get(size, 0) + 1
        return batch

    async def _try_drain(
        self,
        model_name: str,
        *,
        force: bool = False,
    ) -> InferenceBatch | None:
        """Drain the buffer if ready (or forced). Returns None when empty."""
        async with self._lock:
            buf = self._buffers.get(model_name)
            if not buf:
                return None
            if force:
                return self._drain(model_name)
            created_at = self._buffer_created_at.get(model_name, time.monotonic())
            elapsed_ms = (time.monotonic() - created_at) * 1000.0
            if len(buf) >= self.max_batch_size or elapsed_ms >= self.max_wait_ms:
                return self._drain(model_name)
            return None
