"""
Continuous batching engine with PagedAttention-style KV-cache management.

Implements a scheduler similar to vLLM's continuous batching: sequences are
scheduled in FCFS order and can be preempted (swapped back to WAITING) when
KV-cache blocks run out, allowing higher-priority sequences to proceed.

Key components
--------------
Sequence      – tracks a single inference request
BlockTable    – paged KV-cache allocator (fixed-size blocks)
ContinuousBatcher – scheduling loop: schedule → step → drain
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Sequence state machine
# ──────────────────────────────────────────────────────────────────────────────


class SequenceState(Enum):
    WAITING = auto()     # queued, not yet scheduled
    RUNNING = auto()     # currently in the active batch
    FINISHED = auto()    # generation complete (EOS or max_tokens reached)
    PREEMPTED = auto()   # was running, swapped out due to memory pressure


@dataclass
class Sequence:
    """Tracks a single inference request through its lifetime."""

    id: int
    prompt_tokens: list[int]
    max_tokens: int
    priority: float = 0.0          # higher → scheduled first among ties
    arrival_time: float = field(default_factory=time.monotonic)
    generated_tokens: list[int] = field(default_factory=list)
    state: SequenceState = SequenceState.WAITING

    # ── derived helpers ──────────────────────────────────────────────────────

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_tokens)

    @property
    def num_generated_tokens(self) -> int:
        return len(self.generated_tokens)

    @property
    def total_tokens(self) -> int:
        """Current total token count (prompt + generated)."""
        return self.num_prompt_tokens + self.num_generated_tokens

    @property
    def is_finished(self) -> bool:
        return self.state == SequenceState.FINISHED

    @property
    def tokens_remaining(self) -> int:
        """How many more tokens may be generated."""
        return max(0, self.max_tokens - self.num_generated_tokens)


# ──────────────────────────────────────────────────────────────────────────────
# Paged KV-cache block allocator
# ──────────────────────────────────────────────────────────────────────────────


class BlockTable:
    """
    Manages a pool of fixed-size KV-cache blocks.

    Each block holds ``block_size`` token slots.  Sequences are allocated a
    contiguous-in-address-space but potentially non-contiguous set of block
    IDs.  This mimics vLLM's PagedAttention memory layout without requiring a
    GPU.
    """

    def __init__(self, num_blocks: int = 512, block_size: int = 16) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        self.num_blocks = num_blocks
        self.block_size = block_size

        # Pool of free block IDs (block 0 … num_blocks-1)
        self._free_blocks: set[int] = set(range(num_blocks))
        # seq_id → list[block_id]
        self._allocations: dict[int, list[int]] = {}

    # ── public API ────────────────────────────────────────────────────────────

    def allocate(self, seq_id: int, num_blocks: int) -> list[int]:
        """
        Allocate *num_blocks* blocks for *seq_id*.

        Raises ``MemoryError`` if there are not enough free blocks.
        If *seq_id* already has blocks, the new blocks are appended.
        """
        if num_blocks <= 0:
            return []
        if num_blocks > len(self._free_blocks):
            raise MemoryError(
                f"Cannot allocate {num_blocks} blocks; only "
                f"{len(self._free_blocks)} available"
            )

        allocated = []
        for _ in range(num_blocks):
            block_id = self._free_blocks.pop()
            allocated.append(block_id)

        self._allocations.setdefault(seq_id, []).extend(allocated)
        return allocated

    def free(self, seq_id: int) -> int:
        """
        Return all blocks owned by *seq_id* to the pool.

        Returns the number of blocks released.  Safe to call for unknown IDs.
        """
        blocks = self._allocations.pop(seq_id, [])
        self._free_blocks.update(blocks)
        return len(blocks)

    def blocks_for(self, seq_id: int) -> list[int]:
        """Return the list of block IDs currently allocated to *seq_id*."""
        return list(self._allocations.get(seq_id, []))

    def num_blocks_for(self, seq_id: int) -> int:
        return len(self._allocations.get(seq_id, []))

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - len(self._free_blocks)

    def blocks_needed(self, num_tokens: int) -> int:
        """Ceiling division: how many blocks to hold *num_tokens* tokens."""
        return (num_tokens + self.block_size - 1) // self.block_size

    def ensure_capacity(self, seq_id: int, total_tokens: int) -> int:
        """
        Grow the allocation for *seq_id* so it can hold *total_tokens*.

        Returns the number of newly allocated blocks (0 if already sufficient).
        Raises ``MemoryError`` if insufficient free blocks remain.
        """
        needed = self.blocks_needed(total_tokens)
        have = self.num_blocks_for(seq_id)
        extra = needed - have
        if extra <= 0:
            return 0
        self.allocate(seq_id, extra)
        return extra


# ──────────────────────────────────────────────────────────────────────────────
# Continuous batcher / scheduler
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _ThroughputStats:
    """Rolling throughput counters."""
    total_tokens_generated: int = 0
    total_steps: int = 0
    step_start_times: list[float] = field(default_factory=list)
    step_token_counts: list[int] = field(default_factory=list)
    batch_sizes: list[int] = field(default_factory=list)
    max_batch_capacity: int = 1  # set by ContinuousBatcher

    @property
    def tokens_per_second(self) -> float:
        """Estimate throughput using the last N steps."""
        if len(self.step_start_times) < 2:
            return 0.0
        elapsed = self.step_start_times[-1] - self.step_start_times[0]
        if elapsed <= 0:
            return 0.0
        # tokens produced in all steps except the first (whose start anchors time)
        total = sum(self.step_token_counts[1:])
        return total / elapsed

    @property
    def batch_utilization(self) -> float:
        """Average fraction of max_batch_size used across recorded steps."""
        if not self.batch_sizes:
            return 0.0
        return sum(self.batch_sizes) / (len(self.batch_sizes) * self.max_batch_capacity)


class ContinuousBatcher:
    """
    Continuous batching scheduler with FCFS ordering and memory-pressure preemption.

    Scheduling policy
    -----------------
    1. Sequences in WAITING state are sorted by (arrival_time ASC, priority DESC)
       — i.e. FCFS with priority as tie-breaker.
    2. Up to ``max_batch_size`` sequences are promoted to RUNNING.
    3. If KV-cache is exhausted while trying to admit new sequences, the
       lowest-priority RUNNING sequence(s) are preempted (→ WAITING) until
       enough blocks are free.
    4. ``step()`` appends generated tokens and marks sequences FINISHED when
       they hit ``max_tokens``.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_tokens_per_step: int,
        block_size: int = 16,
        num_blocks: int = 512,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if max_tokens_per_step <= 0:
            raise ValueError("max_tokens_per_step must be positive")

        self.max_batch_size = max_batch_size
        self.max_tokens_per_step = max_tokens_per_step

        self.block_table = BlockTable(num_blocks=num_blocks, block_size=block_size)

        # WAITING queue (insertion order preserved via deque for FCFS)
        self._waiting: deque[Sequence] = deque()
        # Currently RUNNING sequences (seq_id → Sequence)
        self._running: dict[int, Sequence] = {}
        # Completed sequences not yet drained
        self._finished: list[Sequence] = []

        self._stats = _ThroughputStats(max_batch_capacity=max_batch_size)

    # ── public API ────────────────────────────────────────────────────────────

    def add_request(self, seq: Sequence) -> None:
        """Enqueue a new sequence.  Must be in WAITING state."""
        if seq.state != SequenceState.WAITING:
            raise ValueError(
                f"add_request expects WAITING sequence; got {seq.state}"
            )
        self._waiting.append(seq)

    def schedule(self) -> list[Sequence]:
        """
        Select the batch for the next forward pass.

        Returns the list of RUNNING sequences after scheduling decisions.
        May preempt existing RUNNING sequences when memory is tight.
        """
        # ── Step 1: try to admit WAITING sequences ────────────────────────────
        # Sort waiting by FCFS (arrival_time asc), break ties by priority desc.
        sorted_waiting = sorted(
            self._waiting,
            key=lambda s: (s.arrival_time, -s.priority),
        )

        admitted: list[Sequence] = []
        still_waiting: list[Sequence] = []

        for seq in sorted_waiting:
            if len(self._running) + len(admitted) >= self.max_batch_size:
                still_waiting.append(seq)
                continue

            # How many blocks needed for the full sequence so far?
            blocks_needed = self.block_table.blocks_needed(seq.total_tokens + 1)
            current_blocks = self.block_table.num_blocks_for(seq.id)
            extra_needed = max(0, blocks_needed - current_blocks)

            if extra_needed > self.block_table.num_free_blocks:
                # Try preemption to free space.
                freed = self._preempt_to_free(extra_needed)
                if freed < extra_needed:
                    # Still not enough — leave this sequence waiting.
                    still_waiting.append(seq)
                    continue

            # Allocate blocks and admit.
            self.block_table.ensure_capacity(seq.id, seq.total_tokens + 1)
            seq.state = SequenceState.RUNNING
            admitted.append(seq)

        # Rebuild the waiting deque (preserve FCFS by using still_waiting order).
        self._waiting = deque(still_waiting)

        # Register newly admitted sequences.
        for seq in admitted:
            self._running[seq.id] = seq

        # ── Step 2: ensure existing RUNNING sequences have enough blocks ───────
        # They each need one more block for the token they are about to generate.
        for seq in list(self._running.values()):
            needed = self.block_table.blocks_needed(seq.total_tokens + 1)
            have = self.block_table.num_blocks_for(seq.id)
            if needed > have:
                extra = needed - have
                if extra > self.block_table.num_free_blocks:
                    freed = self._preempt_to_free(extra, exclude_id=seq.id)
                    if freed < extra:
                        # Preempt this sequence itself.
                        self._preempt(seq)
                        continue
                try:
                    self.block_table.allocate(seq.id, extra)
                except MemoryError:
                    self._preempt(seq)

        # ── Step 3: enforce max_tokens_per_step ───────────────────────────────
        # Cap the batch so total tokens generated in this step ≤ max_tokens_per_step.
        batch: list[Sequence] = []
        token_budget = self.max_tokens_per_step

        # Stable FCFS order within the running set.
        running_sorted = sorted(
            self._running.values(),
            key=lambda s: (s.arrival_time, -s.priority),
        )
        for seq in running_sorted:
            if token_budget <= 0:
                break
            batch.append(seq)
            token_budget -= 1  # one token per sequence per step

        # Record stats.
        self._stats.batch_sizes.append(len(batch))
        self._stats.step_start_times.append(time.monotonic())
        self._stats.total_steps += 1

        return batch

    def step(
        self,
        batch: list[Sequence],
        generated_token_ids: list[int],
    ) -> None:
        """
        Update sequences after a forward pass.

        Parameters
        ----------
        batch:
            The list returned by the most recent ``schedule()`` call.
        generated_token_ids:
            One token ID per sequence in *batch* (same order).
        """
        if len(batch) != len(generated_token_ids):
            raise ValueError(
                f"batch length ({len(batch)}) != generated_token_ids "
                f"length ({len(generated_token_ids)})"
            )

        tokens_this_step = 0

        for seq, token_id in zip(batch, generated_token_ids):
            if seq.state != SequenceState.RUNNING:
                # Sequence was preempted between schedule() and step() — skip.
                continue

            seq.generated_tokens.append(token_id)
            tokens_this_step += 1

            # Grow KV-cache allocation to cover the new token (best-effort;
            # block may already have been allocated by schedule()).
            try:
                self.block_table.ensure_capacity(seq.id, seq.total_tokens)
            except MemoryError:
                pass  # will be caught next schedule() call

            # Check termination conditions.
            if seq.num_generated_tokens >= seq.max_tokens:
                self._finish(seq)

        self._stats.total_tokens_generated += tokens_this_step
        self._stats.step_token_counts.append(tokens_this_step)

    def get_finished(self) -> list[Sequence]:
        """Drain and return all completed sequences."""
        done = list(self._finished)
        self._finished.clear()
        return done

    # ── throughput properties ─────────────────────────────────────────────────

    @property
    def tokens_per_second(self) -> float:
        return self._stats.tokens_per_second

    @property
    def batch_utilization(self) -> float:
        return self._stats.batch_utilization

    # ── queue inspection ──────────────────────────────────────────────────────

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    @property
    def num_running(self) -> int:
        return len(self._running)

    @property
    def num_finished_pending(self) -> int:
        return len(self._finished)

    # ── internals ─────────────────────────────────────────────────────────────

    def _finish(self, seq: Sequence) -> None:
        """Mark *seq* as FINISHED, release its blocks, remove from running."""
        seq.state = SequenceState.FINISHED
        self.block_table.free(seq.id)
        self._running.pop(seq.id, None)
        self._finished.append(seq)

    def _preempt(self, seq: Sequence) -> None:
        """Preempt *seq*: release blocks and return it to the WAITING queue."""
        seq.state = SequenceState.PREEMPTED
        self.block_table.free(seq.id)
        self._running.pop(seq.id, None)
        # Re-queue at the front so it is retried soon (PREEMPTED has priority).
        self._waiting.appendleft(seq)
        # Reset to WAITING so the scheduler will pick it up again.
        seq.state = SequenceState.WAITING

    def _preempt_to_free(
        self,
        target_blocks: int,
        exclude_id: Optional[int] = None,
    ) -> int:
        """
        Preempt lowest-priority RUNNING sequences until *target_blocks* are freed.

        Returns the total number of blocks freed.
        """
        # Sort running by (priority asc, arrival_time desc) → victims first.
        candidates = sorted(
            [s for s in self._running.values() if s.id != exclude_id],
            key=lambda s: (s.priority, -s.arrival_time),
        )
        freed = 0
        for seq in candidates:
            if freed >= target_blocks:
                break
            blocks = self.block_table.num_blocks_for(seq.id)
            self._preempt(seq)
            freed += blocks
        return freed
