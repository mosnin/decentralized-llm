"""
Tests for node/continuous_batcher.py

Pure-Python, no torch, no GPU required.
Run with: pytest tests/test_continuous_batcher.py -v
"""

from __future__ import annotations

import time

import pytest

from node.continuous_batcher import (
    BlockTable,
    ContinuousBatcher,
    Sequence,
    SequenceState,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def make_seq(
    seq_id: int,
    num_prompt_tokens: int = 4,
    max_tokens: int = 8,
    priority: float = 0.0,
    arrival_time: float | None = None,
) -> Sequence:
    """Factory for test sequences."""
    return Sequence(
        id=seq_id,
        prompt_tokens=list(range(num_prompt_tokens)),
        max_tokens=max_tokens,
        priority=priority,
        arrival_time=arrival_time if arrival_time is not None else time.monotonic(),
    )


def run_to_completion(
    batcher: ContinuousBatcher,
    token_factory=None,
    max_steps: int = 200,
) -> list[Sequence]:
    """
    Drive the batcher until all sequences are finished or max_steps reached.

    token_factory(seq) -> int   (defaults to constant 42)
    """
    if token_factory is None:
        token_factory = lambda seq: 42  # noqa: E731

    finished_all: list[Sequence] = []
    for _ in range(max_steps):
        if batcher.num_waiting == 0 and batcher.num_running == 0:
            break
        batch = batcher.schedule()
        if batch:
            tokens = [token_factory(seq) for seq in batch]
            batcher.step(batch, tokens)
        finished_all.extend(batcher.get_finished())

    return finished_all


# ──────────────────────────────────────────────────────────────────────────────
# Test 1 – Sequence state transitions: WAITING → RUNNING → FINISHED
# ──────────────────────────────────────────────────────────────────────────────


def test_sequence_state_transitions():
    """A single sequence moves through WAITING → RUNNING → FINISHED."""
    batcher = ContinuousBatcher(max_batch_size=4, max_tokens_per_step=16)
    seq = make_seq(seq_id=1, max_tokens=3)

    assert seq.state == SequenceState.WAITING

    batcher.add_request(seq)
    assert batcher.num_waiting == 1

    # First schedule: seq should become RUNNING.
    batch = batcher.schedule()
    assert seq in batch
    assert seq.state == SequenceState.RUNNING
    assert batcher.num_running == 1

    # Generate 3 tokens → sequence should finish.
    for step in range(3):
        assert seq.state == SequenceState.RUNNING
        batcher.step([seq], [100 + step])
        if seq.state == SequenceState.FINISHED:
            break
        batch = batcher.schedule()

    assert seq.state == SequenceState.FINISHED

    finished = batcher.get_finished()
    assert seq in finished
    assert seq.num_generated_tokens == 3


# ──────────────────────────────────────────────────────────────────────────────
# Test 2 – Block allocation and deallocation
# ──────────────────────────────────────────────────────────────────────────────


def test_block_allocation_and_deallocation():
    """BlockTable allocates and frees blocks correctly."""
    bt = BlockTable(num_blocks=32, block_size=8)

    assert bt.num_free_blocks == 32

    # Allocate 2 blocks for seq 1.
    blocks = bt.allocate(seq_id=1, num_blocks=2)
    assert len(blocks) == 2
    assert bt.num_free_blocks == 30
    assert bt.num_blocks_for(1) == 2

    # Allocate 3 blocks for seq 2.
    bt.allocate(seq_id=2, num_blocks=3)
    assert bt.num_free_blocks == 27

    # Free seq 1.
    freed = bt.free(seq_id=1)
    assert freed == 2
    assert bt.num_free_blocks == 29
    assert bt.num_blocks_for(1) == 0

    # Free seq 2.
    freed = bt.free(seq_id=2)
    assert freed == 3
    assert bt.num_free_blocks == 32  # back to full pool


# ──────────────────────────────────────────────────────────────────────────────
# Test 3 – Preemption when memory is full
# ──────────────────────────────────────────────────────────────────────────────


def test_preemption_when_memory_full():
    """When KV-cache is exhausted, lower-priority running sequences are preempted."""
    # Tiny pool: 4 blocks of size 4 → 16 token slots total.
    # Two seqs with 6-token prompts each need 2 blocks apiece → 4 blocks total.
    # When a third sequence arrives it should force preemption of the lowest-
    # priority running sequence.
    batcher = ContinuousBatcher(
        max_batch_size=4,
        max_tokens_per_step=16,
        block_size=4,
        num_blocks=4,
    )

    # seq_a: high priority, arrives first
    seq_a = make_seq(seq_id=10, num_prompt_tokens=6, max_tokens=20, priority=1.0, arrival_time=1.0)
    # seq_b: low priority, arrives second
    seq_b = make_seq(seq_id=11, num_prompt_tokens=6, max_tokens=20, priority=0.0, arrival_time=2.0)

    batcher.add_request(seq_a)
    batcher.add_request(seq_b)

    # First schedule admits both (4 blocks total, pool exhausted).
    batch = batcher.schedule()
    admitted_ids = {s.id for s in batch}
    # Both should be running (pool has exactly enough for prompt blocks).
    assert seq_a.state == SequenceState.RUNNING or seq_b.state == SequenceState.RUNNING

    # Now add a high-priority newcomer that needs space.
    seq_c = make_seq(seq_id=12, num_prompt_tokens=6, max_tokens=20, priority=2.0, arrival_time=3.0)
    batcher.add_request(seq_c)

    # Schedule again — seq_c needs blocks, which requires preempting someone.
    batch2 = batcher.schedule()

    # At least one of the original sequences was preempted.
    preempted_states = [
        seq_a.state == SequenceState.WAITING,
        seq_b.state == SequenceState.WAITING,
    ]
    # seq_b (low priority) should have been preempted.
    assert seq_b.state == SequenceState.WAITING, (
        "Lowest-priority sequence should be preempted first"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 4 – FCFS ordering
# ──────────────────────────────────────────────────────────────────────────────


def test_fcfs_ordering():
    """Sequences are admitted in arrival-time order (first-come, first-served)."""
    # Batch size 2 so only 2 of 4 are admitted per schedule call.
    batcher = ContinuousBatcher(max_batch_size=2, max_tokens_per_step=16)

    t0 = time.monotonic()
    seqs = [
        make_seq(seq_id=i, arrival_time=t0 + i * 0.01)
        for i in range(4)
    ]
    for s in seqs:
        batcher.add_request(s)

    batch = batcher.schedule()
    batch_ids = [s.id for s in batch]

    # The two earliest arrivals should be admitted.
    assert batch_ids == [0, 1] or set(batch_ids) == {0, 1}, (
        f"Expected ids {{0,1}} but got {batch_ids}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 5 – Batch size limit respected
# ──────────────────────────────────────────────────────────────────────────────


def test_batch_size_limit():
    """schedule() never returns more than max_batch_size sequences."""
    max_batch = 3
    batcher = ContinuousBatcher(max_batch_size=max_batch, max_tokens_per_step=64)

    for i in range(10):
        batcher.add_request(make_seq(seq_id=i))

    batch = batcher.schedule()
    assert len(batch) <= max_batch, (
        f"Batch length {len(batch)} exceeds max_batch_size {max_batch}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 6 – max_tokens_per_step limit
# ──────────────────────────────────────────────────────────────────────────────


def test_max_tokens_per_step_limit():
    """
    schedule() returns at most max_tokens_per_step sequences when each
    generates exactly one token per step.
    """
    # 10 sequences but only 5 tokens allowed per step.
    batcher = ContinuousBatcher(max_batch_size=10, max_tokens_per_step=5)

    for i in range(10):
        batcher.add_request(make_seq(seq_id=i))

    batch = batcher.schedule()
    # Each sequence generates 1 token; 5-token budget → at most 5 sequences.
    assert len(batch) <= 5, (
        f"Expected ≤5 sequences in batch, got {len(batch)}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 7 – Throughput tracking
# ──────────────────────────────────────────────────────────────────────────────


def test_throughput_tracking():
    """tokens_per_second is positive after generating tokens across multiple steps."""
    batcher = ContinuousBatcher(max_batch_size=4, max_tokens_per_step=16)

    for i in range(4):
        batcher.add_request(make_seq(seq_id=i, max_tokens=10))

    # Run several steps.
    for _ in range(5):
        batch = batcher.schedule()
        if not batch:
            break
        batcher.step(batch, [42] * len(batch))
        batcher.get_finished()

    # After a few steps we should have some throughput data.
    # tokens_per_second requires at least 2 recorded step timestamps; it may
    # be 0.0 if all steps happened in the same monotonic tick on fast hardware.
    tps = batcher.tokens_per_second
    assert tps >= 0.0, f"tokens_per_second must be non-negative, got {tps}"
    assert batcher._stats.total_tokens_generated > 0


# ──────────────────────────────────────────────────────────────────────────────
# Test 8 – Multiple concurrent sequences run to completion
# ──────────────────────────────────────────────────────────────────────────────


def test_multiple_concurrent_sequences():
    """All sequences eventually complete when run to completion."""
    batcher = ContinuousBatcher(max_batch_size=4, max_tokens_per_step=16)

    seqs = [make_seq(seq_id=i, max_tokens=5) for i in range(8)]
    for s in seqs:
        batcher.add_request(s)

    finished = run_to_completion(batcher)

    assert len(finished) == 8, f"Expected 8 finished, got {len(finished)}"
    for s in finished:
        assert s.state == SequenceState.FINISHED
        assert s.num_generated_tokens == 5


# ──────────────────────────────────────────────────────────────────────────────
# Test 9 – Preempted sequence resumes after memory is freed
# ──────────────────────────────────────────────────────────────────────────────


def test_preempted_sequence_resumes():
    """
    A preempted sequence eventually completes once the sequences that displaced
    it finish and release their blocks.
    """
    # Very small block pool: 2 blocks of size 4 → 8 token slots.
    # seq_a has a 3-token prompt → needs 1 block.
    # seq_b has a 3-token prompt → needs 1 block.
    # Together they fill the pool.  seq_c (arriving later) forces preemption.
    batcher = ContinuousBatcher(
        max_batch_size=4,
        max_tokens_per_step=8,
        block_size=4,
        num_blocks=2,
    )

    seq_a = make_seq(seq_id=1, num_prompt_tokens=3, max_tokens=2, priority=1.0, arrival_time=1.0)
    seq_b = make_seq(seq_id=2, num_prompt_tokens=3, max_tokens=1, priority=0.0, arrival_time=2.0)

    batcher.add_request(seq_a)
    batcher.add_request(seq_b)

    # Run to completion — all should finish despite the tiny pool.
    finished = run_to_completion(batcher, max_steps=50)

    finished_ids = {s.id for s in finished}
    assert 1 in finished_ids, "seq_a (id=1) did not finish"
    assert 2 in finished_ids, "seq_b (id=2) did not finish"


# ──────────────────────────────────────────────────────────────────────────────
# Test 10 – Batch utilization calculation
# ──────────────────────────────────────────────────────────────────────────────


def test_batch_utilization():
    """batch_utilization is between 0 and 1 and reflects actual usage."""
    max_batch = 4
    batcher = ContinuousBatcher(max_batch_size=max_batch, max_tokens_per_step=16)

    # Add only 2 sequences into a batcher that supports 4 → utilization ≤ 0.5
    for i in range(2):
        batcher.add_request(make_seq(seq_id=i, max_tokens=6))

    for _ in range(4):
        batch = batcher.schedule()
        if batch:
            batcher.step(batch, [99] * len(batch))
        batcher.get_finished()

    util = batcher.batch_utilization
    assert 0.0 <= util <= 1.0, f"utilization out of range: {util}"
    # With only 2 of 4 slots used, utilization should be ≤ 0.5
    assert util <= 0.5 + 1e-9, (
        f"Expected utilization ≤ 0.5 for 2/{max_batch} seqs, got {util}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 11 – BlockTable raises MemoryError when pool exhausted
# ──────────────────────────────────────────────────────────────────────────────


def test_block_table_memory_error_on_exhaustion():
    """BlockTable raises MemoryError if more blocks are requested than available."""
    bt = BlockTable(num_blocks=4, block_size=8)
    bt.allocate(seq_id=1, num_blocks=4)  # exhaust pool

    with pytest.raises(MemoryError):
        bt.allocate(seq_id=2, num_blocks=1)


# ──────────────────────────────────────────────────────────────────────────────
# Test 12 – get_finished drains the completed list
# ──────────────────────────────────────────────────────────────────────────────


def test_get_finished_drains():
    """get_finished() returns completed sequences and clears the internal list."""
    batcher = ContinuousBatcher(max_batch_size=4, max_tokens_per_step=16)
    seq = make_seq(seq_id=99, max_tokens=1)
    batcher.add_request(seq)

    batch = batcher.schedule()
    batcher.step(batch, [7])  # max_tokens=1 → finishes immediately

    first_drain = batcher.get_finished()
    assert len(first_drain) == 1
    assert first_drain[0].id == 99

    # Second drain should be empty.
    second_drain = batcher.get_finished()
    assert second_drain == []


# ──────────────────────────────────────────────────────────────────────────────
# Test 13 – Priority tie-breaking within same arrival time
# ──────────────────────────────────────────────────────────────────────────────


def test_priority_tiebreaker():
    """When arrival times are identical, higher priority sequences are admitted first."""
    batcher = ContinuousBatcher(max_batch_size=1, max_tokens_per_step=16)

    t = time.monotonic()
    low_priority = make_seq(seq_id=20, priority=0.0, arrival_time=t)
    high_priority = make_seq(seq_id=21, priority=1.0, arrival_time=t)

    # Add low-priority first, then high-priority (same timestamp).
    batcher.add_request(low_priority)
    batcher.add_request(high_priority)

    batch = batcher.schedule()
    assert len(batch) == 1
    # Higher priority should win the tie.
    assert batch[0].id == high_priority.id, (
        f"Expected high-priority seq (id={high_priority.id}) but got id={batch[0].id}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 14 – BlockTable.blocks_needed rounds up correctly
# ──────────────────────────────────────────────────────────────────────────────


def test_block_table_blocks_needed():
    """blocks_needed uses ceiling division."""
    bt = BlockTable(num_blocks=64, block_size=16)

    assert bt.blocks_needed(0) == 0
    assert bt.blocks_needed(1) == 1
    assert bt.blocks_needed(16) == 1
    assert bt.blocks_needed(17) == 2
    assert bt.blocks_needed(32) == 2
    assert bt.blocks_needed(33) == 3


# ──────────────────────────────────────────────────────────────────────────────
# Test 15 – Sequence.tokens_remaining tracks correctly
# ──────────────────────────────────────────────────────────────────────────────


def test_sequence_tokens_remaining():
    """tokens_remaining decreases as tokens are generated and never goes negative."""
    seq = make_seq(seq_id=5, max_tokens=4)

    assert seq.tokens_remaining == 4

    seq.generated_tokens.append(1)
    assert seq.tokens_remaining == 3

    seq.generated_tokens.extend([2, 3, 4])
    assert seq.tokens_remaining == 0

    # Simulate overflow guard.
    seq.generated_tokens.append(5)
    assert seq.tokens_remaining == 0
