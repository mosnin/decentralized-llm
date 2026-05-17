"""
Tests for node/batcher.py — DynamicBatcher and InferenceBatch.

All tests are pure-Python / asyncio; no torch or GPU required.
"""

import asyncio

import pytest

from node.batcher import DynamicBatcher, InferenceBatch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODEL_A = "model-a"
MODEL_B = "model-b"


def _batcher(max_batch_size: int = 4, max_wait_ms: float = 50.0) -> DynamicBatcher:
    return DynamicBatcher(max_batch_size=max_batch_size, max_wait_ms=max_wait_ms)


# ---------------------------------------------------------------------------
# Test: single job dispatched after timeout
# ---------------------------------------------------------------------------


class TestSingleJobDispatchedAfterTimeout:
    """A lone job must be returned once max_wait_ms expires."""

    @pytest.mark.asyncio
    async def test_single_job_dispatched_after_timeout(self):
        batcher = _batcher(max_batch_size=8, max_wait_ms=30.0)

        await batcher.add_job(job_id=1, prompt="hello", max_tokens=10, model_name=MODEL_A)

        # next_batch should block briefly then return the single job.
        batch = await batcher.next_batch(MODEL_A, timeout_s=0.5)

        assert batch is not None
        assert isinstance(batch, InferenceBatch)
        assert batch.job_ids == [1]
        assert batch.prompts == ["hello"]
        assert batch.max_tokens == [10]


# ---------------------------------------------------------------------------
# Test: batch forms immediately when full
# ---------------------------------------------------------------------------


class TestBatchFormsWhenFull:
    """next_batch must return as soon as max_batch_size jobs are queued."""

    @pytest.mark.asyncio
    async def test_batch_forms_when_full(self):
        batcher = _batcher(max_batch_size=3, max_wait_ms=10_000.0)  # very long wait

        for i in range(3):
            await batcher.add_job(job_id=i, prompt=f"p{i}", max_tokens=5, model_name=MODEL_A)

        # Should come back immediately (well within 1 s) even though max_wait_ms
        # is 10 seconds, because the buffer is already full.
        batch = await asyncio.wait_for(batcher.next_batch(MODEL_A, timeout_s=1.0), timeout=1.0)

        assert batch is not None
        assert len(batch.job_ids) == 3


# ---------------------------------------------------------------------------
# Test: jobs grouped by model
# ---------------------------------------------------------------------------


class TestJobsGroupedByModel:
    """Jobs for different models must end up in separate batches."""

    @pytest.mark.asyncio
    async def test_jobs_grouped_by_model(self):
        batcher = _batcher(max_batch_size=8, max_wait_ms=30.0)

        await batcher.add_job(job_id=10, prompt="a-prompt", max_tokens=5, model_name=MODEL_A)
        await batcher.add_job(job_id=20, prompt="b-prompt", max_tokens=5, model_name=MODEL_B)
        await batcher.add_job(job_id=11, prompt="a-prompt2", max_tokens=5, model_name=MODEL_A)

        batch_a = await batcher.next_batch(MODEL_A, timeout_s=0.5)
        batch_b = await batcher.next_batch(MODEL_B, timeout_s=0.5)

        assert batch_a is not None
        assert batch_b is not None

        # MODEL_A batch should contain only MODEL_A job ids.
        assert set(batch_a.job_ids) == {10, 11}
        # MODEL_B batch should contain only MODEL_B job id.
        assert set(batch_b.job_ids) == {20}


# ---------------------------------------------------------------------------
# Test: empty batch returns None on timeout
# ---------------------------------------------------------------------------


class TestEmptyBatchReturnsNoneOnTimeout:
    """next_batch should return None when no jobs arrive before timeout_s."""

    @pytest.mark.asyncio
    async def test_empty_batch_returns_none_on_timeout(self):
        batcher = _batcher(max_batch_size=4, max_wait_ms=10.0)

        # Don't add any jobs — just wait.
        result = await batcher.next_batch(MODEL_A, timeout_s=0.08)

        assert result is None


# ---------------------------------------------------------------------------
# Test: batch preserves job order
# ---------------------------------------------------------------------------


class TestBatchPreservesJobOrder:
    """Jobs must appear in the batch in the order they were added."""

    @pytest.mark.asyncio
    async def test_batch_preserves_job_order(self):
        batcher = _batcher(max_batch_size=8, max_wait_ms=30.0)

        job_ids = [100, 101, 102, 103]
        for jid in job_ids:
            await batcher.add_job(
                job_id=jid, prompt=f"prompt-{jid}", max_tokens=jid, model_name=MODEL_A
            )

        batch = await batcher.next_batch(MODEL_A, timeout_s=0.5)

        assert batch is not None
        assert batch.job_ids == job_ids
        assert batch.prompts == [f"prompt-{jid}" for jid in job_ids]
        assert batch.max_tokens == job_ids


# ---------------------------------------------------------------------------
# Test: max_batch_size respected (never exceeded)
# ---------------------------------------------------------------------------


class TestMaxBatchSizeRespected:
    """Dispatched batch must never contain more than max_batch_size jobs."""

    @pytest.mark.asyncio
    async def test_max_batch_size_respected(self):
        max_size = 3
        batcher = _batcher(max_batch_size=max_size, max_wait_ms=10_000.0)

        # Add more jobs than max_batch_size.
        for i in range(5):
            await batcher.add_job(job_id=i, prompt=f"p{i}", max_tokens=1, model_name=MODEL_A)

        batch = await asyncio.wait_for(batcher.next_batch(MODEL_A, timeout_s=1.0), timeout=1.0)

        assert batch is not None
        assert len(batch.job_ids) <= max_size, (
            f"Batch size {len(batch.job_ids)} exceeded max_batch_size {max_size}"
        )


# ---------------------------------------------------------------------------
# Test: concurrent producers
# ---------------------------------------------------------------------------


class TestConcurrentProducers:
    """Multiple coroutines adding jobs concurrently should not lose any jobs."""

    @pytest.mark.asyncio
    async def test_concurrent_producers(self):
        n_producers = 5
        jobs_per_producer = 4
        total_jobs = n_producers * jobs_per_producer

        batcher = _batcher(max_batch_size=total_jobs + 1, max_wait_ms=200.0)

        async def producer(producer_id: int) -> None:
            for i in range(jobs_per_producer):
                job_id = producer_id * 100 + i
                await batcher.add_job(
                    job_id=job_id,
                    prompt=f"producer-{producer_id}-job-{i}",
                    max_tokens=10,
                    model_name=MODEL_A,
                )
                # Small yield to interleave with other producers.
                await asyncio.sleep(0)

        # Launch all producers concurrently.
        await asyncio.gather(*(producer(pid) for pid in range(n_producers)))

        # Collect all jobs (may span multiple batches if timing causes early dispatch).
        collected_ids: list[int] = []
        while True:
            batch = await batcher.next_batch(MODEL_A, timeout_s=0.3)
            if batch is None:
                break
            collected_ids.extend(batch.job_ids)

        assert len(collected_ids) == total_jobs, (
            f"Expected {total_jobs} total jobs, collected {len(collected_ids)}"
        )
        # No duplicates.
        assert len(set(collected_ids)) == total_jobs
