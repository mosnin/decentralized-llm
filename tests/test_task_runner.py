import asyncio

from node.task_runner import TaskResult, TaskRunner, TaskStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _ok() -> str:
    return "ok"


async def _fail() -> None:
    raise ValueError("boom")


async def _sleep(seconds: float = 0.01) -> str:
    await asyncio.sleep(seconds)
    return "done"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_submit_success():
    async def run():
        runner = TaskRunner()
        result = await runner.submit("t1", _ok)
        assert result.status == TaskStatus.DONE
        assert result.result == "ok"

    asyncio.run(run())


def test_submit_failure():
    async def run():
        runner = TaskRunner()
        result = await runner.submit("t1", _fail)
        assert result.status == TaskStatus.FAILED

    asyncio.run(run())


def test_submit_stores_error():
    async def run():
        runner = TaskRunner()
        result = await runner.submit("t1", _fail)
        assert isinstance(result.error, ValueError)
        assert str(result.error) == "boom"

    asyncio.run(run())


def test_submit_records_duration():
    async def run():
        runner = TaskRunner()
        result = await runner.submit("t1", _sleep)
        assert result.duration_ms > 0

    asyncio.run(run())


def test_get_result_known():
    async def run():
        runner = TaskRunner()
        submitted = await runner.submit("t1", _ok)
        fetched = runner.get_result("t1")
        assert fetched is submitted

    asyncio.run(run())


def test_get_result_unknown():
    runner = TaskRunner()
    assert runner.get_result("missing") is None


def test_stats_counts():
    async def run():
        runner = TaskRunner()
        await runner.submit("t1", _ok)
        await runner.submit("t2", _ok)
        await runner.submit("t3", _fail)
        s = runner.stats()
        assert s["done"] == 2
        assert s["failed"] == 1
        assert s["total"] == 3
        assert s["pending"] == 0
        assert s["running"] == 0

    asyncio.run(run())


def test_max_concurrency_respected():
    """Only max_concurrency tasks should run simultaneously."""

    async def run():
        max_concurrency = 3
        runner = TaskRunner(max_concurrency=max_concurrency)

        active = 0
        peak = 0

        async def tracked_task():
            nonlocal active, peak
            active += 1
            if active > peak:
                peak = active
            await asyncio.sleep(0.05)
            active -= 1
            return "done"

        total_tasks = max_concurrency + 1
        tasks = [(f"t{i}", lambda: tracked_task()) for i in range(total_tasks)]
        await runner.submit_many(tasks)

        assert peak <= max_concurrency

    asyncio.run(run())


def test_submit_many_returns_all():
    async def run():
        runner = TaskRunner()
        tasks = [(f"t{i}", _ok) for i in range(5)]
        results = await runner.submit_many(tasks)
        assert len(results) == 5
        assert all(isinstance(r, TaskResult) for r in results)

    asyncio.run(run())


def test_submit_many_order_preserved():
    async def run():
        runner = TaskRunner()
        task_ids = [f"t{i}" for i in range(5)]
        tasks = [(tid, _ok) for tid in task_ids]
        results = await runner.submit_many(tasks)
        assert [r.task_id for r in results] == task_ids

    asyncio.run(run())
