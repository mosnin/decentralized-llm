import asyncio
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from enum import Enum
from typing import Any


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Exception | None = None
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def duration_ms(self) -> float:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at) * 1000
        return 0.0


class TaskRunner:
    """
    Runs async tasks with a bounded concurrency semaphore.
    Tracks task state and results.
    """

    def __init__(self, max_concurrency: int = 4):
        self._sem = asyncio.Semaphore(max_concurrency)
        self._results: dict[str, TaskResult] = {}
        self._max_concurrency = max_concurrency

    async def submit(self, task_id: str, coro_factory: Callable[[], Coroutine]) -> TaskResult:
        """
        Submit a task. Blocks until a concurrency slot is available, then runs it.
        Stores and returns the TaskResult.
        """
        result = TaskResult(task_id=task_id, status=TaskStatus.PENDING)
        self._results[task_id] = result

        async with self._sem:
            result.status = TaskStatus.RUNNING
            result.started_at = time.monotonic()
            try:
                result.result = await coro_factory()
                result.status = TaskStatus.DONE
            except asyncio.CancelledError:
                result.status = TaskStatus.CANCELLED
                raise
            except Exception as exc:
                result.error = exc
                result.status = TaskStatus.FAILED
            finally:
                result.finished_at = time.monotonic()

        return result

    async def submit_many(self, tasks: list[tuple[str, Callable]]) -> list[TaskResult]:
        """Run all tasks concurrently (limited by max_concurrency). Returns results in order."""
        return await asyncio.gather(*[self.submit(tid, fn) for tid, fn in tasks])

    def get_result(self, task_id: str) -> TaskResult | None:
        return self._results.get(task_id)

    def stats(self) -> dict:
        counts = {s: 0 for s in TaskStatus}
        for r in self._results.values():
            counts[r.status] += 1
        return {
            "max_concurrency": self._max_concurrency,
            "total": len(self._results),
            "done": counts[TaskStatus.DONE],
            "failed": counts[TaskStatus.FAILED],
            "pending": counts[TaskStatus.PENDING],
            "running": counts[TaskStatus.RUNNING],
            "cancelled": counts[TaskStatus.CANCELLED],
        }
