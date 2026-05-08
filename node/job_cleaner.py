"""Periodic cleanup of stale/expired jobs from the local job tracking state."""

import asyncio
import logging
import time

logger = logging.getLogger(__name__)


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
