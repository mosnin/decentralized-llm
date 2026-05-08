import time
from collections import deque
from dataclasses import dataclass


@dataclass
class EarningRecord:
    job_id: int
    amount_lamports: int
    timestamp: float
    model_name: str = ""
    tokens_generated: int = 0


class EarningsTracker:
    """
    Tracks per-job earnings with rolling time windows.
    Thread-safe for single-threaded asyncio usage.
    """

    def __init__(self, window_seconds: float = 86400.0):
        """window_seconds: how far back to include in windowed stats (default 24h)"""
        self._records: deque[EarningRecord] = deque()
        self._window_seconds = window_seconds
        self._total_lamports: int = 0
        self._total_jobs: int = 0

    def record(
        self,
        job_id: int,
        amount_lamports: int,
        model_name: str = "",
        tokens_generated: int = 0,
    ) -> None:
        """Record earnings for a completed job."""
        rec = EarningRecord(
            job_id=job_id,
            amount_lamports=amount_lamports,
            timestamp=time.time(),
            model_name=model_name,
            tokens_generated=tokens_generated,
        )
        self._records.append(rec)
        self._total_lamports += amount_lamports
        self._total_jobs += 1

    def _prune(self) -> None:
        """Remove records older than window_seconds."""
        cutoff = time.time() - self._window_seconds
        while self._records and self._records[0].timestamp < cutoff:
            self._records.popleft()

    def window_summary(self) -> dict:
        """
        Return stats for the rolling window:
        {
            "window_seconds": float,
            "jobs_completed": int,
            "total_lamports": int,
            "total_tokens": int,
            "avg_lamports_per_job": float,
            # top_models: [{"model": name, "jobs": n, "lamports": n}] top 5 by lamports desc
            "top_models": list[dict]
        }
        """
        self._prune()

        jobs_completed = 0
        total_lamports = 0
        total_tokens = 0
        model_stats: dict[str, dict] = {}

        for rec in self._records:
            jobs_completed += 1
            total_lamports += rec.amount_lamports
            total_tokens += rec.tokens_generated
            name = rec.model_name
            if name not in model_stats:
                model_stats[name] = {"model": name, "jobs": 0, "lamports": 0}
            model_stats[name]["jobs"] += 1
            model_stats[name]["lamports"] += rec.amount_lamports

        avg_lamports_per_job = total_lamports / jobs_completed if jobs_completed > 0 else 0.0

        top_models = sorted(model_stats.values(), key=lambda x: x["lamports"], reverse=True)[:5]

        return {
            "window_seconds": self._window_seconds,
            "jobs_completed": jobs_completed,
            "total_lamports": total_lamports,
            "total_tokens": total_tokens,
            "avg_lamports_per_job": avg_lamports_per_job,
            "top_models": top_models,
        }

    def lifetime_summary(self) -> dict:
        """
        Return lifetime totals (not windowed):
        {
            "total_jobs": int,
            "total_lamports": int,
        }
        """
        return {
            "total_jobs": self._total_jobs,
            "total_lamports": self._total_lamports,
        }

    def recent_jobs(self, n: int = 10) -> list[EarningRecord]:
        """Return the N most recent EarningRecords from the window."""
        self._prune()
        records = list(self._records)
        return records[-n:]
