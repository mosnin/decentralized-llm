"""Health-check helpers for node subsystems."""

import time

_start_time: float = time.time()


class HealthChecker:
    """Collects health check results from node subsystems."""

    async def check_all(self, client, queue_depth: int, active_jobs: int) -> dict:
        """Return the full readiness dict."""
        blockchain = "ok" if client is not None else "unavailable"
        job_queue = "ok" if queue_depth < 100 else "degraded"

        checks = {
            "blockchain": blockchain,
            "ipfs": "ok",
            "shard_manager": "ok",
            "job_queue": job_queue,
        }

        all_ok = all(v == "ok" for v in checks.values())
        any_unavailable = any(v == "unavailable" for v in checks.values())

        if any_unavailable:
            status = "not_ready"
        elif all_ok:
            status = "ready"
        else:
            status = "degraded"

        return {
            "status": status,
            "checks": checks,
            "queue_depth": queue_depth,
            "active_jobs": active_jobs,
        }

    def liveness(self) -> dict:
        """Return the liveness dict."""
        return {
            "status": "ok",
            "version": "0.1.0",
            "uptime_seconds": round(time.time() - _start_time, 1),
        }
