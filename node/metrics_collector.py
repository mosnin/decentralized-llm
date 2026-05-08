"""
In-memory inference metrics collector.

Works standalone — does NOT require prometheus_client.
"""

import time
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class InferenceMetricEvent:
    model_name: str
    tokens_generated: int
    latency_ms: float
    success: bool
    job_id: int


class MetricsCollector:
    """
    Aggregates inference metrics in memory.
    Works standalone — does NOT require prometheus_client.
    """

    def __init__(self) -> None:
        self._start_time = time.time()
        self._total_requests: int = 0
        self._failed_requests: int = 0
        self._total_tokens: int = 0
        self._total_latency_ms: float = 0.0
        self._per_model: dict[str, dict] = defaultdict(
            lambda: {"requests": 0, "failures": 0, "tokens": 0, "latency_ms": 0.0}
        )
        self._latency_buckets: list[float] = []  # for percentile calc

    def record(self, event: InferenceMetricEvent) -> None:
        """Record a completed inference event."""
        self._total_requests += 1
        if not event.success:
            self._failed_requests += 1
        self._total_tokens += event.tokens_generated
        self._total_latency_ms += event.latency_ms
        self._latency_buckets.append(event.latency_ms)
        # Keep only last 10000 latency samples to bound memory
        if len(self._latency_buckets) > 10_000:
            self._latency_buckets = self._latency_buckets[-10_000:]

        m = self._per_model[event.model_name]
        m["requests"] += 1
        if not event.success:
            m["failures"] += 1
        m["tokens"] += event.tokens_generated
        m["latency_ms"] += event.latency_ms

    def percentile(self, p: float) -> float:
        """Return the p-th percentile latency (0-100). Returns 0.0 if no data."""
        if not self._latency_buckets:
            return 0.0
        sorted_buckets = sorted(self._latency_buckets)
        idx = max(0, int(len(sorted_buckets) * p / 100) - 1)
        return sorted_buckets[idx]

    def snapshot(self) -> dict:
        """
        Return current aggregated metrics::

            {
                "uptime_seconds": float,
                "total_requests": int,
                "failed_requests": int,
                "success_rate": float,    # 0.0-1.0
                "total_tokens": int,
                "avg_latency_ms": float,
                "p50_latency_ms": float,
                "p95_latency_ms": float,
                "p99_latency_ms": float,
                "per_model": {model_name: {"requests": int, "failures": int, "tokens": int}},
            }
        """
        if self._total_requests > 0:
            success_rate = (self._total_requests - self._failed_requests) / self._total_requests
            avg_latency_ms = self._total_latency_ms / self._total_requests
        else:
            success_rate = 1.0
            avg_latency_ms = 0.0

        per_model: dict[str, dict] = {}
        for model_name, data in self._per_model.items():
            per_model[model_name] = {
                "requests": data["requests"],
                "failures": data["failures"],
                "tokens": data["tokens"],
            }

        return {
            "uptime_seconds": time.time() - self._start_time,
            "total_requests": self._total_requests,
            "failed_requests": self._failed_requests,
            "success_rate": success_rate,
            "total_tokens": self._total_tokens,
            "avg_latency_ms": avg_latency_ms,
            "p50_latency_ms": self.percentile(50),
            "p95_latency_ms": self.percentile(95),
            "p99_latency_ms": self.percentile(99),
            "per_model": per_model,
        }

    def reset(self) -> None:
        """Clear all metrics (useful in tests)."""
        self.__init__()
