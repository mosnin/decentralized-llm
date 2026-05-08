"""Tests for MetricsCollector."""

import pytest

from node.metrics_collector import InferenceMetricEvent, MetricsCollector


def make_event(
    model_name: str = "test-model",
    tokens_generated: int = 100,
    latency_ms: float = 50.0,
    success: bool = True,
    job_id: int = 1,
) -> InferenceMetricEvent:
    return InferenceMetricEvent(
        model_name=model_name,
        tokens_generated=tokens_generated,
        latency_ms=latency_ms,
        success=success,
        job_id=job_id,
    )


class TestMetricsCollector:
    def setup_method(self):
        self.collector = MetricsCollector()

    def test_record_increments_total_requests(self):
        self.collector.record(make_event())
        self.collector.record(make_event())
        assert self.collector.snapshot()["total_requests"] == 2

    def test_record_failure_increments_failed(self):
        self.collector.record(make_event(success=True))
        self.collector.record(make_event(success=False))
        snap = self.collector.snapshot()
        assert snap["failed_requests"] == 1

    def test_success_rate_all_success(self):
        for _ in range(5):
            self.collector.record(make_event(success=True))
        assert self.collector.snapshot()["success_rate"] == 1.0

    def test_success_rate_mixed(self):
        for _ in range(2):
            self.collector.record(make_event(success=True))
        for _ in range(2):
            self.collector.record(make_event(success=False))
        assert self.collector.snapshot()["success_rate"] == pytest.approx(0.5)

    def test_success_rate_no_requests(self):
        # With no requests, success_rate should be 1.0 (no failures)
        assert self.collector.snapshot()["success_rate"] == 1.0

    def test_total_tokens_accumulates(self):
        self.collector.record(make_event(tokens_generated=100))
        self.collector.record(make_event(tokens_generated=200))
        self.collector.record(make_event(tokens_generated=50))
        assert self.collector.snapshot()["total_tokens"] == 350

    def test_avg_latency_correct(self):
        self.collector.record(make_event(latency_ms=100.0))
        self.collector.record(make_event(latency_ms=200.0))
        snap = self.collector.snapshot()
        assert snap["avg_latency_ms"] == pytest.approx(150.0)

    def test_percentile_p50(self):
        # Record 10 events with latencies 10, 20, ..., 100 ms
        for i in range(1, 11):
            self.collector.record(make_event(latency_ms=float(i * 10)))
        p50 = self.collector.snapshot()["p50_latency_ms"]
        # p50 of 10 items: index = max(0, int(10 * 50/100) - 1) = max(0, 4) = 4 → sorted[4] = 50.0
        assert p50 == pytest.approx(50.0)

    def test_percentile_p95(self):
        # Record 100 events with latencies 1..100 ms
        for i in range(1, 101):
            self.collector.record(make_event(latency_ms=float(i)))
        p95 = self.collector.snapshot()["p95_latency_ms"]
        # index = max(0, int(100 * 95/100) - 1) = max(0, 94) = 94 → sorted[94] = 95.0
        assert p95 == pytest.approx(95.0)

    def test_percentile_empty_returns_zero(self):
        assert self.collector.percentile(50) == 0.0
        assert self.collector.snapshot()["p50_latency_ms"] == 0.0

    def test_per_model_tracked_separately(self):
        self.collector.record(make_event(model_name="modelA", tokens_generated=100, success=True))
        self.collector.record(make_event(model_name="modelA", tokens_generated=50, success=False))
        self.collector.record(make_event(model_name="modelB", tokens_generated=200, success=True))

        per_model = self.collector.snapshot()["per_model"]

        assert per_model["modelA"]["requests"] == 2
        assert per_model["modelA"]["failures"] == 1
        assert per_model["modelA"]["tokens"] == 150

        assert per_model["modelB"]["requests"] == 1
        assert per_model["modelB"]["failures"] == 0
        assert per_model["modelB"]["tokens"] == 200

    def test_snapshot_has_all_keys(self):
        snap = self.collector.snapshot()
        expected_keys = {
            "uptime_seconds",
            "total_requests",
            "failed_requests",
            "success_rate",
            "total_tokens",
            "avg_latency_ms",
            "p50_latency_ms",
            "p95_latency_ms",
            "p99_latency_ms",
            "per_model",
        }
        assert set(snap.keys()) == expected_keys

    def test_reset_clears_all(self):
        for _ in range(5):
            self.collector.record(make_event(tokens_generated=100, latency_ms=50.0))
        self.collector.reset()
        snap = self.collector.snapshot()
        assert snap["total_requests"] == 0
        assert snap["failed_requests"] == 0
        assert snap["total_tokens"] == 0
        assert snap["avg_latency_ms"] == 0.0
        assert snap["p50_latency_ms"] == 0.0
        assert snap["per_model"] == {}

    def test_latency_buckets_capped(self):
        for i in range(10_001):
            self.collector.record(make_event(latency_ms=float(i)))
        # pylint: disable=protected-access
        assert len(self.collector._latency_buckets) <= 10_000
