"""
Tests for node.prometheus_exporter — pure-Python Prometheus text-format exporter.
"""

from __future__ import annotations

import re
import threading

from node.prometheus_exporter import (
    HistogramMetric,
    MetricType,
    NetworkMetrics,
    PrometheusMetric,
    PrometheusRegistry,
    get_metrics,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_counter(name: str, value: int) -> PrometheusMetric:
    return PrometheusMetric(
        name=name,
        help_text=f"Help for {name}",
        metric_type=MetricType.COUNTER,
        labels=[],
        value_fn=lambda: value,
    )


def make_gauge(name: str, value: float) -> PrometheusMetric:
    return PrometheusMetric(
        name=name,
        help_text=f"Help for {name}",
        metric_type=MetricType.GAUGE,
        labels=[],
        value_fn=lambda: value,
    )


# ---------------------------------------------------------------------------
# 1. Empty registry renders empty string
# ---------------------------------------------------------------------------


def test_empty_registry_renders_empty_string() -> None:
    registry = PrometheusRegistry()
    assert registry.render() == ""


# ---------------------------------------------------------------------------
# 2. Registry renders correct Prometheus format (HELP, TYPE, value lines)
# ---------------------------------------------------------------------------


def test_registry_renders_correct_format() -> None:
    registry = PrometheusRegistry()
    registry.register(make_counter("dllm_test_total", 7))
    output = registry.render()

    assert "# HELP dllm_test_total Help for dllm_test_total" in output
    assert "# TYPE dllm_test_total counter" in output
    assert "dllm_test_total 7" in output


# ---------------------------------------------------------------------------
# 3. Counter increments are reflected in render output
# ---------------------------------------------------------------------------


def test_counter_increments_reflected() -> None:
    state = {"v": 0}

    metric = PrometheusMetric(
        name="dllm_req_total",
        help_text="Requests",
        metric_type=MetricType.COUNTER,
        labels=[],
        value_fn=lambda: state["v"],
    )
    registry = PrometheusRegistry()
    registry.register(metric)

    assert "dllm_req_total 0" in registry.render()

    state["v"] = 42
    assert "dllm_req_total 42" in registry.render()


# ---------------------------------------------------------------------------
# 4. Gauge set/get
# ---------------------------------------------------------------------------


def test_gauge_set_and_get() -> None:
    nm = NetworkMetrics()
    nm.set_active_nodes(99)
    assert nm.snapshot_active_nodes() == 99

    nm.set_active_nodes(0)
    assert nm.snapshot_active_nodes() == 0


# ---------------------------------------------------------------------------
# 5. Labels rendered in {key="val"} format
# ---------------------------------------------------------------------------


def test_labels_rendered_correctly() -> None:
    registry = PrometheusRegistry()
    registry.register(
        PrometheusMetric(
            name="dllm_infer_total",
            help_text="Total",
            metric_type=MetricType.COUNTER,
            labels=["model", "status"],
            value_fn=lambda: {("llama-3", "success"): 42},
        )
    )
    output = registry.render()
    assert 'dllm_infer_total{model="llama-3",status="success"} 42' in output


# ---------------------------------------------------------------------------
# 6. Histogram buckets filled correctly
# ---------------------------------------------------------------------------


def test_histogram_buckets_filled_correctly() -> None:
    hist = HistogramMetric(buckets=[0.1, 0.5, 1.0])
    hist.observe(0.05)  # bucket <=0.1, <=0.5, <=1.0, +Inf
    hist.observe(0.3)  # bucket <=0.5, <=1.0, +Inf
    hist.observe(0.8)  # bucket <=1.0, +Inf
    hist.observe(2.0)  # bucket +Inf only

    # _counts is cumulative: each observe(v) increments every bucket where v <= bound
    # observe(0.05) -> [1,1,1,1]; observe(0.3) -> [1,2,2,2];
    # observe(0.8)  -> [1,2,3,3]; observe(2.0) -> [1,2,3,4]
    assert hist._counts[0] == 1  # <=0.1: only 0.05 fits
    assert hist._counts[1] == 2  # <=0.5: 0.05 and 0.3
    assert hist._counts[2] == 3  # <=1.0: 0.05, 0.3, 0.8
    assert hist._counts[3] == 4  # +Inf: all four
    assert hist._total == 4
    assert abs(hist._sum - (0.05 + 0.3 + 0.8 + 2.0)) < 1e-9


# ---------------------------------------------------------------------------
# 7. Histogram renders _bucket, _sum, _count lines
# ---------------------------------------------------------------------------


def test_histogram_renders_bucket_sum_count() -> None:
    hist = HistogramMetric(buckets=[1.0, 5.0])
    hist.observe(0.5)
    hist.observe(3.0)

    rendered = hist.render("dllm_latency_seconds", "", "Latency")

    # observe(0.5) -> le=1:1, le=5:1, +Inf:1
    # observe(3.0) -> le=1:1, le=5:2, +Inf:2
    assert "# HELP dllm_latency_seconds Latency" in rendered
    assert "# TYPE dllm_latency_seconds histogram" in rendered
    assert 'dllm_latency_seconds_bucket{le="1"} 1' in rendered
    assert 'dllm_latency_seconds_bucket{le="5"} 2' in rendered
    assert 'dllm_latency_seconds_bucket{le="+Inf"} 2' in rendered
    assert "dllm_latency_seconds_sum" in rendered
    assert "dllm_latency_seconds_count 2" in rendered


# ---------------------------------------------------------------------------
# 8. record_inference updates counters and histogram
# ---------------------------------------------------------------------------


def test_record_inference_updates_all() -> None:
    nm = NetworkMetrics()
    nm.record_inference("llama-3", latency_s=0.5, success=True, tokens=100)
    nm.record_inference("llama-3", latency_s=1.5, success=False, tokens=0)
    nm.record_inference("gpt-j", latency_s=0.2, success=True, tokens=50)

    reqs = nm.snapshot_inference_requests()
    assert reqs[("llama-3", "success")] == 1
    assert reqs[("llama-3", "failure")] == 1
    assert reqs[("gpt-j", "success")] == 1

    tokens = nm.snapshot_tokens_generated()
    assert tokens[("llama-3",)] == 100
    assert tokens[("gpt-j",)] == 50

    # histogram received 3 observations
    assert nm.inference_latency_seconds._total == 3
    assert abs(nm.inference_latency_seconds._sum - (0.5 + 1.5 + 0.2)) < 1e-9


# ---------------------------------------------------------------------------
# 9. record_settlement increments counter
# ---------------------------------------------------------------------------


def test_record_settlement_increments_counter() -> None:
    nm = NetworkMetrics()
    assert nm.snapshot_settlement() == 0
    nm.record_settlement(1_000_000)
    nm.record_settlement(500_000)
    assert nm.snapshot_settlement() == 1_500_000


# ---------------------------------------------------------------------------
# 10. set_active_nodes updates gauge in registry render
# ---------------------------------------------------------------------------


def test_set_active_nodes_updates_render() -> None:
    nm = NetworkMetrics()
    nm.set_active_nodes(12)
    registry = nm.build_registry()
    output = registry.render()
    assert "dllm_active_nodes_total 12" in output


# ---------------------------------------------------------------------------
# 11. render output is parseable — every non-comment, non-empty line has a value
# ---------------------------------------------------------------------------


def test_render_output_parseable() -> None:
    nm = NetworkMetrics()
    nm.record_inference("m1", 0.3, True, 10)
    nm.set_active_nodes(5)
    nm.record_settlement(100)

    registry = nm.build_registry()
    output = registry.render()

    metric_line_re = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*(\{[^}]*\})? \S+$")

    for line in output.splitlines():
        if not line or line.startswith("#"):
            continue
        assert metric_line_re.match(line), f"Unparseable line: {line!r}"


# ---------------------------------------------------------------------------
# 12. Thread safety — concurrent observations don't corrupt state
# ---------------------------------------------------------------------------


def test_thread_safety_concurrent_observations() -> None:
    nm = NetworkMetrics()
    n_threads = 20
    n_per_thread = 500

    errors: list[Exception] = []

    def worker() -> None:
        try:
            for _ in range(n_per_thread):
                nm.record_inference("model-x", 0.1, True, 1)
                nm.record_settlement(1)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
    total_reqs = nm.snapshot_inference_requests()[("model-x", "success")]
    assert total_reqs == n_threads * n_per_thread
    assert nm.snapshot_settlement() == n_threads * n_per_thread


# ---------------------------------------------------------------------------
# 13. set_active_jobs updates gauge per status
# ---------------------------------------------------------------------------


def test_set_active_jobs_per_status() -> None:
    nm = NetworkMetrics()
    nm.set_active_jobs("running", 3)
    nm.set_active_jobs("pending", 7)

    jobs = nm.snapshot_active_jobs()
    assert jobs[("running",)] == 3
    assert jobs[("pending",)] == 7

    registry = nm.build_registry()
    output = registry.render()
    assert 'dllm_active_jobs_total{status="running"} 3' in output
    assert 'dllm_active_jobs_total{status="pending"} 7' in output


# ---------------------------------------------------------------------------
# 14. get_metrics() returns module-level singleton
# ---------------------------------------------------------------------------


def test_get_metrics_returns_singleton() -> None:
    m1 = get_metrics()
    m2 = get_metrics()
    assert m1 is m2


# ---------------------------------------------------------------------------
# 15. Histogram with labels renders label in bucket lines
# ---------------------------------------------------------------------------


def test_histogram_with_labels_string() -> None:
    hist = HistogramMetric(buckets=[1.0])
    hist.observe(0.5)
    rendered = hist.render("dllm_latency", 'model="llama-3"', "Latency")
    assert 'le="1",le=' not in rendered
    assert 'model="llama-3"' in rendered
    assert 'dllm_latency_sum{model="llama-3"}' in rendered
    assert 'dllm_latency_count{model="llama-3"} 1' in rendered
