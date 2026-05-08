import pytest

from node.timeout_estimator import TimeoutEstimator


def test_estimate_default_with_no_samples():
    est = TimeoutEstimator(default_timeout_ms=30_000.0)
    result = est.estimate("unknown-model")
    assert result == 30_000.0


def test_estimate_default_with_few_samples():
    est = TimeoutEstimator(default_timeout_ms=30_000.0)
    for i in range(3):
        est.record("model_a", latency_ms=100.0 * (i + 1))
    result = est.estimate("model_a")
    assert result == 30_000.0


def test_estimate_with_enough_samples():
    est = TimeoutEstimator()
    for i in range(10):
        est.record("model_a", latency_ms=200.0)
    result = est.estimate("model_a")
    assert result > 0


def test_estimate_clamped_to_min():
    est = TimeoutEstimator(min_timeout_ms=1_000.0, safety_multiplier=1.5)
    # All very fast samples — p95 * 1.5 should still be below min
    for _ in range(10):
        est.record("fast_model", latency_ms=0.1)
    result = est.estimate("fast_model")
    assert result >= 1_000.0


def test_estimate_clamped_to_max():
    est = TimeoutEstimator(max_timeout_ms=300_000.0, safety_multiplier=1.5)
    # Very slow samples: 500_000ms each → p95 * 1.5 = 750_000 > max
    for _ in range(10):
        est.record("slow_model", latency_ms=500_000.0)
    result = est.estimate("slow_model")
    assert result <= 300_000.0


def test_estimate_scales_with_p95():
    est = TimeoutEstimator(min_timeout_ms=0.0, max_timeout_ms=float("inf"), safety_multiplier=1.0)
    # 9 samples at 100ms, 1 sample at 900ms
    # With 10 values, p95 index = max(0, int(10 * 95 / 100) - 1) = max(0, 9 - 1) = 8
    # sorted: [100]*9 + [900] → index 8 = 100ms (not the max)
    # But let's use a clear spread: 5 samples at 50, 5 at 200
    # sorted: [50,50,50,50,50,200,200,200,200,200]
    # p95 index = max(0, int(10*95/100)-1) = max(0,9-1) = 8 → value = 200
    for _ in range(5):
        est.record("test_model", latency_ms=50.0)
    for _ in range(5):
        est.record("test_model", latency_ms=200.0)
    result = est.estimate("test_model")
    # p95 = 200ms, multiplier = 1.0 → raw = 200ms
    assert result == pytest.approx(200.0)


def test_record_increments_sample_count():
    est = TimeoutEstimator()
    assert est.sample_count("model_x") == 0
    for _ in range(3):
        est.record("model_x", latency_ms=50.0)
    assert est.sample_count("model_x") == 3


def test_window_size_respected():
    est = TimeoutEstimator(window_size=5)
    for i in range(10):
        est.record("model_a", latency_ms=float(i * 10))
    assert est.sample_count("model_a") == 5


def test_reset_specific_model():
    est = TimeoutEstimator()
    for _ in range(5):
        est.record("model_a", latency_ms=100.0)
    for _ in range(5):
        est.record("model_b", latency_ms=200.0)
    est.reset("model_a")
    assert est.sample_count("model_a") == 0
    assert est.sample_count("model_b") == 5


def test_reset_all():
    est = TimeoutEstimator()
    for _ in range(5):
        est.record("model_a", latency_ms=100.0)
    for _ in range(5):
        est.record("model_b", latency_ms=200.0)
    est.reset()
    assert est.sample_count("model_a") == 0
    assert est.sample_count("model_b") == 0


def test_estimate_safety_multiplier_applied():
    # p95=100ms, multiplier=2.0 → raw=200ms; min=0 so result==200ms
    est = TimeoutEstimator(
        min_timeout_ms=0.0,
        max_timeout_ms=float("inf"),
        safety_multiplier=2.0,
    )
    # 10 identical samples at 100ms → p95 = 100ms
    for _ in range(10):
        est.record("model_a", latency_ms=100.0)
    result = est.estimate("model_a")
    assert result == pytest.approx(200.0)
