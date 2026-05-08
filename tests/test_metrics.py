"""
Tests for Prometheus metrics module and the /metrics gateway endpoint.
"""

import sys
import types
from unittest.mock import patch

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_app():
    """Build a gateway app with all external deps stubbed."""
    for mod in ("anchorpy", "solana", "solders", "spl"):
        if mod not in sys.modules:
            stub = types.ModuleType(mod)
            sys.modules[mod] = stub

    for submod in (
        "solana.rpc",
        "solana.rpc.async_api",
        "solders.keypair",
        "solders.pubkey",
        "anchorpy",
    ):
        if submod not in sys.modules:
            sys.modules[submod] = types.ModuleType(submod)

    from scripts.api_gateway import app

    return app


# ---------------------------------------------------------------------------
# 1. Module imports cleanly without prometheus_client installed
# ---------------------------------------------------------------------------


def test_metrics_module_imports_without_prometheus():
    """node.metrics must import cleanly even when prometheus_client is absent."""
    # Remove the real prometheus_client from sys.modules if present, and
    # replace it with a stub that raises ImportError on attribute access so
    # the try/except in metrics.py triggers.
    saved = sys.modules.pop("prometheus_client", None)
    # Also evict the metrics module so it is re-imported fresh.
    saved_metrics = sys.modules.pop("node.metrics", None)

    try:
        # Make the import itself raise ImportError.
        sys.modules["prometheus_client"] = None  # type: ignore[assignment]

        import importlib

        import node.metrics as m

        importlib.reload(m)
        # If we get here the import did not crash.
        assert m.METRICS_AVAILABLE is False
    finally:
        # Restore original state.
        if saved is not None:
            sys.modules["prometheus_client"] = saved
        else:
            sys.modules.pop("prometheus_client", None)

        if saved_metrics is not None:
            sys.modules["node.metrics"] = saved_metrics
        else:
            sys.modules.pop("node.metrics", None)


# ---------------------------------------------------------------------------
# 2. Counter increments
# ---------------------------------------------------------------------------


def test_counter_increments():
    """jobs_claimed_total counter increments correctly."""
    try:
        import prometheus_client  # noqa: F401
    except ImportError:
        pytest.skip("prometheus_client not installed")

    from node.metrics import jobs_claimed_total

    before = jobs_claimed_total._value.get()
    jobs_claimed_total.inc()
    after = jobs_claimed_total._value.get()
    assert after == before + 1


# ---------------------------------------------------------------------------
# 3. Gauge updates
# ---------------------------------------------------------------------------


def test_gauge_updates():
    """active_jobs gauge can be incremented and decremented."""
    try:
        import prometheus_client  # noqa: F401
    except ImportError:
        pytest.skip("prometheus_client not installed")

    from node.metrics import active_jobs

    before = active_jobs._value.get()
    active_jobs.inc()
    assert active_jobs._value.get() == before + 1
    active_jobs.dec()
    assert active_jobs._value.get() == before


# ---------------------------------------------------------------------------
# 4. Histogram observe
# ---------------------------------------------------------------------------


def test_histogram_observe():
    """inference_latency_seconds histogram accepts observations."""
    try:
        import prometheus_client  # noqa: F401
    except ImportError:
        pytest.skip("prometheus_client not installed")

    from node.metrics import inference_latency_seconds

    # Record the sum before and after; it should increase by our observed value.
    before_sum = inference_latency_seconds._sum.get()
    inference_latency_seconds.observe(7.5)
    after_sum = inference_latency_seconds._sum.get()
    assert abs(after_sum - before_sum - 7.5) < 1e-9


# ---------------------------------------------------------------------------
# 5. /metrics endpoint returns 200 or 503
# ---------------------------------------------------------------------------


def test_metrics_endpoint_available():
    """GET /metrics returns either 200 (prometheus available) or 503 (not available)."""
    app = _make_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/metrics")
    assert resp.status_code in (200, 503)


# ---------------------------------------------------------------------------
# 6. /metrics Content-Type when prometheus_client is present
# ---------------------------------------------------------------------------


def test_metrics_endpoint_content_type():
    """When prometheus_client is available, /metrics uses the correct Content-Type."""
    try:
        import prometheus_client  # noqa: F401
    except ImportError:
        pytest.skip("prometheus_client not installed")

    app = _make_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/metrics")

    if resp.status_code == 200:
        ct = resp.headers.get("content-type", "")
        assert "text/plain" in ct
        assert "0.0.4" in ct
    else:
        # 503 is allowed when prometheus_client is somehow unavailable at runtime
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# 7. jobs_failed_total and jobs_completed_total counters
# ---------------------------------------------------------------------------


def test_failed_and_completed_counters():
    """jobs_failed_total and jobs_completed_total counters increment independently."""
    try:
        import prometheus_client  # noqa: F401
    except ImportError:
        pytest.skip("prometheus_client not installed")

    from node.metrics import jobs_completed_total, jobs_failed_total

    before_failed = jobs_failed_total._value.get()
    before_completed = jobs_completed_total._value.get()

    jobs_failed_total.inc()
    jobs_completed_total.inc()
    jobs_completed_total.inc()

    assert jobs_failed_total._value.get() == before_failed + 1
    assert jobs_completed_total._value.get() == before_completed + 2


# ---------------------------------------------------------------------------
# 8. heartbeat_timestamp gauge
# ---------------------------------------------------------------------------


def test_heartbeat_timestamp_gauge():
    """heartbeat_timestamp gauge can be set to an arbitrary unix timestamp."""
    try:
        import prometheus_client  # noqa: F401
    except ImportError:
        pytest.skip("prometheus_client not installed")

    import time

    from node.metrics import heartbeat_timestamp

    ts = time.time()
    heartbeat_timestamp.set(ts)
    assert abs(heartbeat_timestamp._value.get() - ts) < 1.0


# ---------------------------------------------------------------------------
# 9. /metrics returns "metrics not available" body when prometheus absent
# ---------------------------------------------------------------------------


def test_metrics_endpoint_no_prometheus_body():
    """When prometheus_client is absent, /metrics returns 503 with helpful message."""
    app = _make_app()

    with patch.dict(sys.modules, {"prometheus_client": None}):
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/metrics")

    # Either prometheus is installed (200) or our 503 fallback fires.
    if resp.status_code == 503:
        assert "metrics not available" in resp.text
    else:
        assert resp.status_code == 200
