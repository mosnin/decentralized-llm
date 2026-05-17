"""Tests for NetworkStatsCollector and the /v1/network/stats API endpoint."""

import time
from unittest.mock import patch

import pytest

from node.network_stats import NetworkSnapshot, NetworkStatsCollector

# ── unit tests ────────────────────────────────────────────────────────────────


class TestNetworkStatsCollector:
    def test_empty_snapshot_defaults(self):
        collector = NetworkStatsCollector()
        snap = collector.snapshot()

        assert isinstance(snap, NetworkSnapshot)
        assert snap.total_nodes == 0
        assert snap.active_nodes == 0
        assert snap.total_jobs_24h == 0
        assert snap.completed_jobs_24h == 0
        assert snap.failed_jobs_24h == 0
        assert snap.avg_latency_ms == 0.0
        assert snap.total_staked_tokens == 0
        assert snap.models_available == []

    def test_record_job_success_increments_counter(self):
        collector = NetworkStatsCollector()
        collector.record_job(success=True, latency_ms=100.0, model="llama-3.2-3b")
        collector.record_job(success=True, latency_ms=200.0, model="llama-3.2-3b")

        snap = collector.snapshot()
        assert snap.total_jobs_24h == 2
        assert snap.completed_jobs_24h == 2
        assert snap.failed_jobs_24h == 0

    def test_record_job_failure_increments_counter(self):
        collector = NetworkStatsCollector()
        collector.record_job(success=True, latency_ms=100.0, model="llama-3.2-3b")
        collector.record_job(success=False, latency_ms=0.0, model="llama-3.2-3b")
        collector.record_job(success=False, latency_ms=0.0, model="mistral-7b")

        snap = collector.snapshot()
        assert snap.total_jobs_24h == 3
        assert snap.completed_jobs_24h == 1
        assert snap.failed_jobs_24h == 2

    def test_active_node_count_window(self):
        collector = NetworkStatsCollector()

        now = time.time()
        # One node with a fresh heartbeat, one node that last checked in 10 min ago
        collector._heartbeats["node-a"] = now - 60  # 1 minute ago — active
        collector._heartbeats["node-b"] = now - 700  # ~11.7 minutes ago — inactive

        assert collector.active_node_count(window_seconds=300) == 1
        assert collector.active_node_count(window_seconds=900) == 2

    def test_old_jobs_pruned_from_24h_window(self):
        collector = NetworkStatsCollector()

        now = time.time()
        # Add a recent job and a job from 25 hours ago
        collector._jobs.append((now - 90000, True, 500.0, "llama-3.2-3b"))  # 25 h old
        collector._jobs.append((now - 3600, True, 200.0, "mistral-7b"))  # 1 h old
        collector._jobs.append((now - 100, False, 0.0, "llama-3.2-3b"))  # recent

        snap = collector.snapshot()
        # 25-h-old entry must be pruned
        assert snap.total_jobs_24h == 2
        assert snap.completed_jobs_24h == 1
        assert snap.failed_jobs_24h == 1

    def test_snapshot_success_rate_calculation(self):
        collector = NetworkStatsCollector()
        for _ in range(9):
            collector.record_job(success=True, latency_ms=100.0, model="llama-3.2-3b")
        collector.record_job(success=False, latency_ms=0.0, model="llama-3.2-3b")

        snap = collector.snapshot()
        assert snap.total_jobs_24h == 10
        assert snap.completed_jobs_24h == 9
        rate = snap.completed_jobs_24h / snap.total_jobs_24h
        assert abs(rate - 0.9) < 1e-9

    def test_models_available_deduplicated(self):
        collector = NetworkStatsCollector()
        collector.record_node_registered(
            "node-1", staked=1000, models=["llama-3.2-3b", "mistral-7b"]
        )
        collector.record_node_registered(
            "node-2", staked=2000, models=["mistral-7b", "llama-3.1-8b"]
        )

        snap = collector.snapshot()
        # Deduplication — each model listed once
        assert len(snap.models_available) == len(set(snap.models_available))
        assert "llama-3.2-3b" in snap.models_available
        assert "mistral-7b" in snap.models_available
        assert "llama-3.1-8b" in snap.models_available

    def test_total_staked_tokens_aggregated(self):
        collector = NetworkStatsCollector()
        collector.record_node_registered("node-1", staked=500, models=[])
        collector.record_node_registered("node-2", staked=1500, models=[])

        snap = collector.snapshot()
        assert snap.total_staked_tokens == 2000
        assert snap.total_nodes == 2


# ── integration test: API endpoint ────────────────────────────────────────────

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


def _make_app():
    """Build a gateway app with all external deps stubbed."""
    import sys
    import types

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


class TestNetworkStatsEndpoint:
    def test_network_stats_endpoint_returns_200(self):
        app = _make_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/v1/network/stats")

        assert resp.status_code == 200
        data = resp.json()
        assert "timestamp" in data
        assert "total_nodes" in data
        assert "active_nodes" in data
        assert "total_jobs_24h" in data
        assert "completed_jobs_24h" in data
        assert "success_rate" in data
        assert "avg_latency_ms" in data
        assert "models_available" in data

    def test_network_stats_endpoint_empty_defaults(self):
        app = _make_app()

        fresh_collector = NetworkStatsCollector()
        with patch("scripts.api_gateway._stats_collector", fresh_collector):
            with TestClient(app, raise_server_exceptions=False) as client:
                resp = client.get("/v1/network/stats")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total_nodes"] == 0
        assert data["active_nodes"] == 0
        assert data["success_rate"] == 0.0
        assert data["models_available"] == []

    def test_network_stats_endpoint_success_rate_with_data(self):
        app = _make_app()

        collector = NetworkStatsCollector()
        collector.record_node_registered("n1", staked=1000, models=["llama-3.2-3b", "mistral-7b"])
        collector.record_node_heartbeat("n1")
        for _ in range(148):
            collector.record_job(success=True, latency_ms=3200.0, model="llama-3.2-3b")
        for _ in range(2):
            collector.record_job(success=False, latency_ms=0.0, model="llama-3.2-3b")

        with patch("scripts.api_gateway._stats_collector", collector):
            with TestClient(app, raise_server_exceptions=False) as client:
                resp = client.get("/v1/network/stats")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total_jobs_24h"] == 150
        assert data["completed_jobs_24h"] == 148
        assert abs(data["success_rate"] - 148 / 150) < 1e-9
        assert abs(data["avg_latency_ms"] - 3200.0) < 1e-6
        assert "llama-3.2-3b" in data["models_available"]
        assert "mistral-7b" in data["models_available"]
