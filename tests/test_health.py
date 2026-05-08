"""
Tests for /health, /ready endpoints and the HealthChecker helper.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


def _make_app():
    """Build a gateway app with all external deps stubbed."""
    for mod in ("anchorpy", "solana", "solders", "spl"):
        if mod not in sys.modules:
            sys.modules[mod] = types.ModuleType(mod)

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


# ─────────────────────────── /health ─────────────────────────────────────────


class TestLivenessEndpoint:
    def test_liveness_returns_200(self):
        app = _make_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/health")
        assert resp.status_code == 200

    def test_liveness_has_uptime_field(self):
        app = _make_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/health")
        data = resp.json()
        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], float | int)

    def test_liveness_has_version_and_status(self):
        app = _make_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"


# ─────────────────────────── /ready ──────────────────────────────────────────


class TestReadinessEndpoint:
    def test_readiness_when_client_connected(self):
        app = _make_app()
        mock_client = MagicMock()
        with patch("scripts.api_gateway._client", mock_client):
            with TestClient(app, raise_server_exceptions=False) as client:
                resp = client.get("/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["checks"]["blockchain"] == "ok"

    def test_readiness_when_client_none_is_degraded(self):
        app = _make_app()
        with patch("scripts.api_gateway._client", None):
            with TestClient(app, raise_server_exceptions=False) as client:
                resp = client.get("/ready")
        assert resp.status_code == 503
        data = resp.json()
        assert data["checks"]["blockchain"] == "unavailable"
        assert data["status"] == "not_ready"

    def test_readiness_queue_full_is_degraded(self):
        from node.health import HealthChecker

        checker = HealthChecker()

        async def _run():
            return await checker.check_all(client=MagicMock(), queue_depth=100, active_jobs=5)

        import asyncio

        result = asyncio.run(_run())
        assert result["checks"]["job_queue"] == "degraded"
        assert result["status"] == "degraded"


# ─────────────────────────── HealthChecker unit tests ────────────────────────


class TestHealthCheckerUnit:
    def test_health_checker_check_all_returns_dict(self):
        from node.health import HealthChecker

        checker = HealthChecker()

        async def _run():
            return await checker.check_all(client=MagicMock(), queue_depth=0, active_jobs=0)

        import asyncio

        result = asyncio.run(_run())
        assert isinstance(result, dict)
        assert "status" in result
        assert "checks" in result
        assert "queue_depth" in result
        assert "active_jobs" in result

    def test_health_checker_liveness_returns_dict(self):
        from node.health import HealthChecker

        checker = HealthChecker()
        result = checker.liveness()
        assert isinstance(result, dict)
        assert result["status"] == "ok"
        assert "uptime_seconds" in result
        assert "version" in result

    def test_health_checker_ipfs_always_ok(self):
        from node.health import HealthChecker

        checker = HealthChecker()

        async def _run():
            return await checker.check_all(client=None, queue_depth=0, active_jobs=0)

        import asyncio

        result = asyncio.run(_run())
        assert result["checks"]["ipfs"] == "ok"
        assert result["checks"]["shard_manager"] == "ok"
