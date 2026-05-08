"""
Tests for GET /v1/dashboard endpoint.
"""

import sys
import types

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


class TestDashboardEndpoint:
    def test_dashboard_returns_200(self):
        app = _make_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/v1/dashboard")
        assert resp.status_code == 200

    def test_dashboard_has_inference_key(self):
        app = _make_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/v1/dashboard")
        assert "inference" in resp.json()

    def test_dashboard_has_network_key(self):
        app = _make_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/v1/dashboard")
        assert "network" in resp.json()

    def test_dashboard_has_health_key(self):
        app = _make_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/v1/dashboard")
        assert "health" in resp.json()

    def test_dashboard_has_timestamp(self):
        app = _make_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/v1/dashboard")
        data = resp.json()
        assert "timestamp" in data
        assert isinstance(data["timestamp"], float | int)

    def test_dashboard_inference_success_rate_is_float(self):
        app = _make_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/v1/dashboard")
        data = resp.json()
        assert isinstance(data["inference"]["success_rate"], float)
