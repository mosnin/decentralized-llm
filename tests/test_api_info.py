"""
Tests for the /v1/info and /v1/models API discovery endpoints.
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


class TestInfoEndpoint:
    def test_info_returns_200(self):
        app = _make_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/v1/info")
        assert resp.status_code == 200

    def test_info_has_api_version(self):
        app = _make_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/v1/info")
        data = resp.json()
        assert "api_version" in data
        assert data["api_version"] == "1.0.0"

    def test_info_has_endpoints_dict(self):
        app = _make_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/v1/info")
        data = resp.json()
        assert "endpoints" in data
        assert isinstance(data["endpoints"], dict)
        assert "inference" in data["endpoints"]
        assert "health" in data["endpoints"]

    def test_info_has_features_list(self):
        app = _make_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/v1/info")
        data = resp.json()
        assert "features" in data
        assert isinstance(data["features"], list)
        assert len(data["features"]) > 0


class TestModelsEndpoint:
    def test_models_returns_200(self):
        app = _make_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/v1/models")
        assert resp.status_code == 200

    def test_models_has_models_list(self):
        app = _make_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/v1/models")
        data = resp.json()
        assert "models" in data
        assert isinstance(data["models"], list)
        assert len(data["models"]) > 0

    def test_models_each_has_id(self):
        app = _make_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/v1/models")
        data = resp.json()
        for model in data["models"]:
            assert "id" in model
