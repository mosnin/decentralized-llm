"""
Tests for OpenAPI response models and schema improvements in the API gateway.
"""

import sys
import types

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


def _make_app():
    """Build a gateway app with all external deps stubbed."""
    # Stub solana/anchorpy before importing gateway
    for mod in ("anchorpy", "solana", "solders", "spl"):
        if mod not in sys.modules:
            stub = types.ModuleType(mod)
            sys.modules[mod] = stub

    # Provide minimal stubs for sub-modules referenced at import time
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


class TestHealthResponseModel:
    def test_health_response_model_valid(self):
        from scripts.api_gateway import HealthResponse

        resp = HealthResponse(status="ok", version="0.1.0", uptime_seconds=12.5)
        assert resp.status == "ok"
        assert resp.version == "0.1.0"
        assert resp.uptime_seconds == 12.5

    def test_health_endpoint_returns_model_shape(self):
        app = _make_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "version" in data
        assert "uptime_seconds" in data


class TestReadinessResponseModel:
    def test_readiness_response_model_valid(self):
        from scripts.api_gateway import ReadinessResponse

        resp = ReadinessResponse(
            status="ready",
            checks={"blockchain": "ok", "model": "ok"},
            queue_depth=0,
            active_jobs=0,
        )
        assert resp.status == "ready"
        assert resp.checks["blockchain"] == "ok"
        assert resp.queue_depth == 0
        assert resp.active_jobs == 0

    def test_readiness_response_model_not_ready(self):
        from scripts.api_gateway import ReadinessResponse

        resp = ReadinessResponse(
            status="not_ready",
            checks={"blockchain": "unavailable"},
            queue_depth=5,
            active_jobs=2,
        )
        assert resp.status == "not_ready"
        assert resp.queue_depth == 5
        assert resp.active_jobs == 2


class TestCompletionRequestDefaults:
    def test_completion_request_defaults(self):
        from scripts.api_gateway import CompletionRequest

        req = CompletionRequest(prompt="hello")
        assert req.model == "llama-3.2-3b"
        assert req.max_tokens == 512
        assert req.payment_amount is None
        assert req.stream is False

    def test_completion_request_custom_values(self):
        from scripts.api_gateway import CompletionRequest

        req = CompletionRequest(
            model="llama-3.1-8b",
            prompt="test",
            max_tokens=1024,
            payment_amount=100,
            stream=True,
        )
        assert req.model == "llama-3.1-8b"
        assert req.max_tokens == 1024
        assert req.payment_amount == 100
        assert req.stream is True


class TestJobStatusResponseOptionalFields:
    def test_job_status_response_optional_fields(self):
        from scripts.api_gateway import JobStatusResponse

        # result_cid and node should be optional (None allowed)
        resp = JobStatusResponse(job_id=42, status="pending", result_cid=None, node=None)
        assert resp.job_id == 42
        assert resp.status == "pending"
        assert resp.result_cid is None
        assert resp.node is None

    def test_job_status_response_with_values(self):
        from scripts.api_gateway import JobStatusResponse

        resp = JobStatusResponse(
            job_id=7,
            status="completed",
            result_cid="bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi",
            node="NodePubkey123",
        )
        assert resp.job_id == 7
        assert resp.result_cid == "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
        assert resp.node == "NodePubkey123"


class TestOpenAPISchema:
    def test_openapi_schema_has_inference_tag(self):
        app = _make_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        # Collect all tags used across all paths
        tags_used = set()
        for path_item in schema.get("paths", {}).values():
            for operation in path_item.values():
                if isinstance(operation, dict):
                    for tag in operation.get("tags", []):
                        tags_used.add(tag)
        assert "inference" in tags_used

    def test_openapi_schema_has_ops_tag(self):
        app = _make_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        tags_used = set()
        for path_item in schema.get("paths", {}).values():
            for operation in path_item.values():
                if isinstance(operation, dict):
                    for tag in operation.get("tags", []):
                        tags_used.add(tag)
        assert "ops" in tags_used

    def test_openapi_schema_has_response_models(self):
        app = _make_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        # Verify HealthResponse and JobStatusResponse appear in components
        components = schema.get("components", {}).get("schemas", {})
        assert "HealthResponse" in components
        assert "JobStatusResponse" in components
