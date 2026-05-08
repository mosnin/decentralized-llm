"""
Tests for the GET /v1/stream/{job_id} SSE endpoint.
"""

import sys
import types
from unittest.mock import patch

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


class TestStreamEndpoint:
    def test_stream_endpoint_404_when_no_stream(self):
        """GET /v1/stream/999 with no stream registered returns 404."""
        app = _make_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/v1/stream/999")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_stream_endpoint_yields_sse_events(self):
        """A stream with 2 tokens produces correct SSE data lines."""
        import asyncio

        from node.token_streamer import TokenStreamRegistry

        registry = TokenStreamRegistry()
        stream = registry.create(job_id=1)

        # Pre-populate the stream so TestClient can drain it synchronously
        asyncio.run(stream.push("token1"))
        asyncio.run(stream.push("token2"))
        asyncio.run(stream.finish())

        app = _make_app()
        with patch("scripts.api_gateway._token_stream_registry", registry):
            with TestClient(app, raise_server_exceptions=False) as client:
                resp = client.get("/v1/stream/1")

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = resp.text
        assert "data: token1\n\n" in body
        assert "data: token2\n\n" in body
        assert "data: [DONE]\n\n" in body

    def test_stream_endpoint_cache_control_headers(self):
        """The SSE response must carry no-cache and buffering-disable headers."""
        import asyncio

        from node.token_streamer import TokenStreamRegistry

        registry = TokenStreamRegistry()
        stream = registry.create(job_id=2)
        asyncio.run(stream.finish())

        app = _make_app()
        with patch("scripts.api_gateway._token_stream_registry", registry):
            with TestClient(app, raise_server_exceptions=False) as client:
                resp = client.get("/v1/stream/2")

        assert resp.status_code == 200
        assert resp.headers.get("cache-control") == "no-cache"
        assert resp.headers.get("x-accel-buffering") == "no"

    def test_stream_endpoint_done_sentinel_present_with_no_tokens(self):
        """A stream that finishes immediately still sends [DONE]."""
        import asyncio

        from node.token_streamer import TokenStreamRegistry

        registry = TokenStreamRegistry()
        stream = registry.create(job_id=3)
        asyncio.run(stream.finish())

        app = _make_app()
        with patch("scripts.api_gateway._token_stream_registry", registry):
            with TestClient(app, raise_server_exceptions=False) as client:
                resp = client.get("/v1/stream/3")

        assert "data: [DONE]\n\n" in resp.text
