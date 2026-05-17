"""
Tests for the REST API gateway.

Tests the FastAPI application directly using httpx AsyncClient — no network,
no Solana node, no model loading required.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


def _make_app():
    """Build a gateway app with all external deps stubbed."""
    import sys
    import types

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


class TestHealthAndMetrics:
    def test_health_returns_ok(self):
        app = _make_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "uptime_seconds" in data

    def test_metrics_returns_prometheus_format(self):
        app = _make_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/metrics")
        # 200 when prometheus_client is installed; 503 when it is not.
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            assert "# TYPE" in resp.text


class TestModels:
    def test_list_models_returns_all_four(self):
        app = _make_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        ids = [m["id"] for m in data["data"]]
        assert "llama-3.2-3b" in ids
        assert "llama-3.1-8b" in ids
        assert "mistral-7b" in ids

    def test_model_object_has_required_fields(self):
        app = _make_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/v1/models")
        for model in resp.json()["data"]:
            assert "id" in model
            assert "object" in model
            assert "owned_by" in model


class TestChatML:
    def test_build_chatml_single_user_message(self):
        from scripts.api_gateway import ChatMessage, _build_chatml

        msgs = [ChatMessage(role="user", content="Hello")]
        result = _build_chatml(msgs)
        assert "<|im_start|>user" in result
        assert "Hello" in result
        assert "<|im_end|>" in result
        assert result.endswith("<|im_start|>assistant\n")

    def test_build_chatml_system_and_user(self):
        from scripts.api_gateway import ChatMessage, _build_chatml

        msgs = [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="What is 2+2?"),
        ]
        result = _build_chatml(msgs)
        assert "<|im_start|>system" in result
        assert "<|im_start|>user" in result
        assert "You are a helpful assistant." in result
        assert "What is 2+2?" in result


class TestCompletionEndpoint:
    def test_completion_success(self):
        from client.python.client import CompletionResponse

        mock_result = CompletionResponse(
            text="The answer is 42.",
            job_id=1,
            model="llama-3.2-3b",
            tokens_used=5,
            total_paid=512,
            node="NodePubkey123",
        )

        app = _make_app()
        with patch("scripts.api_gateway._client") as mock_client:
            mock_client.complete = AsyncMock(return_value=mock_result)
            with TestClient(app, raise_server_exceptions=False) as client:
                resp = client.post(
                    "/v1/completions",
                    json={"prompt": "What is 6x7?", "model": "llama-3.2-3b"},
                )

        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "text_completion"
        assert data["choices"][0]["text"] == "The answer is 42."
        assert data["usage"]["completion_tokens"] == 5

    def test_completion_timeout_returns_504(self):
        app = _make_app()
        with patch("scripts.api_gateway._client") as mock_client:
            mock_client.complete = AsyncMock(side_effect=TimeoutError("deadline"))
            with TestClient(app, raise_server_exceptions=False) as client:
                resp = client.post(
                    "/v1/completions",
                    json={"prompt": "hello"},
                )
        assert resp.status_code == 504

    def test_completion_server_error_returns_500(self):
        app = _make_app()
        with patch("scripts.api_gateway._client") as mock_client:
            mock_client.complete = AsyncMock(side_effect=RuntimeError("node offline"))
            with TestClient(app, raise_server_exceptions=False) as client:
                resp = client.post(
                    "/v1/completions",
                    json={"prompt": "hello"},
                )
        assert resp.status_code == 500


class TestChatCompletionEndpoint:
    def test_chat_completion_success(self):
        from client.python.client import CompletionResponse

        mock_result = CompletionResponse(
            text="Paris.",
            job_id=2,
            model="llama-3.2-3b",
            tokens_used=1,
            total_paid=256,
            node="NodePubkey456",
        )

        app = _make_app()
        with patch("scripts.api_gateway._client") as mock_client:
            mock_client.complete = AsyncMock(return_value=mock_result)
            with TestClient(app, raise_server_exceptions=False) as client:
                resp = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "llama-3.2-3b",
                        "messages": [{"role": "user", "content": "Capital of France?"}],
                    },
                )

        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "Paris."

    def test_chat_completion_increments_metrics(self):
        from client.python.client import CompletionResponse
        from scripts.api_gateway import _metrics

        before = _metrics["requests_total"]
        mock_result = CompletionResponse(
            text="ok", job_id=3, model="llama-3.2-3b", tokens_used=1, total_paid=100, node="node"
        )

        app = _make_app()
        with patch("scripts.api_gateway._client") as mock_client:
            mock_client.complete = AsyncMock(return_value=mock_result)
            with TestClient(app, raise_server_exceptions=False) as client:
                client.post(
                    "/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )

        assert _metrics["requests_total"] > before


class TestWebhook:
    def test_valid_paysh_webhook(self):
        from integrations.paysh.handler import PaymentEvent

        mock_event = PaymentEvent(
            payment_id="pay_123",
            customer_wallet="CustomerWallet",
            amount_usd_cents=1000,
            tokens_to_mint=100_000,
            status="completed",
        )

        app = _make_app()
        with patch("scripts.api_gateway._paysh") as mock_paysh:
            mock_paysh.process_webhook = MagicMock(return_value=mock_event)
            with TestClient(app, raise_server_exceptions=False) as client:
                resp = client.post(
                    "/webhooks/paysh",
                    content=b'{"payment_id":"pay_123"}',
                    headers={"X-Paysh-Signature": "sig"},
                )

        assert resp.status_code == 200
        assert resp.json()["tokens_minted"] == 100_000

    def test_invalid_signature_returns_401(self):
        app = _make_app()
        with patch("scripts.api_gateway._paysh") as mock_paysh:
            mock_paysh.process_webhook = MagicMock(side_effect=ValueError("bad sig"))
            with TestClient(app, raise_server_exceptions=False) as client:
                resp = client.post(
                    "/webhooks/paysh",
                    content=b"{}",
                    headers={"X-Paysh-Signature": "wrong"},
                )
        assert resp.status_code == 401
