"""Tests for Pay.sh webhook handler."""

import hashlib
import hmac
import json
import time
import pytest

from integrations.paysh.handler import PayshHandler, PaymentEvent, TOKENS_PER_USD_CENT


def _make_signature(body: bytes, secret: str) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


class TestPayshHandler:
    SECRET = "test-webhook-secret-abc123"

    def _handler(self, on_payment=None):
        return PayshHandler(
            api_key="test-api-key",
            webhook_secret=self.SECRET,
            on_payment=on_payment,
        )

    def _payload(self, **overrides) -> dict:
        base = {
            "id": "pay_test_001",
            "amount": 500,  # $5.00
            "currency": "usd",
            "status": "completed",
            "created_at": int(time.time()),
            "metadata": {"solana_wallet": "AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQq"},
        }
        base.update(overrides)
        return base

    def _make_webhook(self, payload: dict) -> tuple[bytes, str]:
        body = json.dumps(payload).encode()
        sig = _make_signature(body, self.SECRET)
        return body, sig

    def test_valid_webhook_returns_event(self):
        body, sig = self._make_webhook(self._payload())
        event = self._handler().process_webhook(body, sig)

        assert isinstance(event, PaymentEvent)
        assert event.payment_id == "pay_test_001"
        assert event.amount_usd_cents == 500
        assert event.tokens_to_mint == 500 * TOKENS_PER_USD_CENT
        assert event.status == "completed"

    def test_invalid_signature_raises(self):
        body, _ = self._make_webhook(self._payload())
        with pytest.raises(ValueError, match="Invalid Pay.sh webhook signature"):
            self._handler().process_webhook(body, "bad-signature")

    def test_callback_called_on_completed(self):
        received = []
        body, sig = self._make_webhook(self._payload(status="completed"))
        self._handler(on_payment=received.append).process_webhook(body, sig)
        assert len(received) == 1
        assert received[0].tokens_to_mint == 500 * TOKENS_PER_USD_CENT

    def test_callback_not_called_on_failed(self):
        received = []
        body, sig = self._make_webhook(self._payload(status="failed"))
        self._handler(on_payment=received.append).process_webhook(body, sig)
        assert len(received) == 0

    def test_token_calculation(self):
        body, sig = self._make_webhook(self._payload(amount=100))  # $1.00
        event = self._handler().process_webhook(body, sig)
        assert event.tokens_to_mint == 100 * TOKENS_PER_USD_CENT
