"""
Pay.sh integration — fiat-to-token gateway for inference payments.

Flow:
  1. User initiates a payment through Pay.sh (card, bank, crypto)
  2. Pay.sh calls our webhook with a payment confirmation
  3. We verify the HMAC signature
  4. We mint/transfer the equivalent tokens to the user's Solana wallet
  5. User can then call client.complete() with those tokens

Pay.sh docs: https://docs.pay.sh
Set PAYSH_API_KEY and PAYSH_WEBHOOK_SECRET in your environment.
"""

import hashlib
import hmac
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Token price in USD cents (1 token = $0.0001 → 100 tokens per $0.01)
# Adjust based on actual tokenomics
TOKENS_PER_USD_CENT = 100


@dataclass
class PaymentEvent:
    payment_id: str
    amount_usd_cents: int
    customer_wallet: str  # Solana wallet address to receive tokens
    tokens_to_mint: int
    status: str  # "completed", "refunded", "failed"
    currency: str = "usd"  # "usd", "eur", "sol", "usdc", etc.
    timestamp: int = 0


class PayshHandler:
    """
    Handles incoming Pay.sh webhook events and triggers on-chain token minting.

    Example FastAPI integration:
        from fastapi import Request, HTTPException
        from integrations.paysh import PayshHandler

        handler = PayshHandler(
            api_key=os.environ["PAYSH_API_KEY"],
            webhook_secret=os.environ["PAYSH_WEBHOOK_SECRET"],
            on_payment=mint_tokens_to_wallet,
        )

        @app.post("/webhooks/paysh")
        async def paysh_webhook(request: Request):
            body = await request.body()
            sig = request.headers.get("X-Paysh-Signature", "")
            event = handler.process_webhook(body, sig)
            return {"status": "ok"}
    """

    def __init__(
        self,
        api_key: str,
        webhook_secret: str,
        on_payment: Callable[[PaymentEvent], None] | None = None,
    ):
        self.api_key = api_key
        self.webhook_secret = webhook_secret
        self.on_payment = on_payment

    def process_webhook(self, raw_body: bytes, signature: str) -> PaymentEvent:
        """
        Validate and parse an incoming Pay.sh webhook.
        Raises ValueError if signature is invalid.
        """
        self._verify_signature(raw_body, signature)

        payload = json.loads(raw_body)
        event = self._parse_event(payload)

        logger.info(
            "Pay.sh payment %s: %d USD cents → %d tokens → %s",
            event.payment_id,
            event.amount_usd_cents,
            event.tokens_to_mint,
            event.customer_wallet,
        )

        if event.status == "completed" and self.on_payment:
            self.on_payment(event)

        return event

    def create_payment_link(
        self,
        amount_usd_cents: int,
        customer_wallet: str,
        description: str = "Decentralized LLM inference tokens",
        redirect_url: str = "",
    ) -> dict:
        """
        Create a Pay.sh hosted payment page link.
        Returns the URL and payment ID for tracking.

        In production, call Pay.sh REST API:
          POST https://api.pay.sh/v1/payment-links
        """
        import urllib.request

        payload = json.dumps(
            {
                "amount": amount_usd_cents,
                "currency": "usd",
                "description": description,
                "metadata": {
                    "solana_wallet": customer_wallet,
                    "tokens_to_receive": amount_usd_cents * TOKENS_PER_USD_CENT,
                },
                "redirect_url": redirect_url,
            }
        ).encode()

        req = urllib.request.Request(
            "https://api.pay.sh/v1/payment-links",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read())
        except Exception as exc:
            logger.error("Pay.sh API call failed: %s", exc)
            raise

    # ────────────────────────── private ──────────────────────────────────────

    def _verify_signature(self, body: bytes, signature: str) -> None:
        expected = hmac.new(self.webhook_secret.encode(), body, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, signature):
            raise ValueError("Invalid Pay.sh webhook signature")

    def _parse_event(self, payload: dict) -> PaymentEvent:
        amount_cents = int(payload.get("amount", 0))
        return PaymentEvent(
            payment_id=payload["id"],
            amount_usd_cents=amount_cents,
            currency=payload.get("currency", "usd"),
            customer_wallet=payload.get("metadata", {}).get("solana_wallet", ""),
            tokens_to_mint=amount_cents * TOKENS_PER_USD_CENT,
            timestamp=int(payload.get("created_at", time.time())),
            status=payload.get("status", ""),
        )
