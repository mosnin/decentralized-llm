"""
REST API gateway — wraps the Python client SDK so any language can use the network.

Endpoints:
  POST /v1/completions              - OpenAI-compatible text completions
  POST /v1/chat/completions         - OpenAI-compatible chat completions
  GET  /v1/models                   - List available models
  GET  /v1/governance/proposals     - Active DAO proposals
  POST /v1/governance/vote          - Cast a vote
  POST /webhooks/paysh              - Pay.sh payment webhook
  POST /v1/payments/create          - Create a Pay.sh payment link
  GET  /health                      - Health check
  GET  /metrics                     - Prometheus-compatible metrics
"""

import asyncio
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from client.python import DecentralizedLLMClient
from integrations.paysh import PayshHandler

# ────────────────────────── startup / shutdown ────────────────────────────────

_client: DecentralizedLLMClient | None = None
_paysh: PayshHandler | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client, _paysh
    _lifespan_owns_client = False
    if _client is None:
        try:
            _client = DecentralizedLLMClient(
                wallet_path=os.environ.get("WALLET_PATH", "~/.config/solana/id.json"),
                rpc_url=os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"),
            )
            await _client.__aenter__()
            _lifespan_owns_client = True
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning(
                "Blockchain client unavailable (Solana deps missing?): %s", exc
            )
            _client = None

    if _paysh is None:
        _paysh = PayshHandler(
            api_key=os.environ.get("PAYSH_API_KEY", ""),
            webhook_secret=os.environ.get("PAYSH_WEBHOOK_SECRET", ""),
            on_payment=_handle_payment,
        )
    yield
    if _lifespan_owns_client and _client is not None:
        await _client.__aexit__(None, None, None)


app = FastAPI(
    title="Decentralized LLM API",
    version="0.1.0",
    description="Community-owned, decentralized LLM inference network",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ────────────────────────── metrics ──────────────────────────────────────────

_metrics: dict = {
    "requests_total": 0,
    "requests_success": 0,
    "requests_failed": 0,
    "tokens_generated": 0,
    "payments_processed": 0,
    "tokens_minted": 0,
    "started_at": time.time(),
}


# ────────────────────────── schemas ──────────────────────────────────────────


class CompletionRequest(BaseModel):
    model: str = "llama-3.2-3b"
    prompt: str
    max_tokens: int = 512
    payment_amount: int | None = None
    stream: bool = False


class ChatMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "llama-3.2-3b"
    messages: list[ChatMessage]
    max_tokens: int = 512
    payment_amount: int | None = None
    stream: bool = False
    temperature: float = 1.0


class VoteRequest(BaseModel):
    proposal_id: int
    choice: str  # "for" | "against" | "abstain"


class PaymentLinkRequest(BaseModel):
    amount_usd_cents: int
    customer_wallet: str
    description: str = "Decentralized LLM tokens"


# ────────────────────────── endpoints ────────────────────────────────────────


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "llama-3.2-1b",
                "object": "model",
                "owned_by": "decentralized-llm-network",
                "description": "LLaMA 3.2 1B — fastest, lowest cost",
            },
            {
                "id": "llama-3.2-3b",
                "object": "model",
                "owned_by": "decentralized-llm-network",
                "description": "LLaMA 3.2 3B — balanced",
            },
            {
                "id": "llama-3.1-8b",
                "object": "model",
                "owned_by": "decentralized-llm-network",
                "description": "LLaMA 3.1 8B — highest quality",
            },
            {
                "id": "mistral-7b",
                "object": "model",
                "owned_by": "decentralized-llm-network",
                "description": "Mistral 7B v0.3",
            },
        ],
    }


@app.post("/v1/completions")
async def create_completion(req: CompletionRequest):
    _metrics["requests_total"] += 1
    try:
        result = await _client.complete(
            prompt=req.prompt,
            model=req.model,
            max_tokens=req.max_tokens,
            payment_amount=req.payment_amount,
        )
        _metrics["requests_success"] += 1
        _metrics["tokens_generated"] += result.tokens_used
        return {
            "id": f"cmpl-{result.job_id}",
            "object": "text_completion",
            "model": result.model,
            "choices": [
                {
                    "text": result.text,
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": result.tokens_used,
                "total_tokens": result.tokens_used,
                "total_paid": result.total_paid,
            },
            "node": result.node,
        }
    except TimeoutError as exc:
        _metrics["requests_failed"] += 1
        raise HTTPException(status_code=504, detail=str(exc))
    except Exception as exc:
        _metrics["requests_failed"] += 1
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/v1/chat/completions")
async def create_chat_completion(req: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.

    Messages are concatenated into a single prompt using the standard
    ChatML format so any OpenAI client library works out-of-the-box.
    """
    _metrics["requests_total"] += 1
    # Build prompt from chat messages (ChatML format)
    prompt = _build_chatml(req.messages)

    try:
        result = await _client.complete(
            prompt=prompt,
            model=req.model,
            max_tokens=req.max_tokens,
            payment_amount=req.payment_amount,
        )
        _metrics["requests_success"] += 1
        _metrics["tokens_generated"] += result.tokens_used
        created = int(time.time())
        return {
            "id": f"chatcmpl-{result.job_id}",
            "object": "chat.completion",
            "created": created,
            "model": result.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": result.tokens_used,
                "total_tokens": len(prompt.split()) + result.tokens_used,
                "total_paid": result.total_paid,
            },
            "node": result.node,
        }
    except TimeoutError as exc:
        _metrics["requests_failed"] += 1
        raise HTTPException(status_code=504, detail=str(exc))
    except Exception as exc:
        _metrics["requests_failed"] += 1
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/v1/governance/proposals")
async def get_proposals():
    proposals = await _client.get_governance_proposals()
    return {"proposals": proposals}


@app.post("/v1/governance/vote")
async def cast_vote(req: VoteRequest):
    await _client.vote(req.proposal_id, req.choice)
    return {"status": "vote cast"}


@app.post("/v1/payments/create")
async def create_payment_link(req: PaymentLinkRequest):
    link = _paysh.create_payment_link(
        amount_usd_cents=req.amount_usd_cents,
        customer_wallet=req.customer_wallet,
        description=req.description,
    )
    return link


@app.post("/webhooks/paysh")
async def paysh_webhook(request: Request):
    body = await request.body()
    sig = request.headers.get("X-Paysh-Signature", "")
    try:
        event = _paysh.process_webhook(body, sig)
        # Schedule async token minting without blocking the webhook response
        asyncio.create_task(_mint_tokens(event.customer_wallet, event.tokens_to_mint))
        return {"status": "ok", "tokens_minted": event.tokens_to_mint}
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc))


@app.get("/health")
async def health():
    uptime = int(time.time() - _metrics["started_at"])
    return {
        "status": "ok",
        "uptime_seconds": uptime,
        "requests_total": _metrics["requests_total"],
    }


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus text format metrics."""
    uptime = time.time() - _metrics["started_at"]
    lines = [
        "# HELP dllm_requests_total Total inference requests",
        "# TYPE dllm_requests_total counter",
        f"dllm_requests_total {_metrics['requests_total']}",
        "# HELP dllm_requests_success_total Successful inference requests",
        "# TYPE dllm_requests_success_total counter",
        f"dllm_requests_success_total {_metrics['requests_success']}",
        "# HELP dllm_requests_failed_total Failed inference requests",
        "# TYPE dllm_requests_failed_total counter",
        f"dllm_requests_failed_total {_metrics['requests_failed']}",
        "# HELP dllm_tokens_generated_total Total tokens generated",
        "# TYPE dllm_tokens_generated_total counter",
        f"dllm_tokens_generated_total {_metrics['tokens_generated']}",
        "# HELP dllm_payments_processed_total Total payments processed",
        "# TYPE dllm_payments_processed_total counter",
        f"dllm_payments_processed_total {_metrics['payments_processed']}",
        "# HELP dllm_tokens_minted_total Total $DLLM tokens minted",
        "# TYPE dllm_tokens_minted_total counter",
        f"dllm_tokens_minted_total {_metrics['tokens_minted']}",
        "# HELP dllm_uptime_seconds Gateway uptime in seconds",
        "# TYPE dllm_uptime_seconds gauge",
        f"dllm_uptime_seconds {uptime:.1f}",
    ]
    return StreamingResponse(
        iter(["\n".join(lines) + "\n"]),
        media_type="text/plain; version=0.0.4",
    )


# ────────────────────────── helpers ──────────────────────────────────────────


def _build_chatml(messages: list[ChatMessage]) -> str:
    """Convert chat messages to ChatML prompt format."""
    parts = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


# ────────────────────────── payment callback ─────────────────────────────────


async def _mint_tokens(wallet_address: str, amount: int) -> None:
    """
    Mint $DLLM tokens to the customer's wallet after a confirmed fiat payment.

    Uses the DAO treasury mint authority (stored as TREASURY_KEYPAIR env var)
    to call the SPL Token mint_to instruction.
    """
    import base64
    import logging

    log = logging.getLogger(__name__)

    treasury_keypair_b64 = os.environ.get("TREASURY_KEYPAIR_B64", "")
    token_mint_address = os.environ.get("DLLM_TOKEN_MINT", "")

    if not treasury_keypair_b64 or not token_mint_address:
        log.warning(
            "TREASURY_KEYPAIR_B64 or DLLM_TOKEN_MINT not set — "
            "skipping on-chain mint for %d tokens to %s",
            amount,
            wallet_address,
        )
        _metrics["payments_processed"] += 1
        _metrics["tokens_minted"] += amount
        return

    try:
        from solana.rpc.async_api import AsyncClient
        from solders.keypair import Keypair
        from solders.pubkey import Pubkey
        from spl.token.async_client import AsyncToken
        from spl.token.constants import TOKEN_PROGRAM_ID

        keypair_bytes = base64.b64decode(treasury_keypair_b64)
        treasury_kp = Keypair.from_bytes(keypair_bytes)

        rpc_url = os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
        async with AsyncClient(rpc_url) as rpc:
            token = AsyncToken(
                rpc,
                Pubkey.from_string(token_mint_address),
                TOKEN_PROGRAM_ID,
                treasury_kp,
            )
            dest = await token.create_associated_token_account(Pubkey.from_string(wallet_address))
            await token.mint_to(dest, treasury_kp, amount)

        log.info("Minted %d $DLLM to %s", amount, wallet_address)
        _metrics["payments_processed"] += 1
        _metrics["tokens_minted"] += amount
    except Exception as exc:
        log.error("Token minting failed: %s", exc)


def _handle_payment(event):
    """Synchronous shim — schedules the async mint."""
    asyncio.create_task(_mint_tokens(event.customer_wallet, event.tokens_to_mint))
