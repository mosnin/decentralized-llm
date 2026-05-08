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
import json
import os
import time
import uuid
from collections import defaultdict
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

# ────────────────────────── rate limiting ────────────────────────────────────

_rate_buckets: dict = defaultdict(lambda: {"count": 0, "window_start": time.time()})
_RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_MIN", "60"))


def _check_rate_limit(ip: str) -> None:
    bucket = _rate_buckets[ip]
    now = time.time()
    if now - bucket["window_start"] > 60:
        bucket["count"] = 0
        bucket["window_start"] = now
    bucket["count"] += 1
    if bucket["count"] > _RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({_RATE_LIMIT} req/min). Slow down.",
            headers={"Retry-After": "60"},
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
async def create_completion(req: CompletionRequest, request: Request):
    _check_rate_limit(request.client.host if request.client else "unknown")
    _metrics["requests_total"] += 1
    completion_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    try:
        result = await _client.complete(
            prompt=req.prompt,
            model=req.model,
            max_tokens=req.max_tokens,
            payment_amount=req.payment_amount,
        )
        _metrics["requests_success"] += 1
        _metrics["tokens_generated"] += result.tokens_used

        if req.stream:
            return StreamingResponse(
                _stream_completion(completion_id, result, object_type="text_completion"),
                media_type="text/event-stream",
            )

        return {
            "id": completion_id,
            "object": "text_completion",
            "model": result.model,
            "choices": [{"text": result.text, "index": 0, "finish_reason": "stop"}],
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
async def create_chat_completion(req: ChatCompletionRequest, request: Request):
    """
    OpenAI-compatible chat completions endpoint with optional SSE streaming.

    Messages are concatenated into a single prompt using the standard
    ChatML format so any OpenAI client library works out-of-the-box.
    """
    _check_rate_limit(request.client.host if request.client else "unknown")
    _metrics["requests_total"] += 1
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
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

        if req.stream:
            return StreamingResponse(
                _stream_chat_completion(chat_id, result, prompt),
                media_type="text/event-stream",
            )

        created = int(time.time())
        return {
            "id": chat_id,
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
    try:
        from prometheus_client import generate_latest

        output = generate_latest()
        return StreamingResponse(
            iter([output]),
            media_type="text/plain; version=0.0.4",
        )
    except ImportError:
        from fastapi.responses import PlainTextResponse

        return PlainTextResponse("metrics not available", status_code=503)


@app.get("/v1/jobs/{job_id}")
async def get_job_status(job_id: int):
    """Poll on-chain job status. Useful for clients that prefer polling over waiting."""
    if _client is None:
        raise HTTPException(status_code=503, detail="Blockchain client not connected")
    try:
        job_pda = _client._job_pda(job_id)
        job = await _client._program.account["Job"].fetch(job_pda)
        return {
            "job_id": job_id,
            "status": str(job.status),
            "node": str(job.node),
            "result_cid": job.result_cid,
        }
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Job not found: {exc}")


# ────────────────────────── helpers ──────────────────────────────────────────


async def _stream_completion(completion_id: str, result, object_type: str):
    """Yield SSE chunks for a text completion (word-by-word)."""
    words = result.text.split(" ")
    for i, word in enumerate(words):
        chunk_text = word if i == len(words) - 1 else word + " "
        chunk = {
            "id": completion_id,
            "object": object_type,
            "model": result.model,
            "choices": [{"text": chunk_text, "index": 0, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0)  # yield control back to event loop
    # Final chunk with finish_reason
    final = {
        "id": completion_id,
        "object": object_type,
        "model": result.model,
        "choices": [{"text": "", "index": 0, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


async def _stream_chat_completion(chat_id: str, result, prompt: str):
    """Yield SSE chunks for a chat completion (word-by-word)."""
    words = result.text.split(" ")
    for i, word in enumerate(words):
        chunk_content = word if i == len(words) - 1 else word + " "
        chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "model": result.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": chunk_content},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0)
    final = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "model": result.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


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
