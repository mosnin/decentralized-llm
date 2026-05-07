"""
REST API gateway — wraps the Python client SDK so any language can use the network.

Endpoints:
  POST /v1/completions          - Standard inference (OpenAI-compatible schema)
  GET  /v1/models               - List available models
  GET  /v1/governance/proposals - Active DAO proposals
  POST /v1/governance/vote      - Cast a vote
  POST /webhooks/paysh          - Pay.sh payment webhook
  POST /v1/payments/create      - Create a Pay.sh payment link
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from client.python import DecentralizedLLMClient
from integrations.paysh import PayshHandler

# ────────────────────────── startup / shutdown ────────────────────────────────

_client: DecentralizedLLMClient | None = None
_paysh: PayshHandler | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client, _paysh
    _client = DecentralizedLLMClient(
        wallet_path=os.environ.get("WALLET_PATH", "~/.config/solana/id.json"),
        rpc_url=os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"),
    )
    await _client.__aenter__()

    _paysh = PayshHandler(
        api_key=os.environ.get("PAYSH_API_KEY", ""),
        webhook_secret=os.environ.get("PAYSH_WEBHOOK_SECRET", ""),
        on_payment=_handle_payment,
    )
    yield
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


# ────────────────────────── schemas ──────────────────────────────────────────


class CompletionRequest(BaseModel):
    model: str = "llama-3.2-3b"
    prompt: str
    max_tokens: int = 512
    payment_amount: int | None = None


class VoteRequest(BaseModel):
    proposal_id: int
    choice: str  # "for" | "against" | "abstain"
    wallet: str


class PaymentLinkRequest(BaseModel):
    amount_usd_cents: int
    customer_wallet: str
    description: str = "Decentralized LLM tokens"


# ────────────────────────── endpoints ────────────────────────────────────────


@app.get("/v1/models")
async def list_models():
    return {
        "models": [
            {"id": "llama-3.2-1b", "description": "LLaMA 3.2 1B — fastest, lowest cost"},
            {"id": "llama-3.2-3b", "description": "LLaMA 3.2 3B — balanced"},
            {"id": "llama-3.1-8b", "description": "LLaMA 3.1 8B — highest quality"},
            {"id": "mistral-7b", "description": "Mistral 7B v0.3"},
        ]
    }


@app.post("/v1/completions")
async def create_completion(req: CompletionRequest):
    try:
        result = await _client.complete(
            prompt=req.prompt,
            model=req.model,
            max_tokens=req.max_tokens,
            payment_amount=req.payment_amount,
        )
        return {
            "id": f"job_{result.job_id}",
            "model": result.model,
            "choices": [{"text": result.text, "finish_reason": "stop"}],
            "usage": {"completion_tokens": result.tokens_used, "total_paid": result.total_paid},
            "node": result.node,
        }
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc))
    except Exception as exc:
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
        return {"status": "ok", "tokens_minted": event.tokens_to_mint}
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc))


@app.get("/health")
async def health():
    return {"status": "ok"}


# ────────────────────────── payment callback ─────────────────────────────────


def _handle_payment(event):
    """Called when Pay.sh confirms a completed payment. Mint tokens on-chain."""
    # TODO: call SPL token mint instruction for event.tokens_to_mint
    # to event.customer_wallet using the DAO treasury mint authority
    print(f"Minting {event.tokens_to_mint} tokens to {event.customer_wallet}")
