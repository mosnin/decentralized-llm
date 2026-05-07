"""
Python client SDK for the Decentralized LLM network.

Usage:
    from client.python import DecentralizedLLMClient

    client = DecentralizedLLMClient(wallet_path="~/.config/solana/id.json")
    response = await client.complete("Explain quantum entanglement", model="llama-3.2-3b")
    print(response.text)
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path


try:
    from solders.keypair import Keypair
    from solders.pubkey import Pubkey
    from anchorpy import Program, Provider, Wallet
    from solana.rpc.async_api import AsyncClient
    SOLANA_AVAILABLE = True
except ImportError:
    SOLANA_AVAILABLE = False


INFERENCE_MARKET_PROGRAM = "5YQyZqXkJHy6V3JMxKqXyLqfP9V2A3j8Rk7mN4oD1eW"
COMPUTE_REGISTRY_PROGRAM = "8KpR2mT6uLqVwNzS4eBfY9oA3cJ7iGxH1nD5sW0qF2M"

MODEL_IDS = {
    "llama-3.2-1b": hashlib.sha256(b"meta-llama/Llama-3.2-1B").digest(),
    "llama-3.2-3b": hashlib.sha256(b"meta-llama/Llama-3.2-3B").digest(),
    "llama-3.1-8b": hashlib.sha256(b"meta-llama/Llama-3.1-8B").digest(),
    "mistral-7b":   hashlib.sha256(b"mistralai/Mistral-7B-v0.3").digest(),
}


@dataclass
class CompletionResponse:
    text: str
    job_id: int
    model: str
    tokens_used: int
    total_paid: int  # in base token units
    node: str        # Solana pubkey of the node that served the request


class DecentralizedLLMClient:
    def __init__(
        self,
        wallet_path: str = "~/.config/solana/id.json",
        rpc_url: str = "https://api.mainnet-beta.solana.com",
    ):
        if not SOLANA_AVAILABLE:
            raise RuntimeError("Install solana deps: pip install anchorpy solders solana")

        self.rpc_url = rpc_url
        keypair = Keypair.from_json(Path(wallet_path).expanduser().read_text())
        self._wallet = Wallet(keypair)
        self._client: AsyncClient | None = None
        self._program: Program | None = None

    async def __aenter__(self):
        self._client = AsyncClient(self.rpc_url)
        provider = Provider(self._client, self._wallet)
        self._program = await Program.at(
            Pubkey.from_string(INFERENCE_MARKET_PROGRAM), provider
        )
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.close()

    async def complete(
        self,
        prompt: str,
        model: str = "llama-3.2-3b",
        max_tokens: int = 512,
        payment_amount: int | None = None,
        deadline_seconds: int = 120,
    ) -> CompletionResponse:
        """
        Post an inference job on-chain and wait for a node to complete it.

        The prompt is NOT stored on-chain — only its SHA-256 hash is.
        The actual prompt is delivered to the claiming node through a
        secure off-chain P2P channel (implementation in node/server.py).
        """
        model_id = MODEL_IDS.get(model)
        if model_id is None:
            raise ValueError(f"Unknown model '{model}'. Available: {list(MODEL_IDS)}")

        if payment_amount is None:
            payment_amount = self._estimate_cost(max_tokens)

        prompt_hash = list(hashlib.sha256(prompt.encode()).digest())
        model_id_list = list(model_id)
        deadline = int(time.time()) + deadline_seconds

        job_id = await self._post_job(
            model_id=model_id_list,
            prompt_hash=prompt_hash,
            max_tokens=max_tokens,
            payment_amount=payment_amount,
            deadline=deadline,
        )

        result = await self._wait_for_result(job_id, deadline)

        return CompletionResponse(
            text=result["text"],
            job_id=job_id,
            model=model,
            tokens_used=result["tokens_used"],
            total_paid=payment_amount,
            node=result["node"],
        )

    async def get_governance_proposals(self) -> list[dict]:
        """Fetch all active governance proposals."""
        from anchorpy import Program
        from solders.pubkey import Pubkey

        gov_client = await Program.at(
            Pubkey.from_string("3CvE7tX9rMwPfBgY2nKjH6oL4sQ8uZaD5mR1iW0eN9T"),
            Provider(self._client, self._wallet),
        )
        proposals = await gov_client.account["Proposal"].all()
        return [
            {
                "id": p.account.id,
                "title": p.account.title,
                "status": str(p.account.status),
                "votes_for": p.account.votes_for,
                "votes_against": p.account.votes_against,
                "voting_ends_at": p.account.voting_ends_at,
            }
            for p in proposals
        ]

    async def vote(self, proposal_id: int, choice: str) -> None:
        """Vote on a governance proposal. choice: 'for' | 'against' | 'abstain'"""
        choice_map = {"for": {"for": {}}, "against": {"against": {}}, "abstain": {"abstain": {}}}
        if choice not in choice_map:
            raise ValueError("choice must be 'for', 'against', or 'abstain'")
        # Implementation calls governance program cast_vote instruction
        raise NotImplementedError("Governance voting via SDK coming in Phase 3")

    # ────────────────────────── private ──────────────────────────────────────

    async def _post_job(self, **kwargs) -> int:
        market = await self._program.account["Market"].fetch(
            self._market_pda()
        )
        job_id = market.total_jobs

        await self._program.rpc["post_job"](
            kwargs["model_id"],
            kwargs["prompt_hash"],
            kwargs["max_tokens"],
            kwargs["payment_amount"],
            kwargs["deadline"],
            ctx=self._program.context(accounts={"client": self._wallet.public_key}),
        )
        return job_id

    async def _wait_for_result(self, job_id: int, deadline: int) -> dict:
        """Poll for job completion. In production, subscribe to Solana websocket events."""
        job_pda = self._job_pda(job_id)
        while time.time() < deadline:
            try:
                job = await self._program.account["Job"].fetch(job_pda)
                status = str(job.status)
                if "PendingAcceptance" in status or "Completed" in status:
                    # Fetch result from IPFS
                    result_text = await self._fetch_from_ipfs(job.result_cid)
                    return {
                        "text": result_text,
                        "tokens_used": len(result_text.split()),
                        "node": str(job.node),
                    }
            except Exception:
                pass
            await asyncio.sleep(2.0)

        raise TimeoutError(f"Job {job_id} did not complete before deadline")

    async def _fetch_from_ipfs(self, cid: str) -> str:
        """Retrieve result from IPFS/Arweave using the CID."""
        # Integration point: use ipfshttpclient or requests to a gateway
        raise NotImplementedError(f"IPFS fetch for CID {cid} not yet implemented")

    def _estimate_cost(self, max_tokens: int) -> int:
        # 1 token ≈ 0.001 base units; minimum 1000 base units per request
        return max(1000, max_tokens * 1)

    def _market_pda(self) -> "Pubkey":
        return Pubkey.find_program_address(
            [b"market"],
            Pubkey.from_string(INFERENCE_MARKET_PROGRAM),
        )[0]

    def _job_pda(self, job_id: int) -> "Pubkey":
        return Pubkey.find_program_address(
            [b"job", job_id.to_bytes(8, "little")],
            Pubkey.from_string(INFERENCE_MARKET_PROGRAM),
        )[0]
