"""
Python client SDK for the Decentralized LLM network.

Usage:
    from client.python import DecentralizedLLMClient

    client = DecentralizedLLMClient(wallet_path="~/.config/solana/id.json")
    response = await client.complete("Explain quantum entanglement", model="llama-3.2-3b")
    print(response.text)

Privacy guarantee:
    The prompt is encrypted with the claiming node's Ed25519 public key
    (converted to X25519) before leaving the client.  The on-chain job
    account only stores sha256(prompt) — the node can't substitute a
    different prompt without detection.
"""

import asyncio
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from anchorpy import Program, Provider, Wallet
    from solana.rpc.async_api import AsyncClient
    from solders.keypair import Keypair
    from solders.pubkey import Pubkey

    SOLANA_AVAILABLE = True
except ImportError:
    SOLANA_AVAILABLE = False


INFERENCE_MARKET_PROGRAM = "5YQyZqXkJHy6V3JMxKqXyLqfP9V2A3j8Rk7mN4oD1eW"
COMPUTE_REGISTRY_PROGRAM = "8KpR2mT6uLqVwNzS4eBfY9oA3cJ7iGxH1nD5sW0qF2M"
GOVERNANCE_PROGRAM = "3CvE7tX9rMwPfBgY2nKjH6oL4sQ8uZaD5mR1iW0eN9T"

MODEL_IDS = {
    "llama-3.2-1b": hashlib.sha256(b"meta-llama/Llama-3.2-1B").digest(),
    "llama-3.2-3b": hashlib.sha256(b"meta-llama/Llama-3.2-3B").digest(),
    "llama-3.1-8b": hashlib.sha256(b"meta-llama/Llama-3.1-8B").digest(),
    "mistral-7b": hashlib.sha256(b"mistralai/Mistral-7B-v0.3").digest(),
}

# Public IPFS gateways for reading results (no API key needed)
IPFS_GATEWAYS = [
    "https://gateway.lighthouse.storage/ipfs",
    "https://ipfs.io/ipfs",
    "https://cloudflare-ipfs.com/ipfs",
]


@dataclass
class CompletionResponse:
    text: str
    job_id: int
    model: str
    tokens_used: int
    total_paid: int  # in base token units
    node: str  # Solana pubkey of the node that served the request


class DecentralizedLLMClient:
    def __init__(
        self,
        wallet_path: str = "~/.config/solana/id.json",
        rpc_url: str = "https://api.mainnet-beta.solana.com",
        lighthouse_api_key: str = "",
    ):
        if not SOLANA_AVAILABLE:
            raise RuntimeError("Install solana deps: pip install anchorpy solders solana")

        self.rpc_url = rpc_url
        self._lighthouse_api_key = lighthouse_api_key
        self._wallet_path = Path(wallet_path).expanduser()
        keypair = Keypair.from_json(self._wallet_path.read_text())
        self._wallet = Wallet(keypair)
        self._client: AsyncClient | None = None
        self._program: Program | None = None
        self._registry: Program | None = None

    async def __aenter__(self):
        self._client = AsyncClient(self.rpc_url)
        provider = Provider(self._client, self._wallet)
        self._program = await Program.at(Pubkey.from_string(INFERENCE_MARKET_PROGRAM), provider)
        self._registry = await Program.at(Pubkey.from_string(COMPUTE_REGISTRY_PROGRAM), provider)
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

        End-to-end privacy flow:
          1. Query compute registry for the best available node for this model
          2. Encrypt prompt with that node's Ed25519 wallet pubkey (ECIES/X25519)
          3. Upload encrypted blob to IPFS (CID committed on-chain alongside hash)
          4. Post job → node downloads blob, decrypts, runs inference
          5. Node uploads result to IPFS, submits result_hash on-chain
          6. Client downloads result from IPFS and returns it
        """
        model_id = MODEL_IDS.get(model)
        if model_id is None:
            raise ValueError(f"Unknown model '{model}'. Available: {list(MODEL_IDS)}")

        if payment_amount is None:
            payment_amount = self._estimate_cost(max_tokens)

        # Step 1: find best node for this model
        node_pubkey_bytes = await self._find_best_node(model_id)

        # Step 2: encrypt prompt for that node
        from node.encryption import encrypt_prompt

        prompt_bytes = prompt.encode("utf-8")
        prompt_hash = list(hashlib.sha256(prompt_bytes).digest())
        encrypted_blob = encrypt_prompt(prompt, node_pubkey_bytes)

        # Step 3: upload encrypted blob to IPFS
        prompt_cid = await self._upload_prompt(encrypted_blob)

        deadline = int(time.time()) + deadline_seconds

        # Step 4: post job on-chain
        job_id = await self._post_job(
            model_id=list(model_id),
            prompt_hash=prompt_hash,
            prompt_cid=prompt_cid,
            max_tokens=max_tokens,
            payment_amount=payment_amount,
            deadline=deadline,
        )

        # Step 5+6: wait for result and download it
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
        provider = Provider(self._client, self._wallet)
        gov_program = await Program.at(Pubkey.from_string(GOVERNANCE_PROGRAM), provider)
        proposals = await gov_program.account["Proposal"].all()
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
        valid = {"for", "against", "abstain"}
        if choice not in valid:
            raise ValueError(f"choice must be one of {valid}")
        if self._client is None:
            raise RuntimeError("Not connected — use 'async with client'")

        provider = Provider(self._client, self._wallet)
        gov_program = await Program.at(Pubkey.from_string(GOVERNANCE_PROGRAM), provider)

        # Derive proposal PDA
        proposal_pda = Pubkey.find_program_address(
            [b"proposal", proposal_id.to_bytes(8, "little")],
            Pubkey.from_string(GOVERNANCE_PROGRAM),
        )[0]

        # Derive vote record PDA (one per voter per proposal)
        vote_record_pda = Pubkey.find_program_address(
            [
                b"vote",
                bytes(proposal_pda),
                bytes(self._wallet.public_key),
            ],
            Pubkey.from_string(GOVERNANCE_PROGRAM),
        )[0]

        # Find voter's token account
        voter_token_account = await self._find_token_account(self._wallet.public_key)

        vote_choice = {
            "for": {"for": {}},
            "against": {"against": {}},
            "abstain": {"abstain": {}},
        }[choice]

        await gov_program.rpc["cast_vote"](
            vote_choice,
            ctx=gov_program.context(
                accounts={
                    "proposal": proposal_pda,
                    "vote_record": vote_record_pda,
                    "voter_token_account": voter_token_account,
                    "voter": self._wallet.public_key,
                    "system_program": Pubkey.from_string(
                        "11111111111111111111111111111111"
                    ),
                }
            ),
        )

    async def _find_token_account(self, owner: "Pubkey") -> "Pubkey":
        """Find the associated token account for the $DLLM token."""
        from spl.token.instructions import get_associated_token_address

        # TOKEN_MINT is stored in the market state; resolve lazily
        market = await self._program.account["Market"].fetch(self._market_pda())
        return get_associated_token_address(owner, market.token_mint)

    # ────────────────────────── private ──────────────────────────────────────

    async def _find_best_node(self, model_id: bytes) -> bytes:
        """
        Query compute registry for registered nodes that serve this model.
        Returns the Ed25519 pubkey bytes of the best node (highest reputation).
        """
        if self._registry is None:
            raise RuntimeError("Not connected — use 'async with client'")
        try:
            nodes = await self._registry.account["NodeRecord"].all()
            eligible = [
                n
                for n in nodes
                if any(bytes(mid) == model_id for mid in n.account.model_ids)
                and n.account.staked_amount > 0
            ]
            if not eligible:
                raise RuntimeError(f"No registered nodes found for model {model_id.hex()[:8]}…")
            # Pick highest reputation
            best = max(eligible, key=lambda n: n.account.reputation)
            return bytes(best.public_key)
        except Exception as exc:
            raise RuntimeError(f"Failed to find node: {exc}") from exc

    async def _upload_prompt(self, blob: bytes) -> str:
        """
        Upload encrypted prompt blob to IPFS.
        Requires LIGHTHOUSE_API_KEY for writes; falls back to a content-hash
        placeholder if no key is configured (for testing without real uploads).
        """
        if self._lighthouse_api_key:
            from node.storage import StorageClient

            storage = StorageClient(api_key=self._lighthouse_api_key)
            return await storage.upload(blob, filename="prompt.bin")

        # Deterministic placeholder for testing/dev (not a real IPFS CID)
        h = hashlib.sha256(blob).hexdigest()
        return f"bafkrei{h[:32]}"

    async def _post_job(self, **kwargs) -> int:
        if self._program is None:
            raise RuntimeError("Not connected — use 'async with client'")
        market = await self._program.account["Market"].fetch(self._market_pda())
        job_id = market.total_jobs

        await self._program.rpc["post_job"](
            kwargs["model_id"],
            kwargs["prompt_hash"],
            kwargs["prompt_cid"],
            kwargs["max_tokens"],
            kwargs["payment_amount"],
            kwargs["deadline"],
            ctx=self._program.context(accounts={"client": self._wallet.public_key}),
        )
        return job_id

    async def _wait_for_result(self, job_id: int, deadline: int) -> dict:
        """Poll for job completion then download result from IPFS."""
        if self._program is None:
            raise RuntimeError("Not connected — use 'async with client'")
        job_pda = self._job_pda(job_id)
        while time.time() < deadline:
            try:
                job = await self._program.account["Job"].fetch(job_pda)
                status = str(job.status)
                if "PendingAcceptance" in status or "Completed" in status:
                    result_text = await self._fetch_result_from_ipfs(job.result_cid)
                    return {
                        "text": result_text,
                        "tokens_used": len(result_text.split()),
                        "node": str(job.node),
                    }
            except Exception:
                pass
            await asyncio.sleep(2.0)

        raise TimeoutError(f"Job {job_id} did not complete before deadline")

    async def _fetch_result_from_ipfs(self, cid: str) -> str:
        """Download result from any available IPFS gateway."""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            for gateway in IPFS_GATEWAYS:
                try:
                    async with session.get(
                        f"{gateway}/{cid}",
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp:
                        if resp.status == 200:
                            return (await resp.read()).decode("utf-8")
                except Exception:
                    continue
        raise RuntimeError(f"Could not fetch result CID {cid} from any gateway")

    def _estimate_cost(self, max_tokens: int) -> int:
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
