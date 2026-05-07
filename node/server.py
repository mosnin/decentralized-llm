"""
Main compute node server.

Startup sequence:
  1. Load model shard into GPU memory
  2. Connect to Solana RPC and verify wallet
  3. Register in the compute-registry (if not already)
  4. Join the P2P DHT network and announce this shard
  5. Start job-polling loop: claim → infer → upload result → submit on-chain

For rented GPU environments (io.net, Vast.ai, RunPod), set PUBLIC_HOST to
the provider-assigned public IP before launching.
"""

import asyncio
import hashlib
import io
import json
import logging
import os
import signal
import sys
import time


import torch

from .blockchain import BlockchainClient, OpenJob
from .config import NodeConfig
from .p2p import P2PLayer
from .shard_manager import ShardManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


class Node:
    def __init__(self, config: NodeConfig | None = None):
        self.config = config or NodeConfig()
        self.shard_mgr = ShardManager(self.config)
        self.blockchain = BlockchainClient(self.config)
        self.p2p: P2PLayer | None = None
        self._running = False
        self._active_jobs: dict[int, asyncio.Task] = {}

    # ────────────────────────── lifecycle ────────────────────────────────────

    async def start(self) -> None:
        logger.info("=== Decentralized LLM Node starting ===")
        logger.info("Model: %s  Shard: %d/%d", self.config.model_name,
                    self.config.shard_index, self.config.num_shards)

        self.shard_mgr.load()

        await self.blockchain.connect()
        await self._ensure_registered()

        self.p2p = P2PLayer(self.config, self.shard_mgr)
        await self.p2p.start()

        self._running = True
        logger.info("Node ready. Starting job poll loop.")
        await self._job_loop()

    async def stop(self) -> None:
        logger.info("Shutting down node…")
        self._running = False
        for task in self._active_jobs.values():
            task.cancel()
        if self.p2p:
            await self.p2p.stop()
        await self.blockchain.close()

    # ────────────────────────── job loop ─────────────────────────────────────

    async def _job_loop(self) -> None:
        model_id = self._model_id_bytes()
        while self._running:
            if len(self._active_jobs) < self.config.max_concurrent_jobs:
                jobs = await self.blockchain.fetch_open_jobs(model_id)
                for job in jobs:
                    if job.job_id not in self._active_jobs:
                        task = asyncio.create_task(self._handle_job(job))
                        self._active_jobs[job.job_id] = task
                        task.add_done_callback(
                            lambda t, jid=job.job_id: self._active_jobs.pop(jid, None)
                        )

            await asyncio.sleep(self.config.job_poll_interval_seconds)

    async def _handle_job(self, job: OpenJob) -> None:
        logger.info("Handling job %d (payment: %d tokens)", job.job_id, job.payment_amount)

        claimed = await self.blockchain.claim_job(job)
        if not claimed:
            return

        try:
            result_text = await self._run_inference(job)
        except Exception as exc:
            logger.error("Inference failed for job %d: %s", job.job_id, exc)
            return

        # Store result on decentralized storage (IPFS via web3.storage or Arweave)
        result_cid = await self._upload_result(job.job_id, result_text)

        await self.blockchain.submit_result(
            job, result_text.encode(), result_cid
        )

    # ────────────────────────── inference ────────────────────────────────────

    async def _run_inference(self, job: OpenJob) -> str:
        """
        Execute a pipeline-parallel inference pass.

        - Shard 0: embed input → forward its layers → send to shard 1
        - Middle shards: receive activations → forward → send to next
        - Last shard: decode logits → sample tokens → return text

        The actual prompt is retrieved off-chain (delivered encrypted to the
        claiming node via the client's P2P channel or a content-addressed store).
        """
        # In a full implementation, the client sends the encrypted prompt
        # directly to the claiming node via a secure P2P channel keyed to
        # the node's wallet public key. Here we show the structural skeleton.

        loop = asyncio.get_event_loop()

        if self.config.shard_index == 0:
            prompt = await self._fetch_prompt(job)
            input_ids = await loop.run_in_executor(
                None, self._tokenize, prompt
            )
            hidden = await loop.run_in_executor(None, self.shard_mgr.embed, input_ids)
        else:
            hidden = await self._receive_activations(job.job_id)

        hidden = await loop.run_in_executor(None, self.shard_mgr.forward, hidden)

        if self.config.shard_index < self.config.num_shards - 1:
            next_shard = await self.p2p.get_next_shard_peer(self.config.shard_index + 1)
            if next_shard is None:
                raise RuntimeError(f"Shard {self.config.shard_index + 1} not available")
            result = await next_shard.forward(hidden)
            return str(result)  # downstream shard returns final text
        else:
            # Last shard: generate tokens autoregressively
            return await loop.run_in_executor(
                None, self._generate, hidden, job.max_tokens
            )

    def _tokenize(self, prompt: str) -> torch.Tensor:
        tokens = self.shard_mgr.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )
        device = next(self.shard_mgr.model.parameters()).device
        return tokens["input_ids"].to(device)

    def _generate(self, hidden: torch.Tensor, max_new_tokens: int) -> str:
        logits = self.shard_mgr.decode(hidden)
        # Greedy decoding for simplicity; swap for sampling/beam search as needed
        generated = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens = [generated.item()]

        for _ in range(max_new_tokens - 1):
            hidden = self.shard_mgr.forward(
                self.shard_mgr.model.layers[-1](hidden)[0]
                if hasattr(self.shard_mgr.model, "layers")
                else hidden
            )
            logits = self.shard_mgr.decode(hidden)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens.append(next_tok.item())
            if next_tok.item() == self.shard_mgr.tokenizer.eos_token_id:
                break

        return self.shard_mgr.tokenizer.decode(tokens, skip_special_tokens=True)

    # ────────────────────────── helpers ──────────────────────────────────────

    async def _fetch_prompt(self, job: OpenJob) -> str:
        """
        Retrieve the actual prompt for a job. The client delivers it either:
          a) directly to the claiming node over an encrypted P2P channel, or
          b) as an encrypted blob stored on IPFS (CID derived from prompt_hash).
        """
        # Placeholder: in production, open a noise-encrypted channel to the client
        raise NotImplementedError("Prompt delivery not yet implemented")

    async def _receive_activations(self, job_id: int) -> torch.Tensor:
        """Receive activations streamed from the previous shard."""
        raise NotImplementedError("Activation streaming not yet implemented")

    async def _upload_result(self, job_id: int, result_text: str) -> str:
        """Upload result to IPFS/Arweave and return the CID."""
        # Placeholder: integrate with web3.storage, nft.storage, or Arweave SDK
        content_hash = hashlib.sha256(result_text.encode()).hexdigest()
        logger.info("Result for job %d would be uploaded (hash: %s)", job_id, content_hash[:16])
        return f"bafkreifake{content_hash[:32]}"  # replace with real IPFS upload

    async def _ensure_registered(self) -> None:
        model_id = self._model_id_bytes()
        gpu_count = torch.cuda.device_count() or 1
        vram_gb = 0
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)

        public_host = self.config.public_host or self.config.listen_host
        endpoint = f"{public_host}:{self.config.listen_port}"

        await self.blockchain.register_node(
            endpoint=endpoint,
            vram_gb=vram_gb,
            gpu_count=gpu_count,
            model_ids=[model_id],
            stake_amount=0,  # stake managed separately via CLI
        )

    def _model_id_bytes(self) -> bytes:
        return hashlib.sha256(self.config.model_name.encode()).digest()


# ────────────────────────── entrypoint ───────────────────────────────────────

def main():
    config = NodeConfig()
    node = Node(config)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(node.stop()))

    try:
        loop.run_until_complete(node.start())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
