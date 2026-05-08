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
import logging
import signal
import time

from .blockchain import BlockchainClient, OpenJob
from .config import NodeConfig
from .encryption import decrypt_prompt
from .integrity import IntegrityError, compute_model_id, verify_model_id, verify_prompt_hash
from .logging_config import set_correlation_id
from .metrics import METRICS_AVAILABLE
from .model_registry import ModelRegistry
from .p2p import P2PLayer
from .shard_manager import ShardManager
from .storage import StorageClient
from .timeout_manager import JobTimeoutError, TimeoutManager
from .verifier import ResultVerifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

_MAX_RETRIES = 2
_RETRY_DELAY = 5.0
_IPFS_CIRCUIT_BREAKER_THRESHOLD = 3


class Node:
    def __init__(self, config: NodeConfig | None = None):
        self.config = config or NodeConfig()
        self.shard_mgr = ShardManager(self.config)
        self._model_registry = ModelRegistry()
        self.blockchain = BlockchainClient(self.config)
        self.storage: StorageClient | None = None
        self.p2p: P2PLayer | None = None
        self._running = False
        # Maps job_id → True for all jobs currently queued or being handled.
        # Used for deduplication across both the queue and active workers.
        self._active_jobs: dict[int, bool] = {}
        # Priority queue: items are (priority_key, job) where priority_key is
        # the negated payment_amount so highest-paying jobs sort first.
        self._job_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()

    # ────────────────────────── lifecycle ────────────────────────────────────

    async def start(self) -> None:
        logger.info("=== Decentralized LLM Node starting ===")
        logger.info(
            "Model: %s  Shard: %d/%d",
            self.config.model_name,
            self.config.shard_index,
            self.config.num_shards,
        )

        self.shard_mgr.load()

        # Load all supported models into the registry concurrently.
        # Falls back to [model_name] when supported_models is empty so that
        # single-model deployments continue to work without extra config.
        models_to_load = self.config.supported_models or [self.config.model_name]
        await self._model_registry.load_all(models_to_load, self.config)

        if self.config.lighthouse_api_key:
            self.storage = StorageClient(api_key=self.config.lighthouse_api_key)

        await self.blockchain.connect()
        await self._ensure_registered()

        self.p2p = P2PLayer(self.config, self.shard_mgr)
        await self.p2p.start()

        self._running = True
        logger.info(
            "Node ready. Starting job poll loop with %d workers.",
            self.config.max_concurrent_jobs,
        )

        # Start N worker coroutines that drain the priority queue.
        worker_tasks = [
            asyncio.create_task(self._job_worker(worker_id=i))
            for i in range(self.config.max_concurrent_jobs)
        ]
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        try:
            await self._job_loop()
        finally:
            heartbeat_task.cancel()
            for task in worker_tasks:
                task.cancel()

    async def stop(self) -> None:
        logger.info("Shutting down node…")
        self._running = False
        if self.p2p:
            await self.p2p.stop()
        await self.blockchain.close()

    # ────────────────────────── job loop ─────────────────────────────────────

    async def _heartbeat_loop(self) -> None:
        """Update on-chain endpoint every 60 s and permissionlessly settle expired jobs."""
        public_host = self.config.public_host or self.config.listen_host
        endpoint = f"{public_host}:{self.config.listen_port}"
        while self._running:
            await self.blockchain.heartbeat(endpoint)
            await self.blockchain.auto_settle_expired_jobs()
            if METRICS_AVAILABLE:
                from .metrics import heartbeat_timestamp

                heartbeat_timestamp.set(time.time())
            await asyncio.sleep(60)

    async def _job_loop(self) -> None:
        """Poll the blockchain for open jobs and enqueue new ones (deduplication here)."""
        model_id = self._model_id_bytes()
        while self._running:
            jobs = await self.blockchain.fetch_open_jobs(model_id)
            for job in jobs:
                if job.job_id not in self._active_jobs:
                    # Mark as seen immediately to prevent re-enqueuing on the
                    # next poll before the worker picks it up.
                    self._active_jobs[job.job_id] = True
                    # Negate payment_amount: lower value = higher priority in
                    # asyncio.PriorityQueue (min-heap).
                    priority_key = -job.payment_amount
                    await self._job_queue.put((priority_key, job))
                    logger.debug("Enqueued job %d (payment: %d)", job.job_id, job.payment_amount)
                    if METRICS_AVAILABLE:
                        from .metrics import queue_depth

                        queue_depth.set(self._job_queue.qsize())

            await asyncio.sleep(self.config.job_poll_interval_seconds)

    async def _job_worker(self, worker_id: int) -> None:
        """Drain the priority queue and process jobs one at a time per worker."""
        logger.debug("Job worker %d started", worker_id)
        while self._running:
            try:
                _priority_key, job = await asyncio.wait_for(self._job_queue.get(), timeout=1.0)
            except TimeoutError:
                continue

            if METRICS_AVAILABLE:
                from .metrics import queue_depth

                queue_depth.set(self._job_queue.qsize())

            try:
                await self._handle_job(job)
            except Exception as exc:
                logger.error(
                    "Unhandled exception in worker %d for job %d: %s",
                    worker_id,
                    job.job_id,
                    exc,
                )
            finally:
                self._active_jobs.pop(job.job_id, None)
                self._job_queue.task_done()

        logger.debug("Job worker %d stopping", worker_id)

    async def _handle_job(self, job: OpenJob) -> None:
        set_correlation_id(f"job-{job.job_id}")
        logger.info("Handling job %d (payment: %d tokens)", job.job_id, job.payment_amount)

        # Resolve the ShardManager for this job's model via the registry.
        # Fall back to the legacy self.shard_mgr when the registry has no entry.
        shard_mgr = self._model_registry.get_by_model_id(job.model_id) or self.shard_mgr

        # Dead-on-arrival check: skip jobs whose deadline has already passed.
        if TimeoutManager.is_expired(job.deadline):
            logger.warning(
                "Skipping job %d — deadline already expired (deadline=%d)",
                job.job_id,
                job.deadline,
            )
            return

        claimed = await self.blockchain.claim_job(job)
        if not claimed:
            return

        if METRICS_AVAILABLE:
            from .metrics import active_jobs, jobs_claimed_total

            jobs_claimed_total.inc()
            active_jobs.inc()

        claim_time = time.monotonic()

        # Retry logic: up to _MAX_RETRIES additional attempts after the first.
        result_text: str | None = None
        last_exc: Exception | None = None
        for attempt in range(1 + _MAX_RETRIES):
            try:
                result_text = await self._run_inference(job, shard_mgr)
                break
            except IntegrityError as exc:
                logger.error(
                    "Prompt integrity check failed for job %d: %s — skipping tampered job",
                    job.job_id,
                    exc,
                )
                if METRICS_AVAILABLE:
                    from .metrics import active_jobs, jobs_failed_total

                    jobs_failed_total.inc()
                    active_jobs.dec()
                return
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    logger.warning(
                        "Inference failed for job %d (attempt %d/%d): %s — retrying in %.0fs",
                        job.job_id,
                        attempt + 1,
                        1 + _MAX_RETRIES,
                        exc,
                        _RETRY_DELAY,
                    )
                    await asyncio.sleep(_RETRY_DELAY)
                else:
                    logger.error(
                        "Inference failed for job %d after %d attempts: %s",
                        job.job_id,
                        1 + _MAX_RETRIES,
                        exc,
                    )

        if result_text is None:
            logger.error("Giving up on job %d after all retries: %s", job.job_id, last_exc)
            if METRICS_AVAILABLE:
                from .metrics import active_jobs, jobs_failed_total

                jobs_failed_total.inc()
                active_jobs.dec()
            return

        # Store result on decentralized storage (IPFS via web3.storage or Arweave).
        # Both operations are wrapped in the job's on-chain deadline.
        async def _infer_and_upload() -> str:
            return await self._upload_result(job.job_id, result_text)  # type: ignore[arg-type]

        try:
            result_cid = await TimeoutManager().run_with_deadline(_infer_and_upload(), job.deadline)
        except JobTimeoutError:
            logger.warning(
                "Job %d timed out during upload (deadline=%d) — abandoning",
                job.job_id,
                job.deadline,
            )
            if METRICS_AVAILABLE:
                from .metrics import active_jobs, jobs_failed_total

                jobs_failed_total.inc()
                active_jobs.dec()
            return

        result_bytes = result_text.encode()
        claimed_hash = hashlib.sha256(result_bytes).digest()
        is_valid, reason = ResultVerifier.verify_result(
            result_bytes,
            claimed_hash,
            result_cid,
            job.max_tokens,
        )
        if not is_valid:
            logger.error(
                "Result verification failed for job %d: %s — skipping on-chain submission",
                job.job_id,
                reason,
            )
            if METRICS_AVAILABLE:
                from .metrics import active_jobs, jobs_failed_total

                jobs_failed_total.inc()
                active_jobs.dec()
            return

        await self.blockchain.submit_result(job, result_bytes, result_cid)

        if METRICS_AVAILABLE:
            from .metrics import active_jobs, inference_latency_seconds, jobs_completed_total

            jobs_completed_total.inc()
            active_jobs.dec()
            inference_latency_seconds.observe(time.monotonic() - claim_time)

    # ────────────────────────── inference ────────────────────────────────────

    async def _run_inference(self, job: OpenJob, shard_mgr: ShardManager | None = None) -> str:
        """
        Execute a pipeline-parallel inference pass.

        - Shard 0: embed input → forward its layers → send to shard 1
        - Middle shards: receive activations → forward → send to next
        - Last shard: decode logits → sample tokens → return text

        The actual prompt is retrieved off-chain (delivered encrypted to the
        claiming node via the client's P2P channel or a content-addressed store).

        *shard_mgr* selects the model to use; falls back to ``self.shard_mgr``
        when not provided so existing callers keep working.
        """
        # In a full implementation, the client sends the encrypted prompt
        # directly to the claiming node via a secure P2P channel keyed to
        # the node's wallet public key. Here we show the structural skeleton.

        mgr = shard_mgr if shard_mgr is not None else self.shard_mgr
        loop = asyncio.get_event_loop()

        if self.config.shard_index == 0:
            prompt = await self._fetch_prompt(job)
            input_ids = await loop.run_in_executor(None, self._tokenize_with, mgr, prompt)
            hidden = await loop.run_in_executor(None, mgr.embed, input_ids)
        else:
            hidden = await self._receive_activations(job.job_id)

        hidden = await loop.run_in_executor(None, mgr.forward, hidden)

        if self.config.shard_index < self.config.num_shards - 1:
            # Push activations into the DHT for the next shard to pick up
            await self._push_activations(job.job_id, hidden)
            # Block until the final shard publishes the result to the DHT
            return await self._await_result(job.job_id)
        else:
            # Last shard: generate tokens autoregressively and publish result
            result = await loop.run_in_executor(
                None, self._generate_with, mgr, hidden, job.max_tokens
            )
            await self._publish_result(job.job_id, result)
            return result

    def _tokenize(self, prompt: str):
        return self._tokenize_with(self.shard_mgr, prompt)

    def _tokenize_with(self, mgr: ShardManager, prompt: str):
        tokens = mgr.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        device = next(mgr.model.parameters()).device
        return tokens["input_ids"].to(device)

    def _generate(self, hidden, max_new_tokens: int) -> str:
        return self._generate_with(self.shard_mgr, hidden, max_new_tokens)

    def _generate_with(self, mgr: ShardManager, hidden, max_new_tokens: int) -> str:
        logits = mgr.decode(hidden)
        # Greedy decoding for simplicity; swap for sampling/beam search as needed
        generated = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens = [generated.item()]

        for _ in range(max_new_tokens - 1):
            hidden = mgr.forward(
                mgr.model.layers[-1](hidden)[0] if hasattr(mgr.model, "layers") else hidden
            )
            logits = mgr.decode(hidden)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens.append(next_tok.item())
            if next_tok.item() == mgr.tokenizer.eos_token_id:
                break

        return mgr.tokenizer.decode(tokens, skip_special_tokens=True)

    # ────────────────────────── helpers ──────────────────────────────────────

    async def _fetch_prompt(self, job: OpenJob) -> str:
        """
        Retrieve and decrypt the prompt for a job.

        The client uploads the ECIES-encrypted blob to IPFS at job submission
        time and stores the CID in the on-chain job account.  We download the
        blob, decrypt it with our wallet private key, and verify the SHA-256
        matches the on-chain prompt_hash commitment.
        """
        if self.storage is None:
            raise RuntimeError(
                "LIGHTHOUSE_API_KEY not set — cannot fetch encrypted prompt from IPFS"
            )

        blob = await self.storage.download(job.prompt_cid)

        verify_prompt_hash(blob, job.prompt_hash)

        wallet_seed = self.config.wallet_private_key_bytes()
        return decrypt_prompt(blob, wallet_seed, job.prompt_hash)

    async def _receive_activations(self, job_id: int):
        """
        Receive activations streamed from the previous shard via the DHT.

        Middle shards block here until shard (index-1) pushes the tensor
        into the DHT under key "activations.<job_id>.<shard_index>".
        """
        if self.p2p is None or self.p2p.dht is None:
            raise RuntimeError("P2P layer not started")

        import asyncio

        key = f"activations.{job_id}.{self.config.shard_index}"
        deadline = asyncio.get_event_loop().time() + 300  # 5-minute timeout

        while asyncio.get_event_loop().time() < deadline:
            result = await asyncio.get_event_loop().run_in_executor(None, self.p2p.dht.get, key)
            if result is not None:
                import torch

                tensor_bytes = result["tensor"]
                shape = result["shape"]
                dtype_str = result["dtype"]
                dtype = getattr(torch, dtype_str)
                return torch.frombuffer(bytearray(tensor_bytes), dtype=dtype).reshape(shape)
            await asyncio.sleep(0.5)

        raise TimeoutError(f"Timed out waiting for activations for job {job_id}")

    async def _push_activations(self, job_id: int, hidden) -> None:
        """
        Push this shard's output tensor into the DHT so the next shard can read it.
        Key: "activations.<job_id>.<next_shard_index>"
        """
        if self.p2p is None or self.p2p.dht is None:
            raise RuntimeError("P2P layer not started")

        import hivemind

        next_idx = self.config.shard_index + 1
        key = f"activations.{job_id}.{next_idx}"
        value = {
            "tensor": list(hidden.cpu().numpy().tobytes()),
            "shape": list(hidden.shape),
            "dtype": str(hidden.dtype).replace("torch.", ""),
        }
        await asyncio.get_event_loop().run_in_executor(
            None,
            self.p2p.dht.store,
            key,
            value,
            hivemind.get_dht_time() + 300,  # 5-minute TTL
        )
        logger.debug("Pushed activations for job %d → shard %d", job_id, next_idx)

    async def _publish_result(self, job_id: int, result_text: str) -> None:
        """Publish final result text into DHT so shard 0 can collect it."""
        if self.p2p is None or self.p2p.dht is None:
            return

        import hivemind

        key = f"result.{job_id}"
        await asyncio.get_event_loop().run_in_executor(
            None,
            self.p2p.dht.store,
            key,
            {"text": result_text},
            hivemind.get_dht_time() + 300,
        )

    async def _await_result(self, job_id: int, timeout: float = 300.0) -> str:
        """
        Non-final shards wait here until the last shard publishes the result.
        """
        if self.p2p is None or self.p2p.dht is None:
            raise RuntimeError("P2P layer not started")

        key = f"result.{job_id}"
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            val = await asyncio.get_event_loop().run_in_executor(None, self.p2p.dht.get, key)
            if val is not None:
                return val["text"]
            await asyncio.sleep(0.5)

        raise TimeoutError(f"Timed out waiting for result of job {job_id}")

    async def _upload_result(self, job_id: int, result_text: str) -> str:
        """
        Upload inference result to Lighthouse (IPFS+Filecoin) and return the CID.

        Circuit breaker: if the IPFS upload fails
        _IPFS_CIRCUIT_BREAKER_THRESHOLD times, fall back to a deterministic
        content-hash placeholder so the job can still be submitted on-chain.
        """
        if self.storage is None:
            content_hash = hashlib.sha256(result_text.encode()).hexdigest()
            logger.warning(
                "No storage client — result for job %d not persisted (hash: %s)",
                job_id,
                content_hash[:16],
            )
            return f"bafkrei{content_hash[:32]}"  # deterministic placeholder

        last_exc: Exception | None = None
        for attempt in range(1, _IPFS_CIRCUIT_BREAKER_THRESHOLD + 1):
            try:
                upload_start = time.monotonic()
                cid = await self.storage.upload_result(result_text, job_id)
                if METRICS_AVAILABLE:
                    from .metrics import ipfs_upload_duration_seconds

                    ipfs_upload_duration_seconds.observe(time.monotonic() - upload_start)
                logger.info("Job %d result uploaded: %s", job_id, cid)
                return cid
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "IPFS upload failed for job %d (attempt %d/%d): %s",
                    job_id,
                    attempt,
                    _IPFS_CIRCUIT_BREAKER_THRESHOLD,
                    exc,
                )

        # All upload attempts exhausted — use the content-hash placeholder.
        content_hash = hashlib.sha256(result_text.encode()).hexdigest()
        logger.error(
            "IPFS circuit breaker tripped for job %d after %d attempts (%s). "
            "Falling back to content-hash placeholder.",
            job_id,
            _IPFS_CIRCUIT_BREAKER_THRESHOLD,
            last_exc,
        )
        return f"bafkrei{content_hash[:32]}"

    async def _ensure_registered(self) -> None:
        import torch

        model_id = self._model_id_bytes()
        # Sanity-check: verify our own model_id round-trips correctly.
        if not verify_model_id(self.config.model_name, model_id):
            raise RuntimeError(
                f"Model ID mismatch for '{self.config.model_name}' — integrity check failed"
            )
        gpu_count = torch.cuda.device_count() or 1
        vram_gb = 0
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory // (1024**3)

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
        return compute_model_id(self.config.model_name)


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
