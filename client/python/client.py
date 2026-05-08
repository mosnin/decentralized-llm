"""
Decentralized LLM Python client SDK.

Usage:
    async with DecentralizedLLMClient(config) as client:
        result = await client.infer("llama", "What is 2+2?", max_tokens=64)
"""

import asyncio
import hashlib
import json
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import AsyncIterator  # noqa: F401  (re-exported for callers)
from dataclasses import dataclass, field

from client.python.reputation import ReputationCache
from client.python.retry import RetryConfig, with_retry  # noqa: F401  (re-exported)

# ---------------------------------------------------------------------------
# Backward-compatibility constants used by existing tests / gateway
# ---------------------------------------------------------------------------

MODEL_IDS = {
    "llama-3.2-1b": hashlib.sha256(b"meta-llama/Llama-3.2-1B").digest(),
    "llama-3.2-3b": hashlib.sha256(b"meta-llama/Llama-3.2-3B").digest(),
    "llama-3.1-8b": hashlib.sha256(b"meta-llama/Llama-3.1-8B").digest(),
    "mistral-7b": hashlib.sha256(b"mistralai/Mistral-7B-v0.3").digest(),
}


@dataclass
class CompletionResponse:
    """Backward-compatible response dataclass (used by existing tests/gateway)."""

    text: str
    job_id: int
    model: str
    tokens_used: int
    total_paid: int  # in base token units
    node: str = ""  # Solana pubkey of the node that served the request


# ---------------------------------------------------------------------------
# New HTTP-based SDK
# ---------------------------------------------------------------------------


@dataclass
class InferenceJob:
    job_id: str
    model_name: str
    prompt: str
    max_tokens: int
    status: str = "pending"  # pending | submitted | running | done | failed
    result: str | None = None
    submitted_at: float = field(default_factory=time.time)
    completed_at: float | None = None

    @property
    def prompt_hash(self) -> bytes:
        return hashlib.sha256(self.prompt.encode()).digest()


@dataclass
class ClientConfig:
    gateway_url: str = "http://localhost:8080"
    max_poll_attempts: int = 60
    poll_interval_s: float = 2.0
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    timeout_s: float = 120.0


class DecentralizedLLMClient:
    """
    Async client for the decentralized LLM network.

    Usage:
        async with DecentralizedLLMClient(config) as client:
            result = await client.infer("llama", "What is 2+2?", max_tokens=64)
    """

    def __init__(self, config: ClientConfig | None = None):
        self.config = config or ClientConfig()
        self._reputation = ReputationCache()
        self._jobs: dict[str, InferenceJob] = {}
        self._session_id = str(uuid.uuid4())

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def _make_request(self, method: str, path: str, body: dict | None = None) -> dict:
        """Synchronous HTTP request (run in executor for async use)."""
        url = f"{self.config.gateway_url}{path}"
        data = json.dumps(body).encode() if body else None
        headers = {
            "Content-Type": "application/json",
            "X-Session-Id": self._session_id,
        }
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
            return json.loads(resp.read())

    async def submit_job(
        self,
        model_name: str,
        prompt: str,
        max_tokens: int = 256,
        payment_lamports: int = 1000,
    ) -> InferenceJob:
        """Submit an inference job and return immediately."""
        job_id = str(uuid.uuid4())
        job = InferenceJob(
            job_id=job_id, model_name=model_name, prompt=prompt, max_tokens=max_tokens
        )
        self._jobs[job_id] = job

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._make_request,
            "POST",
            "/v1/jobs",
            {
                "job_id": job_id,
                "model_name": model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "payment_lamports": payment_lamports,
            },
        )
        job.status = "submitted"
        return job

    async def poll_result(self, job_id: str) -> InferenceJob:
        """Poll until job completes. Raises TimeoutError if max_poll_attempts exceeded."""
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"Unknown job: {job_id}")

        loop = asyncio.get_event_loop()
        for _ in range(self.config.max_poll_attempts):
            data = await loop.run_in_executor(
                None, self._make_request, "GET", f"/v1/jobs/{job_id}", None
            )
            status = data.get("status", "unknown")
            if status == "done":
                job.status = "done"
                job.result = data.get("result")
                job.completed_at = time.time()
                return job
            if status == "failed":
                job.status = "failed"
                raise RuntimeError(f"Job {job_id} failed: {data.get('error', 'unknown')}")
            await asyncio.sleep(self.config.poll_interval_s)

        raise TimeoutError(
            f"Job {job_id} did not complete within {self.config.max_poll_attempts} polls"
        )

    async def infer(
        self,
        model_name: str,
        prompt: str,
        max_tokens: int = 256,
        payment_lamports: int = 1000,
    ) -> str:
        """Submit and wait for inference result. Returns result text."""
        job = await self.submit_job(model_name, prompt, max_tokens, payment_lamports)
        completed = await self.poll_result(job.job_id)
        return completed.result or ""

    def list_jobs(self) -> list[InferenceJob]:
        return list(self._jobs.values())
