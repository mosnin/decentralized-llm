"""
Tests for client.python.client — DecentralizedLLMClient SDK.

All network calls are mocked so no real HTTP server is required.
"""

import asyncio
import hashlib
from unittest.mock import patch

import pytest

from client.python.client import (
    ClientConfig,
    DecentralizedLLMClient,
    InferenceJob,
)
from client.python.retry import RetryConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run(coro):
    """Run a coroutine synchronously."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# InferenceJob tests
# ---------------------------------------------------------------------------


def test_inference_job_status_pending():
    job = InferenceJob(job_id="abc", model_name="llama", prompt="hi", max_tokens=64)
    assert job.status == "pending"


def test_prompt_hash_is_sha256():
    prompt = "What is the capital of France?"
    job = InferenceJob(job_id="x", model_name="llama", prompt=prompt, max_tokens=32)
    expected = hashlib.sha256(prompt.encode()).digest()
    assert job.prompt_hash == expected


# ---------------------------------------------------------------------------
# ClientConfig tests
# ---------------------------------------------------------------------------


def test_client_config_defaults():
    cfg = ClientConfig()
    assert cfg.gateway_url == "http://localhost:8080"
    assert cfg.max_poll_attempts == 60
    assert cfg.poll_interval_s == 2.0
    assert cfg.timeout_s == 120.0
    assert isinstance(cfg.retry_config, RetryConfig)


# ---------------------------------------------------------------------------
# submit_job tests
# ---------------------------------------------------------------------------


def test_client_submit_job():
    """submit_job calls _make_request and returns an InferenceJob."""
    client = DecentralizedLLMClient()

    with patch.object(client, "_make_request", return_value={"status": "submitted"}) as mock_req:
        job = run(client.submit_job("llama", "Hello", max_tokens=64))

    assert isinstance(job, InferenceJob)
    assert job.model_name == "llama"
    assert job.prompt == "Hello"
    assert job.max_tokens == 64
    assert job.status == "submitted"
    mock_req.assert_called_once()


def test_client_job_stored_after_submit():
    """After submit_job, list_jobs() contains the submitted job."""
    client = DecentralizedLLMClient()

    with patch.object(client, "_make_request", return_value={}):
        job = run(client.submit_job("llama", "Tell me a joke", max_tokens=128))

    jobs = client.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].job_id == job.job_id


# ---------------------------------------------------------------------------
# poll_result tests
# ---------------------------------------------------------------------------


def test_client_poll_result_done():
    """When GET returns status=done, poll_result returns the completed job."""
    client = DecentralizedLLMClient(ClientConfig(poll_interval_s=0.0))

    # Pre-register a job
    job = InferenceJob(job_id="job-1", model_name="llama", prompt="2+2?", max_tokens=16)
    job.status = "submitted"
    client._jobs["job-1"] = job

    done_response = {"status": "done", "result": "4"}

    with patch.object(client, "_make_request", return_value=done_response):
        completed = run(client.poll_result("job-1"))

    assert completed.status == "done"
    assert completed.result == "4"
    assert completed.completed_at is not None


def test_client_poll_result_failed_raises():
    """When GET returns status=failed, poll_result raises RuntimeError."""
    client = DecentralizedLLMClient(ClientConfig(poll_interval_s=0.0))

    job = InferenceJob(job_id="job-2", model_name="llama", prompt="boom", max_tokens=8)
    job.status = "submitted"
    client._jobs["job-2"] = job

    failed_response = {"status": "failed", "error": "OOM"}

    with patch.object(client, "_make_request", return_value=failed_response):
        with pytest.raises(RuntimeError, match="job-2"):
            run(client.poll_result("job-2"))


def test_client_poll_timeout_raises():
    """When status never becomes done/failed, poll_result raises TimeoutError."""
    cfg = ClientConfig(max_poll_attempts=3, poll_interval_s=0.0)
    client = DecentralizedLLMClient(cfg)

    job = InferenceJob(job_id="job-3", model_name="llama", prompt="slow", max_tokens=8)
    job.status = "submitted"
    client._jobs["job-3"] = job

    running_response = {"status": "running"}

    with patch.object(client, "_make_request", return_value=running_response):
        with pytest.raises(TimeoutError):
            run(client.poll_result("job-3"))


# ---------------------------------------------------------------------------
# infer end-to-end test
# ---------------------------------------------------------------------------


def test_client_infer_returns_text():
    """infer() submits a job, polls until done, and returns the result string."""
    cfg = ClientConfig(poll_interval_s=0.0)
    client = DecentralizedLLMClient(cfg)

    submit_response = {"status": "submitted"}
    done_response = {"status": "done", "result": "The answer is 42."}

    call_count = 0

    def fake_make_request(method, path, body=None):
        nonlocal call_count
        call_count += 1
        if method == "POST":
            return submit_response
        return done_response

    with patch.object(client, "_make_request", side_effect=fake_make_request):
        result = run(client.infer("llama", "What is the answer?", max_tokens=32))

    assert result == "The answer is 42."
    assert call_count >= 2  # at least one POST + one GET


# ---------------------------------------------------------------------------
# Context manager test
# ---------------------------------------------------------------------------


def test_context_manager():
    """async with DecentralizedLLMClient() works without errors."""

    async def _use_cm():
        async with DecentralizedLLMClient() as client:
            assert client is not None

    run(_use_cm())
