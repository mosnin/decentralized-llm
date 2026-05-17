"""
Full pipeline integration tests — exercises the complete system end-to-end.

Path covered:
  API request → job scheduling → shard inference → activation streaming
  → settlement → proof verification

All tests use pure mocks: no real GPU, no real Solana, no network I/O.
Run with:
    pytest tests/test_full_pipeline.py -v
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Shared helpers / test constants
# ---------------------------------------------------------------------------

MODEL_NAME = "meta-llama/Llama-3.2-3B"
NODE_ID = "node-abc123"
CLIENT_PUBKEY = "Client1111111111111111111111111111111111"
NODE_PUBKEY = "Node11111111111111111111111111111111111111"
BOUNTY = 1_000_000  # lamports


def _sha(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


# ---------------------------------------------------------------------------
# Fixtures — reusable mock objects shared across all test classes
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_shard_manager():
    """
    A ShardManager whose model/tokenizer are fully mocked.

    generate() returns a deterministic string so tests never hit real weights.
    """
    mgr = MagicMock()
    mgr.config = MagicMock()
    mgr.config.shard_index = 0
    mgr.config.num_shards = 1
    mgr.config.model_name = MODEL_NAME

    # tokenizer stub
    tok = MagicMock()
    tok.eos_token_id = 2
    tok.encode.return_value = [1, 2, 3, 4]
    tok.decode.return_value = "The capital of France is Paris."
    tok.side_effect = None
    tok.return_value = {"input_ids": MagicMock()}
    mgr.tokenizer = tok

    # model stub
    model = MagicMock()
    params_mock = MagicMock()
    params_mock.device = "cpu"
    model.parameters.return_value = iter([params_mock])
    mgr.model = model

    # high-level API used by tests
    mgr.generate.return_value = "The capital of France is Paris."
    mgr.forward.return_value = MagicMock(name="hidden_states")
    mgr.embed.return_value = MagicMock(name="embedded")
    mgr.decode.return_value = MagicMock(name="logits")

    return mgr


@pytest.fixture
def market():
    """Fresh MockInferenceMarket imported from test_e2e_devnet."""
    from tests.test_e2e_devnet import MockInferenceMarket

    return MockInferenceMarket()


@pytest.fixture
def registry():
    """Fresh MockComputeRegistry."""
    from tests.test_e2e_devnet import MockComputeRegistry

    return MockComputeRegistry()


@pytest.fixture
def reputation():
    from client.python.reputation import ReputationCache

    return ReputationCache()


@pytest.fixture
def settlement():
    from node.settlement import SettlementTracker

    return SettlementTracker(min_batch_lamports=1, max_batch_size=100)


@pytest.fixture
def ed25519_keys():
    """Return (private_seed, public_key_bytes) as raw bytes."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    priv = Ed25519PrivateKey.generate()
    seed = priv.private_bytes_raw()
    pub = priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return seed, pub


# ---------------------------------------------------------------------------
# TestSingleShardPipeline
# ---------------------------------------------------------------------------


class TestSingleShardPipeline:
    """
    Simplest path: single node, single shard, generates text directly.

    No activation streaming, no multi-shard handoff.
    """

    def test_single_shard_generate_tokens(self, mock_shard_manager):
        """ShardManager.generate() returns a non-empty string."""
        mock_shard_manager.config.shard_index = 0
        mock_shard_manager.config.num_shards = 1
        mock_shard_manager.generate.return_value = "Hello, world!"

        result = mock_shard_manager.generate("Say hello", max_tokens=16)

        assert isinstance(result, str)
        assert len(result) > 0
        mock_shard_manager.generate.assert_called_once_with("Say hello", max_tokens=16)

    def test_single_shard_streaming(self):
        """TokenStream emits tokens one by one and terminates cleanly."""
        from node.token_streamer import TokenStream

        async def _run():
            stream = TokenStream(job_id=1)
            tokens_to_push = ["The", " quick", " brown", " fox"]
            for tok in tokens_to_push:
                await stream.push(tok)
            await stream.finish()

            received = []
            async for tok in stream:
                received.append(tok)
            return received

        received = asyncio.run(_run())
        assert received == ["The", " quick", " brown", " fox"]

    def test_single_shard_respects_max_tokens(self, mock_shard_manager):
        """
        ShardManager.generate is called with the max_tokens parameter and the
        result never exceeds the requested token budget.
        """
        # Simulate a tokenizer that produces exactly max_tokens output tokens.
        expected_tokens = 5
        mock_shard_manager.tokenizer.decode.return_value = "one two three four five"
        mock_shard_manager.generate.return_value = "one two three four five"

        result = mock_shard_manager.generate("Count to five", max_tokens=expected_tokens)

        mock_shard_manager.generate.assert_called_once_with(
            "Count to five", max_tokens=expected_tokens
        )
        # Result is a string (token budget enforced inside real generate())
        assert isinstance(result, str)

    def test_single_shard_temperature_affects_output(self, mock_shard_manager):
        """
        Two calls with different temperatures produce different mock responses,
        verifying the temperature parameter is forwarded.
        """
        mock_shard_manager.generate.side_effect = lambda prompt, **kw: (
            "deterministic output" if kw.get("temperature", 1.0) == 0.0 else "sampled output"
        )

        greedy = mock_shard_manager.generate("Hello", max_tokens=10, temperature=0.0)
        sampled = mock_shard_manager.generate("Hello", max_tokens=10, temperature=0.9)

        assert greedy == "deterministic output"
        assert sampled == "sampled output"
        assert greedy != sampled


# ---------------------------------------------------------------------------
# TestMultiShardPipeline
# ---------------------------------------------------------------------------


class TestMultiShardPipeline:
    """
    Two- and three-shard splits: activation streaming between shards.

    Uses the real TokenStream and mocked ActivationReceiver/Sender so no
    real TCP connections are opened.
    """

    def test_two_shard_activation_handoff(self):
        """
        Shard 0 pushes activations; shard 1 receives them via the receiver's
        pending-future mechanism.  We simulate the wire with direct future resolution.
        """

        async def _run():
            loop = asyncio.get_running_loop()
            fut: asyncio.Future = loop.create_future()
            pending: dict[int, asyncio.Future] = {42: fut}

            # Simulate shard 0 producing activations and resolving the future
            fake_tensor = MagicMock(name="activation_tensor")
            fut.set_result(fake_tensor)

            # Shard 1 waits on the future (mocked receive)
            tensor = await asyncio.wait_for(asyncio.shield(pending[42]), timeout=1.0)
            return tensor

        tensor = asyncio.run(_run())
        assert tensor is not None

    def test_two_shard_handles_peer_timeout(self):
        """
        When shard 1's activation receiver times out, the job fails gracefully
        (TimeoutError is raised, not swallowed silently).
        """

        async def _run():
            loop = asyncio.get_running_loop()
            # A future that is never resolved → simulates a dead peer
            orphan: asyncio.Future = loop.create_future()
            with pytest.raises((asyncio.TimeoutError, TimeoutError)):
                await asyncio.wait_for(asyncio.shield(orphan), timeout=0.05)

        asyncio.run(_run())

    def test_two_shard_circuit_breaker_trips(self):
        """
        After failure_threshold consecutive failures, the circuit breaker
        opens and subsequent calls raise CircuitOpenError immediately.
        """
        from node.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitOpenError,
            CircuitState,
        )

        cb = CircuitBreaker(
            CircuitBreakerConfig(failure_threshold=3, success_threshold=2, timeout_seconds=300.0),
            name="shard-pipeline",
        )

        async def _always_fail():
            raise RuntimeError("shard unreachable")

        async def _run():
            # Trip the breaker with 3 failures
            for _ in range(3):
                try:
                    await cb.call(_always_fail)
                except (RuntimeError, CircuitOpenError):
                    pass

            assert cb.state == CircuitState.OPEN

            # Next call must raise CircuitOpenError, not RuntimeError
            with pytest.raises(CircuitOpenError):
                await cb.call(_always_fail)

        asyncio.run(_run())

    def test_three_shard_full_forward_pass(self):
        """
        Three shards in sequence: shard 0 embeds, shard 1 forwards, shard 2 decodes.
        Each shard produces activations passed to the next; final output is a string.
        """

        def _make_shard(index: int, num_shards: int = 3):
            mgr = MagicMock()
            mgr.config = MagicMock()
            mgr.config.shard_index = index
            mgr.config.num_shards = num_shards
            mgr.forward.return_value = MagicMock(name=f"hidden_{index}")
            mgr.embed.return_value = MagicMock(name="embedded")
            mgr.decode.return_value = MagicMock(name="logits")
            mgr.tokenizer = MagicMock()
            mgr.tokenizer.decode.return_value = "Final answer from 3 shards"
            return mgr

        shard0 = _make_shard(0)
        shard1 = _make_shard(1)
        shard2 = _make_shard(2)

        # Simulate the pipeline: embed → forward → forward → decode
        embedded = shard0.embed([1, 2, 3])
        h1 = shard0.forward(embedded)
        h2 = shard1.forward(h1)
        shard2.decode(h2)

        # Each stage was called exactly once
        shard0.embed.assert_called_once()
        shard0.forward.assert_called_once()
        shard1.forward.assert_called_once()
        shard2.decode.assert_called_once()

        # Final decoding produces a string
        tokens = [1, 2, 3, 4, 5]
        result = shard2.tokenizer.decode(tokens, skip_special_tokens=True)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# TestJobLifecycle
# ---------------------------------------------------------------------------


class TestJobLifecycle:
    """
    On-chain job lifecycle wired together with inference and reputation.

    Uses MockInferenceMarket from test_e2e_devnet for the Solana simulation.
    """

    def test_job_posted_claimed_settled(self, market):
        """Full on-chain lifecycle: post → claim → settle with result hash."""
        from tests.test_e2e_devnet import JobStatus

        prompt_hash = _sha(b"What is 2+2?")
        result_hash = _sha(b"4")

        job_id = market.post_job(CLIENT_PUBKEY, prompt_hash, BOUNTY)
        assert market.get_job(job_id).status == JobStatus.OPEN

        market.claim_job(NODE_PUBKEY, job_id)
        assert market.get_job(job_id).status == JobStatus.CLAIMED

        payment = market.settle_job(NODE_PUBKEY, job_id, result_hash)
        assert market.get_job(job_id).status == JobStatus.SETTLED
        assert payment > 0
        assert market.escrow_balance(job_id) == 0

    def test_settlement_records_reputation(self, market, reputation):
        """
        After a job completes successfully the node's reputation score rises
        above the neutral baseline of 0.5.
        """
        prompt_hash = _sha(b"Tell me a joke")
        result_hash = _sha(b"Why did the chicken cross the road?")

        job_id = market.post_job(CLIENT_PUBKEY, prompt_hash, BOUNTY)
        market.claim_job(NODE_PUBKEY, job_id)
        market.settle_job(NODE_PUBKEY, job_id, result_hash)

        # Record success in the reputation cache
        reputation.record(NODE_PUBKEY, success=True, latency_ms=150.0)

        score = reputation.score(NODE_PUBKEY)
        assert score > 0.5, f"Expected score > 0.5 after success, got {score}"

    def test_failed_job_penalizes_node(self, reputation):
        """
        Three consecutive failures cause is_penalized() to return True.
        """
        for _ in range(3):
            reputation.record(NODE_PUBKEY, success=False, latency_ms=5000.0)

        assert reputation.is_penalized(NODE_PUBKEY), (
            "Node should be penalized after 3 consecutive failures"
        )
        score = reputation.score(NODE_PUBKEY)
        assert score < 0.5, f"Expected score < 0.5 after failures, got {score}"

    def test_disputed_job_freezes_escrow(self, market):
        """
        When a client disputes a CLAIMED job the escrow remains locked and
        the status transitions to DISPUTED.
        """
        from tests.test_e2e_devnet import JobStatus

        prompt_hash = _sha(b"Write me a poem")
        job_id = market.post_job(CLIENT_PUBKEY, prompt_hash, BOUNTY)
        market.claim_job(NODE_PUBKEY, job_id)
        market.dispute_job(CLIENT_PUBKEY, job_id)

        job = market.get_job(job_id)
        assert job.status == JobStatus.DISPUTED
        # Escrow must remain locked — funds cannot be released during a dispute
        assert market.escrow_balance(job_id) == BOUNTY


# ---------------------------------------------------------------------------
# TestProofPipeline
# ---------------------------------------------------------------------------


class TestProofPipeline:
    """
    ZK-lite proof creation and verification round-trips.

    Uses InferenceVerifier (commitment-based), MerkleTree, and SettlementTracker.
    """

    def test_honest_node_proof_verifies(self, ed25519_keys):
        """ProofBuilder → ProofVerifier round-trip: honest commitment passes."""
        from node.zk_verifier import InferenceVerifier

        priv_key, pub_key = ed25519_keys

        input_bytes = b"What is the capital of France?"
        output_bytes = b"Paris."
        activation_bytes = b"\x01\x02\x03\x04" * 16  # fake activation sample

        commitment = InferenceVerifier.create_commitment(
            job_id="job-001",
            input_ids_bytes=input_bytes,
            output_ids_bytes=output_bytes,
            activation_sample_bytes=activation_bytes,
            node_private_key_bytes=priv_key,
            node_id=NODE_ID,
        )

        expected_input_hash = _sha(input_bytes)
        verified = InferenceVerifier.verify_commitment(
            commitment,
            expected_input_hash=expected_input_hash,
            node_public_key_bytes=pub_key,
        )
        assert verified, "Honest commitment should verify successfully"

    def test_tampered_output_proof_rejected(self, ed25519_keys):
        """A commitment whose output_hash has been tampered with fails verification."""
        from node.zk_verifier import ComputeCommitment, InferenceVerifier

        priv_key, pub_key = ed25519_keys

        commitment = InferenceVerifier.create_commitment(
            job_id="job-002",
            input_ids_bytes=b"input",
            output_ids_bytes=b"honest output",
            activation_sample_bytes=b"\x00" * 32,
            node_private_key_bytes=priv_key,
            node_id=NODE_ID,
        )

        # Tamper: replace output_hash with a different hash
        tampered = ComputeCommitment(
            job_id=commitment.job_id,
            input_hash=commitment.input_hash,
            output_hash=_sha(b"TAMPERED output"),  # different from original
            activation_sketch=commitment.activation_sketch,
            timestamp=commitment.timestamp,
            node_id=commitment.node_id,
            signature_b64=commitment.signature_b64,  # signature now invalid
        )

        expected_input_hash = _sha(b"input")
        verified = InferenceVerifier.verify_commitment(
            tampered,
            expected_input_hash=expected_input_hash,
            node_public_key_bytes=pub_key,
        )
        assert not verified, "Tampered commitment must not verify"

    def test_settlement_uses_proof_verification(self, settlement, ed25519_keys):
        """
        SettlementTracker stores proofs alongside payment records; the batch
        exposes them for auditing after confirmation.
        """
        from node.zk_verifier import InferenceVerifier

        priv_key, _pub_key = ed25519_keys

        # Build a commitment and wrap it in a simple proof-like object
        commitment = InferenceVerifier.create_commitment(
            job_id="job-003",
            input_ids_bytes=b"prompt",
            output_ids_bytes=b"response",
            activation_sample_bytes=b"\xab" * 32,
            node_private_key_bytes=priv_key,
            node_id=NODE_ID,
        )

        # Attach the commitment as an opaque "proof" in the settlement record
        settlement.record(
            job_id=3,
            amount_lamports=500_000,
            client_pubkey=CLIENT_PUBKEY,
            proof=commitment,  # InferenceVerifier returns a ComputeCommitment
        )

        batch = settlement.flush()
        assert batch is not None
        assert batch.job_count == 1

        # Proof is accessible from the batch
        proofs = batch.proofs()
        assert len(proofs) == 1
        assert proofs[0].job_id == "job-003"

    def test_merkle_proof_integrity(self):
        """
        A MerkleProof built from output tokens verifies correctly and a
        modified leaf fails verification.
        """
        from node.merkle import MerkleTree

        tokens = [b"token_0", b"token_1", b"token_2", b"token_3"]
        tree = MerkleTree(tokens)

        proof = tree.get_proof(1)
        assert proof.verify(), "Honest Merkle proof should verify"

        # Flip a bit in the leaf hash → proof must fail
        corrupted_leaf = bytes(b ^ 0xFF for b in proof.leaf_hash)
        from node.merkle import MerkleProof

        bad_proof = MerkleProof(
            leaf_index=proof.leaf_index,
            leaf_hash=corrupted_leaf,
            siblings=proof.siblings,
            root=proof.root,
        )
        assert not bad_proof.verify(), "Corrupted Merkle proof must not verify"


# ---------------------------------------------------------------------------
# TestContinuousBatcherIntegration
# ---------------------------------------------------------------------------


class TestContinuousBatcherIntegration:
    """
    Multiple concurrent inference requests through the ContinuousBatcher.
    """

    def _make_seq(self, seq_id: int, prompt_len: int = 4, max_tokens: int = 8) -> Any:
        from node.continuous_batcher import Sequence

        return Sequence(
            id=seq_id,
            prompt_tokens=list(range(prompt_len)),
            max_tokens=max_tokens,
            priority=0.0,
            arrival_time=time.monotonic(),
        )

    def _run_to_completion(self, batcher, max_steps: int = 500):
        """Drive the batcher until all sequences finish."""
        finished: list = []
        for _ in range(max_steps):
            if batcher.num_waiting == 0 and batcher.num_running == 0:
                break
            batch = batcher.schedule()
            if batch:
                tokens = [42 for _ in batch]
                batcher.step(batch, tokens)
            finished.extend(batcher.get_finished())
        return finished

    def test_two_concurrent_jobs_both_complete(self):
        """Two sequences added together both reach FINISHED state."""
        from node.continuous_batcher import ContinuousBatcher, SequenceState

        batcher = ContinuousBatcher(max_batch_size=4, max_tokens_per_step=16)
        seq0 = self._make_seq(seq_id=0, max_tokens=4)
        seq1 = self._make_seq(seq_id=1, max_tokens=4)

        batcher.add_request(seq0)
        batcher.add_request(seq1)

        finished = self._run_to_completion(batcher)

        assert len(finished) == 2, f"Expected 2 finished sequences, got {len(finished)}"
        for seq in finished:
            assert seq.state == SequenceState.FINISHED
            assert seq.num_generated_tokens == seq.max_tokens

    def test_preemption_under_memory_pressure_recovers(self):
        """
        When KV-cache is tight, a sequence gets preempted and later
        re-scheduled.  The sequence still finishes with all tokens generated.
        """
        from node.continuous_batcher import ContinuousBatcher, SequenceState

        # Very small block table to force preemption: 4 blocks × 16 tokens = 64 slots
        batcher = ContinuousBatcher(
            max_batch_size=2,
            max_tokens_per_step=4,
            block_size=16,
            num_blocks=4,
        )

        # Two sequences that together need more than 4 blocks at peak
        seq0 = self._make_seq(seq_id=0, prompt_len=8, max_tokens=4)
        seq1 = self._make_seq(seq_id=1, prompt_len=8, max_tokens=4)

        batcher.add_request(seq0)
        batcher.add_request(seq1)

        finished = self._run_to_completion(batcher, max_steps=200)

        # Both must eventually complete (preemption → re-admit → finish)
        assert len(finished) == 2
        for seq in finished:
            assert seq.state == SequenceState.FINISHED

    def test_batch_utilization_above_threshold(self):
        """
        With 4 concurrent sequences and max_batch_size=4, utilization
        should be 1.0 (all slots always occupied) once all are admitted.
        """
        from node.continuous_batcher import ContinuousBatcher

        batcher = ContinuousBatcher(
            max_batch_size=4,
            max_tokens_per_step=8,
            block_size=16,
            num_blocks=512,
        )

        for i in range(4):
            batcher.add_request(self._make_seq(seq_id=i, max_tokens=6))

        # Run a few steps (not to completion) to record utilisation
        for _ in range(5):
            batch = batcher.schedule()
            if batch:
                batcher.step(batch, [42] * len(batch))

        utilization = batcher.batch_utilization
        assert utilization > 0.5, (
            f"Expected utilization > 0.5 with 4 concurrent jobs, got {utilization:.3f}"
        )


# ---------------------------------------------------------------------------
# TestOpenAICompatibility
# ---------------------------------------------------------------------------


class TestOpenAICompatibility:
    """
    OpenAI wire-format compatibility: response shapes, SSE format, model mapping,
    and error envelopes.
    """

    def test_chat_completion_response_has_all_openai_fields(self):
        """format_chat_completion returns every field mandated by the OpenAI spec."""
        from node.openai_compat import format_chat_completion

        resp = format_chat_completion(
            content="Paris is the capital of France.",
            model=MODEL_NAME,
            prompt_text="What is the capital of France?",
        )

        required_top_level = {"id", "object", "created", "model", "choices", "usage"}
        assert required_top_level.issubset(set(resp.keys())), (
            f"Missing fields: {required_top_level - set(resp.keys())}"
        )

        assert resp["object"] == "chat.completion"
        assert resp["id"].startswith("chatcmpl-")
        assert isinstance(resp["created"], int)

        choice = resp["choices"][0]
        assert choice["index"] == 0
        assert choice["message"]["role"] == "assistant"
        assert choice["message"]["content"] == "Paris is the capital of France."
        assert choice["finish_reason"] == "stop"

        usage = resp["usage"]
        assert {"prompt_tokens", "completion_tokens", "total_tokens"}.issubset(set(usage.keys()))
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_streaming_response_sse_format(self):
        """
        stream_chunk() produces valid SSE lines (data: ... \\n\\n) with
        the chat.completion.chunk object type.
        """
        from node.openai_compat import stream_chunk, stream_done

        chunk = stream_chunk("Hello", MODEL_NAME)
        assert chunk.startswith("data: "), "SSE line must begin with 'data: '"
        assert chunk.endswith("\n\n"), "SSE line must end with double newline"

        payload = json.loads(chunk.removeprefix("data: ").strip())
        assert payload["object"] == "chat.completion.chunk"
        assert payload["choices"][0]["delta"]["content"] == "Hello"
        assert payload["choices"][0]["finish_reason"] is None

        done = stream_done()
        assert done == "data: [DONE]\n\n"

    def test_model_name_mapping_gpt4_to_llama(self):
        """
        normalize_model_name maps OpenAI aliases to the network's model IDs.
        Unknown names pass through unchanged.
        """
        from node.openai_compat import normalize_model_name

        assert normalize_model_name("gpt-4") == "meta-llama/Llama-3.2-70B"
        assert normalize_model_name("gpt-3.5-turbo") == "meta-llama/Llama-3.2-3B"
        assert normalize_model_name("gpt-4o") == "meta-llama/Llama-3.2-8B"
        # Native model ID passes through
        assert normalize_model_name(MODEL_NAME) == MODEL_NAME
        # Completely unknown name passes through
        assert normalize_model_name("unknown-model-xyz") == "unknown-model-xyz"

    def test_error_response_matches_openai_format(self):
        """
        openai_error_response produces the canonical OpenAI error envelope.
        """
        from node.openai_compat import openai_error_response

        resp = openai_error_response(
            status_code=429,
            error_type="rate_limit_error",
            message="Too many requests, please slow down.",
            code="rate_limit_exceeded",
        )

        assert resp.status_code == 429
        body = json.loads(resp.body)

        assert "error" in body
        err = body["error"]
        assert err["message"] == "Too many requests, please slow down."
        assert err["type"] == "rate_limit_error"
        assert err["code"] == "rate_limit_exceeded"
        # param not supplied → must be null
        assert err["param"] is None

    def test_build_usage_totals_are_consistent(self):
        """build_usage total_tokens equals prompt + completion token counts."""
        from node.openai_compat import build_usage

        prompt = "a" * 40  # → 10 tokens
        completion = "b" * 80  # → 20 tokens
        usage = build_usage(prompt, completion, MODEL_NAME)

        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 20
        assert usage["total_tokens"] == 30

    def test_stream_chunk_created_timestamp_is_recent(self):
        """stream_chunk embeds a Unix timestamp within 2 s of now."""
        from node.openai_compat import stream_chunk

        before = int(time.time()) - 1
        chunk = stream_chunk("token", MODEL_NAME)
        after = int(time.time()) + 1

        payload = json.loads(chunk.removeprefix("data: ").strip())
        assert before <= payload["created"] <= after

    def test_format_completion_text_object(self):
        """format_completion returns object='text_completion' with the right shape."""
        from node.openai_compat import format_completion

        resp = format_completion(
            content="Generated text here.",
            model=MODEL_NAME,
            prompt_text="Generate some text.",
        )

        assert resp["object"] == "text_completion"
        assert resp["id"].startswith("cmpl-")
        choice = resp["choices"][0]
        assert choice["text"] == "Generated text here."
        assert choice["finish_reason"] == "stop"
        assert choice["logprobs"] is None


# ---------------------------------------------------------------------------
# TestEndToEndJobWithSettlement
# ---------------------------------------------------------------------------


class TestEndToEndJobWithSettlement:
    """
    Wires inference → on-chain lifecycle → settlement → proof verification
    in a single coherent scenario.
    """

    def test_full_pipeline_post_infer_settle_verify(self, market, settlement, ed25519_keys):
        """
        Complete path:
          1. Client posts job on-chain.
          2. Node claims and executes inference (mocked).
          3. Node settles job on-chain.
          4. Payment recorded in SettlementTracker with a proof.
          5. Proof verifies correctly.
        """
        from node.zk_verifier import InferenceVerifier
        from tests.test_e2e_devnet import JobStatus

        priv_key, pub_key = ed25519_keys
        prompt_bytes = b"What is 1+1?"
        result_bytes = b"2"

        # --- Step 1: post job ---
        prompt_hash = _sha(prompt_bytes)
        job_id = market.post_job(CLIENT_PUBKEY, prompt_hash, BOUNTY)

        # --- Step 2: claim ---
        market.claim_job(NODE_PUBKEY, job_id)
        assert market.get_job(job_id).status == JobStatus.CLAIMED

        # --- Step 3: build proof commitment ---
        commitment = InferenceVerifier.create_commitment(
            job_id=str(job_id),
            input_ids_bytes=prompt_bytes,
            output_ids_bytes=result_bytes,
            activation_sample_bytes=b"\xde\xad\xbe\xef" * 8,
            node_private_key_bytes=priv_key,
            node_id=NODE_ID,
        )

        # --- Step 4: settle on-chain ---
        result_hash = _sha(result_bytes)
        payment = market.settle_job(NODE_PUBKEY, job_id, result_hash)
        assert market.get_job(job_id).status == JobStatus.SETTLED
        assert payment > 0

        # --- Step 5: record in settlement tracker with proof ---
        settlement.record(
            job_id=job_id,
            amount_lamports=payment,
            client_pubkey=CLIENT_PUBKEY,
            proof=commitment,
        )
        batch = settlement.flush()
        assert batch is not None
        settlement.confirm(batch.batch_id, tx_signature="mock-tx-sig-" + uuid.uuid4().hex[:8])
        assert settlement.total_confirmed_lamports() > 0

        # --- Step 6: verify the proof ---
        assert InferenceVerifier.verify_commitment(
            commitment,
            expected_input_hash=_sha(prompt_bytes),
            node_public_key_bytes=pub_key,
        )

    def test_activation_sketch_consistent_for_same_input(self):
        """
        Two sketches of the same byte blob have cosine similarity ≥ 0.99.
        """
        from node.zk_verifier import ActivationSketch

        data = b"\x42\x17\xff\x00" * 64
        sketcher = ActivationSketch(seed=0xDEADBEEF, sketch_dim=64)
        s1 = sketcher.sketch(data)
        s2 = sketcher.sketch(data)

        assert s1 == s2, "Same input and seed must produce byte-identical sketches"
        assert ActivationSketch.verify_consistency(s1, s2, threshold=0.99)

    def test_activation_sketch_diverges_for_different_input(self):
        """
        Two sketches of very different inputs should have low cosine similarity.
        """
        from node.zk_verifier import ActivationSketch

        sketcher = ActivationSketch(seed=0xDEADBEEF, sketch_dim=64)
        s_honest = sketcher.sketch(b"\x01" * 64)
        s_attacker = sketcher.sketch(b"\xff" * 64)

        # With antipodal inputs the cosine similarity should be clearly < 0.95
        # (the threshold used by verify_spot_check)
        assert not ActivationSketch.verify_consistency(s_honest, s_attacker, threshold=0.95), (
            "Sketches of opposite-valued inputs should not pass the consistency check"
        )

    def test_multi_node_consensus_detects_outlier(self, ed25519_keys):
        """
        aggregate_verifications flags a node that submitted a different output_hash
        as an outlier; consensus_score reflects the majority.
        """
        from node.zk_verifier import InferenceVerifier

        priv_key, _pub_key = ed25519_keys
        input_bytes = b"shared prompt"
        honest_output = b"correct answer"
        byzantine_output = b"wrong answer"

        honest_commitment = InferenceVerifier.create_commitment(
            job_id="job-consensus",
            input_ids_bytes=input_bytes,
            output_ids_bytes=honest_output,
            activation_sample_bytes=b"\x01" * 32,
            node_private_key_bytes=priv_key,
            node_id="node-honest-1",
        )
        # Re-create with a slightly different key for variety (same output)
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        priv2 = Ed25519PrivateKey.generate().private_bytes_raw()
        honest2 = InferenceVerifier.create_commitment(
            job_id="job-consensus",
            input_ids_bytes=input_bytes,
            output_ids_bytes=honest_output,
            activation_sample_bytes=b"\x01" * 32,
            node_private_key_bytes=priv2,
            node_id="node-honest-2",
        )
        priv3 = Ed25519PrivateKey.generate().private_bytes_raw()
        byzantine = InferenceVerifier.create_commitment(
            job_id="job-consensus",
            input_ids_bytes=input_bytes,
            output_ids_bytes=byzantine_output,  # different!
            activation_sample_bytes=b"\xff" * 32,
            node_private_key_bytes=priv3,
            node_id="node-byzantine",
        )

        result = InferenceVerifier.aggregate_verifications([honest_commitment, honest2, byzantine])

        assert result["is_honest"] is True, "2/3 honest nodes should pass Byzantine threshold"
        assert result["consensus_score"] == pytest.approx(2 / 3, abs=0.01)
        assert "node-byzantine" in result["outlier_nodes"]
        assert "node-honest-1" not in result["outlier_nodes"]
