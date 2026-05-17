"""
Tests for the verification pipeline that wires the ZK verifier and Merkle
proof system into the job settlement flow.

All tests are self-contained — no external services, network calls, or
pre-generated key files are required.
"""

from __future__ import annotations

import hashlib
import os

import pytest

from node.verification_pipeline import (
    InferenceProof,
    ProofBuilder,
    ProofVerifier,
    SettlementVerifier,
)
from node.zk_verifier import ComputeCommitment

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------


def _sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _generate_key_pair() -> tuple[bytes, bytes]:
    """Return (private_key_bytes, public_key_bytes).

    Uses Ed25519 when the ``cryptography`` package is available, otherwise
    returns the same 32-byte secret for both keys (HMAC-SHA256 fallback).
    """
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        priv = Ed25519PrivateKey.generate()
        priv_bytes = priv.private_bytes_raw()
        pub_bytes = priv.public_key().public_bytes_raw()
        return priv_bytes, pub_bytes
    except ImportError:
        secret = os.urandom(32)
        return secret, secret  # symmetric fallback


# Shared key pair for tests that just need a stable node identity
_PRIV, _PUB = _generate_key_pair()

# Stable inputs used by many tests
_INPUT_IDS_BYTES = b"\x01\x02\x03\x04" * 8
_OUTPUT_TOKEN_IDS = [10, 20, 30, 40, 50]
_ACTIVATION_BYTES = b"\xde\xad\xbe\xef" * 32
_NODE_ID = "test-node-001"
_JOB_ID = 42


def _build_proof(
    job_id: int = _JOB_ID,
    input_ids_bytes: bytes = _INPUT_IDS_BYTES,
    output_token_ids: list[int] | None = None,
    activation_bytes: bytes = _ACTIVATION_BYTES,
    node_id: str = _NODE_ID,
    priv: bytes | None = None,
) -> InferenceProof:
    """Convenience factory that calls ProofBuilder.build_proof."""
    if output_token_ids is None:
        output_token_ids = _OUTPUT_TOKEN_IDS
    if priv is None:
        priv = _PRIV
    builder = ProofBuilder(priv)
    return builder.build_proof(
        job_id=job_id,
        input_ids_bytes=input_ids_bytes,
        output_token_ids=output_token_ids,
        activation_sample_bytes=activation_bytes,
        node_id=node_id,
    )


# ---------------------------------------------------------------------------
# 1. ProofBuilder.build_proof returns an InferenceProof
# ---------------------------------------------------------------------------


def test_build_proof_returns_inference_proof():
    proof = _build_proof()

    assert isinstance(proof, InferenceProof)
    assert proof.job_id == _JOB_ID
    assert proof.node_id == _NODE_ID
    assert proof.token_count == len(_OUTPUT_TOKEN_IDS)
    assert isinstance(proof.output_merkle_root, str)
    assert len(proof.output_merkle_root) == 64  # hex SHA-256 → 32 bytes → 64 hex chars
    assert isinstance(proof.commitment, ComputeCommitment)
    assert proof.proof_version == 1
    assert isinstance(proof.timestamp, float) and proof.timestamp > 0


# ---------------------------------------------------------------------------
# 2. Merkle root in the proof is valid
# ---------------------------------------------------------------------------


def test_proof_has_valid_merkle_root():
    """The stored Merkle root must equal a freshly built tree's root."""
    from node.merkle import MerkleTree

    proof = _build_proof()
    leaves = [tid.to_bytes(4, "big") for tid in _OUTPUT_TOKEN_IDS]
    tree = MerkleTree(leaves)
    assert proof.output_merkle_root == tree.root.hex()


# ---------------------------------------------------------------------------
# 3. Commitment in the proof verifies correctly
# ---------------------------------------------------------------------------


def test_proof_commitment_verified():
    """The embedded ComputeCommitment must pass InferenceVerifier.verify_commitment."""
    from node.zk_verifier import InferenceVerifier

    proof = _build_proof()
    expected_input_hash = _sha256(_INPUT_IDS_BYTES)

    ok = InferenceVerifier.verify_commitment(
        commitment=proof.commitment,
        expected_input_hash=expected_input_hash,
        node_public_key_bytes=_PUB,
    )
    assert ok is True


# ---------------------------------------------------------------------------
# 4. ProofVerifier.verify_proof passes for an honest node
# ---------------------------------------------------------------------------


def test_verify_proof_passes_for_honest_node():
    proof = _build_proof()
    verifier = ProofVerifier()

    result = verifier.verify_proof(
        proof=proof,
        expected_input_hash=_sha256(_INPUT_IDS_BYTES),
        node_public_key_bytes=_PUB,
        output_token_ids=_OUTPUT_TOKEN_IDS,
    )
    assert result is True


# ---------------------------------------------------------------------------
# 5. ProofVerifier.verify_proof fails when output tokens are tampered
# ---------------------------------------------------------------------------


def test_verify_proof_fails_tampered_output():
    """Supplying different output token IDs must cause Merkle root mismatch."""
    proof = _build_proof(output_token_ids=[10, 20, 30, 40, 50])
    verifier = ProofVerifier()

    tampered_token_ids = [10, 20, 30, 40, 99]  # last token changed

    result = verifier.verify_proof(
        proof=proof,
        expected_input_hash=_sha256(_INPUT_IDS_BYTES),
        node_public_key_bytes=_PUB,
        output_token_ids=tampered_token_ids,
    )
    assert result is False


# ---------------------------------------------------------------------------
# 6. ProofVerifier.verify_token_at_index
# ---------------------------------------------------------------------------


def test_verify_token_at_index():
    output_token_ids = [100, 200, 300, 400]
    proof = _build_proof(output_token_ids=output_token_ids)
    verifier = ProofVerifier()

    # All valid positions must verify
    for idx, tok in enumerate(output_token_ids):
        assert verifier.verify_token_at_index(proof, tok, idx, output_token_ids) is True

    # Wrong token_id at a valid index must fail
    assert verifier.verify_token_at_index(proof, 9999, 0, output_token_ids) is False

    # Out-of-range index must return False, not raise
    assert verifier.verify_token_at_index(proof, 100, 999, output_token_ids) is False


# ---------------------------------------------------------------------------
# 7. ProofVerifier.spot_check passes for honest activations
# ---------------------------------------------------------------------------


def test_spot_check_passes_for_honest_activations():
    """A fresh activation sample identical to the committed one must pass."""
    proof = _build_proof(activation_bytes=_ACTIVATION_BYTES)
    verifier = ProofVerifier()

    # Same bytes → same sketch → cosine similarity = 1.0
    result = verifier.spot_check(proof, fresh_activation_bytes=_ACTIVATION_BYTES)
    assert result is True


# ---------------------------------------------------------------------------
# 8. SettlementVerifier settles without a proof (client opted out)
# ---------------------------------------------------------------------------


def test_settlement_verifier_settles_without_proof():
    sv = SettlementVerifier()
    ok, reason = sv.should_settle(
        job_id=_JOB_ID,
        proof=None,
        node_pubkey_bytes=_PUB,
        expected_input_hash=_sha256(_INPUT_IDS_BYTES),
    )
    assert ok is True
    assert reason == "no_proof"


# ---------------------------------------------------------------------------
# 9. SettlementVerifier rejects a tampered proof
# ---------------------------------------------------------------------------


def test_settlement_verifier_rejects_tampered_proof():
    """A proof with an invalid commitment signature must not be settled."""
    proof = _build_proof()

    # Tamper with the commitment's output_hash — signature will no longer match

    tampered_commitment = ComputeCommitment(
        job_id=proof.commitment.job_id,
        input_hash=proof.commitment.input_hash,
        output_hash=_sha256(b"malicious output"),  # different
        activation_sketch=proof.commitment.activation_sketch,
        timestamp=proof.commitment.timestamp,
        node_id=proof.commitment.node_id,
        signature_b64=proof.commitment.signature_b64,  # original sig — now invalid
    )
    tampered_proof = InferenceProof(
        job_id=proof.job_id,
        commitment=tampered_commitment,
        output_merkle_root=proof.output_merkle_root,
        token_count=proof.token_count,
        node_id=proof.node_id,
        timestamp=proof.timestamp,
    )

    sv = SettlementVerifier()
    ok, reason = sv.should_settle(
        job_id=_JOB_ID,
        proof=tampered_proof,
        node_pubkey_bytes=_PUB,
        expected_input_hash=_sha256(_INPUT_IDS_BYTES),
    )
    assert ok is False
    assert reason == "invalid_commitment"


# ---------------------------------------------------------------------------
# 10. SettlementVerifier.aggregate_peer_proofs — consensus score
# ---------------------------------------------------------------------------


def test_aggregate_peer_proofs_consensus_score():
    sv = SettlementVerifier()

    # Build several proofs with identical output — full consensus
    tokens_a = [1, 2, 3, 4, 5]
    tokens_b = [9, 8, 7, 6, 5]  # different output for dissenting node

    honest_proofs: list[InferenceProof] = []
    for i in range(4):
        priv, _ = _generate_key_pair()
        honest_proofs.append(_build_proof(job_id=99, output_token_ids=tokens_a, priv=priv))

    dissenting_proof = _build_proof(job_id=99, output_token_ids=tokens_b)

    # 4 out of 5 agree → score = 0.8
    score = sv.aggregate_peer_proofs(honest_proofs + [dissenting_proof])
    assert score == pytest.approx(0.8)

    # All agree → score = 1.0
    score_full = sv.aggregate_peer_proofs(honest_proofs)
    assert score_full == pytest.approx(1.0)

    # Empty list → score = 0.0
    score_empty = sv.aggregate_peer_proofs([])
    assert score_empty == pytest.approx(0.0)
