"""
Tests for the zero-knowledge-lite inference verification system.

All tests are self-contained — no external services, pre-generated key files,
or network calls are required.  Keys are generated fresh in each test (or
shared via module-level fixtures for speed).
"""

from __future__ import annotations

import hashlib
import os
import struct
import time

import pytest

from node.zk_verifier import ActivationSketch, ComputeCommitment, InferenceVerifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _generate_key_pair() -> tuple[bytes, bytes]:
    """
    Return (private_key_bytes, public_key_bytes) for Ed25519 (or HMAC fallback).

    For Ed25519 the private key is a 32-byte seed; the public key is the
    corresponding 32-byte compressed point.  For the HMAC fallback both
    "keys" are the same 32-byte secret (symmetric scheme).
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


def _make_commitment(
    job_id: str = "job-001",
    input_bytes: bytes = b"\x01\x02\x03\x04",
    output_bytes: bytes = b"\x05\x06\x07\x08",
    activation_bytes: bytes = b"\x10" * 128,
    priv: bytes | None = None,
    pub: bytes | None = None,
    node_id: str = "node-a",
) -> tuple[ComputeCommitment, bytes, bytes]:
    """
    Convenience factory.  Returns (commitment, priv_key, pub_key).
    """
    if priv is None or pub is None:
        priv, pub = _generate_key_pair()
    commitment = InferenceVerifier.create_commitment(
        job_id=job_id,
        input_ids_bytes=input_bytes,
        output_ids_bytes=output_bytes,
        activation_sample_bytes=activation_bytes,
        node_private_key_bytes=priv,
        node_id=node_id,
    )
    return commitment, priv, pub


# ---------------------------------------------------------------------------
# 1. Commitment creation
# ---------------------------------------------------------------------------

class TestCommitmentCreation:
    def test_fields_populated(self):
        """create_commitment populates all required fields."""
        commitment, _priv, _pub = _make_commitment()

        assert commitment.job_id == "job-001"
        assert isinstance(commitment.input_hash, bytes) and len(commitment.input_hash) == 32
        assert isinstance(commitment.output_hash, bytes) and len(commitment.output_hash) == 32
        assert isinstance(commitment.activation_sketch, bytes)
        assert len(commitment.activation_sketch) == InferenceVerifier._SKETCH_DIM * 4
        assert isinstance(commitment.timestamp, float)
        assert commitment.node_id == "node-a"
        assert isinstance(commitment.signature_b64, str) and len(commitment.signature_b64) > 0

    def test_input_hash_matches_sha256(self):
        """input_hash must equal SHA-256 of the supplied input bytes."""
        input_bytes = b"\xAA\xBB\xCC" * 10
        commitment, _priv, _pub = _make_commitment(input_bytes=input_bytes)

        assert commitment.input_hash == _sha256(input_bytes)

    def test_output_hash_matches_sha256(self):
        """output_hash must equal SHA-256 of the supplied output bytes."""
        output_bytes = b"\x11\x22\x33" * 10
        commitment, _priv, _pub = _make_commitment(output_bytes=output_bytes)

        assert commitment.output_hash == _sha256(output_bytes)

    def test_different_jobs_produce_different_commitments(self):
        """Two different jobs should not share output hashes (even with same content)."""
        priv, pub = _generate_key_pair()
        c1, _, _ = _make_commitment(job_id="job-A", priv=priv, pub=pub)
        c2, _, _ = _make_commitment(job_id="job-B", priv=priv, pub=pub)

        # job_ids differ
        assert c1.job_id != c2.job_id
        # signatures differ because payload includes job_id
        assert c1.signature_b64 != c2.signature_b64


# ---------------------------------------------------------------------------
# 2. Commitment verification — valid path
# ---------------------------------------------------------------------------

class TestCommitmentVerificationValid:
    def test_valid_commitment_passes(self):
        """A freshly created commitment should pass verify_commitment."""
        input_bytes = b"prompt tokens here"
        commitment, _priv, pub = _make_commitment(input_bytes=input_bytes)

        result = InferenceVerifier.verify_commitment(
            commitment=commitment,
            expected_input_hash=_sha256(input_bytes),
            node_public_key_bytes=pub,
        )

        assert result is True

    def test_verify_is_deterministic(self):
        """verify_commitment must return the same answer on repeated calls."""
        input_bytes = b"repeatable"
        commitment, _priv, pub = _make_commitment(input_bytes=input_bytes)
        expected = _sha256(input_bytes)

        r1 = InferenceVerifier.verify_commitment(commitment, expected, pub)
        r2 = InferenceVerifier.verify_commitment(commitment, expected, pub)

        assert r1 == r2 == True  # noqa: E712


# ---------------------------------------------------------------------------
# 3. Commitment rejection — tampered output hash
# ---------------------------------------------------------------------------

class TestCommitmentRejectionTamperedOutput:
    def test_tampered_output_hash_fails_signature(self):
        """
        If the output_hash field is altered after signing the signature check
        must fail because the canonical payload will no longer match.
        """
        input_bytes = b"original prompt"
        commitment, _priv, pub = _make_commitment(input_bytes=input_bytes)

        # Tamper with output_hash
        tampered = ComputeCommitment(
            job_id=commitment.job_id,
            input_hash=commitment.input_hash,
            output_hash=_sha256(b"fake output"),   # different!
            activation_sketch=commitment.activation_sketch,
            timestamp=commitment.timestamp,
            node_id=commitment.node_id,
            signature_b64=commitment.signature_b64,
        )

        result = InferenceVerifier.verify_commitment(
            commitment=tampered,
            expected_input_hash=_sha256(input_bytes),
            node_public_key_bytes=pub,
        )

        assert result is False


# ---------------------------------------------------------------------------
# 4. Commitment rejection — bad signature
# ---------------------------------------------------------------------------

class TestCommitmentRejectionBadSignature:
    def test_wrong_public_key_fails(self):
        """Using a different public key should cause signature verification to fail."""
        input_bytes = b"prompt"
        commitment, _priv, _pub = _make_commitment(input_bytes=input_bytes)

        _other_priv, other_pub = _generate_key_pair()

        result = InferenceVerifier.verify_commitment(
            commitment=commitment,
            expected_input_hash=_sha256(input_bytes),
            node_public_key_bytes=other_pub,
        )

        assert result is False

    def test_corrupted_signature_fails(self):
        """A bit-flipped signature must not verify."""
        input_bytes = b"honest prompt"
        commitment, _priv, pub = _make_commitment(input_bytes=input_bytes)

        import base64
        raw_sig = bytearray(base64.b64decode(commitment.signature_b64))
        raw_sig[0] ^= 0xFF  # flip first byte
        bad_sig_b64 = base64.b64encode(bytes(raw_sig)).decode()

        tampered = ComputeCommitment(
            job_id=commitment.job_id,
            input_hash=commitment.input_hash,
            output_hash=commitment.output_hash,
            activation_sketch=commitment.activation_sketch,
            timestamp=commitment.timestamp,
            node_id=commitment.node_id,
            signature_b64=bad_sig_b64,
        )

        result = InferenceVerifier.verify_commitment(
            commitment=tampered,
            expected_input_hash=_sha256(input_bytes),
            node_public_key_bytes=pub,
        )

        assert result is False

    def test_wrong_input_hash_fails(self):
        """Supplying the wrong expected_input_hash must fail even with a valid sig."""
        input_bytes = b"real prompt"
        commitment, _priv, pub = _make_commitment(input_bytes=input_bytes)

        result = InferenceVerifier.verify_commitment(
            commitment=commitment,
            expected_input_hash=_sha256(b"completely different prompt"),
            node_public_key_bytes=pub,
        )

        assert result is False


# ---------------------------------------------------------------------------
# 5. Sketch consistency — same input → same sketch
# ---------------------------------------------------------------------------

class TestSketchConsistencySameInput:
    def test_identical_inputs_produce_identical_sketches(self):
        """sketch() is deterministic: same bytes + same seed → identical output."""
        sketcher = ActivationSketch(seed=42)
        data = b"\x01\x02\x03" * 64

        s1 = sketcher.sketch(data)
        s2 = sketcher.sketch(data)

        assert s1 == s2

    def test_verify_consistency_returns_true_for_same_sketch(self):
        """verify_consistency of a sketch with itself must return True."""
        sketcher = ActivationSketch(seed=12345)
        data = os.urandom(256)
        sketch = sketcher.sketch(data)

        assert ActivationSketch.verify_consistency(sketch, sketch) is True

    def test_high_similarity_near_identical_data(self):
        """
        Two activations that differ in only a few bytes should produce sketches
        with cosine similarity above the default 0.95 threshold.
        """
        seed = 7
        sketcher = ActivationSketch(seed=seed, sketch_dim=64)

        base = b"\x80" * 512
        # Introduce a tiny perturbation in one float (4 bytes)
        perturbed = bytearray(base)
        perturbed[0] = 0x81
        perturbed = bytes(perturbed)

        s1 = sketcher.sketch(base)
        s2 = sketcher.sketch(perturbed)

        assert ActivationSketch.verify_consistency(s1, s2, threshold=0.95) is True


# ---------------------------------------------------------------------------
# 6. Sketch inconsistency detection — different inputs → low similarity
# ---------------------------------------------------------------------------

class TestSketchInconsistencyDetection:
    def test_completely_different_inputs_below_threshold(self):
        """
        Completely different activation tensors should fail the consistency check
        (cosine similarity well below 0.95).
        """
        sketcher = ActivationSketch(seed=99, sketch_dim=64)
        # All zeros vs all ones — maximally different float representations
        s1 = sketcher.sketch(b"\x00" * 256)
        s2 = sketcher.sketch(b"\xff" * 256)

        # With a 64-dim sketch these should be clearly distinguishable
        assert ActivationSketch.verify_consistency(s1, s2, threshold=0.95) is False

    def test_different_seeds_same_input_are_inconsistent(self):
        """
        Two sketchers with *different* seeds project to incompatible spaces
        and should fail consistency even on the same input.
        """
        data = b"\xAA" * 256
        s1 = ActivationSketch(seed=1).sketch(data)
        s2 = ActivationSketch(seed=2).sketch(data)

        # Different projection matrices → different fingerprints
        assert s1 != s2


# ---------------------------------------------------------------------------
# 7. Spot-check passes for honest node
# ---------------------------------------------------------------------------

class TestSpotCheckHonestNode:
    def test_spot_check_passes_with_same_activations(self):
        """
        If the fresh activation sample matches what was committed, the spot
        check must pass.
        """
        activation_bytes = b"\xDE\xAD\xBE\xEF" * 64
        commitment, _priv, _pub = _make_commitment(activation_bytes=activation_bytes)

        # The "fresh" sample is the same bytes (honest node)
        result = InferenceVerifier.verify_spot_check(
            commitment=commitment,
            fresh_activation_bytes=activation_bytes,
        )

        assert result is True

    def test_spot_check_passes_with_near_identical_activations(self):
        """
        Minor floating-point noise in activations (a few bits flipped) must not
        cause a false failure given the 0.95 cosine threshold.
        """
        base = b"\x3F\x80\x00\x00" * 64  # many 1.0 floats
        perturbed = bytearray(base)
        perturbed[4] = 0x3F  # barely change one float
        commitment, _priv, _pub = _make_commitment(activation_bytes=base)

        result = InferenceVerifier.verify_spot_check(
            commitment=commitment,
            fresh_activation_bytes=bytes(perturbed),
            threshold=0.95,
        )

        assert result is True


# ---------------------------------------------------------------------------
# 8. Spot-check fails for dishonest node
# ---------------------------------------------------------------------------

class TestSpotCheckDishonestNode:
    def test_spot_check_fails_with_different_activations(self):
        """
        A dishonest node commits a fabricated sketch but cannot reproduce the
        corresponding activations on demand; the fresh sample will be different.
        """
        real_activations = b"\x00" * 256   # what was actually computed
        fake_activations = b"\xFF" * 256   # what the dishonest node committed

        # Node committed based on *fake* activations
        commitment, _priv, _pub = _make_commitment(activation_bytes=fake_activations)

        # Verifier requests fresh activations — honest retrieval gives *real* ones
        result = InferenceVerifier.verify_spot_check(
            commitment=commitment,
            fresh_activation_bytes=real_activations,
            threshold=0.95,
        )

        assert result is False


# ---------------------------------------------------------------------------
# 9. aggregate_verifications reaches consensus
# ---------------------------------------------------------------------------

class TestAggregateVerificationsConsensus:
    def _build_commitments(
        self,
        n: int,
        output_bytes: bytes,
        activation_bytes: bytes = b"\xAA" * 128,
    ) -> list[ComputeCommitment]:
        commitments = []
        for i in range(n):
            priv, pub = _generate_key_pair()
            c = InferenceVerifier.create_commitment(
                job_id="job-consensus",
                input_ids_bytes=b"same input",
                output_ids_bytes=output_bytes,
                activation_sample_bytes=activation_bytes,
                node_private_key_bytes=priv,
                node_id=f"node-{i}",
            )
            commitments.append(c)
        return commitments

    def test_unanimous_consensus(self):
        """All nodes agree → consensus_score=1.0, no outliers, is_honest=True."""
        commitments = self._build_commitments(5, output_bytes=b"correct output")
        result = InferenceVerifier.aggregate_verifications(commitments)

        assert result["consensus_score"] == 1.0
        assert result["outlier_nodes"] == []
        assert result["is_honest"] is True

    def test_majority_consensus_above_threshold(self):
        """4 out of 5 nodes agree → consensus_score=0.8 ≥ 2/3, is_honest=True."""
        good = self._build_commitments(4, output_bytes=b"correct output")
        bad = self._build_commitments(1, output_bytes=b"wrong output")

        result = InferenceVerifier.aggregate_verifications(good + bad)

        assert result["consensus_score"] == pytest.approx(0.8)
        assert result["is_honest"] is True
        assert len(result["outlier_nodes"]) == 1

    def test_empty_commitments_returns_not_honest(self):
        """No commitments → is_honest=False, consensus_score=0."""
        result = InferenceVerifier.aggregate_verifications([])

        assert result["is_honest"] is False
        assert result["consensus_score"] == 0.0
        assert result["outlier_nodes"] == []

    def test_single_honest_node(self):
        """A single commitment with no outliers counts as honest."""
        commitments = self._build_commitments(1, output_bytes=b"solo output")
        result = InferenceVerifier.aggregate_verifications(commitments)

        assert result["is_honest"] is True
        assert result["consensus_score"] == 1.0


# ---------------------------------------------------------------------------
# 10. aggregate_verifications flags outlier nodes
# ---------------------------------------------------------------------------

class TestAggregateVerificationsOutliers:
    def test_minority_nodes_flagged_as_outliers(self):
        """Nodes with a different output_hash appear in outlier_nodes."""
        honest_output = b"the real answer"
        fake_output = b"fabricated answer"

        honest_nodes: list[ComputeCommitment] = []
        for i in range(4):
            priv, _ = _generate_key_pair()
            c = InferenceVerifier.create_commitment(
                job_id="job-outlier",
                input_ids_bytes=b"input",
                output_ids_bytes=honest_output,
                activation_sample_bytes=b"\x00" * 64,
                node_private_key_bytes=priv,
                node_id=f"honest-{i}",
            )
            honest_nodes.append(c)

        dishonest_nodes: list[ComputeCommitment] = []
        for i in range(2):
            priv, _ = _generate_key_pair()
            c = InferenceVerifier.create_commitment(
                job_id="job-outlier",
                input_ids_bytes=b"input",
                output_ids_bytes=fake_output,
                activation_sample_bytes=b"\x00" * 64,
                node_private_key_bytes=priv,
                node_id=f"dishonest-{i}",
            )
            dishonest_nodes.append(c)

        result = InferenceVerifier.aggregate_verifications(honest_nodes + dishonest_nodes)

        assert set(result["outlier_nodes"]) == {"dishonest-0", "dishonest-1"}
        assert result["consensus_score"] == pytest.approx(4 / 6)
        assert result["is_honest"] is True  # 4/6 ≈ 0.667 ≥ 2/3

    def test_split_vote_below_threshold_not_honest(self):
        """
        When no output_hash commands 2/3 supermajority the result is not honest.
        3 nodes each with a different output → majority count = 1 out of 3.
        """
        outputs = [b"answer A", b"answer B", b"answer C"]
        commitments: list[ComputeCommitment] = []
        for i, out in enumerate(outputs):
            priv, _ = _generate_key_pair()
            c = InferenceVerifier.create_commitment(
                job_id="job-split",
                input_ids_bytes=b"same input",
                output_ids_bytes=out,
                activation_sample_bytes=b"\x55" * 64,
                node_private_key_bytes=priv,
                node_id=f"node-{i}",
            )
            commitments.append(c)

        result = InferenceVerifier.aggregate_verifications(commitments)

        # Best any single hash can do is 1/3
        assert result["consensus_score"] == pytest.approx(1 / 3)
        assert result["is_honest"] is False
        # Two of the three nodes are outliers relative to whichever wins
        assert len(result["outlier_nodes"]) == 2

    def test_outlier_node_ids_are_strings(self):
        """outlier_nodes must be a list of strings (node_id values)."""
        priv1, _ = _generate_key_pair()
        priv2, _ = _generate_key_pair()

        c1 = InferenceVerifier.create_commitment(
            "j", b"in", b"out-1", b"\x00" * 64, priv1, node_id="n1"
        )
        c2 = InferenceVerifier.create_commitment(
            "j", b"in", b"out-2", b"\x00" * 64, priv2, node_id="n2"
        )

        result = InferenceVerifier.aggregate_verifications([c1, c2])

        for nid in result["outlier_nodes"]:
            assert isinstance(nid, str)
