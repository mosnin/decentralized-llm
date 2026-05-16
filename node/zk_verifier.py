"""
Zero-knowledge-lite inference verification.

Lets clients verify that compute nodes produced honest outputs without
re-running the full inference.  Uses three complementary mechanisms:

1. Cryptographic commitments — every output is signed by the node's Ed25519
   key, binding the node identity to a specific (input_hash, output_hash) pair.
2. Activation sketches — a seeded random-projection fingerprint of a sample of
   intermediate activations.  Two honest runs of the same model on the same
   input produce sketches that are nearly identical; a lying node that swaps
   activations will diverge.
3. Consensus aggregation — collect commitments from multiple nodes and flag
   statistical outliers.

Signing backend
---------------
Uses ``cryptography`` Ed25519 when available (strongly recommended in
production).  Falls back to HMAC-SHA256 keyed on the private-key bytes when
the library is absent, which provides authenticity but not non-repudiation.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import struct
import time
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Optional cryptography import — lazy so the module loads without it installed
# ---------------------------------------------------------------------------

def _try_import_ed25519() -> Any:
    """Return (Ed25519PrivateKey, Ed25519PublicKey) or None on ImportError."""
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
            Ed25519PublicKey,
        )
        return Ed25519PrivateKey, Ed25519PublicKey
    except ImportError:  # pragma: no cover
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _dot_product(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: list[float]) -> float:
    return sum(x * x for x in v) ** 0.5


def _cosine(a: list[float], b: list[float]) -> float:
    na, nb = _norm(a), _norm(b)
    if na == 0.0 or nb == 0.0:
        return 1.0 if na == nb else 0.0
    return _dot_product(a, b) / (na * nb)


# ---------------------------------------------------------------------------
# ComputeCommitment
# ---------------------------------------------------------------------------

@dataclass
class ComputeCommitment:
    """Cryptographic commitment to one inference computation."""

    job_id: str
    """Unique identifier for the inference job."""

    input_hash: bytes
    """SHA-256 of the serialised input token ids (prompt)."""

    output_hash: bytes
    """SHA-256 of the serialised output token ids (generated sequence)."""

    activation_sketch: bytes
    """64-dimensional random-projection fingerprint of intermediate activations."""

    timestamp: float
    """Unix timestamp (seconds) at commitment creation."""

    node_id: str
    """Identifier of the compute node that produced this commitment."""

    signature_b64: str
    """Base-64-encoded signature over the commitment payload."""


# ---------------------------------------------------------------------------
# ActivationSketch
# ---------------------------------------------------------------------------

class ActivationSketch:
    """
    Seeded random-projection sketch of neural-network activation tensors.

    A random projection matrix R (sketch_dim × input_dim) is generated from
    the given seed.  For any activation byte-blob the sketch is:

        sketch = sign(R @ x)  where x = float32 interpretation of the bytes

    ``sign`` collapses each projection to ±1 so that the sketch is a
    locality-sensitive hash: two identical inputs produce identical sketches,
    and similar inputs produce high cosine similarity sketches.
    """

    def __init__(self, seed: int, sketch_dim: int = 64) -> None:
        self.seed = seed
        self.sketch_dim = sketch_dim
        # Seeded state used to generate projection rows on demand
        self._rng_state = seed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lcg_floats(self, n: int, offset: int = 0) -> list[float]:
        """
        Generate *n* pseudo-random floats in (−1, 1) using a linear
        congruential generator seeded from ``self.seed + offset``.
        Cheap and pure-Python so there is no numpy dependency.
        """
        # LCG parameters (same as glibc)
        a, c, m = 1103515245, 12345, 2**31
        state = (self.seed + offset) & 0xFFFFFFFF
        results: list[float] = []
        for _ in range(n):
            state = (a * state + c) % m
            # Map [0, m) → (−1, 1)
            results.append((state / (m / 2.0)) - 1.0)
        return results

    def _bytes_to_floats(self, data: bytes) -> list[float]:
        """Interpret *data* as little-endian float32 values, padding if needed."""
        # Pad to multiple of 4 bytes
        remainder = len(data) % 4
        if remainder:
            data = data + b"\x00" * (4 - remainder)
        n = len(data) // 4
        if n == 0:
            return [0.0]
        return list(struct.unpack_from(f"<{n}f", data))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sketch(self, tensor_bytes: bytes) -> bytes:
        """
        Project *tensor_bytes* through the seeded random matrix and return a
        ``sketch_dim``-float fingerprint encoded as ``sketch_dim * 4`` bytes
        (little-endian float32).
        """
        x = self._bytes_to_floats(tensor_bytes)
        input_dim = len(x)

        projection: list[float] = []
        for row_idx in range(self.sketch_dim):
            # Each row of R: input_dim weights seeded by (seed, row_idx)
            row = self._lcg_floats(input_dim, offset=row_idx * 999983)
            val = sum(r * xi for r, xi in zip(row, x))
            projection.append(val)

        return struct.pack(f"<{self.sketch_dim}f", *projection)

    @staticmethod
    def verify_consistency(sketch1: bytes, sketch2: bytes, threshold: float = 0.95) -> bool:
        """
        Return True if two sketches are consistent (cosine similarity ≥ threshold).

        Two sketches of the *same* activation tensor should be byte-identical
        (similarity = 1.0).  Threshold < 1.0 provides tolerance for tiny
        floating-point differences that may arise from different serialisation
        paths.
        """
        n = len(sketch1) // 4
        if n == 0 or len(sketch2) // 4 != n:
            return False
        v1 = list(struct.unpack_from(f"<{n}f", sketch1))
        v2 = list(struct.unpack_from(f"<{n}f", sketch2))
        return _cosine(v1, v2) >= threshold


# ---------------------------------------------------------------------------
# InferenceVerifier
# ---------------------------------------------------------------------------

class InferenceVerifier:
    """
    Create and verify zero-knowledge-lite inference commitments.

    All methods are stateless class methods so the verifier can be used
    as a singleton or instantiated per-request.
    """

    # Default sketch seed — in production derive this from a shared VRF output
    _DEFAULT_SKETCH_SEED: int = 0xDEADBEEF
    _SKETCH_DIM: int = 64

    # ------------------------------------------------------------------
    # Signing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sign(payload: bytes, private_key_bytes: bytes) -> bytes:
        """Sign *payload* with Ed25519 or fall back to HMAC-SHA256."""
        ed = _try_import_ed25519()
        if ed is not None:
            Ed25519PrivateKey, _ = ed
            priv = Ed25519PrivateKey.from_private_bytes(private_key_bytes)
            return priv.sign(payload)
        # Fallback: HMAC-SHA256
        return hmac.new(private_key_bytes, payload, hashlib.sha256).digest()

    @staticmethod
    def _verify_sig(
        payload: bytes,
        signature: bytes,
        public_key_bytes: bytes,
    ) -> bool:
        """
        Verify *signature* over *payload* using Ed25519 or HMAC-SHA256.

        For the HMAC fallback the caller must pass the *private* key bytes as
        ``public_key_bytes`` (symmetric scheme — the verifier must be trusted).
        """
        ed = _try_import_ed25519()
        if ed is not None:
            _, Ed25519PublicKey = ed
            try:
                pub = Ed25519PublicKey.from_public_bytes(public_key_bytes)
                pub.verify(signature, payload)
                return True
            except Exception:
                return False
        # Fallback: constant-time HMAC comparison
        expected = hmac.new(public_key_bytes, payload, hashlib.sha256).digest()
        return hmac.compare_digest(expected, signature)

    # ------------------------------------------------------------------
    # Payload construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_payload(
        job_id: str,
        input_hash: bytes,
        output_hash: bytes,
        activation_sketch: bytes,
        timestamp: float,
        node_id: str,
    ) -> bytes:
        """Canonical byte representation that is signed / verified."""
        parts = [
            job_id.encode(),
            b"\x00",
            input_hash,
            output_hash,
            activation_sketch,
            struct.pack("<d", timestamp),
            node_id.encode(),
        ]
        return b"".join(parts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def create_commitment(
        cls,
        job_id: str,
        input_ids_bytes: bytes,
        output_ids_bytes: bytes,
        activation_sample_bytes: bytes,
        node_private_key_bytes: bytes,
        node_id: str = "",
        sketch_seed: int | None = None,
        timestamp: float | None = None,
    ) -> ComputeCommitment:
        """
        Build a signed :class:`ComputeCommitment` for one inference job.

        Parameters
        ----------
        job_id:
            Unique job identifier (e.g. UUID string).
        input_ids_bytes:
            Raw bytes of the prompt token id sequence.
        output_ids_bytes:
            Raw bytes of the generated token id sequence.
        activation_sample_bytes:
            A sample of intermediate layer activations (arbitrary byte blob).
        node_private_key_bytes:
            32-byte Ed25519 seed (or any 32-byte secret for HMAC fallback).
        node_id:
            Human-readable node identifier.  Derived from the public key when
            empty.
        sketch_seed:
            RNG seed for the projection matrix.  Defaults to
            ``_DEFAULT_SKETCH_SEED``.
        timestamp:
            Unix timestamp.  Defaults to ``time.time()``.
        """
        if sketch_seed is None:
            sketch_seed = cls._DEFAULT_SKETCH_SEED
        if timestamp is None:
            timestamp = time.time()

        input_hash = _sha256(input_ids_bytes)
        output_hash = _sha256(output_ids_bytes)

        sketcher = ActivationSketch(seed=sketch_seed, sketch_dim=cls._SKETCH_DIM)
        activation_sketch = sketcher.sketch(activation_sample_bytes)

        # Derive a node_id from the public key when not supplied
        if not node_id:
            ed = _try_import_ed25519()
            if ed is not None:
                Ed25519PrivateKey, _ = ed
                priv = Ed25519PrivateKey.from_private_bytes(node_private_key_bytes)
                pub_bytes = priv.public_key().public_bytes_raw()
                node_id = pub_bytes.hex()[:16]
            else:
                node_id = _sha256(node_private_key_bytes).hex()[:16]

        payload = cls._build_payload(
            job_id, input_hash, output_hash, activation_sketch, timestamp, node_id
        )
        signature = cls._sign(payload, node_private_key_bytes)
        signature_b64 = base64.b64encode(signature).decode()

        return ComputeCommitment(
            job_id=job_id,
            input_hash=input_hash,
            output_hash=output_hash,
            activation_sketch=activation_sketch,
            timestamp=timestamp,
            node_id=node_id,
            signature_b64=signature_b64,
        )

    @classmethod
    def verify_commitment(
        cls,
        commitment: ComputeCommitment,
        expected_input_hash: bytes,
        node_public_key_bytes: bytes,
    ) -> bool:
        """
        Verify a :class:`ComputeCommitment`.

        Checks:

        1. ``commitment.input_hash == expected_input_hash``
        2. The Ed25519 (or HMAC) signature is valid for the commitment payload.

        Returns ``True`` iff both checks pass.
        """
        # Check 1: input hash must match what the client sent
        if commitment.input_hash != expected_input_hash:
            return False

        # Check 2: signature must be valid
        payload = cls._build_payload(
            commitment.job_id,
            commitment.input_hash,
            commitment.output_hash,
            commitment.activation_sketch,
            commitment.timestamp,
            commitment.node_id,
        )
        try:
            signature = base64.b64decode(commitment.signature_b64)
        except Exception:
            return False

        return cls._verify_sig(payload, signature, node_public_key_bytes)

    @classmethod
    def verify_spot_check(
        cls,
        commitment: ComputeCommitment,
        fresh_activation_bytes: bytes,
        sketch_seed: int | None = None,
        threshold: float = 0.95,
    ) -> bool:
        """
        Spot-check honesty by re-sketching a fresh activation sample.

        The verifier (or a third-party auditor) obtains a fresh activation
        sample from the node and checks whether its sketch is consistent with
        the one recorded in the commitment.  A dishonest node that fabricated
        the original sketch will fail this check.

        Parameters
        ----------
        commitment:
            The commitment to check.
        fresh_activation_bytes:
            A freshly retrieved activation sample from the same layer / token
            that was used during commitment creation.
        sketch_seed:
            Must match the seed used at commitment time.
        threshold:
            Cosine-similarity threshold (default 0.95).
        """
        if sketch_seed is None:
            sketch_seed = cls._DEFAULT_SKETCH_SEED

        sketcher = ActivationSketch(seed=sketch_seed, sketch_dim=cls._SKETCH_DIM)
        fresh_sketch = sketcher.sketch(fresh_activation_bytes)
        return ActivationSketch.verify_consistency(
            commitment.activation_sketch, fresh_sketch, threshold=threshold
        )

    @classmethod
    def aggregate_verifications(
        cls,
        commitments: list[ComputeCommitment],
    ) -> dict[str, Any]:
        """
        Aggregate multiple commitments for the *same* job and assess honesty.

        Strategy
        --------
        1. Compute the majority ``output_hash`` (the value held by the most
           nodes).  Nodes that deviate are flagged as outliers.
        2. ``consensus_score`` = fraction of nodes that agree with the majority.
        3. ``is_honest`` = True iff consensus_score ≥ 2/3 (Byzantine threshold)
           and at least one commitment is present.

        Returns
        -------
        dict with keys:
            ``consensus_score`` (float 0–1),
            ``outlier_nodes`` (list[str] of node_id values),
            ``is_honest`` (bool).
        """
        if not commitments:
            return {"consensus_score": 0.0, "outlier_nodes": [], "is_honest": False}

        # Count votes per output_hash
        vote_counts: dict[bytes, int] = {}
        for c in commitments:
            vote_counts[c.output_hash] = vote_counts.get(c.output_hash, 0) + 1

        majority_hash = max(vote_counts, key=lambda h: vote_counts[h])
        majority_count = vote_counts[majority_hash]
        total = len(commitments)
        consensus_score = majority_count / total

        outlier_nodes = [
            c.node_id for c in commitments if c.output_hash != majority_hash
        ]

        is_honest = consensus_score >= (2 / 3) and total > 0

        return {
            "consensus_score": consensus_score,
            "outlier_nodes": outlier_nodes,
            "is_honest": is_honest,
        }
