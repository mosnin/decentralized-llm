"""
Verification pipeline — wires the ZK verifier and Merkle proof system into the
job settlement flow.

Components
----------
InferenceProof
    Dataclass bundling a ``ComputeCommitment`` with the Merkle root of the
    output token IDs and associated metadata.

ProofBuilder
    Node-side: builds an ``InferenceProof`` for a completed inference job.

ProofVerifier
    Client/auditor-side: verifies an ``InferenceProof`` and supports per-token
    inclusion checks and activation spot-checks.

SettlementVerifier
    Settlement layer: decides whether to proceed with on-chain settlement based
    on proof validity, and aggregates consensus across multiple peer proofs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from node.merkle import MerkleTree
from node.zk_verifier import ComputeCommitment, InferenceVerifier

# ---------------------------------------------------------------------------
# InferenceProof
# ---------------------------------------------------------------------------


@dataclass
class InferenceProof:
    """Proof of honest inference for one job."""

    job_id: int
    """Numeric job identifier (matches the on-chain job account)."""

    commitment: ComputeCommitment
    """ZK-lite cryptographic commitment created by the compute node."""

    output_merkle_root: str
    """Hex-encoded Merkle root over the output token IDs."""

    token_count: int
    """Number of output tokens (unpadded leaf count)."""

    node_id: str
    """Identifier of the node that produced this proof."""

    timestamp: float
    """Unix timestamp at proof creation."""

    proof_version: int = 1
    """Schema version — bump when the proof format changes."""


# ---------------------------------------------------------------------------
# ProofBuilder  (compute-node side)
# ---------------------------------------------------------------------------


class ProofBuilder:
    """Build ``InferenceProof`` objects for completed inference jobs."""

    def __init__(self, node_private_key_bytes: bytes) -> None:
        self._private_key_bytes = node_private_key_bytes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_proof(
        self,
        job_id: int,
        input_ids_bytes: bytes,
        output_token_ids: list[int],
        activation_sample_bytes: bytes,
        node_id: str,
    ) -> InferenceProof:
        """
        Build and return an ``InferenceProof`` for a completed inference.

        Parameters
        ----------
        job_id:
            Numeric on-chain job identifier.
        input_ids_bytes:
            Raw bytes of the prompt token id sequence (used for input_hash).
        output_token_ids:
            List of integer token IDs produced by the model.
        activation_sample_bytes:
            Sample of intermediate activations for the spot-check sketch.
        node_id:
            Human-readable identifier for the producing node.
        """
        # Serialise output token IDs as 4-byte big-endian words
        output_ids_bytes = b"".join(tid.to_bytes(4, "big") for tid in output_token_ids)

        # Build cryptographic commitment
        commitment = InferenceVerifier.create_commitment(
            job_id=str(job_id),
            input_ids_bytes=input_ids_bytes,
            output_ids_bytes=output_ids_bytes,
            activation_sample_bytes=activation_sample_bytes,
            node_private_key_bytes=self._private_key_bytes,
            node_id=node_id,
        )

        # Build Merkle tree over token IDs (each token = 4-byte big-endian leaf)
        leaves = [tid.to_bytes(4, "big") for tid in output_token_ids]
        tree = MerkleTree(leaves)
        merkle_root_hex = tree.root.hex()

        return InferenceProof(
            job_id=job_id,
            commitment=commitment,
            output_merkle_root=merkle_root_hex,
            token_count=len(output_token_ids),
            node_id=node_id,
            timestamp=time.time(),
        )

    def proof_to_dict(self, proof: InferenceProof) -> dict:
        """
        Serialise an ``InferenceProof`` to a JSON-serialisable dictionary.

        ``bytes`` fields are hex-encoded; ``commitment`` is inlined.
        """
        c = proof.commitment
        return {
            "job_id": proof.job_id,
            "proof_version": proof.proof_version,
            "node_id": proof.node_id,
            "timestamp": proof.timestamp,
            "token_count": proof.token_count,
            "output_merkle_root": proof.output_merkle_root,
            "commitment": {
                "job_id": c.job_id,
                "input_hash": c.input_hash.hex(),
                "output_hash": c.output_hash.hex(),
                "activation_sketch": c.activation_sketch.hex(),
                "timestamp": c.timestamp,
                "node_id": c.node_id,
                "signature_b64": c.signature_b64,
            },
        }


# ---------------------------------------------------------------------------
# ProofVerifier  (client / auditor side)
# ---------------------------------------------------------------------------


class ProofVerifier:
    """Verify ``InferenceProof`` objects produced by compute nodes."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rebuild_merkle_tree(proof: InferenceProof, output_token_ids: list[int]) -> MerkleTree:
        leaves = [tid.to_bytes(4, "big") for tid in output_token_ids]
        return MerkleTree(leaves)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify_proof(
        self,
        proof: InferenceProof,
        expected_input_hash: bytes,
        node_public_key_bytes: bytes,
        output_token_ids: list[int] | None = None,
    ) -> bool:
        """
        Verify an ``InferenceProof``.

        Checks performed
        ----------------
        1. ``InferenceVerifier.verify_commitment()`` — validates the signature
           and input hash binding in the ``ComputeCommitment``.
        2. If ``output_token_ids`` are supplied, rebuilds the Merkle tree and
           confirms that its root matches ``proof.output_merkle_root``.

        Returns True only if all applicable checks pass.
        """
        # Check 1: cryptographic commitment
        commitment_ok = InferenceVerifier.verify_commitment(
            commitment=proof.commitment,
            expected_input_hash=expected_input_hash,
            node_public_key_bytes=node_public_key_bytes,
        )
        if not commitment_ok:
            return False

        # Check 2: Merkle root consistency (when token IDs are available)
        if output_token_ids is not None:
            if len(output_token_ids) == 0:
                return False
            tree = self._rebuild_merkle_tree(proof, output_token_ids)
            if tree.root.hex() != proof.output_merkle_root:
                return False

        return True

    def verify_token_at_index(
        self,
        proof: InferenceProof,
        token_id: int,
        token_index: int,
        output_token_ids: list[int],
    ) -> bool:
        """
        Verify that ``token_id`` at position ``token_index`` is included in the
        Merkle tree whose root is stored in ``proof.output_merkle_root``.

        Parameters
        ----------
        proof:
            The ``InferenceProof`` containing the authoritative Merkle root.
        token_id:
            The integer token ID to verify.
        token_index:
            Zero-based position of the token in the output sequence.
        output_token_ids:
            Full list of output token IDs (needed to reconstruct the tree).
        """
        if not output_token_ids:
            return False
        try:
            tree = self._rebuild_merkle_tree(proof, output_token_ids)
            # Confirm the token at the requested index matches what was supplied
            if output_token_ids[token_index] != token_id:
                return False
            merkle_proof = tree.get_proof(token_index)
            # The proof must verify AND the root must match the stored root
            return merkle_proof.verify() and tree.root.hex() == proof.output_merkle_root
        except (IndexError, ValueError):
            return False

    def spot_check(
        self,
        proof: InferenceProof,
        fresh_activation_bytes: bytes,
    ) -> bool:
        """
        Spot-check activation honesty by re-sketching a fresh activation sample
        and comparing it with the sketch recorded in ``proof.commitment``.

        Delegates to ``InferenceVerifier.verify_spot_check()``.
        """
        return InferenceVerifier.verify_spot_check(
            commitment=proof.commitment,
            fresh_activation_bytes=fresh_activation_bytes,
        )


# ---------------------------------------------------------------------------
# SettlementVerifier  (settlement layer)
# ---------------------------------------------------------------------------


class SettlementVerifier:
    """
    Decide whether to proceed with on-chain settlement based on proof validity.

    ``should_settle()`` is the primary entry-point.  It applies the following
    policy:

    * If no proof is provided the job is settled unconditionally (the client
      opted out of verification).
    * If a proof is provided it must pass ``verify_proof()``.  When
      ``activation_sketch`` is available a spot-check is also run.
    """

    def __init__(self, min_proof_score: float = 0.9) -> None:
        self._min_proof_score = min_proof_score
        self._verifier = ProofVerifier()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_settle(
        self,
        job_id: int,
        proof: InferenceProof | None,
        node_pubkey_bytes: bytes,
        expected_input_hash: bytes,
        output_token_ids: list[int] | None = None,
        fresh_activation_bytes: bytes | None = None,
    ) -> tuple[bool, str]:
        """
        Determine whether job ``job_id`` should be settled.

        Parameters
        ----------
        job_id:
            Numeric job identifier (used for logging / sanity checks).
        proof:
            The ``InferenceProof`` submitted by the compute node, or ``None``
            if the client opted out of verification.
        node_pubkey_bytes:
            Ed25519 public key (or HMAC secret) of the compute node.
        expected_input_hash:
            SHA-256 of the prompt token IDs as submitted by the client.
        output_token_ids:
            Optional full output sequence for Merkle root verification.
        fresh_activation_bytes:
            Optional fresh activation sample for the spot-check.

        Returns
        -------
        ``(True, "verified")``   — proof passed; proceed with settlement.
        ``(True, "no_proof")``   — client opted out; settle unconditionally.
        ``(False, "<reason>")``  — proof invalid; withhold settlement.
        """
        # No proof supplied — client opted out
        if proof is None:
            return True, "no_proof"

        # Sanity: proof job_id must match
        if proof.job_id != job_id:
            return False, "job_id_mismatch"

        # Core commitment + Merkle check
        commitment_valid = self._verifier.verify_proof(
            proof=proof,
            expected_input_hash=expected_input_hash,
            node_public_key_bytes=node_pubkey_bytes,
            output_token_ids=output_token_ids,
        )
        if not commitment_valid:
            return False, "invalid_commitment"

        # Optional spot-check when fresh activations are available
        if fresh_activation_bytes is not None:
            spot_ok = self._verifier.spot_check(proof, fresh_activation_bytes)
            if not spot_ok:
                return False, "spot_check_failed"

        return True, "verified"

    def aggregate_peer_proofs(self, proofs: list[InferenceProof]) -> float:
        """
        Compute a consensus score across multiple ``InferenceProof`` objects
        for the *same* job.

        Strategy
        --------
        The consensus metric is the fraction of proofs whose
        ``output_merkle_root`` matches the majority value.  This mirrors the
        approach used in ``InferenceVerifier.aggregate_verifications()`` but
        operates on Merkle roots rather than raw output hashes.

        Returns
        -------
        float in [0, 1] — 1.0 means full consensus, 0.0 means no proofs.
        """
        if not proofs:
            return 0.0

        # Count votes per output_merkle_root
        vote_counts: dict[str, int] = {}
        for p in proofs:
            vote_counts[p.output_merkle_root] = vote_counts.get(p.output_merkle_root, 0) + 1

        majority_count = max(vote_counts.values())
        return majority_count / len(proofs)
