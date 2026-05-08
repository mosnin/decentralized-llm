"""
Result verification for submitted inference outputs.

Provides integrity and format checks before results are committed on-chain,
preventing the node from submitting malformed or incorrect data.
"""

import hashlib
import math


class ResultVerifier:
    """
    Verifies that submitted inference results are consistent with the job parameters.

    Checks:
    1. result_hash == SHA-256(result_bytes) — the claimed hash matches the actual result
    2. len(result_tokens) <= job.max_tokens — result doesn't exceed token budget
    3. result_cid is a valid IPFS CID format (starts with "bafy", "bafk", or "Qm")
    4. Result text is valid UTF-8
    5. Result is not empty
    """

    @staticmethod
    def verify_result(
        result_bytes: bytes,
        claimed_hash: bytes,
        result_cid: str,
        max_tokens: int,
    ) -> tuple[bool, str]:
        """
        Returns (is_valid, reason). reason is empty string on success.
        """
        # Check 5: result must not be empty
        if not result_bytes:
            return False, "result is empty"

        # Check 4: result must be valid UTF-8
        try:
            result_text = result_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            return False, f"result is not valid UTF-8: {exc}"

        # Check 1: hash must match
        actual_hash = hashlib.sha256(result_bytes).digest()
        if actual_hash != claimed_hash:
            return False, "result_hash does not match SHA-256(result_bytes)"

        # Check 2: token count must be within budget
        estimated = ResultVerifier.estimate_tokens(result_text)
        if estimated > max_tokens:
            return False, (f"estimated token count {estimated} exceeds max_tokens {max_tokens}")

        # Check 3: CID format must be valid
        if not ResultVerifier.verify_cid_format(result_cid):
            return False, f"result_cid '{result_cid}' is not a valid IPFS CID format"

        return True, ""

    @staticmethod
    def verify_cid_format(cid: str) -> bool:
        """Return True if cid looks like a valid IPFS CID."""
        if not cid:
            return False
        return cid.startswith("bafy") or cid.startswith("bafk") or cid.startswith("Qm")

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token count estimate: len(text.split()) * 1.3"""
        return math.ceil(len(text.split()) * 1.3)
