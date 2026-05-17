"""Prompt and result integrity checks for on-chain hash binding."""

import hashlib


class IntegrityError(Exception):
    """Raised when a downloaded blob's hash doesn't match the on-chain commitment."""


def verify_prompt_hash(blob: bytes, expected_hash: bytes | list[int]) -> None:
    """
    SHA-256 the blob and compare to expected_hash.
    Raises IntegrityError if they don't match.
    expected_hash can be bytes or list[int] (as returned by AnchorPy).
    """
    if isinstance(expected_hash, list):
        expected_hash = bytes(expected_hash)
    actual = hashlib.sha256(blob).digest()
    if actual != expected_hash:
        raise IntegrityError(
            f"Prompt hash mismatch: expected {expected_hash.hex()}, got {actual.hex()}"
        )


def verify_result_hash(result_bytes: bytes, expected_hash: bytes | list[int]) -> None:
    """Same but for result blobs."""
    if isinstance(expected_hash, list):
        expected_hash = bytes(expected_hash)
    actual = hashlib.sha256(result_bytes).digest()
    if actual != expected_hash:
        raise IntegrityError(
            f"Result hash mismatch: expected {expected_hash.hex()}, got {actual.hex()}"
        )


def compute_model_id(model_name: str) -> bytes:
    """Returns sha256(model_name.encode('utf-8')) — matches MODEL_IDS in client/python/client.py."""
    return hashlib.sha256(model_name.encode("utf-8")).digest()


def verify_model_id(model_name: str, on_chain_model_id: bytes | list[int]) -> bool:
    """Returns True if sha256(model_name) == on_chain_model_id."""
    if isinstance(on_chain_model_id, list):
        on_chain_model_id = bytes(on_chain_model_id)
    return compute_model_id(model_name) == on_chain_model_id
