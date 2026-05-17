"""Tests for node.integrity — prompt/result hash verification and model ID helpers."""

import hashlib

import pytest

from node.integrity import (
    IntegrityError,
    compute_model_id,
    verify_model_id,
    verify_prompt_hash,
    verify_result_hash,
)

# ──────────────────────────── verify_prompt_hash ─────────────────────────────


def test_verify_prompt_hash_correct_bytes():
    blob = b"Hello, decentralized world!"
    expected = hashlib.sha256(blob).digest()
    # Should not raise
    verify_prompt_hash(blob, expected)


def test_verify_prompt_hash_correct_list_of_ints():
    blob = b"Encrypted prompt payload"
    expected_bytes = hashlib.sha256(blob).digest()
    expected_list = list(expected_bytes)
    # AnchorPy returns list[int]; should still pass
    verify_prompt_hash(blob, expected_list)


def test_verify_prompt_hash_tampered_raises():
    blob = b"Original blob content"
    tampered_hash = hashlib.sha256(b"Different content").digest()
    with pytest.raises(IntegrityError, match="Prompt hash mismatch"):
        verify_prompt_hash(blob, tampered_hash)


# ──────────────────────────── verify_result_hash ─────────────────────────────


def test_verify_result_hash_correct():
    result_bytes = b"The answer is 42."
    expected = hashlib.sha256(result_bytes).digest()
    # Should not raise
    verify_result_hash(result_bytes, expected)


def test_verify_result_hash_wrong_raises():
    result_bytes = b"Legitimate result"
    wrong_hash = hashlib.sha256(b"Tampered result").digest()
    with pytest.raises(IntegrityError, match="Result hash mismatch"):
        verify_result_hash(result_bytes, wrong_hash)


# ──────────────────────────── compute_model_id ───────────────────────────────


def test_compute_model_id_matches_client_constant():
    """sha256('meta-llama/Llama-3.2-3B') must equal MODEL_IDS['llama-3.2-3b']."""
    from client.python.client import MODEL_IDS

    expected = MODEL_IDS["llama-3.2-3b"]
    result = compute_model_id("meta-llama/Llama-3.2-3B")
    assert result == expected


# ──────────────────────────── verify_model_id ────────────────────────────────


def test_verify_model_id_true():
    model_name = "meta-llama/Llama-3.2-3B"
    on_chain_id = compute_model_id(model_name)
    assert verify_model_id(model_name, on_chain_id) is True


def test_verify_model_id_false():
    model_name = "meta-llama/Llama-3.2-3B"
    wrong_id = compute_model_id("mistralai/Mistral-7B-v0.3")
    assert verify_model_id(model_name, wrong_id) is False


def test_verify_model_id_list_of_ints():
    """verify_model_id should also accept list[int] as returned by AnchorPy."""
    model_name = "meta-llama/Llama-3.1-8B"
    on_chain_id_list = list(compute_model_id(model_name))
    assert verify_model_id(model_name, on_chain_id_list) is True
