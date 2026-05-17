"""Tests for ECIES prompt encryption/decryption."""

import hashlib

import pytest

from node.encryption import decrypt_prompt, encrypt_prompt


def _make_ed25519_keypair():
    """Generate a random Ed25519 seed and derive a simple pubkey for testing."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    priv = Ed25519PrivateKey.generate()
    priv_seed = priv.private_bytes_raw()
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    pub_bytes = priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return priv_seed, pub_bytes


class TestEncryption:
    def test_roundtrip(self):
        priv_seed, pub_bytes = _make_ed25519_keypair()
        prompt = "What is the capital of France?"

        blob = encrypt_prompt(prompt, pub_bytes)
        recovered = decrypt_prompt(blob, priv_seed, hashlib.sha256(prompt.encode()).digest())
        assert recovered == prompt

    def test_wrong_hash_raises(self):
        priv_seed, pub_bytes = _make_ed25519_keypair()
        prompt = "Hello world"

        blob = encrypt_prompt(prompt, pub_bytes)
        wrong_hash = hashlib.sha256(b"different prompt").digest()

        with pytest.raises(ValueError, match="hash does not match"):
            decrypt_prompt(blob, priv_seed, wrong_hash)

    def test_tampered_ciphertext_raises(self):
        priv_seed, pub_bytes = _make_ed25519_keypair()
        prompt = "Test prompt"

        blob = bytearray(encrypt_prompt(prompt, pub_bytes))
        blob[-1] ^= 0xFF  # flip last byte

        with pytest.raises(Exception):
            decrypt_prompt(
                bytes(blob),
                priv_seed,
                hashlib.sha256(prompt.encode()).digest(),
            )

    def test_different_nodes_cannot_decrypt(self):
        _, pub1 = _make_ed25519_keypair()
        priv2, _ = _make_ed25519_keypair()  # different node
        prompt = "Secret prompt"

        blob = encrypt_prompt(prompt, pub1)
        with pytest.raises(Exception):
            decrypt_prompt(blob, priv2, hashlib.sha256(prompt.encode()).digest())

    def test_unicode_prompt(self):
        priv_seed, pub_bytes = _make_ed25519_keypair()
        prompt = "Explain 量子纠缠 (quantum entanglement) in simple terms 🔬"

        blob = encrypt_prompt(prompt, pub_bytes)
        recovered = decrypt_prompt(blob, priv_seed, hashlib.sha256(prompt.encode("utf-8")).digest())
        assert recovered == prompt

    def test_blob_structure(self):
        _, pub_bytes = _make_ed25519_keypair()
        prompt = "Short prompt"

        blob = encrypt_prompt(prompt, pub_bytes)
        # 32 (eph pubkey) + 12 (nonce) + len(ciphertext) + 16 (GCM tag)
        assert len(blob) == 32 + 12 + len(prompt.encode()) + 16
