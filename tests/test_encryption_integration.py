"""
Integration tests: full ECIES encrypt→decrypt round-trips plus edge cases.

These tests are self-contained — no network, no GPU, no Solana.
"""

import hashlib

import pytest

from node.encryption import (
    decrypt_prompt,
    ed25519_privkey_to_x25519,
    ed25519_pubkey_to_x25519,
    encrypt_prompt,
)


def _random_keypair():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    priv = Ed25519PrivateKey.generate()
    seed = priv.private_bytes_raw()
    pub = priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return seed, pub


class TestKeyConversion:
    def test_ed25519_to_x25519_pubkey_is_32_bytes(self):
        _, pub = _random_keypair()
        x_pub = ed25519_pubkey_to_x25519(pub)
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

        raw = x_pub.public_bytes(Encoding.Raw, PublicFormat.Raw)
        assert len(raw) == 32

    def test_ed25519_to_x25519_privkey_is_deterministic(self):
        from cryptography.hazmat.primitives.serialization import (
            Encoding,
            NoEncryption,
            PrivateFormat,
        )

        seed, _ = _random_keypair()
        k1 = ed25519_privkey_to_x25519(seed)
        k2 = ed25519_privkey_to_x25519(seed)
        raw = lambda k: k.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())  # noqa: E731
        assert raw(k1) == raw(k2)

    def test_pubkey_from_privkey_matches(self):
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

        seed, pub = _random_keypair()
        x_priv = ed25519_privkey_to_x25519(seed)
        x_pub_from_ed = ed25519_pubkey_to_x25519(pub)
        raw = lambda k: k.public_bytes(Encoding.Raw, PublicFormat.Raw)  # noqa: E731
        assert raw(x_priv.public_key()) == raw(x_pub_from_ed)


class TestEncryptDecryptEdgeCases:
    def test_empty_prompt_roundtrip(self):
        seed, pub = _random_keypair()
        blob = encrypt_prompt("", pub)
        recovered = decrypt_prompt(blob, seed, hashlib.sha256(b"").digest())
        assert recovered == ""

    def test_large_prompt_roundtrip(self):
        seed, pub = _random_keypair()
        prompt = "A" * 10_000
        blob = encrypt_prompt(prompt, pub)
        recovered = decrypt_prompt(blob, seed, hashlib.sha256(prompt.encode()).digest())
        assert recovered == prompt

    def test_blob_grows_linearly_with_prompt(self):
        _, pub = _random_keypair()
        overhead = 32 + 12 + 16  # eph_pub + nonce + GCM tag
        assert len(encrypt_prompt("x" * 100, pub)) == 100 + overhead
        assert len(encrypt_prompt("x" * 200, pub)) == 200 + overhead

    def test_multiple_encryptions_produce_different_blobs(self):
        _, pub = _random_keypair()
        prompt = "same prompt"
        # Ephemeral key is re-generated each call → ciphertexts must differ
        assert encrypt_prompt(prompt, pub) != encrypt_prompt(prompt, pub)

    def test_wrong_key_cannot_decrypt(self):
        seed1, pub1 = _random_keypair()
        seed2, _ = _random_keypair()
        blob = encrypt_prompt("secret", pub1)
        with pytest.raises(Exception):
            decrypt_prompt(blob, seed2, hashlib.sha256(b"secret").digest())

    def test_truncated_blob_raises(self):
        seed, pub = _random_keypair()
        blob = encrypt_prompt("hi", pub)
        with pytest.raises(Exception):
            decrypt_prompt(blob[:20], seed, hashlib.sha256(b"hi").digest())
