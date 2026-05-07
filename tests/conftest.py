"""Shared pytest fixtures."""

import pytest


@pytest.fixture
def ed25519_keypair():
    """Returns (private_seed_bytes, public_key_bytes) for testing."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    priv = Ed25519PrivateKey.generate()
    seed = priv.private_bytes_raw()
    pub = priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return seed, pub
