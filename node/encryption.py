"""
End-to-end encrypted prompt delivery using ECIES with Solana keypairs.

The node's Ed25519 Solana wallet key is converted to X25519 for ECDH.
Since the node's pubkey is already registered on-chain in the compute-registry,
no separate key infrastructure is required — the chain IS the key directory.

Encrypt flow (client side):
  1. Convert node's on-chain Ed25519 pubkey → X25519 pubkey
  2. Generate ephemeral X25519 keypair
  3. ECDH(ephemeral_priv, node_pub) → shared secret
  4. HKDF(shared) → 32-byte AES key
  5. AES-256-GCM encrypt prompt
  6. Deliver blob (eph_pub || nonce || ciphertext) to node via DHT or P2P

Decrypt flow (node side):
  1. Convert wallet Ed25519 privkey → X25519 privkey
  2. ECDH(node_priv, eph_pub) → same shared secret
  3. AES-256-GCM decrypt → plaintext prompt
  4. Verify SHA-256 matches on-chain prompt_hash
"""

import hashlib
import os

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

_HKDF_INFO = b"decentralized-llm-prompt-v1"


def ed25519_pubkey_to_x25519(ed25519_pubkey_bytes: bytes) -> X25519PublicKey:
    """
    Convert a 32-byte Ed25519 public key to X25519.

    This uses the birational map between Ed25519 and Curve25519:
      u = (1 + y) / (1 - y)  where y is the Ed25519 y-coordinate.

    The sign bit of the Ed25519 key encodes the x-sign; X25519 only
    needs the u-coordinate (Montgomery form), so we strip it.
    """
    # The standard conversion: interpret as Edwards y, convert to Montgomery u
    # Reference: https://www.rfc-editor.org/rfc/rfc7748#section-4.1
    p = 2**255 - 19
    y_bytes = bytearray(ed25519_pubkey_bytes)
    y_bytes[31] &= 0x7F  # clear sign bit
    y = int.from_bytes(bytes(y_bytes), "little")
    u = (1 + y) * pow(1 - y, p - 2, p) % p
    u_bytes = u.to_bytes(32, "little")
    return X25519PublicKey.from_public_bytes(u_bytes)


def ed25519_privkey_to_x25519(ed25519_privkey_bytes: bytes) -> X25519PrivateKey:
    """
    Convert a 32-byte Ed25519 scalar (seed) to an X25519 private key.

    Ed25519 private key = SHA-512(seed); the first 32 bytes (clamped)
    form the scalar used in Curve25519 operations.
    """
    h = hashlib.sha512(ed25519_privkey_bytes).digest()
    scalar = bytearray(h[:32])
    scalar[0] &= 248
    scalar[31] &= 127
    scalar[31] |= 64
    return X25519PrivateKey.from_private_bytes(bytes(scalar))


def encrypt_prompt(prompt: str, node_ed25519_pubkey_bytes: bytes) -> bytes:
    """
    Encrypt a prompt for a specific compute node.

    Returns a blob: eph_pub(32) || nonce(12) || ciphertext(variable)
    """
    node_x25519_pub = ed25519_pubkey_to_x25519(node_ed25519_pubkey_bytes)

    eph_priv = X25519PrivateKey.generate()
    shared_secret = eph_priv.exchange(node_x25519_pub)

    aes_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=_HKDF_INFO,
    ).derive(shared_secret)

    nonce = os.urandom(12)
    ciphertext = AESGCM(aes_key).encrypt(nonce, prompt.encode("utf-8"), None)

    eph_pub_bytes = eph_priv.public_key().public_bytes(
        encoding=Encoding.Raw, format=PublicFormat.Raw
    )
    return eph_pub_bytes + nonce + ciphertext


def decrypt_prompt(
    blob: bytes,
    node_ed25519_privkey_seed: bytes,
    expected_hash: bytes,
) -> str:
    """
    Decrypt an encrypted prompt blob using this node's wallet private key.

    Raises ValueError if the decrypted prompt doesn't match the on-chain hash.
    """
    node_x25519_priv = ed25519_privkey_to_x25519(node_ed25519_privkey_seed)

    eph_pub = X25519PublicKey.from_public_bytes(blob[:32])
    nonce = blob[32:44]
    ciphertext = blob[44:]

    shared_secret = node_x25519_priv.exchange(eph_pub)
    aes_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=_HKDF_INFO,
    ).derive(shared_secret)

    plaintext_bytes = AESGCM(aes_key).decrypt(nonce, ciphertext, None)
    prompt = plaintext_bytes.decode("utf-8")

    actual_hash = hashlib.sha256(prompt.encode("utf-8")).digest()
    if actual_hash != expected_hash:
        raise ValueError(
            "Decrypted prompt hash does not match on-chain commitment — "
            "possible prompt substitution attack"
        )

    return prompt
