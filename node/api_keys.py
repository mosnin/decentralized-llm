"""API key validation (stdlib only)."""

import hashlib
import secrets
from dataclasses import dataclass


@dataclass
class ApiKey:
    key_id: str
    key_hash: str  # SHA-256 hex of the raw key
    owner: str
    tier: str = "free"  # "free" | "pro" | "enterprise"


class ApiKeyStore:
    """In-memory API key registry (production would use a DB)."""

    def __init__(self):
        self._keys: dict[str, ApiKey] = {}  # key_id → ApiKey

    @staticmethod
    def generate() -> tuple[str, str]:
        """Generate (key_id, raw_key). raw_key is shown once."""
        key_id = secrets.token_hex(8)
        raw_key = secrets.token_hex(32)
        return key_id, raw_key

    def register(self, owner: str, tier: str = "free") -> tuple[str, str]:
        """Register a new key, return (key_id, raw_key)."""
        key_id, raw_key = self.generate()
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        self._keys[key_id] = ApiKey(key_id=key_id, key_hash=key_hash, owner=owner, tier=tier)
        return key_id, raw_key

    def validate(self, key_id: str, raw_key: str) -> "ApiKey | None":
        """Return ApiKey if valid, None otherwise."""
        entry = self._keys.get(key_id)
        if entry is None:
            return None
        expected = hashlib.sha256(raw_key.encode()).hexdigest()
        if not secrets.compare_digest(entry.key_hash, expected):
            return None
        return entry

    def revoke(self, key_id: str) -> bool:
        """Remove a key. Returns True if it existed."""
        return self._keys.pop(key_id, None) is not None

    def __len__(self) -> int:
        return len(self._keys)
