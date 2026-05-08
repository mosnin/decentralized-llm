"""Tests for the API key store."""

import pytest

from node.api_keys import ApiKey, ApiKeyStore


@pytest.fixture
def store():
    return ApiKeyStore()


class TestApiKeyStore:
    def test_generate_returns_unique_keys(self):
        """Two generate() calls should produce different key_ids."""
        id1, raw1 = ApiKeyStore.generate()
        id2, raw2 = ApiKeyStore.generate()
        assert id1 != id2
        assert raw1 != raw2

    def test_register_and_validate_success(self, store):
        """register() then validate() with correct raw_key should return ApiKey."""
        key_id, raw_key = store.register(owner="alice")
        result = store.validate(key_id, raw_key)
        assert result is not None
        assert isinstance(result, ApiKey)
        assert result.key_id == key_id
        assert result.owner == "alice"

    def test_validate_wrong_key_returns_none(self, store):
        """validate() with wrong raw_key should return None."""
        key_id, _ = store.register(owner="alice")
        result = store.validate(key_id, "wrong_raw_key")
        assert result is None

    def test_validate_unknown_key_id(self, store):
        """validate() with unknown key_id should return None."""
        result = store.validate("nonexistent_id", "any_key")
        assert result is None

    def test_revoke_removes_key(self, store):
        """After revoke(), validate() should return None."""
        key_id, raw_key = store.register(owner="bob")
        assert store.validate(key_id, raw_key) is not None
        removed = store.revoke(key_id)
        assert removed is True
        assert store.validate(key_id, raw_key) is None

    def test_revoke_missing_key_returns_false(self, store):
        """revoke() on a non-existent key should return False."""
        result = store.revoke("does_not_exist")
        assert result is False

    def test_len_tracks_registered_count(self, store):
        """len() should increase with register and decrease with revoke."""
        assert len(store) == 0
        key_id1, _ = store.register(owner="alice")
        assert len(store) == 1
        key_id2, _ = store.register(owner="bob")
        assert len(store) == 2
        store.revoke(key_id1)
        assert len(store) == 1
        store.revoke(key_id2)
        assert len(store) == 0

    def test_tier_stored_correctly(self, store):
        """register() with tier='pro' should persist and be returned by validate()."""
        key_id, raw_key = store.register(owner="enterprise_user", tier="pro")
        result = store.validate(key_id, raw_key)
        assert result is not None
        assert result.tier == "pro"

    def test_default_tier_is_free(self, store):
        """register() without explicit tier should default to 'free'."""
        key_id, raw_key = store.register(owner="free_user")
        result = store.validate(key_id, raw_key)
        assert result is not None
        assert result.tier == "free"

    def test_hash_not_stored_as_raw_key(self, store):
        """The stored key_hash should not equal the raw key."""
        key_id, raw_key = store.register(owner="alice")
        entry = store._keys[key_id]
        assert entry.key_hash != raw_key
        assert len(entry.key_hash) == 64  # SHA-256 hex digest length
