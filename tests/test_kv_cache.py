"""Tests for the PagedKVCache (vLLM-inspired paged attention KV-cache)."""

import pytest

from node.kv_cache import PagedKVCache

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cache(
    num_layers: int = 4,
    num_heads: int = 2,
    head_dim: int = 8,
    block_size: int = 4,
    max_blocks: int = 16,
) -> PagedKVCache:
    return PagedKVCache(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_blocks=max_blocks,
        device="cpu",
    )


def _rand_kv(num_heads: int = 2, head_dim: int = 8) -> tuple:
    k = torch.randn(num_heads, head_dim)
    v = torch.randn(num_heads, head_dim)
    return k, v


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAllocateAndFreeBasic:
    """test_allocate_and_free_basic: allocate 32 tokens, verify blocks assigned, free."""

    def test_allocate_and_free_basic(self):
        cache = _make_cache(block_size=4, max_blocks=16)

        block_table = cache.allocate(seq_id=1, num_tokens=32)

        # 32 tokens / block_size 4 = 8 blocks
        assert len(block_table) == 8
        # All block indices should be unique
        assert len(set(block_table)) == len(block_table)
        # Utilization should reflect the 8 allocated blocks
        assert cache.utilization() == pytest.approx(8 / 16)

        cache.free(seq_id=1)

        assert cache.utilization() == pytest.approx(0.0)
        # Free twice is a no-op (should not raise)
        cache.free(seq_id=1)


class TestWriteAndReadRoundtrip:
    """test_write_and_read_roundtrip: write K/V to a layer, read back, assert equal."""

    def test_write_and_read_roundtrip(self):
        cache = _make_cache(num_layers=4, num_heads=2, head_dim=8, block_size=4)
        cache.allocate(seq_id=42, num_tokens=8)

        layer = 2
        written_keys = []
        written_vals = []

        for pos in range(8):
            k, v = _rand_kv()
            written_keys.append(k)
            written_vals.append(v)
            cache.write(seq_id=42, layer_idx=layer, token_pos=pos, key=k, value=v)

        keys_out, vals_out = cache.read(seq_id=42, layer_idx=layer)

        assert keys_out.shape == (8, 2, 8)
        assert vals_out.shape == (8, 2, 8)

        for pos in range(8):
            assert torch.allclose(keys_out[pos], written_keys[pos])
            assert torch.allclose(vals_out[pos], written_vals[pos])

    def test_read_returns_only_filled_tokens(self):
        """Reads should only return the tokens that have been written."""
        cache = _make_cache(block_size=4, max_blocks=8)
        cache.allocate(seq_id=7, num_tokens=8)

        k, v = _rand_kv()
        cache.write(seq_id=7, layer_idx=0, token_pos=0, key=k, value=v)

        keys_out, vals_out = cache.read(seq_id=7, layer_idx=0)
        # Only 1 token was written
        assert keys_out.shape[0] == 1
        assert torch.allclose(keys_out[0], k)


class TestOomRaisesMemoryError:
    """test_oom_raises_memory_error: fill all blocks, verify MemoryError on next allocate."""

    def test_oom_raises_memory_error(self):
        cache = _make_cache(block_size=4, max_blocks=4)

        # 4 blocks * 4 tokens = 16 tokens exactly fills the cache
        cache.allocate(seq_id=1, num_tokens=16)
        assert cache.utilization() == pytest.approx(1.0)

        with pytest.raises(MemoryError):
            cache.allocate(seq_id=2, num_tokens=1)

    def test_oom_after_partial_allocations(self):
        """OOM should trigger even if free space is insufficient but not zero."""
        cache = _make_cache(block_size=4, max_blocks=4)
        cache.allocate(seq_id=1, num_tokens=8)  # uses 2 blocks
        cache.allocate(seq_id=2, num_tokens=4)  # uses 1 block
        # 1 block left; requesting 2 blocks should fail
        with pytest.raises(MemoryError):
            cache.allocate(seq_id=3, num_tokens=8)


class TestUtilizationTracksAllocations:
    """test_utilization_tracks_allocations: utilization increases with allocation."""

    def test_utilization_tracks_allocations(self):
        cache = _make_cache(block_size=4, max_blocks=8)

        assert cache.utilization() == pytest.approx(0.0)

        cache.allocate(seq_id=10, num_tokens=4)  # 1 block
        assert cache.utilization() == pytest.approx(1 / 8)

        cache.allocate(seq_id=11, num_tokens=8)  # 2 blocks
        assert cache.utilization() == pytest.approx(3 / 8)

        cache.free(seq_id=10)
        assert cache.utilization() == pytest.approx(2 / 8)

        cache.free(seq_id=11)
        assert cache.utilization() == pytest.approx(0.0)

    def test_utilization_never_exceeds_one(self):
        cache = _make_cache(block_size=4, max_blocks=4)
        cache.allocate(seq_id=1, num_tokens=16)
        assert cache.utilization() <= 1.0


class TestCopyOnWritePrefixSharing:
    """test_copy_on_write_prefix_sharing: share prefix between two seqs."""

    def test_copy_on_write_prefix_sharing(self):
        cache = _make_cache(num_layers=2, num_heads=2, head_dim=4, block_size=4, max_blocks=8)
        # Allocate src and write 8 tokens
        cache.allocate(seq_id=100, num_tokens=8)
        layer = 0
        src_keys = []
        for pos in range(8):
            k, v = _rand_kv(num_heads=2, head_dim=4)
            src_keys.append(k)
            cache.write(seq_id=100, layer_idx=layer, token_pos=pos, key=k, value=v)

        # Share the first 8 tokens (2 blocks) with a new sequence
        shared_table = cache.share_prefix(src_seq_id=100, dst_seq_id=200, num_shared_tokens=8)

        # Both sequences should point at the same physical blocks
        assert shared_table == cache._block_tables[100][:2]

        # Reading from dst should return the same K tensors as src wrote
        keys_dst, _ = cache.read(seq_id=200, layer_idx=layer)
        assert keys_dst.shape[0] == 8
        for pos in range(8):
            assert torch.allclose(keys_dst[pos], src_keys[pos])

        # Now write to dst at position 0 — should trigger CoW and NOT modify src
        new_k, new_v = _rand_kv(num_heads=2, head_dim=4)
        cache.write(seq_id=200, layer_idx=layer, token_pos=0, key=new_k, value=new_v)

        keys_src_after, _ = cache.read(seq_id=100, layer_idx=layer)
        keys_dst_after, _ = cache.read(seq_id=200, layer_idx=layer)

        # src block 0 should be unchanged
        assert torch.allclose(keys_src_after[0], src_keys[0])
        # dst block 0 should reflect the new write
        assert torch.allclose(keys_dst_after[0], new_k)

    def test_share_prefix_increments_utilization(self):
        """Shared blocks are counted once; no extra blocks allocated for the prefix."""
        cache = _make_cache(block_size=4, max_blocks=8)
        cache.allocate(seq_id=1, num_tokens=8)  # 2 blocks
        util_before = cache.utilization()

        cache.share_prefix(src_seq_id=1, dst_seq_id=2, num_shared_tokens=8)
        # Shared blocks are already allocated; no new physical blocks consumed
        assert cache.utilization() == util_before

    def test_share_prefix_bad_src_raises(self):
        cache = _make_cache()
        with pytest.raises(KeyError):
            cache.share_prefix(src_seq_id=999, dst_seq_id=1, num_shared_tokens=4)
