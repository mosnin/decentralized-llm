"""Tests for ShardManager layer slicing logic (no GPU required)."""

from unittest.mock import MagicMock

from node.config import NodeConfig


class TestShardSlicing:
    """Tests for the layer slice computation — pure math, no model loading."""

    def _manager(self, num_shards: int, shard_index: int):
        from node.shard_manager import ShardManager

        cfg = NodeConfig(
            model_name="meta-llama/Llama-3.2-1B",
            num_shards=num_shards,
            shard_index=shard_index,
        )
        return ShardManager(cfg)

    def test_even_split_4_shards_32_layers(self):
        for i in range(4):
            mgr = self._manager(4, i)
            start, end = mgr._compute_slice(32)
            assert end - start == 8

    def test_uneven_split_remainder_distributed(self):
        # 10 layers / 3 shards → [4, 3, 3]
        mgr0 = self._manager(3, 0)
        mgr1 = self._manager(3, 1)
        mgr2 = self._manager(3, 2)
        s0 = mgr0._compute_slice(10)
        s1 = mgr1._compute_slice(10)
        s2 = mgr2._compute_slice(10)
        assert s0 == (0, 4)
        assert s1 == (4, 7)
        assert s2 == (7, 10)

    def test_slices_cover_all_layers(self):
        num_layers = 28
        num_shards = 4
        slices = []
        for i in range(num_shards):
            mgr = self._manager(num_shards, i)
            slices.append(mgr._compute_slice(num_layers))
        # No gaps, no overlaps
        assert slices[0][0] == 0
        assert slices[-1][1] == num_layers
        for a, b in zip(slices, slices[1:]):
            assert a[1] == b[0]

    def test_single_shard_gets_all_layers(self):
        mgr = self._manager(1, 0)
        assert mgr._compute_slice(32) == (0, 32)

    def test_get_total_layers_llama_config(self):
        from node.shard_manager import ShardManager

        cfg_mock = MagicMock()
        cfg_mock.num_hidden_layers = 32
        assert ShardManager._get_total_layers(cfg_mock) == 32

    def test_get_total_layers_gpt_config(self):
        from node.shard_manager import ShardManager

        cfg_mock = MagicMock(spec=[])
        cfg_mock.n_layer = 24
        assert ShardManager._get_total_layers(cfg_mock) == 24
