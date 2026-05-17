"""Tests for NodeConfig defaults and environment variable parsing."""


class TestNodeConfig:
    def test_defaults_are_reasonable(self):
        from node.config import NodeConfig

        cfg = NodeConfig()
        assert cfg.num_shards >= 1
        assert cfg.shard_index >= 0
        assert cfg.max_concurrent_jobs >= 1
        assert cfg.job_poll_interval_seconds > 0
        assert cfg.lora_rank > 0

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("NUM_SHARDS", "8")
        monkeypatch.setenv("SHARD_INDEX", "3")
        monkeypatch.setenv("MAX_CONCURRENT_JOBS", "2")

        # Re-import to pick up new env values
        import importlib

        import node.config as m

        importlib.reload(m)

        cfg = m.NodeConfig()
        assert cfg.num_shards == 8
        assert cfg.shard_index == 3
        assert cfg.max_concurrent_jobs == 2

    def test_lighthouse_api_key_field_exists(self):
        from node.config import NodeConfig

        cfg = NodeConfig()
        assert hasattr(cfg, "lighthouse_api_key")

    def test_dht_bootstrap_peers_empty_by_default(self, monkeypatch):
        monkeypatch.delenv("DHT_BOOTSTRAP_PEERS", raising=False)
        from node.config import NodeConfig

        cfg = NodeConfig()
        assert cfg.dht_bootstrap_peers == []

    def test_dht_bootstrap_peers_parsed_from_env(self, monkeypatch):
        monkeypatch.setenv(
            "DHT_BOOTSTRAP_PEERS",
            "/ip4/1.2.3.4/tcp/7070,/ip4/5.6.7.8/tcp/7070",
        )
        from node.config import NodeConfig

        cfg = NodeConfig()
        assert len(cfg.dht_bootstrap_peers) == 2
