"""Tests for ModelRegistry — all ShardManager interactions are mocked."""

import asyncio
import hashlib
from unittest.mock import MagicMock, patch

import pytest

from node.config import NodeConfig
from node.model_registry import ModelRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_shard_manager():
    """Return a MagicMock that looks enough like a ShardManager to satisfy the registry."""
    mgr = MagicMock()
    mgr.load = MagicMock()  # synchronous load()
    return mgr


def _make_config(**kwargs):
    return NodeConfig(model_name="base-model", **kwargs)


def _sm_module(mock_mgr=None):
    """Return a fake sys.modules entry for node.shard_manager."""
    if mock_mgr is None:
        mock_mgr = _make_mock_shard_manager()
    sm_module = MagicMock()
    sm_module.ShardManager.return_value = mock_mgr
    return sm_module, mock_mgr


async def _sync_executor(_exc, fn, *args):
    """Replacement for loop.run_in_executor that calls the function synchronously."""
    return fn(*args) if args else fn()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestModelRegistry:
    # ── test_load_and_get ────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_load_and_get(self):
        """Loading a model should make it retrievable via get()."""
        registry = ModelRegistry()
        config = _make_config()
        sm_mod, mock_mgr = _sm_module()

        with patch.dict("sys.modules", {"node.shard_manager": sm_mod}):
            with patch.object(
                asyncio.get_event_loop(), "run_in_executor", side_effect=_sync_executor
            ):
                await registry.load("my-model", config)

        assert registry.get("my-model") is mock_mgr

    # ── test_load_is_idempotent ──────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_load_is_idempotent(self):
        """Calling load() twice for the same model must not create a duplicate entry."""
        registry = ModelRegistry()
        config = _make_config()
        call_count = 0

        def factory(_cfg):
            nonlocal call_count
            call_count += 1
            return _make_mock_shard_manager()

        sm_module = MagicMock()
        sm_module.ShardManager.side_effect = factory

        with patch.dict("sys.modules", {"node.shard_manager": sm_module}):
            with patch.object(
                asyncio.get_event_loop(), "run_in_executor", side_effect=_sync_executor
            ):
                await registry.load("model-a", config)
                await registry.load("model-a", config)  # second call — should no-op

        assert call_count == 1, "ShardManager should only be instantiated once"
        assert registry.list_loaded() == ["model-a"]

    # ── test_unload_removes_model ────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_unload_removes_model(self):
        """After unload(), the model must not be accessible via get()."""
        registry = ModelRegistry()
        config = _make_config()
        sm_mod, _ = _sm_module()

        with patch.dict("sys.modules", {"node.shard_manager": sm_mod}):
            with patch.object(
                asyncio.get_event_loop(), "run_in_executor", side_effect=_sync_executor
            ):
                await registry.load("model-b", config)
                assert registry.is_loaded("model-b")

                await registry.unload("model-b")

        assert not registry.is_loaded("model-b")
        assert registry.get("model-b") is None

    # ── test_get_unknown_returns_none ────────────────────────────────────────

    def test_get_unknown_returns_none(self):
        """get() for a name that was never loaded must return None."""
        registry = ModelRegistry()
        assert registry.get("does-not-exist") is None

    # ── test_model_id_matches_sha256 ─────────────────────────────────────────

    def test_model_id_matches_sha256(self):
        """model_id() must return the SHA-256 digest of the UTF-8 encoded name."""
        name = "meta-llama/Llama-3.2-3B"
        expected = hashlib.sha256(name.encode()).digest()
        assert ModelRegistry.model_id(name) == expected

    # ── test_list_loaded_empty ───────────────────────────────────────────────

    def test_list_loaded_empty(self):
        """A fresh registry reports an empty list of loaded models."""
        registry = ModelRegistry()
        assert registry.list_loaded() == []

    # ── test_list_loaded_after_loading ───────────────────────────────────────

    @pytest.mark.asyncio
    async def test_list_loaded_after_loading(self):
        """list_loaded() must reflect all models loaded so far."""
        registry = ModelRegistry()
        config = _make_config()

        sm_module = MagicMock()
        sm_module.ShardManager.side_effect = lambda _cfg: _make_mock_shard_manager()

        with patch.dict("sys.modules", {"node.shard_manager": sm_module}):
            with patch.object(
                asyncio.get_event_loop(), "run_in_executor", side_effect=_sync_executor
            ):
                await registry.load("model-x", config)
                await registry.load("model-y", config)

        loaded = set(registry.list_loaded())
        assert loaded == {"model-x", "model-y"}

    # ── test_load_all_loads_concurrently ─────────────────────────────────────

    @pytest.mark.asyncio
    async def test_load_all_loads_concurrently(self):
        """load_all() must use asyncio.gather to load models concurrently."""
        registry = ModelRegistry()
        config = _make_config()

        sm_module = MagicMock()
        sm_module.ShardManager.side_effect = lambda _cfg: _make_mock_shard_manager()

        original_gather = asyncio.gather

        async def tracking_gather(*coros, **kw):
            tracking_gather.call_count += 1
            tracking_gather.last_args = coros
            return await original_gather(*coros, **kw)

        tracking_gather.call_count = 0
        tracking_gather.last_args = ()

        with patch.dict("sys.modules", {"node.shard_manager": sm_module}):
            with patch.object(
                asyncio.get_event_loop(), "run_in_executor", side_effect=_sync_executor
            ):
                with patch("node.model_registry.asyncio.gather", side_effect=tracking_gather) as mock_gather:
                    await registry.load_all(["alpha", "beta", "gamma"], config)

        # asyncio.gather was called once with three coroutines
        mock_gather.assert_called_once()
        args, _ = mock_gather.call_args
        assert len(args) == 3

        # All three models are now loaded
        assert set(registry.list_loaded()) == {"alpha", "beta", "gamma"}

    # ── test_is_loaded ───────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_is_loaded(self):
        """is_loaded() returns True only for models that have been loaded."""
        registry = ModelRegistry()
        config = _make_config()
        sm_mod, _ = _sm_module()

        assert not registry.is_loaded("model-z")

        with patch.dict("sys.modules", {"node.shard_manager": sm_mod}):
            with patch.object(
                asyncio.get_event_loop(), "run_in_executor", side_effect=_sync_executor
            ):
                await registry.load("model-z", config)

        assert registry.is_loaded("model-z")
        assert not registry.is_loaded("model-w")
