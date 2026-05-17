"""Registry mapping model IDs to loaded ShardManager instances."""

import asyncio
import hashlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .shard_manager import ShardManager

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Manages a pool of loaded ShardManager instances, one per model.

    Provides concurrent loading, lazy access, and clean shutdown of multiple
    models so a single node can serve several models simultaneously.
    """

    def __init__(self) -> None:
        self._registry: dict[str, ShardManager] = {}
        # Maps sha256(model_name) → model_name for reverse lookup by on-chain ID.
        self._id_to_name: dict[bytes, str] = {}

    async def load(self, model_name: str, config) -> None:
        """Load the model (via ShardManager.load()) and cache it.

        If the model is already loaded, this is a no-op.
        """
        if model_name in self._registry:
            logger.debug("Model %r already loaded — skipping", model_name)
            return

        import dataclasses

        from .shard_manager import ShardManager  # lazy import to avoid GPU at module level

        model_config = dataclasses.replace(config, model_name=model_name)
        shard_manager = ShardManager(model_config)
        await asyncio.get_event_loop().run_in_executor(None, shard_manager.load)
        self._registry[model_name] = shard_manager
        self._id_to_name[hashlib.sha256(model_name.encode()).digest()] = model_name
        logger.info("Loaded model %r into registry", model_name)

    async def unload(self, model_name: str) -> None:
        """Shutdown the ShardManager for *model_name* and remove it from registry."""
        shard_manager = self._registry.pop(model_name, None)
        if shard_manager is None:
            logger.debug("Model %r not in registry — nothing to unload", model_name)
            return

        self._id_to_name.pop(hashlib.sha256(model_name.encode()).digest(), None)

        shutdown = getattr(shard_manager, "shutdown", None)
        if shutdown is not None:
            if asyncio.iscoroutinefunction(shutdown):
                await shutdown()
            else:
                await asyncio.get_event_loop().run_in_executor(None, shutdown)

        logger.info("Unloaded model %r from registry", model_name)

    def get(self, model_name: str) -> "ShardManager | None":
        """Return the loaded ShardManager for *model_name*, or None if not loaded."""
        return self._registry.get(model_name)

    def get_by_model_id(self, model_id: bytes) -> "ShardManager | None":
        """Return the loaded ShardManager for the on-chain *model_id*, or None."""
        name = self._id_to_name.get(model_id)
        if name is None:
            return None
        return self._registry.get(name)

    @staticmethod
    def model_id(model_name: str) -> bytes:
        """Return sha256(model_name.encode()) — the canonical on-chain model ID."""
        return hashlib.sha256(model_name.encode()).digest()

    def list_loaded(self) -> list[str]:
        """Return the names of all currently-loaded models."""
        return list(self._registry)

    def is_loaded(self, model_name: str) -> bool:
        """Return True if *model_name* is currently loaded."""
        return model_name in self._registry

    async def load_all(self, model_names: list[str], config) -> None:
        """Load all *model_names* concurrently via asyncio.gather."""
        await asyncio.gather(*(self.load(name, config) for name in model_names))
        logger.info("Loaded %d model(s): %s", len(model_names), model_names)
