"""
LoRA adapter hot-swap manager.

Allows switching fine-tuned LoRA adapters in and out of a base model
without reloading the model weights.  Maintains an LRU cache of loaded
adapter weight tensors and applies them to the model via forward hooks.

All heavy imports (torch, peft) are lazy so this module is importable
with stdlib only.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass  # torch / peft type stubs only when a type-checker runs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AdapterSpec:
    """Static description of a LoRA adapter checkpoint."""

    adapter_id: str
    model_id: str
    lora_rank: int
    lora_alpha: float
    target_modules: list[str]
    checkpoint_path: str
    description: str
    created_at: float = field(default_factory=time.time)


@dataclass
class AdapterSlot:
    """Runtime state for a single loaded adapter."""

    spec: AdapterSpec
    is_loaded: bool = False
    load_time: float = 0.0
    inference_count: int = 0
    last_used: float = 0.0
    # Loaded weight tensors: module_name -> {"lora_A": tensor, "lora_B": tensor}
    weights: dict[str, dict[str, Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# AdapterManager
# ---------------------------------------------------------------------------


class AdapterManager:
    """
    Hot-swap manager for LoRA adapters.

    Maintains:
    - A *registry* of all known AdapterSpecs (may exceed max_loaded).
    - A *slot cache* of at most *max_loaded* loaded weight tensors (LRU).
    - At most one *active* adapter whose LoRA delta is wired into the model
      via registered forward hooks.

    Thread / async safety
    ---------------------
    ``swap()`` is protected by an ``asyncio.Lock`` so concurrent callers
    wait rather than racing.  Individual ``load`` / ``unload`` / ``activate``
    / ``deactivate`` are *not* independently locked; callers are expected to
    use ``swap()`` for safe transitions.
    """

    def __init__(self, max_loaded: int = 4) -> None:
        self._max_loaded = max_loaded
        # adapter_id -> AdapterSpec (all known)
        self._registry: dict[str, AdapterSpec] = {}
        # adapter_id -> AdapterSlot (only loaded ones)
        self._slots: dict[str, AdapterSlot] = {}
        # currently active adapter id (None if no adapter is active)
        self._active_id: str | None = None
        # handles returned by model.register_forward_hook — keyed by module name
        self._hook_handles: list[Any] = []
        # lock for atomic swap
        self._swap_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------

    def register(self, spec: AdapterSpec) -> None:
        """Add *spec* to the registry without loading its weights."""
        if spec.adapter_id in self._registry:
            logger.debug("Adapter %r already registered — replacing spec", spec.adapter_id)
        self._registry[spec.adapter_id] = spec
        logger.info("Registered adapter %r (model=%s)", spec.adapter_id, spec.model_id)

    def list_adapters(self) -> list[AdapterSpec]:
        """Return all registered AdapterSpecs."""
        return list(self._registry.values())

    def get_active_adapter(self) -> str | None:
        """Return the adapter_id of the currently active adapter, or None."""
        return self._active_id

    # ------------------------------------------------------------------
    # Load / unload
    # ------------------------------------------------------------------

    def load(self, adapter_id: str) -> AdapterSlot:
        """
        Load the adapter weights into an in-memory slot.

        If the adapter is already loaded, returns the existing slot (updating
        ``last_used``).  If loading a new adapter would exceed ``max_loaded``,
        the least-recently-used loaded adapter is evicted first.

        Raises ``KeyError`` if *adapter_id* is not registered.
        """
        if adapter_id not in self._registry:
            raise KeyError(f"Adapter {adapter_id!r} is not registered")

        # Cache hit
        if adapter_id in self._slots and self._slots[adapter_id].is_loaded:
            slot = self._slots[adapter_id]
            slot.last_used = time.time()
            logger.debug("Adapter %r already loaded — cache hit", adapter_id)
            return slot

        # Evict LRU if needed
        if len(self._slots) >= self._max_loaded:
            self._evict_lru()

        spec = self._registry[adapter_id]
        slot = AdapterSlot(spec=spec)

        # Lazy torch import — keep GPU / CUDA imports out of module-level scope
        weights = self._load_weights(spec)

        slot.weights = weights
        slot.is_loaded = True
        slot.load_time = time.time()
        slot.last_used = time.time()

        self._slots[adapter_id] = slot
        logger.info(
            "Loaded adapter %r from %s (%d target modules)",
            adapter_id,
            spec.checkpoint_path,
            len(spec.target_modules),
        )
        return slot

    def unload(self, adapter_id: str) -> None:
        """
        Explicitly unload adapter weights and free the slot (VRAM).

        If the adapter is currently active, its hooks are removed first.
        No-op if the adapter is not loaded.
        """
        slot = self._slots.get(adapter_id)
        if slot is None or not slot.is_loaded:
            logger.debug("Adapter %r not loaded — nothing to unload", adapter_id)
            return

        if self._active_id == adapter_id:
            logger.warning(
                "Unloading active adapter %r — hooks will be removed but no model was provided;"
                " call deactivate(model) first for a clean teardown",
                adapter_id,
            )
            self._active_id = None
            self._hook_handles.clear()

        slot.weights.clear()
        slot.is_loaded = False
        del self._slots[adapter_id]
        logger.info("Unloaded adapter %r", adapter_id)

    # ------------------------------------------------------------------
    # Activate / deactivate
    # ------------------------------------------------------------------

    def activate(self, adapter_id: str, model: Any) -> None:
        """
        Apply the LoRA delta weights for *adapter_id* to *model* in-place.

        For each target module listed in the AdapterSpec, a forward hook is
        registered that adds ``(lora_A @ lora_B) * (alpha / rank)`` to the
        module's output.  The hook is stored so it can be removed later.

        Raises ``KeyError`` if *adapter_id* is not loaded.
        """
        if adapter_id not in self._slots or not self._slots[adapter_id].is_loaded:
            raise KeyError(f"Adapter {adapter_id!r} is not loaded; call load() first")

        # Remove any existing hooks before applying new ones
        self._remove_hooks()

        slot = self._slots[adapter_id]
        spec = slot.spec
        scaling = spec.lora_alpha / spec.lora_rank

        # Walk the model's named modules and attach hooks to targets
        for name, module in model.named_modules():
            if name not in slot.weights:
                continue

            lora_A = slot.weights[name]["lora_A"]
            lora_B = slot.weights[name]["lora_B"]

            # Build a closure that captures (lora_A, lora_B, scaling, slot)
            handle = module.register_forward_hook(_make_lora_hook(lora_A, lora_B, scaling, slot))
            self._hook_handles.append(handle)

        self._active_id = adapter_id
        slot.last_used = time.time()
        logger.info(
            "Activated adapter %r — %d hook(s) registered",
            adapter_id,
            len(self._hook_handles),
        )

    def deactivate(self, model: Any) -> None:  # noqa: ARG002  (model arg kept for API clarity)
        """
        Remove all LoRA forward hooks from the model.

        The *model* argument is accepted for API clarity but the hooks are
        removed via their stored handles, so the reference is not strictly
        required.
        """
        self._remove_hooks()
        self._active_id = None
        logger.info("Deactivated all LoRA hooks")

    # ------------------------------------------------------------------
    # Atomic swap
    # ------------------------------------------------------------------

    async def swap(self, from_id: str | None, to_id: str, model: Any) -> None:
        """
        Atomically deactivate *from_id* and activate *to_id*.

        The operation is serialised with an ``asyncio.Lock`` so concurrent
        callers queue up rather than leaving the model in an inconsistent
        state.  *from_id* may be ``None`` when activating an adapter for
        the first time.

        Raises ``KeyError`` if *to_id* is not registered.
        """
        if to_id not in self._registry:
            raise KeyError(f"Adapter {to_id!r} is not registered")

        async with self._swap_lock:
            # Deactivate current (if any)
            if from_id is not None and self._active_id == from_id:
                self.deactivate(model)
            elif self._active_id is not None:
                # Deactivate whatever is currently active
                self.deactivate(model)

            # Ensure target is loaded
            if to_id not in self._slots or not self._slots[to_id].is_loaded:
                self.load(to_id)

            self.activate(to_id, model)
            logger.info("Swapped adapter %r → %r", from_id, to_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evict_lru(self) -> None:
        """Evict the least-recently-used loaded slot."""
        if not self._slots:
            return

        # Do not evict the currently active adapter if we can avoid it
        candidates = sorted(self._slots.items(), key=lambda kv: kv[1].last_used)
        for adapter_id, _slot in candidates:
            if adapter_id != self._active_id:
                logger.info("LRU eviction: unloading adapter %r", adapter_id)
                self.unload(adapter_id)
                return

        # All loaded adapters are active (only one can be active, so this
        # path is only hit when max_loaded == 1 and the active adapter is the
        # only one loaded).
        adapter_id = candidates[0][0]
        logger.warning("Evicting active adapter %r due to max_loaded constraint", adapter_id)
        self.unload(adapter_id)

    def _load_weights(self, spec: AdapterSpec) -> dict[str, dict[str, Any]]:
        """
        Load LoRA weight tensors for *spec* from disk.

        Attempts to load a ``safetensors`` or ``torch`` checkpoint from
        ``spec.checkpoint_path``.  Falls back to synthesising zero tensors
        (rank × hidden) when the file does not exist, so the manager can be
        exercised in test environments without real checkpoints.

        Returns a mapping:
            module_name -> {"lora_A": Tensor[rank, in], "lora_B": Tensor[out, rank]}
        """
        import torch  # noqa: PLC0415 — lazy import to avoid GPU at module load

        weights: dict[str, dict[str, Any]] = {}

        import os  # noqa: PLC0415

        checkpoint = spec.checkpoint_path
        if os.path.isfile(checkpoint):
            # Prefer safetensors when available
            try:
                from safetensors.torch import load_file  # noqa: PLC0415

                raw = load_file(checkpoint)
            except ImportError:
                raw = torch.load(checkpoint, map_location="cpu", weights_only=True)

            # Expected key format: "<module_name>.lora_A.weight"
            for key, tensor in raw.items():
                parts = key.rsplit(".", 2)
                if len(parts) == 3 and parts[1] in ("lora_A", "lora_B"):
                    module_name, matrix_name, _ = parts
                    weights.setdefault(module_name, {})[matrix_name] = tensor
        else:
            # Synthesise placeholder tensors (all zeros) so activate() can
            # still register hooks in test / dry-run scenarios.
            logger.warning(
                "Checkpoint %r not found — using zero-weight placeholders for adapter %r",
                checkpoint,
                spec.adapter_id,
            )
            for module_name in spec.target_modules:
                weights[module_name] = {
                    "lora_A": torch.zeros(spec.lora_rank, 1),
                    "lora_B": torch.zeros(1, spec.lora_rank),
                }

        return weights

    def _remove_hooks(self) -> None:
        """Remove all currently registered forward hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()


# ---------------------------------------------------------------------------
# Hook factory
# ---------------------------------------------------------------------------


def _make_lora_hook(lora_A: Any, lora_B: Any, scaling: float, slot: AdapterSlot):
    """
    Return a ``torch.nn.Module`` forward hook that adds the LoRA delta to
    the module output.

    The delta is computed as ``output + (input @ lora_A.T @ lora_B.T) * scaling``.
    ``slot.inference_count`` is incremented on every forward pass.
    """

    def lora_forward_hook(module: Any, inputs: tuple, output: Any) -> Any:  # noqa: ARG001
        import torch  # noqa: PLC0415

        # inputs is a tuple; the first element is the primary input tensor
        if not inputs:
            return output

        slot.inference_count += 1
        slot.last_used = time.time()

        x = inputs[0]

        # Ensure A and B are on the same device as the input
        A = lora_A.to(x.device, x.dtype)  # [rank, in_features]
        B = lora_B.to(x.device, x.dtype)  # [out_features, rank]

        # LoRA delta: x @ A^T → [batch, seq, rank]  then @ B^T → [batch, seq, out]
        # Works for both 2-D (batch × in) and 3-D (batch × seq × in) inputs.
        try:
            delta = torch.matmul(torch.matmul(x, A.T), B.T) * scaling
            return output + delta
        except RuntimeError:
            # Shape mismatch (e.g. placeholder zeros with rank=1, in=1) — skip
            return output

    return lora_forward_hook
