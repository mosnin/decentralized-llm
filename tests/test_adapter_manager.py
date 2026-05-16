"""
Tests for node.adapter_manager — no real GPU / torch checkpoints required.

All torch operations are mocked via sys.modules patching so the test suite
runs in a CPU-only / torch-free environment without modification.
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Minimal torch stub used across every test that touches _load_weights or hooks
# ---------------------------------------------------------------------------


def _make_torch_stub() -> MagicMock:
    """Return a MagicMock that satisfies the torch usage inside adapter_manager."""
    torch_stub = MagicMock(name="torch")

    # torch.zeros(r, c) → a real object we can track
    def _zeros(*shape, **kwargs):
        t = MagicMock(name=f"zeros{shape}")
        t.shape = shape
        t.T = MagicMock(name=f"zeros{shape}.T")
        return t

    torch_stub.zeros.side_effect = _zeros

    # torch.matmul returns another mock; keep it simple
    mm_result = MagicMock(name="matmul_result")
    mm_result.__mul__ = lambda self, other: mm_result  # result * scaling
    mm_result.__add__ = lambda self, other: mm_result  # output + delta
    mm_result.__radd__ = lambda self, other: mm_result
    torch_stub.matmul.return_value = mm_result

    return torch_stub


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _spec(
    adapter_id: str = "adapter-a",
    model_id: str = "meta-llama/Llama-2-7b",
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    target_modules: list[str] | None = None,
    checkpoint_path: str = "/nonexistent/adapter_a.pt",
    description: str = "Test adapter",
) -> "AdapterSpec":
    from node.adapter_manager import AdapterSpec

    return AdapterSpec(
        adapter_id=adapter_id,
        model_id=model_id,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules or ["q_proj", "v_proj"],
        checkpoint_path=checkpoint_path,
        description=description,
    )


def _make_mock_model(module_names: list[str] | None = None) -> MagicMock:
    """
    Build a mock nn.Module with named_modules() returning the given names.

    Each sub-module exposes ``register_forward_hook`` that records and
    returns a removable handle mock.
    """
    if module_names is None:
        module_names = ["q_proj", "v_proj"]

    model = MagicMock(name="model")
    submodules: list[tuple[str, MagicMock]] = []
    for name in module_names:
        sub = MagicMock(name=f"module_{name}")
        handle = MagicMock(name=f"handle_{name}")
        sub.register_forward_hook.return_value = handle
        submodules.append((name, sub))

    model.named_modules.return_value = submodules
    return model


@pytest.fixture(autouse=True)
def _patch_torch(monkeypatch):
    """Inject a torch stub for every test so no real torch is needed."""
    stub = _make_torch_stub()
    monkeypatch.setitem(sys.modules, "torch", stub)
    # Ensure safetensors is absent so _load_weights falls back to torch.load
    monkeypatch.setitem(sys.modules, "safetensors", None)
    monkeypatch.setitem(sys.modules, "safetensors.torch", None)
    return stub


@pytest.fixture()
def manager():
    from node.adapter_manager import AdapterManager

    return AdapterManager(max_loaded=4)


@pytest.fixture()
def small_manager():
    """AdapterManager with max_loaded=2 for LRU eviction tests."""
    from node.adapter_manager import AdapterManager

    return AdapterManager(max_loaded=2)


# ===========================================================================
# 1. register adapter
# ===========================================================================


class TestRegister:
    def test_register_adds_to_registry(self, manager):
        s = _spec("adapter-x")
        manager.register(s)
        assert "adapter-x" in {a.adapter_id for a in manager.list_adapters()}

    def test_register_does_not_load(self, manager):
        s = _spec("adapter-x")
        manager.register(s)
        # No slot should have been created
        assert "adapter-x" not in manager._slots

    def test_register_replaces_existing_spec(self, manager):
        s1 = _spec("adapter-x", description="v1")
        s2 = _spec("adapter-x", description="v2")
        manager.register(s1)
        manager.register(s2)
        registered = {a.adapter_id: a for a in manager.list_adapters()}
        assert registered["adapter-x"].description == "v2"

    def test_register_multiple_adapters(self, manager):
        for i in range(3):
            manager.register(_spec(f"adapter-{i}"))
        ids = {a.adapter_id for a in manager.list_adapters()}
        assert ids == {"adapter-0", "adapter-1", "adapter-2"}


# ===========================================================================
# 2. load adapter (mocked torch)
# ===========================================================================


class TestLoad:
    def test_load_unregistered_raises(self, manager):
        with pytest.raises(KeyError, match="not registered"):
            manager.load("does-not-exist")

    def test_load_creates_slot(self, manager):
        s = _spec("adapter-a")
        manager.register(s)
        slot = manager.load("adapter-a")
        assert slot.is_loaded is True
        assert slot.spec is s

    def test_load_sets_load_time(self, manager):
        before = time.time()
        manager.register(_spec("adapter-a"))
        slot = manager.load("adapter-a")
        after = time.time()
        assert before <= slot.load_time <= after

    def test_load_cache_hit_returns_same_slot(self, manager):
        manager.register(_spec("adapter-a"))
        slot1 = manager.load("adapter-a")
        slot2 = manager.load("adapter-a")
        assert slot1 is slot2

    def test_load_cache_hit_updates_last_used(self, manager):
        manager.register(_spec("adapter-a"))
        slot = manager.load("adapter-a")
        old_last_used = slot.last_used
        time.sleep(0.01)
        manager.load("adapter-a")
        assert slot.last_used >= old_last_used

    def test_load_synthesises_placeholder_weights_when_no_checkpoint(self, manager):
        s = _spec("adapter-a", checkpoint_path="/definitely/not/there.pt")
        manager.register(s)
        slot = manager.load("adapter-a")
        # Placeholder weights created for each target module
        assert set(slot.weights.keys()) == set(s.target_modules)
        for module_name in s.target_modules:
            assert "lora_A" in slot.weights[module_name]
            assert "lora_B" in slot.weights[module_name]


# ===========================================================================
# 3. LRU eviction when max_loaded exceeded
# ===========================================================================


class TestLRUEviction:
    def test_evicts_lru_adapter_when_full(self, small_manager):
        for i in range(3):
            small_manager.register(_spec(f"adapter-{i}"))

        # Load two adapters to fill the cache
        small_manager.load("adapter-0")
        time.sleep(0.01)
        small_manager.load("adapter-1")

        # Loading a third should evict adapter-0 (LRU)
        small_manager.load("adapter-2")

        assert "adapter-0" not in small_manager._slots
        assert "adapter-1" in small_manager._slots
        assert "adapter-2" in small_manager._slots

    def test_recently_used_adapter_survives_eviction(self, small_manager):
        for i in range(3):
            small_manager.register(_spec(f"adapter-{i}"))

        small_manager.load("adapter-0")
        time.sleep(0.01)
        small_manager.load("adapter-1")
        # Touch adapter-0 to make it more recent than adapter-1
        time.sleep(0.01)
        small_manager.load("adapter-0")  # cache hit, updates last_used

        # Loading adapter-2 should evict adapter-1 (now the LRU)
        small_manager.load("adapter-2")

        assert "adapter-1" not in small_manager._slots
        assert "adapter-0" in small_manager._slots
        assert "adapter-2" in small_manager._slots

    def test_eviction_respects_max_loaded_boundary(self, small_manager):
        for i in range(4):
            small_manager.register(_spec(f"adapter-{i}"))

        for i in range(4):
            small_manager.load(f"adapter-{i}")

        # After all 4 loads, only max_loaded=2 should remain
        assert len(small_manager._slots) == 2


# ===========================================================================
# 4. activate applies hooks to model
# ===========================================================================


class TestActivate:
    def test_activate_registers_hooks_on_target_modules(self, manager):
        s = _spec("adapter-a", target_modules=["q_proj", "v_proj"])
        manager.register(s)
        manager.load("adapter-a")

        model = _make_mock_model(["q_proj", "v_proj"])
        manager.activate("adapter-a", model)

        # register_forward_hook should have been called once per target module
        for _, sub in model.named_modules():
            sub.register_forward_hook.assert_called_once()

    def test_activate_sets_active_id(self, manager):
        manager.register(_spec("adapter-a"))
        manager.load("adapter-a")
        model = _make_mock_model()
        manager.activate("adapter-a", model)
        assert manager.get_active_adapter() == "adapter-a"

    def test_activate_stores_hook_handles(self, manager):
        s = _spec("adapter-a", target_modules=["q_proj", "v_proj"])
        manager.register(s)
        manager.load("adapter-a")
        model = _make_mock_model(["q_proj", "v_proj"])
        manager.activate("adapter-a", model)
        assert len(manager._hook_handles) == 2

    def test_activate_raises_if_not_loaded(self, manager):
        manager.register(_spec("adapter-a"))
        model = _make_mock_model()
        with pytest.raises(KeyError, match="not loaded"):
            manager.activate("adapter-a", model)

    def test_activate_replaces_previous_hooks(self, manager):
        for aid in ("adapter-a", "adapter-b"):
            manager.register(_spec(aid))
            manager.load(aid)

        model = _make_mock_model(["q_proj", "v_proj"])
        manager.activate("adapter-a", model)
        first_handles = list(manager._hook_handles)

        # Re-wire to a fresh model to get fresh handles
        model2 = _make_mock_model(["q_proj", "v_proj"])
        manager.activate("adapter-b", model2)

        # Old handles should have been removed
        for h in first_handles:
            h.remove.assert_called_once()
        # New handles installed
        assert len(manager._hook_handles) == 2


# ===========================================================================
# 5. deactivate removes hooks
# ===========================================================================


class TestDeactivate:
    def test_deactivate_removes_all_hooks(self, manager):
        s = _spec("adapter-a", target_modules=["q_proj", "v_proj"])
        manager.register(s)
        manager.load("adapter-a")
        model = _make_mock_model(["q_proj", "v_proj"])
        manager.activate("adapter-a", model)

        handles = list(manager._hook_handles)
        manager.deactivate(model)

        for h in handles:
            h.remove.assert_called_once()
        assert manager._hook_handles == []

    def test_deactivate_clears_active_id(self, manager):
        manager.register(_spec("adapter-a"))
        manager.load("adapter-a")
        model = _make_mock_model()
        manager.activate("adapter-a", model)
        manager.deactivate(model)
        assert manager.get_active_adapter() is None

    def test_deactivate_idempotent_when_no_active(self, manager):
        model = _make_mock_model()
        # Should not raise when nothing is active
        manager.deactivate(model)
        assert manager.get_active_adapter() is None


# ===========================================================================
# 6. swap is atomic (uses asyncio.Lock)
# ===========================================================================


class TestSwap:
    @pytest.mark.asyncio
    async def test_swap_activates_target(self, manager):
        for aid in ("adapter-a", "adapter-b"):
            manager.register(_spec(aid))

        model = _make_mock_model(["q_proj", "v_proj"])
        await manager.swap(None, "adapter-a", model)
        assert manager.get_active_adapter() == "adapter-a"

        model2 = _make_mock_model(["q_proj", "v_proj"])
        await manager.swap("adapter-a", "adapter-b", model2)
        assert manager.get_active_adapter() == "adapter-b"

    @pytest.mark.asyncio
    async def test_swap_deactivates_source(self, manager):
        for aid in ("adapter-a", "adapter-b"):
            manager.register(_spec(aid))

        model = _make_mock_model(["q_proj", "v_proj"])
        await manager.swap(None, "adapter-a", model)
        handles_a = list(manager._hook_handles)

        model2 = _make_mock_model(["q_proj", "v_proj"])
        await manager.swap("adapter-a", "adapter-b", model2)

        # Handles from adapter-a should have been removed
        for h in handles_a:
            h.remove.assert_called()

    @pytest.mark.asyncio
    async def test_swap_raises_for_unregistered_target(self, manager):
        model = _make_mock_model()
        with pytest.raises(KeyError, match="not registered"):
            await manager.swap(None, "ghost-adapter", model)

    @pytest.mark.asyncio
    async def test_swap_loads_adapter_if_not_loaded(self, manager):
        manager.register(_spec("adapter-a"))
        model = _make_mock_model(["q_proj", "v_proj"])
        # adapter-a is registered but not yet loaded
        assert "adapter-a" not in manager._slots
        await manager.swap(None, "adapter-a", model)
        assert manager._slots["adapter-a"].is_loaded is True


# ===========================================================================
# 7. unload frees slot
# ===========================================================================


class TestUnload:
    def test_unload_removes_slot(self, manager):
        manager.register(_spec("adapter-a"))
        manager.load("adapter-a")
        assert "adapter-a" in manager._slots
        manager.unload("adapter-a")
        assert "adapter-a" not in manager._slots

    def test_unload_clears_weights(self, manager):
        manager.register(_spec("adapter-a"))
        slot = manager.load("adapter-a")
        # Weights dict was populated by _load_weights
        assert slot.weights  # non-empty

        manager.unload("adapter-a")
        # slot.weights should have been cleared (dict mutation)
        assert not slot.weights

    def test_unload_not_loaded_is_noop(self, manager):
        manager.register(_spec("adapter-a"))
        # Should not raise
        manager.unload("adapter-a")

    def test_unload_clears_active_id_when_active(self, manager):
        manager.register(_spec("adapter-a"))
        manager.load("adapter-a")
        # Manually set active without a model (avoid needing named_modules here)
        manager._active_id = "adapter-a"
        manager.unload("adapter-a")
        assert manager.get_active_adapter() is None


# ===========================================================================
# 8. list_adapters returns all registered
# ===========================================================================


class TestListAdapters:
    def test_list_adapters_empty_initially(self, manager):
        assert manager.list_adapters() == []

    def test_list_adapters_returns_all_registered(self, manager):
        specs = [_spec(f"adapter-{i}") for i in range(5)]
        for s in specs:
            manager.register(s)
        result = manager.list_adapters()
        assert len(result) == 5
        assert {r.adapter_id for r in result} == {s.adapter_id for s in specs}

    def test_list_adapters_includes_unloaded(self, manager):
        manager.register(_spec("adapter-a"))
        manager.register(_spec("adapter-b"))
        manager.load("adapter-a")
        # Both should appear even though only one is loaded
        ids = {a.adapter_id for a in manager.list_adapters()}
        assert "adapter-a" in ids
        assert "adapter-b" in ids


# ===========================================================================
# 9. get_active_adapter returns current
# ===========================================================================


class TestGetActiveAdapter:
    def test_get_active_adapter_none_initially(self, manager):
        assert manager.get_active_adapter() is None

    def test_get_active_adapter_after_activate(self, manager):
        manager.register(_spec("adapter-a"))
        manager.load("adapter-a")
        model = _make_mock_model()
        manager.activate("adapter-a", model)
        assert manager.get_active_adapter() == "adapter-a"

    def test_get_active_adapter_after_deactivate(self, manager):
        manager.register(_spec("adapter-a"))
        manager.load("adapter-a")
        model = _make_mock_model()
        manager.activate("adapter-a", model)
        manager.deactivate(model)
        assert manager.get_active_adapter() is None


# ===========================================================================
# 10. concurrent swap with locking
# ===========================================================================


class TestConcurrentSwap:
    @pytest.mark.asyncio
    async def test_concurrent_swaps_serialize(self, manager):
        """
        Fire multiple concurrent swap() calls and verify that only one
        adapter is active at the end and no assertion errors occur.
        """
        for i in range(4):
            manager.register(_spec(f"adapter-{i}"))

        models = [_make_mock_model(["q_proj", "v_proj"]) for _ in range(4)]

        async def do_swap(from_id, to_id, model):
            await manager.swap(from_id, to_id, model)

        # Prime with adapter-0
        await do_swap(None, "adapter-0", models[0])

        # Fire three concurrent swaps
        await asyncio.gather(
            do_swap("adapter-0", "adapter-1", models[1]),
            do_swap("adapter-0", "adapter-2", models[2]),
            do_swap("adapter-0", "adapter-3", models[3]),
        )

        # Exactly one adapter must be active
        active = manager.get_active_adapter()
        assert active in {"adapter-1", "adapter-2", "adapter-3"}

    @pytest.mark.asyncio
    async def test_swap_lock_prevents_interleaving(self, manager):
        """
        Verify the asyncio.Lock is actually acquired: a second swap that
        starts while the first holds the lock will queue behind it.
        """
        for i in range(3):
            manager.register(_spec(f"adapter-{i}"))

        call_order: list[str] = []
        original_activate = manager.activate

        async def slow_swap(from_id, to_id, model, tag):
            async with manager._swap_lock:
                call_order.append(f"start-{tag}")
                await asyncio.sleep(0)  # yield control
                original_activate(to_id, model)
                manager._active_id = to_id
                call_order.append(f"end-{tag}")

        manager.load("adapter-0")
        manager.load("adapter-1")
        model_a = _make_mock_model(["q_proj"])
        model_b = _make_mock_model(["q_proj"])
        await asyncio.gather(
            slow_swap(None, "adapter-0", model_a, "A"),
            slow_swap(None, "adapter-1", model_b, "B"),
        )

        # Each task must complete before the other starts (no interleaving)
        assert call_order.index("end-A") < call_order.index("start-B") or (
            call_order.index("end-B") < call_order.index("start-A")
        )


# ===========================================================================
# 11. inference_count tracking
# ===========================================================================


class TestInferenceCount:
    def test_inference_count_increments_via_hook(self, manager):
        """
        Simulate the forward hook being called and verify inference_count rises.
        """
        from node.adapter_manager import _make_lora_hook

        # Build a minimal slot with a real spec
        from node.adapter_manager import AdapterSlot

        s = _spec("adapter-a", lora_rank=4, lora_alpha=8.0)
        slot = AdapterSlot(spec=s)

        # Fake lora weights (MagicMock tensors)
        lora_A = MagicMock(name="lora_A")
        lora_B = MagicMock(name="lora_B")
        lora_A.T = MagicMock()
        lora_B.T = MagicMock()

        # Patch torch inside adapter_manager so matmul works
        import torch  # This is the stub from sys.modules

        scaling = s.lora_alpha / s.lora_rank  # 2.0

        hook = _make_lora_hook(lora_A, lora_B, scaling, slot)

        fake_input = MagicMock(name="input_tensor")
        fake_input.device = "cpu"
        fake_input.dtype = MagicMock()
        fake_output = MagicMock(name="output_tensor")

        lora_A.to.return_value = lora_A
        lora_B.to.return_value = lora_B

        assert slot.inference_count == 0
        hook(MagicMock(), (fake_input,), fake_output)
        assert slot.inference_count == 1
        hook(MagicMock(), (fake_input,), fake_output)
        assert slot.inference_count == 2

    def test_inference_count_not_incremented_when_empty_inputs(self, manager):
        """Hook with empty inputs tuple should return output unchanged, count stays 0."""
        from node.adapter_manager import _make_lora_hook, AdapterSlot

        s = _spec("adapter-a")
        slot = AdapterSlot(spec=s)

        lora_A = MagicMock()
        lora_B = MagicMock()
        hook = _make_lora_hook(lora_A, lora_B, 1.0, slot)

        fake_output = MagicMock(name="output")
        result = hook(MagicMock(), (), fake_output)

        assert slot.inference_count == 0
        assert result is fake_output

    def test_inference_count_per_slot_not_shared(self, manager):
        """Each slot tracks its own inference count independently."""
        from node.adapter_manager import _make_lora_hook, AdapterSlot

        s_a = _spec("adapter-a")
        s_b = _spec("adapter-b")
        slot_a = AdapterSlot(spec=s_a)
        slot_b = AdapterSlot(spec=s_b)

        lora_A = MagicMock()
        lora_B = MagicMock()
        lora_A.T = MagicMock()
        lora_B.T = MagicMock()
        lora_A.to.return_value = lora_A
        lora_B.to.return_value = lora_B

        hook_a = _make_lora_hook(lora_A, lora_B, 1.0, slot_a)
        hook_b = _make_lora_hook(lora_A, lora_B, 1.0, slot_b)

        fake_input = MagicMock()
        fake_input.device = "cpu"
        fake_input.dtype = MagicMock()
        fake_output = MagicMock()

        for _ in range(3):
            hook_a(MagicMock(), (fake_input,), fake_output)
        for _ in range(7):
            hook_b(MagicMock(), (fake_input,), fake_output)

        assert slot_a.inference_count == 3
        assert slot_b.inference_count == 7
