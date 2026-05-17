"""
Tests for robust layer discovery in shard_manager.

All tests use mock nn.Module objects — no real model weights are loaded.
"""

import pytest
import torch
import torch.nn as nn

from node.config import NodeConfig
from node.shard_manager import (
    LayerDiscoveryError,
    ShardManager,
    _find_embedding,
    _find_layers,
    _find_lm_head,
)

# ---------------------------------------------------------------------------
# Helpers to build lightweight mock model hierarchies
# ---------------------------------------------------------------------------


def _make_block(has_self_attn: bool = True) -> nn.Module:
    """Return a minimal transformer-block lookalike."""

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)
            if has_self_attn:
                self.self_attn = nn.Linear(4, 4)

    return Block()


def _modulelist(n: int, has_self_attn: bool = True) -> nn.ModuleList:
    return nn.ModuleList([_make_block(has_self_attn) for _ in range(n)])


# ── named mock model classes ──────────────────────────────────────────────────


class _InnerModel(nn.Module):
    """Represents the `.model` sub-object common to most HF causal-LM wrappers."""

    def __init__(self, layers: nn.ModuleList, embed: nn.Module, norm: nn.Module):
        super().__init__()
        self.layers = layers
        self.embed_tokens = embed
        self.norm = norm


class LlamaForCausalLM(nn.Module):
    def __init__(self, n: int = 6):
        super().__init__()
        embed = nn.Embedding(100, 4)
        norm = nn.LayerNorm(4)
        self.model = _InnerModel(_modulelist(n), embed, norm)
        self.lm_head = nn.Linear(4, 100, bias=False)


class MistralForCausalLM(nn.Module):
    def __init__(self, n: int = 6):
        super().__init__()
        embed = nn.Embedding(100, 4)
        norm = nn.LayerNorm(4)
        self.model = _InnerModel(_modulelist(n), embed, norm)
        self.lm_head = nn.Linear(4, 100, bias=False)


class _TransformerH(nn.Module):
    """Represents GPT-2 / Falcon / Bloom's transformer sub-object."""

    def __init__(self, layers: nn.ModuleList, embed: nn.Module, norm: nn.Module):
        super().__init__()
        self.h = layers
        self.wte = embed
        self.ln_f = norm


class GPT2LMHeadModel(nn.Module):
    def __init__(self, n: int = 6):
        super().__init__()
        embed = nn.Embedding(100, 4)
        norm = nn.LayerNorm(4)
        self.transformer = _TransformerH(_modulelist(n), embed, norm)
        self.lm_head = nn.Linear(4, 100, bias=False)


class _GPTNeoxInner(nn.Module):
    def __init__(self, layers: nn.ModuleList, embed: nn.Module):
        super().__init__()
        self.layers = layers
        self.embed_in = embed


class GPTNeoXForCausalLM(nn.Module):
    def __init__(self, n: int = 6):
        super().__init__()
        embed = nn.Embedding(100, 4)
        self.gpt_neox = _GPTNeoxInner(_modulelist(n), embed)
        self.embed_out = nn.Linear(4, 100, bias=False)


class _OPTDecoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, embed: nn.Module):
        super().__init__()
        self.layers = layers
        self.embed_tokens = embed
        self.final_layer_norm = nn.LayerNorm(4)


class _OPTModel(nn.Module):
    def __init__(self, decoder: "_OPTDecoder"):
        super().__init__()
        self.decoder = decoder


class OPTForCausalLM(nn.Module):
    def __init__(self, n: int = 6):
        super().__init__()
        embed = nn.Embedding(100, 4)
        decoder = _OPTDecoder(_modulelist(n), embed)
        self.model = _OPTModel(decoder)
        self.lm_head = nn.Linear(4, 100, bias=False)


class UnknownArchitectureModel(nn.Module):
    """A model whose class name is not in LAYER_ATTR_MAP but has a clear ModuleList."""

    def __init__(self, n: int = 6, has_self_attn: bool = True):
        super().__init__()
        self.weird_attr = _modulelist(n, has_self_attn=has_self_attn)
        self.embed = nn.Embedding(100, 4)
        self.head = nn.Linear(4, 100)


class TinyModuleListModel(nn.Module):
    """Model with only a small (≤2 element) ModuleList — should fall to fallback."""

    def __init__(self):
        super().__init__()
        # Only 2 items — attention-scan ignores these; largest-fallback picks them up
        self.stuff = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])


class NoModuleListModel(nn.Module):
    """Model with no ModuleList at all — should raise."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)


# ── tests for _find_layers ────────────────────────────────────────────────────


class TestFindLayers:
    def test_llama_layers_found(self):
        model = LlamaForCausalLM(n=6)
        layers = _find_layers(model)
        assert isinstance(layers, nn.ModuleList)
        assert len(layers) == 6
        # Must be exactly the same object as model.model.layers
        assert layers is model.model.layers

    def test_mistral_layers_found(self):
        model = MistralForCausalLM(n=8)
        layers = _find_layers(model)
        assert isinstance(layers, nn.ModuleList)
        assert len(layers) == 8
        assert layers is model.model.layers

    def test_gpt2_layers_found(self):
        model = GPT2LMHeadModel(n=12)
        layers = _find_layers(model)
        assert isinstance(layers, nn.ModuleList)
        assert len(layers) == 12
        assert layers is model.transformer.h

    def test_gpt_neox_layers_found(self):
        model = GPTNeoXForCausalLM(n=10)
        layers = _find_layers(model)
        assert isinstance(layers, nn.ModuleList)
        assert len(layers) == 10
        assert layers is model.gpt_neox.layers

    def test_opt_layers_found(self):
        model = OPTForCausalLM(n=12)
        layers = _find_layers(model)
        assert isinstance(layers, nn.ModuleList)
        assert len(layers) == 12
        assert layers is model.model.decoder.layers

    def test_fallback_finds_attention_modules(self):
        """A novel model class should be found via attention-attribute scan."""
        model = UnknownArchitectureModel(n=5, has_self_attn=True)
        layers = _find_layers(model)
        assert isinstance(layers, nn.ModuleList)
        assert len(layers) == 5
        assert layers is model.weird_attr

    def test_fallback_modulelist_scan(self):
        """Largest-ModuleList fallback handles models without attention attrs."""
        model = UnknownArchitectureModel(n=4, has_self_attn=False)
        layers = _find_layers(model)
        assert isinstance(layers, nn.ModuleList)
        assert len(layers) == 4

    def test_raises_on_unknown_model_no_modulelist(self):
        model = NoModuleListModel()
        with pytest.raises(LayerDiscoveryError):
            _find_layers(model)


# ── tests for _find_embedding and _find_lm_head ───────────────────────────────


class TestFindEmbeddingAndHead:
    def test_llama_embedding_found(self):
        model = LlamaForCausalLM()
        emb = _find_embedding(model)
        assert emb is model.model.embed_tokens

    def test_gpt2_embedding_found(self):
        model = GPT2LMHeadModel()
        emb = _find_embedding(model)
        assert emb is model.transformer.wte

    def test_llama_lm_head_found(self):
        model = LlamaForCausalLM()
        head = _find_lm_head(model)
        assert head is model.lm_head

    def test_gpt2_lm_head_found(self):
        model = GPT2LMHeadModel()
        head = _find_lm_head(model)
        assert head is model.lm_head

    def test_gpt_neox_lm_head_found(self):
        model = GPTNeoXForCausalLM()
        head = _find_lm_head(model)
        assert head is model.embed_out


# ── tests for ShardManager._get_total_layers ─────────────────────────────────


class TestGetTotalLayers:
    def _make_cfg(self, **attrs):
        """Return a mock config object with the given attributes."""

        class Cfg:
            pass

        cfg = Cfg()
        for k, v in attrs.items():
            setattr(cfg, k, v)
        return cfg

    def test_total_layers_count_num_hidden_layers(self):
        cfg = self._make_cfg(num_hidden_layers=32)
        assert ShardManager._get_total_layers(cfg) == 32

    def test_total_layers_count_n_layer(self):
        cfg = self._make_cfg(n_layer=24)
        assert ShardManager._get_total_layers(cfg) == 24

    def test_total_layers_count_num_layers(self):
        cfg = self._make_cfg(num_layers=28)
        assert ShardManager._get_total_layers(cfg) == 28

    def test_total_layers_raises_on_unknown_config(self):
        cfg = self._make_cfg(something_else=42)
        with pytest.raises(ValueError):
            ShardManager._get_total_layers(cfg)


# ── tests for shard extraction ────────────────────────────────────────────────


def _make_shard_manager(num_shards: int, shard_index: int) -> ShardManager:
    cfg = NodeConfig(
        model_name="meta-llama/Llama-3.2-1B",
        num_shards=num_shards,
        shard_index=shard_index,
    )
    mgr = ShardManager(cfg)
    return mgr


class TestExtractShard:
    def _build_shard_wrapper(self, n_layers: int, num_shards: int, shard_index: int):
        """
        Build a _ShardWrapper from a mock LlamaForCausalLM using the real
        _extract_shard path but without touching any GPU or tokenizer.
        """
        from node.shard_manager import _ShardWrapper

        model = LlamaForCausalLM(n=n_layers)

        mgr = _make_shard_manager(num_shards, shard_index)
        start, end = mgr._compute_slice(n_layers)
        is_first = shard_index == 0
        is_last = shard_index == num_shards - 1

        wrapper = _ShardWrapper(model, start, end, is_first, is_last, torch)
        return wrapper, start, end

    def test_extract_shard_correct_slice(self):
        """Shard 1 of 3 on a 9-layer model should get layers [3, 4, 5]."""
        wrapper, start, end = self._build_shard_wrapper(9, num_shards=3, shard_index=1)
        assert start == 3
        assert end == 6
        assert len(list(wrapper._module.layers)) == 3

    def test_first_shard_has_embed(self):
        wrapper, _, _ = self._build_shard_wrapper(9, num_shards=3, shard_index=0)
        assert hasattr(wrapper._module, "embed_tokens")

    def test_last_shard_has_norm_and_head(self):
        wrapper, _, _ = self._build_shard_wrapper(9, num_shards=3, shard_index=2)
        assert hasattr(wrapper._module, "norm")
        assert hasattr(wrapper._module, "lm_head")

    def test_middle_shard_has_no_embed_or_head(self):
        wrapper, _, _ = self._build_shard_wrapper(9, num_shards=3, shard_index=1)
        assert not hasattr(wrapper._module, "embed_tokens")
        assert not hasattr(wrapper._module, "lm_head")

    def test_single_shard_gets_all_layers(self):
        wrapper, start, end = self._build_shard_wrapper(12, num_shards=1, shard_index=0)
        assert start == 0
        assert end == 12
        assert len(list(wrapper._module.layers)) == 12
        # Single shard is both first and last
        assert hasattr(wrapper._module, "embed_tokens")
        assert hasattr(wrapper._module, "norm")
        assert hasattr(wrapper._module, "lm_head")
