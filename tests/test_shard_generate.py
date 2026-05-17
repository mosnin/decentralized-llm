"""
Comprehensive tests for ShardManager.generate(), generate_batch(),
_sample_token(), _top_p_filter(), and _extract_shard() layer discovery.

All tests run without a GPU and without downloading real model weights by
constructing lightweight mock objects.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from node.config import NodeConfig
from node.shard_manager import ShardManager, _ShardWrapper

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

VOCAB = 32
HIDDEN = 16
SEQ = 4


def _make_config(shard_index: int = 0, num_shards: int = 1) -> NodeConfig:
    return NodeConfig(
        model_name="mock-model",
        num_shards=num_shards,
        shard_index=shard_index,
    )


class _FakeLayer(nn.Module):
    """Identity transformer layer (just passes hidden states through)."""

    def __init__(self):
        super().__init__()
        self.self_attn = nn.Identity()  # satisfies the fallback discovery check

    def forward(self, x):
        return x


class _FakeEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(VOCAB, HIDDEN))

    def forward(self, ids):
        return self.weight[ids]


class _FakeLMHead(nn.Module):
    """Always returns the same fixed logits regardless of hidden state."""

    def __init__(self, fixed_logits: torch.Tensor):
        super().__init__()
        self._fixed = fixed_logits  # (vocab,)
        # Dummy parameter so .parameters() / .to() work.
        self._p = nn.Parameter(torch.zeros(1))

    def forward(self, hidden):
        # hidden: (B, L, H) → return (B, L, vocab)
        batch, seq_len, _ = hidden.shape
        return self._fixed.expand(batch, seq_len, -1)


def _build_shard_wrapper(
    fixed_logits: torch.Tensor,
    is_first: bool = True,
    is_last: bool = True,
    num_layers: int = 1,
) -> _ShardWrapper:
    """Build a _ShardWrapper bypassing _find_layers using manual wiring."""
    wrapper_module = nn.Module.__new__(nn.Module)
    nn.Module.__init__(wrapper_module)

    wrapper_module.layers = nn.ModuleList([_FakeLayer() for _ in range(num_layers)])
    if is_first:
        wrapper_module.embed_tokens = _FakeEmbedding()
    if is_last:
        wrapper_module.norm = nn.LayerNorm(HIDDEN)
        # LM head always produces `fixed_logits` shape (vocab,) broadcast to (B, L, vocab)
        wrapper_module.lm_head = _FakeLMHead(fixed_logits)

    sw = object.__new__(_ShardWrapper)
    sw._torch = torch
    sw.is_first = is_first
    sw.is_last = is_last
    sw._module = wrapper_module
    return sw


def _build_manager(
    fixed_logits: torch.Tensor,
    shard_index: int = 0,
    num_shards: int = 1,
    eos_id: int | None = None,
    vocab_size: int = VOCAB,
) -> ShardManager:
    """Return a ShardManager whose .model is a _ShardWrapper using _FakeLMHead."""
    cfg = _make_config(shard_index=shard_index, num_shards=num_shards)
    mgr = ShardManager(cfg)

    is_first = shard_index == 0
    is_last = shard_index == num_shards - 1
    mgr.model = _build_shard_wrapper(fixed_logits, is_first=is_first, is_last=is_last)
    mgr.layer_slice = (0, 1)

    if is_first:
        tok = MagicMock()
        tok.encode.side_effect = lambda text, **kw: (
            torch.tensor([[1, 2, 3]]) if kw.get("return_tensors") == "pt" else [1, 2, 3]
        )
        tok.decode.return_value = "hello world"
        tok.eos_token_id = eos_id
        tok.pad_token_id = 0
        mgr.tokenizer = tok

    return mgr


# ─────────────────────────────────────────────────────────────────────────────
# _top_p_filter
# ─────────────────────────────────────────────────────────────────────────────


class TestTopPFilter:
    def test_top_p_filter_keeps_min_one_token(self):
        """Even with top_p=0.0 the top token must survive."""
        probs = torch.tensor([[0.9, 0.05, 0.03, 0.02]])
        result = ShardManager._top_p_filter(probs, top_p=0.0)
        # At least one token must have non-zero probability.
        assert (result > 0).any(), "All tokens were zeroed – must keep at least one."

    def test_top_p_filter_removes_low_prob_tokens(self):
        """With top_p=0.9 and a dominant first token, lower tokens should be zeroed."""
        probs = torch.tensor([[0.91, 0.05, 0.03, 0.01]])
        result = ShardManager._top_p_filter(probs, top_p=0.9)
        # The 0.91 token covers the nucleus; remaining tokens should be zeroed.
        # Index 0 survives (cumsum starts at 0, then jumps past 0.9).
        assert result[0, 0].item() > 0.0, "Top token should survive."
        # Tokens with very low prob that don't contribute to the top-p should be zero.
        assert result[0, -1].item() == 0.0, "Last low-prob token should be zeroed."

    def test_top_p_filter_keeps_all_tokens_with_top_p_1(self):
        """top_p=1.0 should keep all tokens (full distribution)."""
        probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        result = ShardManager._top_p_filter(probs, top_p=1.0)
        assert (result > 0).sum().item() == 4

    def test_top_p_filter_shape_preserved(self):
        """Output shape must match input shape."""
        probs = torch.rand(3, 100)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        result = ShardManager._top_p_filter(probs, top_p=0.9)
        assert result.shape == probs.shape


# ─────────────────────────────────────────────────────────────────────────────
# _sample_token
# ─────────────────────────────────────────────────────────────────────────────


class TestSampleToken:
    def test_sample_token_shape(self):
        """[batch, vocab] logits → [batch] token IDs."""
        batch, vocab = 5, VOCAB
        logits = torch.randn(batch, vocab)
        result = ShardManager._sample_token(logits, temperature=1.0, top_p=0.95)
        assert result.shape == (batch,)

    def test_sample_token_valid_range(self):
        """Sampled token IDs must be in [0, vocab)."""
        logits = torch.randn(4, VOCAB)
        ids = ShardManager._sample_token(logits, temperature=1.0, top_p=0.95)
        assert (ids >= 0).all()
        assert (ids < VOCAB).all()

    def test_temperature_zero_point_one_is_near_greedy(self):
        """Very low temperature should almost always pick the argmax token."""
        vocab = 50
        logits = torch.zeros(1, vocab)
        greedy_idx = 7
        logits[0, greedy_idx] = 20.0  # dominant token
        counts: dict[int, int] = {}
        for _ in range(50):
            tid = int(ShardManager._sample_token(logits, temperature=0.1, top_p=0.95)[0].item())
            counts[tid] = counts.get(tid, 0) + 1
        # The greedy token should win the vast majority of the time.
        assert counts.get(greedy_idx, 0) >= 45, (
            f"Expected greedy token {greedy_idx} to win ≥45/50 times; got {counts}"
        )

    def test_sample_token_single_row(self):
        """Works for a single-row batch (the generate() single-prompt path)."""
        logits = torch.randn(1, VOCAB)
        result = ShardManager._sample_token(logits, temperature=1.0, top_p=0.95)
        assert result.shape == (1,)


# ─────────────────────────────────────────────────────────────────────────────
# generate()
# ─────────────────────────────────────────────────────────────────────────────


class TestGenerate:
    def _uniform_logits(self) -> torch.Tensor:
        """Uniform logits – all tokens equally likely."""
        return torch.zeros(VOCAB)

    def test_generate_returns_string(self):
        """generate() on shard 0 (which is also the last shard) returns a str."""
        fixed = self._uniform_logits()
        mgr = _build_manager(fixed, shard_index=0, num_shards=1)
        result = mgr.generate("hello", max_tokens=3)
        assert isinstance(result, str)

    def test_generate_intermediate_shard_returns_none(self):
        """An intermediate shard returns None."""
        fixed = self._uniform_logits()
        mgr = _build_manager(fixed, shard_index=1, num_shards=3)
        result = mgr.generate("hello", max_tokens=3)
        assert result is None

    def test_generate_stops_at_eos(self):
        """generate() stops early when the model samples the EOS token."""
        eos = 5
        # Logits heavily favour token 5 (EOS).
        fixed = torch.full((VOCAB,), -100.0)
        fixed[eos] = 100.0

        cfg = _make_config(shard_index=0, num_shards=1)
        mgr = ShardManager(cfg)
        mgr.model = _build_shard_wrapper(fixed, is_first=True, is_last=True)
        mgr.layer_slice = (0, 1)

        # Patch the tokenizer so decode tracks how many tokens were decoded.
        decoded_calls: list[list[int]] = []

        tok = MagicMock()
        tok.encode.side_effect = lambda text, **kw: (
            torch.tensor([[1, 2]]) if kw.get("return_tensors") == "pt" else [1, 2]
        )

        def _decode(ids, **kw):
            decoded_calls.append(list(ids))
            return "stopped"

        tok.decode.side_effect = _decode
        tok.eos_token_id = eos
        tok.pad_token_id = 0
        mgr.tokenizer = tok

        result = mgr.generate("hi", max_tokens=20)

        # Should have stopped immediately after the first sampled EOS token.
        assert result == "stopped"
        assert len(decoded_calls) == 1
        # Only one token (the EOS) should have been decoded.
        assert decoded_calls[0] == [eos]

    def test_generate_respects_max_tokens(self):
        """generate() produces at most max_tokens tokens (no EOS in vocab)."""
        # No EOS token defined → runs until max_tokens.
        fixed = self._uniform_logits()
        cfg = _make_config(shard_index=0, num_shards=1)
        mgr = ShardManager(cfg)
        mgr.model = _build_shard_wrapper(fixed, is_first=True, is_last=True)
        mgr.layer_slice = (0, 1)

        generated_counts: list[int] = []

        tok = MagicMock()
        tok.encode.side_effect = lambda text, **kw: (
            torch.tensor([[1]]) if kw.get("return_tensors") == "pt" else [1]
        )

        def _decode(ids, **kw):
            generated_counts.append(len(ids))
            return " ".join(str(i) for i in ids)

        tok.decode.side_effect = _decode
        tok.eos_token_id = None
        tok.pad_token_id = 0
        mgr.tokenizer = tok

        mgr.generate("x", max_tokens=5)
        assert generated_counts[-1] == 5


# ─────────────────────────────────────────────────────────────────────────────
# generate_batch()
# ─────────────────────────────────────────────────────────────────────────────


class TestGenerateBatch:
    def _build_batch_manager(self, eos_id: int | None = None) -> ShardManager:
        fixed = torch.zeros(VOCAB)
        cfg = _make_config(shard_index=0, num_shards=1)
        mgr = ShardManager(cfg)
        mgr.model = _build_shard_wrapper(fixed, is_first=True, is_last=True)
        mgr.layer_slice = (0, 1)

        tok = MagicMock()
        # Each prompt encodes to a different-length list so we exercise padding.
        encode_map: dict[str, list[int]] = {
            "a": [1],
            "ab": [1, 2],
            "abc": [1, 2, 3],
        }
        tok.encode.side_effect = lambda text, **kw: encode_map.get(text, [1])
        tok.decode.side_effect = lambda ids, **kw: "out"
        tok.eos_token_id = eos_id
        tok.pad_token_id = 0
        mgr.tokenizer = tok
        return mgr

    def test_generate_batch_returns_list(self):
        """generate_batch() returns a list of strings (one per prompt)."""
        mgr = self._build_batch_manager()
        results = mgr.generate_batch(["a", "ab", "abc"], max_tokens=2)
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)

    def test_generate_batch_intermediate_shard_returns_nones(self):
        """Intermediate shard returns list of None, one per prompt."""
        fixed = torch.zeros(VOCAB)
        mgr = _build_manager(fixed, shard_index=1, num_shards=3)
        results = mgr.generate_batch(["x", "y"], max_tokens=2)
        assert results == [None, None]

    def test_generate_batch_stops_when_all_finished(self):
        """When all sequences hit EOS the loop terminates before max_tokens."""
        eos = 3
        # Force the model to always sample the EOS token.
        fixed = torch.full((VOCAB,), -100.0)
        fixed[eos] = 100.0

        cfg = _make_config(shard_index=0, num_shards=1)
        mgr = ShardManager(cfg)
        mgr.model = _build_shard_wrapper(fixed, is_first=True, is_last=True)
        mgr.layer_slice = (0, 1)

        step_counts: list[int] = [0]

        tok = MagicMock()
        tok.encode.side_effect = lambda text, **kw: [1, 2]
        tok.decode.side_effect = lambda ids, **kw: "done"
        tok.eos_token_id = eos
        tok.pad_token_id = 0
        mgr.tokenizer = tok

        # Monkeypatch _sample_token to count calls.
        original_sample = ShardManager._sample_token

        def counting_sample(logits, temperature, top_p):
            step_counts[0] += 1
            return original_sample(logits, temperature, top_p)

        with patch.object(ShardManager, "_sample_token", staticmethod(counting_sample)):
            mgr.generate_batch(["p1", "p2", "p3"], max_tokens=10)

        # All three sequences should have stopped after the first step (EOS).
        assert step_counts[0] == 1, (
            f"Expected 1 sampling step (immediate EOS), got {step_counts[0]}"
        )

    def test_generate_batch_length_matches_prompts(self):
        """Output list length must always equal the number of input prompts."""
        mgr = self._build_batch_manager()
        for n in (1, 2, 5):
            prompts = ["x"] * n
            results = mgr.generate_batch(prompts, max_tokens=2)
            assert len(results) == n


# ─────────────────────────────────────────────────────────────────────────────
# _extract_shard / _find_layers — layer discovery
# ─────────────────────────────────────────────────────────────────────────────


def _make_fake_full_model(layers_attr_path: str, num_layers: int = 4):
    """
    Build a minimal nn.Module hierarchy where the transformer layer list lives
    at the given dotted attribute path (e.g. ``"model.layers"`` or
    ``"transformer.h"``).

    Uses only ``add_module()`` / ``setattr()`` – never touches ``.__dict__``
    directly – to avoid PyTorch's ``KeyError: attribute already exists`` guard.
    """

    class _FakeFullModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

    full = _FakeFullModel()
    layer_list = nn.ModuleList([_FakeLayer() for _ in range(num_layers)])

    parts = layers_attr_path.split(".")

    if len(parts) == 1:
        # e.g. "layers" directly on full_model
        full.add_module(parts[0], layer_list)

    elif len(parts) == 2:
        parent_name, child_name = parts
        parent = nn.Module()
        parent.add_module(child_name, layer_list)
        parent.add_module("embed_tokens", _FakeEmbedding())
        parent.add_module("norm", nn.LayerNorm(HIDDEN))
        full.add_module(parent_name, parent)

    elif len(parts) == 3:
        grandparent_name, parent_name, child_name = parts
        leaf = nn.Module()
        leaf.add_module(child_name, layer_list)
        mid = nn.Module()
        mid.add_module(parent_name, leaf)
        mid.add_module("embed_tokens", _FakeEmbedding())
        mid.add_module("norm", nn.LayerNorm(HIDDEN))
        full.add_module(grandparent_name, mid)

    else:
        raise ValueError(f"Unsupported path depth: {layers_attr_path!r}")

    return full


class TestLayerDiscovery:
    def _discover(self, path: str, num_layers: int = 4) -> nn.ModuleList:
        from node.shard_manager import _find_layers

        full = _make_fake_full_model(path, num_layers=num_layers)
        return _find_layers(full)

    def test_layer_discovery_mistral_naming(self):
        """model.layers path (LLaMA / Mistral / Qwen) is resolved correctly."""
        layers = self._discover("model.layers", num_layers=4)
        assert isinstance(layers, nn.ModuleList)
        assert len(layers) == 4

    def test_layer_discovery_gpt2_naming(self):
        """transformer.h path (GPT-2) is resolved correctly."""
        layers = self._discover("transformer.h", num_layers=6)
        assert isinstance(layers, nn.ModuleList)
        assert len(layers) == 6

    def test_layer_discovery_bart_naming(self):
        """model.decoder.layers path (BART / T5) is resolved correctly."""
        layers = self._discover("model.decoder.layers", num_layers=3)
        assert isinstance(layers, nn.ModuleList)
        assert len(layers) == 3

    def test_layer_discovery_pythia_naming(self):
        """gpt_neox.layers path (Pythia / GPT-NeoX) is resolved correctly."""
        layers = self._discover("gpt_neox.layers", num_layers=2)
        assert isinstance(layers, nn.ModuleList)
        assert len(layers) == 2

    def test_layer_discovery_fallback_modulelist(self):
        """
        Fallback: a ModuleList of modules with ``self_attn`` attributes is found
        even when no known attribute path matches.
        """

        class _UnknownLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = nn.Identity()

            def forward(self, x):
                return x

        class _WeirdModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
                # Use an unconventional attribute name.
                self.crazy_blocks = nn.ModuleList([_UnknownLayer() for _ in range(5)])

        from node.shard_manager import _find_layers

        weird = _WeirdModel()
        layers = _find_layers(weird)
        assert isinstance(layers, nn.ModuleList)
        assert len(layers) == 5

    def test_layer_discovery_raises_on_unknown_model(self):
        """_find_layers raises LayerDiscoveryError if no layer list can be found."""
        from node.shard_manager import LayerDiscoveryError, _find_layers

        class _NoLayers(nn.Module):
            def __init__(self):
                super().__init__()
                self.some_linear = nn.Linear(4, 4)

        with pytest.raises(LayerDiscoveryError):
            _find_layers(_NoLayers())
