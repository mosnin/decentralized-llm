"""
Manages downloading, loading, and serving a slice of transformer layers.

The model is split into `num_shards` equal-sized chunks of transformer blocks.
Each node loads exactly one shard. Activations flow through nodes in order:

  shard 0 → shard 1 → shard 2 → ... → shard N-1

Shard 0 additionally runs the embedding layer.
Shard N-1 additionally runs the LM head / final layer norm.
"""

import logging
import operator
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Architecture registry: maps HuggingFace model class names to dotted
# attribute paths for their transformer layer lists, embedding tables, and
# LM heads.  Add new architectures here without touching any other code.
# ---------------------------------------------------------------------------

LAYER_ATTR_MAP: dict[str, str] = {
    "LlamaForCausalLM": "model.layers",
    "MistralForCausalLM": "model.layers",
    "MixtralForCausalLM": "model.layers",
    "GemmaForCausalLM": "model.layers",
    "Gemma2ForCausalLM": "model.layers",
    "Qwen2ForCausalLM": "model.layers",
    "PhiForCausalLM": "model.layers",
    "Phi3ForCausalLM": "model.layers",
    "FalconForCausalLM": "transformer.h",
    "GPT2LMHeadModel": "transformer.h",
    "GPTNeoXForCausalLM": "gpt_neox.layers",
    "GPTJForCausalLM": "transformer.h",
    "BloomForCausalLM": "transformer.h",
    "OPTForCausalLM": "model.decoder.layers",
    "BartForCausalLM": "model.decoder.layers",
    "MPTForCausalLM": "transformer.blocks",
    "RWForCausalLM": "transformer.h",
    "InternLMForCausalLM": "model.layers",
    "BaichuanForCausalLM": "model.layers",
}

EMBED_ATTR_MAP: dict[str, str] = {
    "LlamaForCausalLM": "model.embed_tokens",
    "MistralForCausalLM": "model.embed_tokens",
    "MixtralForCausalLM": "model.embed_tokens",
    "GemmaForCausalLM": "model.embed_tokens",
    "Gemma2ForCausalLM": "model.embed_tokens",
    "Qwen2ForCausalLM": "model.embed_tokens",
    "PhiForCausalLM": "model.embed_tokens",
    "Phi3ForCausalLM": "model.embed_tokens",
    "FalconForCausalLM": "transformer.word_embeddings",
    "GPT2LMHeadModel": "transformer.wte",
    "GPTNeoXForCausalLM": "gpt_neox.embed_in",
    "GPTJForCausalLM": "transformer.wte",
    "BloomForCausalLM": "transformer.word_embeddings",
    "OPTForCausalLM": "model.decoder.embed_tokens",
    "BartForCausalLM": "model.decoder.embed_tokens",
    "MPTForCausalLM": "transformer.wte",
    "RWForCausalLM": "transformer.word_embeddings",
    "InternLMForCausalLM": "model.embed_tokens",
    "BaichuanForCausalLM": "model.embed_tokens",
}

LM_HEAD_ATTR_MAP: dict[str, str] = {
    "LlamaForCausalLM": "lm_head",
    "MistralForCausalLM": "lm_head",
    "MixtralForCausalLM": "lm_head",
    "GemmaForCausalLM": "lm_head",
    "Gemma2ForCausalLM": "lm_head",
    "Qwen2ForCausalLM": "lm_head",
    "PhiForCausalLM": "lm_head",
    "Phi3ForCausalLM": "lm_head",
    "FalconForCausalLM": "lm_head",
    "GPT2LMHeadModel": "lm_head",
    "GPTNeoXForCausalLM": "embed_out",
    "GPTJForCausalLM": "lm_head",
    "BloomForCausalLM": "lm_head",
    "OPTForCausalLM": "lm_head",
    "BartForCausalLM": "lm_head",
    "MPTForCausalLM": "lm_head",
    "RWForCausalLM": "lm_head",
    "InternLMForCausalLM": "lm_head",
    "BaichuanForCausalLM": "lm_head",
}


class LayerDiscoveryError(ValueError):
    """Raised when the transformer layer list cannot be found in a model."""


# ---------------------------------------------------------------------------
# Module-level discovery helpers (testable without instantiating ShardManager)
# ---------------------------------------------------------------------------


def _find_layers(model) -> "torch.nn.ModuleList":  # type: ignore[name-defined]  # noqa: F821
    """
    Return the nn.ModuleList that contains the main transformer blocks.

    Resolution order:
    1. LAYER_ATTR_MAP lookup by class name (fast, deterministic).
    2. Scan named_modules() for the first ModuleList with >2 elements whose
       members have a self-attention-like attribute (handles novel architectures).
    3. Final fallback: the largest ModuleList by element count.

    Raises LayerDiscoveryError if nothing is found.
    """
    import torch.nn as nn

    class_name = type(model).__name__

    # 1. Explicit registry lookup
    if class_name in LAYER_ATTR_MAP:
        try:
            layers = operator.attrgetter(LAYER_ATTR_MAP[class_name])(model)
            if isinstance(layers, nn.ModuleList):
                return layers
        except AttributeError:
            pass  # fall through to scanning

    # 2. Scan for ModuleList whose children look like transformer blocks
    _ATTN_ATTRS = ("self_attn", "attention", "attn")
    for _name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 2:
            first = next(iter(module), None)
            if first is not None and any(hasattr(first, a) for a in _ATTN_ATTRS):
                return module

    # 3. Largest ModuleList fallback
    best: nn.ModuleList | None = None
    for _name, module in model.named_modules():
        if isinstance(module, nn.ModuleList):
            if best is None or len(module) > len(best):
                best = module
    if best is not None and len(best) > 0:
        return best

    raise LayerDiscoveryError(
        f"Cannot find transformer layer list in model of type '{class_name}'. "
        "Add it to LAYER_ATTR_MAP in node/shard_manager.py."
    )


def _find_embedding(model):
    """Return the token-embedding module for *model*."""
    import torch.nn as nn

    class_name = type(model).__name__
    if class_name in EMBED_ATTR_MAP:
        try:
            emb = operator.attrgetter(EMBED_ATTR_MAP[class_name])(model)
            if isinstance(emb, nn.Module):
                return emb
        except AttributeError:
            pass

    # Generic fallbacks
    for candidate in (
        "model.embed_tokens",
        "transformer.wte",
        "transformer.word_embeddings",
        "gpt_neox.embed_in",
        "model.decoder.embed_tokens",
    ):
        try:
            emb = operator.attrgetter(candidate)(model)
            if isinstance(emb, nn.Module):
                return emb
        except AttributeError:
            continue

    raise LayerDiscoveryError(f"Cannot find embedding layer in model of type '{class_name}'.")


def _find_lm_head(model):
    """Return the LM head module for *model*."""
    import torch.nn as nn

    class_name = type(model).__name__
    if class_name in LM_HEAD_ATTR_MAP:
        try:
            head = operator.attrgetter(LM_HEAD_ATTR_MAP[class_name])(model)
            if isinstance(head, nn.Module):
                return head
        except AttributeError:
            pass

    # Generic fallbacks
    for candidate in ("lm_head", "embed_out", "output"):
        try:
            head = operator.attrgetter(candidate)(model)
            if isinstance(head, nn.Module):
                return head
        except AttributeError:
            continue

    raise LayerDiscoveryError(f"Cannot find LM head in model of type '{class_name}'.")


class ShardManager:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.layer_slice: tuple[int, int] | None = None

    def load(self) -> None:
        """Download (if needed) and load this node's model shard into GPU memory."""
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        model_cfg = AutoConfig.from_pretrained(self.config.model_name, cache_dir=cache_dir)

        total_layers = self._get_total_layers(model_cfg)
        self.layer_slice = self._compute_slice(total_layers)

        logger.info(
            "Loading shard %d/%d: layers %d–%d of %s",
            self.config.shard_index,
            self.config.num_shards,
            *self.layer_slice,
            self.config.model_name,
        )

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.dtype, torch.float16)

        # Load full model weights then extract the layers we need.
        # ExLlamaV2 backend (USE_EXLLAMA=1) loads only the assigned layer range
        # natively, avoiding peak memory = full model on CPU.
        full_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
            device_map="cpu",  # load to CPU first, then slice
        )

        self.model = self._extract_shard(full_model, torch)
        del full_model  # free the rest of the weights

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)
        self.model.eval()

        if self.config.shard_index == 0:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name, cache_dir=cache_dir
            )

        logger.info("Shard %d loaded on %s", self.config.shard_index, device)

    def forward(self, hidden_states):
        """
        Run a forward pass through this shard's layers.

        Input/output: float tensor of shape (batch, seq_len, hidden_dim)
        """
        import torch

        with torch.no_grad():
            return self.model(hidden_states)

    def embed(self, input_ids):
        """Embed token IDs → hidden states. Only valid on shard 0."""
        import torch

        assert self.config.shard_index == 0, "embed() only valid on shard 0"
        with torch.no_grad():
            return self.model.embed(input_ids)

    def decode(self, hidden_states):
        """Run LM head to get logits. Only valid on the last shard."""
        import torch

        assert self.config.shard_index == self.config.num_shards - 1
        with torch.no_grad():
            return self.model.decode(hidden_states)

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 1.0,
        top_p: float = 0.95,
    ) -> str | None:
        """
        Generate text for a single prompt.

        Returns the decoded string on the final shard, or None on intermediate shards.
        All torch imports are lazy (inside the method).
        """
        import torch

        is_first = self.config.shard_index == 0
        is_last = self.config.shard_index == self.config.num_shards - 1

        # Tokenize only on shard 0; intermediate/last shards receive activations,
        # not token IDs, via the network layer (this method handles single-node
        # stand-alone use where all shards are present in one process).
        if is_first:
            assert self.tokenizer is not None, "tokenizer not loaded on shard 0"
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)
        else:
            # For intermediate shards in a stand-alone single-process setup the
            # caller is responsible for supplying activations. When called as a
            # pure endpoint (multi-node) this path is reached but the prompt
            # argument is the serialised hidden state, not text. Here we simply
            # return None to signal that this shard does not produce text output.
            return None

        # Determine EOS token id.
        eos_id = None
        if hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
            eos_id = self.tokenizer.eos_token_id

        generated_ids: list[int] = []

        with torch.no_grad():
            # Initial embedding + forward through layers.
            hidden = self.model.embed(input_ids)
            hidden = self.model(hidden)

            for _ in range(max_tokens):
                if is_last:
                    # Decode using the last token's hidden state.
                    logits = self.model.decode(hidden)  # (1, seq_len, vocab)
                    last_logits = logits[:, -1, :]  # (1, vocab)
                    next_id_tensor = self._sample_token(last_logits, temperature, top_p)
                    next_id = int(next_id_tensor[0].item())
                else:
                    # In a single-process multi-shard scenario the caller chains
                    # shards together; this shard only runs its layers.
                    break

                generated_ids.append(next_id)

                if eos_id is not None and next_id == eos_id:
                    break

                # Append the new token and run the next step.
                new_token = torch.tensor([[next_id]], device=hidden.device)
                new_emb = self.model.embed(new_token)  # (1, 1, hidden_dim)
                # Run only the new token through layers (append to sequence).
                new_hidden = self.model(new_emb)
                # Concatenate along sequence dimension for next decode step.
                hidden = torch.cat([hidden, new_hidden], dim=1)

        if is_last:
            return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return None

    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int,
        temperature: float = 1.0,
        top_p: float = 0.95,
    ) -> list[str | None]:
        """
        Generate text for a batch of prompts.

        Pads all prompts to the same length, runs a single batched forward pass
        per autoregressive step, and samples tokens vectorised across the batch.

        Returns a list of decoded strings on the final shard, or a list of None
        on intermediate shards.
        """
        import torch

        is_first = self.config.shard_index == 0
        is_last = self.config.shard_index == self.config.num_shards - 1

        if not is_first:
            return [None] * len(prompts)

        assert self.tokenizer is not None, "tokenizer not loaded on shard 0"

        pad_id = 0
        if hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is not None:
            pad_id = self.tokenizer.pad_token_id
        elif hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
            pad_id = self.tokenizer.eos_token_id

        eos_id = None
        if hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
            eos_id = self.tokenizer.eos_token_id

        # Encode all prompts and pad to equal length.
        encoded = [self.tokenizer.encode(p) for p in prompts]
        max_prompt_len = max(len(e) for e in encoded)
        padded = [
            [pad_id] * (max_prompt_len - len(e)) + e  # left-pad
            for e in encoded
        ]

        device = next(self.model.parameters()).device
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)  # (B, L)

        batch_size = len(prompts)
        generated_ids: list[list[int]] = [[] for _ in range(batch_size)]
        # Track which sequences are still generating.
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        with torch.no_grad():
            hidden = self.model.embed(input_ids)  # (B, L, H)
            hidden = self.model(hidden)  # (B, L, H)

            for _ in range(max_tokens):
                if is_last:
                    logits = self.model.decode(hidden)  # (B, L, vocab)
                    last_logits = logits[:, -1, :]  # (B, vocab)
                    next_ids = self._sample_token(last_logits, temperature, top_p)  # (B,)
                else:
                    break

                for i in range(batch_size):
                    if not finished[i]:
                        generated_ids[i].append(int(next_ids[i].item()))

                if eos_id is not None:
                    finished = finished | (next_ids == eos_id)

                if finished.all():
                    break

                # Embed the new tokens and run forward for next step.
                new_tokens = next_ids.unsqueeze(1)  # (B, 1)
                new_emb = self.model.embed(new_tokens)  # (B, 1, H)
                new_hidden = self.model(new_emb)  # (B, 1, H)
                hidden = torch.cat([hidden, new_hidden], dim=1)  # (B, L+1, H)

        if is_last:
            return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
        return [None] * batch_size

    # ──────────────────── sampling helpers ───────────────────────────────────

    @staticmethod
    def _sample_token(
        logits_2d,
        temperature: float,
        top_p: float,
    ):
        """
        Sample one token per row from logits of shape [batch, vocab].

        Applies temperature scaling then top-p nucleus filtering before sampling.
        All operations are pure torch; import is lazy.

        Returns a 1-D tensor of shape [batch] with the sampled token IDs.
        """
        import torch

        # Temperature scaling — clamp to a small positive value to avoid /0.
        temp = max(temperature, 1e-8)
        scaled = logits_2d / temp

        probs = torch.softmax(scaled, dim=-1)
        probs = ShardManager._top_p_filter(probs, top_p)

        # Re-normalise after filtering (some mass may have been zeroed out).
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)

        next_ids = torch.multinomial(probs, num_samples=1).squeeze(1)  # (batch,)
        return next_ids

    @staticmethod
    def _top_p_filter(probs, top_p: float):
        """
        Zero out tokens whose cumulative probability exceeds *top_p*.

        Sorts probabilities in descending order, computes the cumulative sum,
        and masks tokens beyond the nucleus threshold.  Always keeps at least
        the single most-probable token so that sampling never sees an all-zero
        distribution.

        Returns a tensor of the same shape as *probs* with low-probability
        tokens zeroed out.
        """
        import torch

        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)

        # Shift cumsum right by one so the token that *pushes* cumsum past the
        # threshold is still included.  This is the standard nucleus-sampling
        # trick to keep at least one token.
        shifted = torch.roll(cum_probs, shifts=1, dims=-1)
        # Use -inf for the first position so the top-1 token is *always* kept,
        # even when top_p=0.0 (ensures the distribution is never all-zero).
        shifted[:, 0] = float("-inf")

        # Mask tokens where the shifted cumsum already exceeds top_p.
        remove_mask = shifted >= top_p
        sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)

        # Scatter back to original vocabulary ordering.
        filtered = torch.zeros_like(probs)
        filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_probs)
        return filtered

    # ──────────────────── private helpers ────────────────────────────────────

    def _compute_slice(self, total_layers: int) -> tuple[int, int]:
        base = total_layers // self.config.num_shards
        remainder = total_layers % self.config.num_shards
        start = self.config.shard_index * base + min(self.config.shard_index, remainder)
        end = start + base + (1 if self.config.shard_index < remainder else 0)
        return start, end

    @staticmethod
    def _get_total_layers(cfg) -> int:
        """Determine total layer count from a HuggingFace config object."""
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            if hasattr(cfg, attr):
                return getattr(cfg, attr)
        raise ValueError(f"Cannot determine layer count from config: {cfg}")

    def _extract_shard(self, full_model, torch):
        start, end = self.layer_slice
        is_first = self.config.shard_index == 0
        is_last = self.config.shard_index == self.config.num_shards - 1
        return _ShardWrapper(full_model, start, end, is_first, is_last, torch)


class _ShardWrapper:
    """Holds a slice of transformer blocks plus optional embed/decode layers."""

    def __init__(self, full_model, start: int, end: int, is_first: bool, is_last: bool, torch):
        import torch as _torch  # noqa: PLC0415

        self._torch = _torch
        self.is_first = is_first
        self.is_last = is_last

        # Wrap as a proper nn.Module so .to(device) / .eval() work
        class _Wrapper(_torch.nn.Module):
            pass

        wrapper = _Wrapper()

        # Use the module-level robust layer discovery helpers.
        all_layers = _find_layers(full_model)
        wrapper.layers = _torch.nn.ModuleList(list(all_layers)[start:end])

        if is_first:
            wrapper.embed_tokens = _find_embedding(full_model)

        if is_last:
            # Final layer norm: try common attribute names on the inner model
            inner = full_model.model if hasattr(full_model, "model") else full_model
            for norm_attr in ("norm", "ln_f", "final_layer_norm"):
                if hasattr(inner, norm_attr):
                    wrapper.norm = getattr(inner, norm_attr)
                    break
            else:
                raise LayerDiscoveryError(
                    f"Cannot find final layer norm in model of type '{type(full_model).__name__}'."
                )
            wrapper.lm_head = _find_lm_head(full_model)

        self._module = wrapper

    # ── device / eval delegation ──────────────────────────────────────────────

    def to(self, device):
        self._module = self._module.to(device)
        return self

    def eval(self):
        self._module.eval()
        return self

    def parameters(self):
        return self._module.parameters()

    def embed(self, input_ids):
        return self._module.embed_tokens(input_ids)

    def __call__(self, hidden_states):
        for layer in self._module.layers:
            out = layer(hidden_states)
            hidden_states = out[0] if isinstance(out, tuple) else out
        return hidden_states

    def decode(self, hidden_states):
        hidden_states = self._module.norm(hidden_states)
        return self._module.lm_head(hidden_states)
