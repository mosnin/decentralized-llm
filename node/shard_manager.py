"""
Manages downloading, loading, and serving a slice of transformer layers.

The model is split into `num_shards` equal-sized chunks of transformer blocks.
Each node loads exactly one shard. Activations flow through nodes in order:

  shard 0 → shard 1 → shard 2 → ... → shard N-1

Shard 0 additionally runs the embedding layer.
Shard N-1 additionally runs the LM head / final layer norm.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


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

    def generate(self, prompt: str, max_tokens: int) -> str:
        """
        Generate text for a single prompt.

        This is the single-prompt entry point used by ``generate_batch`` and
        by the autoregressive decoding loop in ``node/server.py``.
        """
        raise NotImplementedError(
            "generate() must be implemented by the concrete shard wrapper or subclass."
        )

    def generate_batch(self, prompts: list[str], max_tokens: list[int]) -> list[str]:
        """
        Generate text for a batch of prompts that share the same model.

        Each (prompt, max_tokens) pair is processed sequentially by delegating
        to ``generate()``.

        # TODO: Replace the sequential loop with true batched forward passes so
        #       all prompts in the batch share a single GPU kernel launch.  This
        #       requires padding inputs to the same sequence length and running a
        #       batched forward pass through the shard layers, which is the
        #       standard approach used by vLLM and similar systems.
        """
        return [self.generate(prompt, n) for prompt, n in zip(prompts, max_tokens)]

    # ──────────────────── private helpers ────────────────────────────────────

    def _compute_slice(self, total_layers: int) -> tuple[int, int]:
        base = total_layers // self.config.num_shards
        remainder = total_layers % self.config.num_shards
        start = self.config.shard_index * base + min(self.config.shard_index, remainder)
        end = start + base + (1 if self.config.shard_index < remainder else 0)
        return start, end

    @staticmethod
    def _get_total_layers(cfg) -> int:
        # Works for LLaMA, Mistral, Falcon, GPT-NeoX families
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

        model = full_model.model if hasattr(full_model, "model") else full_model

        for attr in ("layers", "h", "blocks"):
            if hasattr(model, attr):
                all_layers = getattr(model, attr)
                break
        else:
            raise ValueError("Cannot find transformer layer list in model")

        wrapper.layers = _torch.nn.ModuleList(list(all_layers)[start:end])

        if is_first:
            wrapper.embed_tokens = (
                model.embed_tokens if hasattr(model, "embed_tokens") else model.wte
            )
        if is_last:
            wrapper.norm = model.norm if hasattr(model, "norm") else model.ln_f
            wrapper.lm_head = full_model.lm_head

        self._module = wrapper

    # Delegate device/eval to inner module
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
