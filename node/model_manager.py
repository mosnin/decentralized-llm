"""
Model manager: auto-download weights from HuggingFace, VRAM estimation,
quantization selection, and local cache listing.

All heavy imports (huggingface_hub, hashlib for large files) are lazy so
this module is importable with stdlib only.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "decentralized-llm" / "models"

# Approximate parameter counts keyed by lower-case size tag found in model_id
_PARAM_COUNTS: dict[str, float] = {
    "1b": 1.2e9,
    "3b": 3.2e9,
    "7b": 7.0e9,
    "8b": 8.0e9,
    "13b": 13.0e9,
    "70b": 70.0e9,
}

# bytes-per-parameter for each quantization scheme
_BYTES_PER_PARAM: dict[str, float] = {
    "fp16": 2.0,
    "int8": 1.0,
    "gptq-4bit": 0.5,
    "awq-4bit": 0.5,
    "gguf-q4_k_m": 0.5,
}

_KV_OVERHEAD = 1.20  # 20 % overhead for KV cache + activations

# Quantization options ordered from largest to smallest VRAM usage
_QUANT_PREFERENCE = ["fp16", "int8", "gptq-4bit", "awq-4bit", "gguf-q4_k_m"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    """Describes a model variant (precision, quantization, size)."""

    model_id: str
    revision: str = "main"
    quantization: str = "fp16"
    estimated_vram_gb: float = 0.0
    local_path: Path = field(default_factory=Path)


# ---------------------------------------------------------------------------
# ModelManager
# ---------------------------------------------------------------------------


class ModelManager:
    """
    Manages local model cache and coordinates downloads from HuggingFace.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir: Path = cache_dir or DEFAULT_CACHE_DIR

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def download(
        self,
        model_id: str,
        quantization: str = "fp16",
        revision: str = "main",
    ) -> Path:
        """
        Download *model_id* to the local cache and return the directory path.

        Uses ``huggingface_hub.snapshot_download``; raises ``ImportError``
        with installation instructions if the package is not available.
        """
        try:
            from huggingface_hub import snapshot_download  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for model downloads. "
                "Install it with: pip install huggingface-hub"
            ) from exc

        dest = self._model_dir(model_id)
        dest.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Downloading %s (quant=%s, rev=%s) → %s", model_id, quantization, revision, dest
        )

        loop = _get_event_loop()
        path = await loop.run_in_executor(
            None,
            lambda: snapshot_download(
                repo_id=model_id,
                revision=revision,
                local_dir=str(dest),
                local_dir_use_symlinks=False,
            ),
        )
        return Path(path)

    def list_local(self) -> list[ModelSpec]:
        """
        Scan the local cache directory and return specs for all models found.

        Returns an empty list if the cache directory does not exist.
        """
        if not self.cache_dir.exists():
            return []

        specs: list[ModelSpec] = []
        # Each immediate child that contains at least one file is treated as
        # a cached model.  We reconstruct an approximate model_id from the
        # two-level namespace layout: <org>/<name> maps to <cache>/<org>/<name>
        for entry in sorted(self.cache_dir.iterdir()):
            if not entry.is_dir():
                continue
            # Check for one-level entry (e.g. "gpt2") and two-level (e.g. "meta-llama/Llama-3.2-3B")
            sub_entries = list(entry.iterdir())
            has_files = any(p.is_file() for p in sub_entries)
            has_subdirs = [p for p in sub_entries if p.is_dir()]

            if has_files:
                model_id = entry.name
                quant = _detect_quantization(entry)
                vram = self.estimate_vram(model_id, quant)
                specs.append(
                    ModelSpec(
                        model_id=model_id,
                        quantization=quant,
                        estimated_vram_gb=vram,
                        local_path=entry,
                    )
                )
            else:
                for sub in sorted(has_subdirs):
                    model_id = f"{entry.name}/{sub.name}"
                    quant = _detect_quantization(sub)
                    vram = self.estimate_vram(model_id, quant)
                    specs.append(
                        ModelSpec(
                            model_id=model_id,
                            quantization=quant,
                            estimated_vram_gb=vram,
                            local_path=sub,
                        )
                    )
        return specs

    def estimate_vram(self, model_id: str, quantization: str = "fp16") -> float:
        """
        Return estimated VRAM in GB using parameter-count × dtype-size heuristics.

        Includes a 20 % overhead for KV cache and activations.
        Falls back to 7B parameters when the model size cannot be inferred.
        """
        params = _infer_param_count(model_id)
        bytes_pp = _BYTES_PER_PARAM.get(quantization, 2.0)
        raw_gb = params * bytes_pp / 1e9
        return raw_gb * _KV_OVERHEAD

    def select_quantization(self, available_vram_gb: float, model_id: str) -> str:
        """
        Return the best (highest-quality) quantization that fits in *available_vram_gb*.

        Preference order: fp16 > int8 > gptq-4bit > awq-4bit > gguf-q4_k_m
        Returns ``"gguf-q4_k_m"`` as last resort even if it doesn't fit.
        """
        for quant in _QUANT_PREFERENCE:
            if self.estimate_vram(model_id, quant) <= available_vram_gb:
                return quant
        return _QUANT_PREFERENCE[-1]

    async def verify_checksum(self, path: Path, expected_sha256: str) -> bool:
        """
        Verify the SHA-256 checksum of *path*.

        Returns ``True`` if the digest matches *expected_sha256* (hex string),
        ``False`` otherwise.  Reads the file in 1 MB chunks to avoid loading
        large weight files entirely into memory.
        """
        loop = _get_event_loop()
        actual = await loop.run_in_executor(None, lambda: _sha256_file(path))
        return actual.lower() == expected_sha256.lower()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _model_dir(self, model_id: str) -> Path:
        """Return the local cache directory for *model_id*."""
        # Replace "/" with the OS path separator so namespaced IDs nest nicely.
        return self.cache_dir / model_id.replace("/", "/")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _infer_param_count(model_id: str) -> float:
    """Infer parameter count from size tags embedded in *model_id*."""
    lower = model_id.lower()
    # Check longer tags first to avoid "1b" matching inside "13b"
    for tag in ("70b", "13b", "8b", "7b", "3b", "1b"):
        if tag in lower:
            return _PARAM_COUNTS[tag]
    # Fallback: assume 7B
    return _PARAM_COUNTS["7b"]


def _detect_quantization(model_dir: Path) -> str:
    """
    Heuristically detect the quantization format stored in *model_dir*.

    Looks for well-known file extensions / name fragments.
    """
    for f in model_dir.iterdir():
        name = f.name.lower()
        if name.endswith(".gguf") or "gguf" in name:
            return "gguf-q4_k_m"
        if "awq" in name:
            return "awq-4bit"
        if "gptq" in name:
            return "gptq-4bit"
        if name.endswith(".bin") or name.endswith(".safetensors"):
            # Peek at config if available
            continue
    config_file = model_dir / "config.json"
    if config_file.exists():
        text = config_file.read_text()
        if "awq" in text.lower():
            return "awq-4bit"
        if "gptq" in text.lower():
            return "gptq-4bit"
        if "load_in_8bit" in text or "int8" in text.lower():
            return "int8"
    return "fp16"


def _sha256_file(path: Path) -> str:
    """Compute the hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):  # 1 MB chunks
            h.update(chunk)
    return h.hexdigest()


def _get_event_loop():
    """Return the running asyncio event loop."""
    import asyncio  # noqa: PLC0415

    return asyncio.get_event_loop()
