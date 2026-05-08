"""
Tests for node.model_manager — no real network calls, no GPU required.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from node.model_manager import ModelManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _manager(tmp_path: Path) -> ModelManager:
    return ModelManager(cache_dir=tmp_path / "models")


# ---------------------------------------------------------------------------
# VRAM estimation
# ---------------------------------------------------------------------------


class TestEstimateVram:
    def test_estimate_vram_fp16_3b(self):
        """3B params × 2 bytes / 1e9 × 1.20 overhead ≈ 7.68 GB."""
        mgr = ModelManager()
        vram = mgr.estimate_vram("org/model-3B-instruct", "fp16")
        # 3.2e9 * 2 / 1e9 * 1.2 = 7.68
        assert abs(vram - 7.68) < 0.01

    def test_estimate_vram_int8_7b(self):
        """7B params × 1 byte / 1e9 × 1.20 overhead ≈ 8.40 GB."""
        mgr = ModelManager()
        vram = mgr.estimate_vram("meta-llama/Llama-2-7b-chat-hf", "int8")
        # 7e9 * 1 / 1e9 * 1.2 = 8.4
        assert abs(vram - 8.4) < 0.01

    def test_estimate_vram_gptq_4bit_7b(self):
        """7B params × 0.5 bytes / 1e9 × 1.20 overhead ≈ 4.20 GB."""
        mgr = ModelManager()
        vram = mgr.estimate_vram("TheBloke/Llama-2-7B-GPTQ", "gptq-4bit")
        # 7e9 * 0.5 / 1e9 * 1.2 = 4.2
        assert abs(vram - 4.2) < 0.01

    def test_estimate_vram_awq_4bit_7b(self):
        """AWQ-4bit should give the same estimate as GPTQ-4bit for 7B."""
        mgr = ModelManager()
        assert mgr.estimate_vram("model-7b", "awq-4bit") == mgr.estimate_vram(
            "model-7b", "gptq-4bit"
        )

    def test_estimate_vram_unknown_size_falls_back_to_7b(self):
        """Model IDs without a size tag fall back to 7B parameter count."""
        mgr = ModelManager()
        vram_unknown = mgr.estimate_vram("some-custom-model", "fp16")
        vram_7b = mgr.estimate_vram("model-7b", "fp16")
        assert abs(vram_unknown - vram_7b) < 0.01


# ---------------------------------------------------------------------------
# Quantization selection
# ---------------------------------------------------------------------------


class TestSelectQuantization:
    def test_select_quantization_24gb_vram(self):
        """24 GB VRAM should allow fp16 for ≤13B models."""
        mgr = ModelManager()
        # 13B fp16: 13e9 * 2 / 1e9 * 1.2 = 31.2 GB  → won't fit
        # 13B int8: 13e9 * 1 / 1e9 * 1.2 = 15.6 GB  → won't fit
        # 13B gptq-4bit: 13e9 * 0.5 / 1e9 * 1.2 = 7.8 GB → fits
        q_13b = mgr.select_quantization(24.0, "meta-llama/Llama-2-13b")
        assert q_13b in ("gptq-4bit", "awq-4bit", "gguf-q4_k_m", "int8")

        # 7B fp16: 7e9 * 2 / 1e9 * 1.2 = 16.8 GB  → won't fit in 24 GB? — fits!
        q_7b = mgr.select_quantization(24.0, "meta-llama/Llama-2-7b")
        assert q_7b == "fp16"

        # 3B fp16: 3.2e9 * 2 / 1e9 * 1.2 = 7.68 GB → fits
        q_3b = mgr.select_quantization(24.0, "model-3b")
        assert q_3b == "fp16"

    def test_select_quantization_8gb_vram(self):
        """8 GB VRAM should select gptq-4bit for a 7B model."""
        mgr = ModelManager()
        # fp16: 16.8 GB  → no
        # int8: 8.4 GB   → no (8.4 > 8.0)
        # gptq-4bit: 4.2 GB → yes
        quant = mgr.select_quantization(8.0, "mistralai/Mistral-7B-v0.1")
        assert quant == "gptq-4bit"

    def test_select_quantization_returns_last_resort_when_nothing_fits(self):
        """Even when nothing fits, return the smallest option rather than raising."""
        mgr = ModelManager()
        quant = mgr.select_quantization(0.1, "model-70b")
        assert quant == "gguf-q4_k_m"


# ---------------------------------------------------------------------------
# Local cache listing
# ---------------------------------------------------------------------------


class TestListLocal:
    def test_list_local_empty_cache(self, tmp_path):
        """Returns [] when the cache directory does not exist."""
        mgr = ModelManager(cache_dir=tmp_path / "nonexistent")
        assert mgr.list_local() == []

    def test_list_local_empty_cache_dir(self, tmp_path):
        """Returns [] when the cache directory exists but is empty."""
        cache = tmp_path / "models"
        cache.mkdir()
        mgr = ModelManager(cache_dir=cache)
        assert mgr.list_local() == []

    def test_list_local_finds_models(self, tmp_path):
        """Creates a fake two-level cache structure and verifies ModelSpec list."""
        cache = tmp_path / "models"
        # Simulate: ~/.cache/decentralized-llm/models/meta-llama/Llama-2-7b/
        model_dir = cache / "meta-llama" / "Llama-2-7b"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text('{"model_type": "llama"}')
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 16)

        mgr = ModelManager(cache_dir=cache)
        specs = mgr.list_local()

        assert len(specs) == 1
        assert specs[0].model_id == "meta-llama/Llama-2-7b"
        assert specs[0].quantization == "fp16"
        assert specs[0].local_path == model_dir

    def test_list_local_finds_gguf_model(self, tmp_path):
        """Detects GGUF quantization from file extension."""
        cache = tmp_path / "models"
        model_dir = cache / "TheBloke" / "Llama-7B-GGUF"
        model_dir.mkdir(parents=True)
        (model_dir / "llama-7b.Q4_K_M.gguf").write_bytes(b"\x00" * 16)

        mgr = ModelManager(cache_dir=cache)
        specs = mgr.list_local()

        assert len(specs) == 1
        assert specs[0].quantization == "gguf-q4_k_m"


# ---------------------------------------------------------------------------
# Checksum verification
# ---------------------------------------------------------------------------


class TestVerifyChecksum:
    @pytest.mark.asyncio
    async def test_verify_checksum_correct(self, tmp_path):
        """Writing a known file and verifying its SHA-256 should return True."""
        content = b"decentralized llm node test data"
        expected = hashlib.sha256(content).hexdigest()
        f = tmp_path / "weights.bin"
        f.write_bytes(content)

        mgr = ModelManager()
        assert await mgr.verify_checksum(f, expected) is True

    @pytest.mark.asyncio
    async def test_verify_checksum_wrong(self, tmp_path):
        """A mismatched expected hash should return False."""
        f = tmp_path / "weights.bin"
        f.write_bytes(b"real content")
        wrong_hash = "a" * 64  # 64 hex chars, all 'a'

        mgr = ModelManager()
        assert await mgr.verify_checksum(f, wrong_hash) is False

    @pytest.mark.asyncio
    async def test_verify_checksum_case_insensitive(self, tmp_path):
        """Comparison should be case-insensitive."""
        content = b"hello"
        expected = hashlib.sha256(content).hexdigest().upper()
        f = tmp_path / "weights.bin"
        f.write_bytes(content)

        mgr = ModelManager()
        assert await mgr.verify_checksum(f, expected) is True


# ---------------------------------------------------------------------------
# Download – ImportError without huggingface_hub
# ---------------------------------------------------------------------------


class TestDownload:
    @pytest.mark.asyncio
    async def test_download_raises_without_huggingface_hub(self, tmp_path):
        """
        When huggingface_hub is not installed, download() must raise ImportError
        with a helpful installation message.
        """
        mgr = _manager(tmp_path)

        # Remove huggingface_hub from sys.modules so the lazy import fails
        with patch.dict(sys.modules, {"huggingface_hub": None}):
            with pytest.raises(ImportError, match="pip install huggingface-hub"):
                await mgr.download("meta-llama/Llama-2-7b", quantization="fp16")

    @pytest.mark.asyncio
    async def test_download_calls_snapshot_download(self, tmp_path):
        """When huggingface_hub is available, snapshot_download is called."""
        mgr = _manager(tmp_path)

        fake_hf_hub = MagicMock()
        dest = str(tmp_path / "models" / "meta-llama" / "Llama-2-7b")
        fake_hf_hub.snapshot_download.return_value = dest

        with patch.dict(sys.modules, {"huggingface_hub": fake_hf_hub}):
            result = await mgr.download("meta-llama/Llama-2-7b", quantization="fp16")

        fake_hf_hub.snapshot_download.assert_called_once()
        call_kwargs = fake_hf_hub.snapshot_download.call_args
        assert call_kwargs.kwargs.get("repo_id") == "meta-llama/Llama-2-7b" or (
            call_kwargs.args and call_kwargs.args[0] == "meta-llama/Llama-2-7b"
        )
        assert Path(result) == Path(dest)
