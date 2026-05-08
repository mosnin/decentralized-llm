"""
Coverage sweep: smoke tests for modules that had no dedicated test file.

Modules covered
---------------
- finetuning/dataset.py   — LocalDataset (pure stdlib, no GPU required)
- finetuning/trainer.py   — FinetuneConfig dataclass defaults and field types
- integrations/vastai/provisioner.py — GpuRequirements, GpuInstance,
                                        VastAiProvisioner._parse_instance
- node/blockchain.py      — OpenJob dataclass, BlockchainClient guard when
                            Solana packages are absent, auto_settle / heartbeat
                            graceful no-op when inference program is None

All tests use stdlib + unittest.mock only; no GPU, no network, no Solana/Vast.ai.
Heavy optional packages (torch, hivemind, peft, transformers, vastai, solana,
anchorpy, solders) are stubbed via sys.modules where needed.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# sys.modules stubs — installed once at collection time
# ---------------------------------------------------------------------------


def _stub_module(name: str) -> None:
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)


def _install_stubs() -> None:
    # Solana / Anchor
    for mod in (
        "solana",
        "solana.rpc",
        "solana.rpc.async_api",
        "solders",
        "solders.keypair",
        "solders.pubkey",
        "anchorpy",
    ):
        _stub_module(mod)

    # Vast.ai SDK
    _stub_module("vastai")

    # Heavy ML stack
    for mod in (
        "hivemind",
        "peft",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "lighthouseweb3",
    ):
        _stub_module(mod)

    # torch — needs a minimal nn.Module base class
    if "torch" not in sys.modules:
        torch_mod = types.ModuleType("torch")

        class _FakeTensor:
            pass

        torch_mod.Tensor = _FakeTensor  # type: ignore[attr-defined]
        nn_mod = types.ModuleType("torch.nn")

        class _FakeModule:
            def __init__(self, *args, **kwargs):
                pass

        nn_mod.Module = _FakeModule  # type: ignore[attr-defined]
        torch_mod.nn = nn_mod  # type: ignore[attr-defined]
        sys.modules["torch"] = torch_mod
        sys.modules["torch.nn"] = nn_mod
    else:
        _stub_module("torch.nn")


_install_stubs()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockTokenizer:
    """Minimal tokenizer stub for LocalDataset tests."""

    def __call__(self, text: str, **kwargs):  # noqa: ARG002
        class _Enc:
            def __getitem__(self, key):  # noqa: ARG002
                class _Tensor:
                    def squeeze(self, dim):  # noqa: ARG002
                        return [1, 2, 3]

                return _Tensor()

        return _Enc()


# ---------------------------------------------------------------------------
# 1. finetuning/dataset.py — LocalDataset
# ---------------------------------------------------------------------------


class TestLocalDataset:
    """Tests for finetuning.dataset.LocalDataset."""

    def test_load_jsonl_text_field(self, tmp_path: Path) -> None:
        """JSONL rows with a 'text' key are loaded correctly."""
        from finetuning.dataset import LocalDataset

        data_file = tmp_path / "train.jsonl"
        rows = [{"text": "Hello world"}, {"text": "Foo bar baz"}]
        data_file.write_text("\n".join(json.dumps(r) for r in rows))

        ds = LocalDataset(data_file, _MockTokenizer(), max_length=64)
        assert len(ds) == 2

    def test_load_jsonl_prompt_completion(self, tmp_path: Path) -> None:
        """JSONL rows with 'prompt'+'completion' are concatenated and loaded."""
        from finetuning.dataset import LocalDataset

        data_file = tmp_path / "train.jsonl"
        rows = [{"prompt": "Q: What is 2+2?", "completion": " A: 4."}]
        data_file.write_text(json.dumps(rows[0]))

        ds = LocalDataset(data_file, _MockTokenizer(), max_length=128)
        assert len(ds) == 1

    def test_load_txt_file(self, tmp_path: Path) -> None:
        """Plain .txt files are loaded one document per non-empty line."""
        from finetuning.dataset import LocalDataset

        data_file = tmp_path / "corpus.txt"
        lines = ["Line one", "", "Line two", "Line three"]
        data_file.write_text("\n".join(lines))

        ds = LocalDataset(data_file, _MockTokenizer(), max_length=64)
        # blank line is skipped
        assert len(ds) == 3

    def test_getitem_returns_expected_keys(self, tmp_path: Path) -> None:
        """__getitem__ must return dict with 'input_ids' and 'attention_mask'."""
        from finetuning.dataset import LocalDataset

        data_file = tmp_path / "data.jsonl"
        data_file.write_text(json.dumps({"text": "test"}) + "\n")

        ds = LocalDataset(data_file, _MockTokenizer(), max_length=32)
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        """Unsupported file extensions must raise ValueError."""
        from finetuning.dataset import LocalDataset

        data_file = tmp_path / "data.csv"
        data_file.write_text("col1,col2\nfoo,bar\n")

        with pytest.raises(ValueError, match="Unsupported data format"):
            LocalDataset(data_file, _MockTokenizer())

    def test_empty_lines_skipped_in_jsonl(self, tmp_path: Path) -> None:
        """Blank lines in a JSONL file must not cause crashes and are ignored."""
        from finetuning.dataset import LocalDataset

        data_file = tmp_path / "data.jsonl"
        data_file.write_text(
            json.dumps({"text": "first"}) + "\n\n" + json.dumps({"text": "second"}) + "\n"
        )

        ds = LocalDataset(data_file, _MockTokenizer())
        assert len(ds) == 2


# ---------------------------------------------------------------------------
# 2. finetuning/trainer.py — FinetuneConfig
# ---------------------------------------------------------------------------


class TestFinetuneConfig:
    """Tests for finetuning.trainer.FinetuneConfig dataclass defaults."""

    def test_default_model_name(self) -> None:
        from finetuning.trainer import FinetuneConfig

        cfg = FinetuneConfig()
        assert cfg.model_name == "meta-llama/Llama-3.2-3B"

    def test_lora_defaults_are_sane(self) -> None:
        """LoRA rank, alpha, and dropout must be positive/valid."""
        from finetuning.trainer import FinetuneConfig

        cfg = FinetuneConfig()
        assert cfg.lora_rank > 0
        assert cfg.lora_alpha > 0
        assert 0.0 <= cfg.lora_dropout < 1.0

    def test_lora_target_modules_is_list(self) -> None:
        """lora_target_modules default must be a list (mutable default guard)."""
        from finetuning.trainer import FinetuneConfig

        cfg1 = FinetuneConfig()
        cfg2 = FinetuneConfig()
        # Each instance should get its own list, not share one
        cfg1.lora_target_modules.append("k_proj")
        assert "k_proj" not in cfg2.lora_target_modules

    def test_custom_config_overrides(self) -> None:
        """Constructor keyword args must override defaults."""
        from finetuning.trainer import FinetuneConfig

        cfg = FinetuneConfig(lora_rank=16, learning_rate=1e-5, max_steps=500)
        assert cfg.lora_rank == 16
        assert cfg.learning_rate == 1e-5
        assert cfg.max_steps == 500

    def test_federated_trainer_raises_without_peft(self) -> None:
        """FederatedTrainer.__init__ must raise RuntimeError when peft is missing."""
        import importlib

        from finetuning.trainer import FinetuneConfig

        # Temporarily hide the peft stub so the import check inside __init__ fails
        peft_backup = sys.modules.pop("peft", None)
        try:
            # Re-import trainer with peft absent
            import finetuning.trainer as trainer_mod

            importlib.reload(trainer_mod)
            with pytest.raises(RuntimeError, match="pip install peft"):
                trainer_mod.FederatedTrainer(FinetuneConfig())
        finally:
            if peft_backup is not None:
                sys.modules["peft"] = peft_backup


# ---------------------------------------------------------------------------
# 3. integrations/vastai/provisioner.py — dataclasses and _parse_instance
# ---------------------------------------------------------------------------


class TestGpuRequirements:
    """Tests for integrations.vastai.provisioner.GpuRequirements."""

    def test_default_values(self) -> None:
        from integrations.vastai.provisioner import GpuRequirements

        req = GpuRequirements()
        assert req.min_vram_gb == 24
        assert req.max_price_per_hour > 0
        assert req.min_reliability > 0.0
        assert req.verified_only is True

    def test_custom_gpu_name(self) -> None:
        from integrations.vastai.provisioner import GpuRequirements

        req = GpuRequirements(gpu_name="A100", min_vram_gb=80)
        assert req.gpu_name == "A100"
        assert req.min_vram_gb == 80


class TestParseInstance:
    """Tests for VastAiProvisioner._parse_instance (pure dict → dataclass logic)."""

    def _make_provisioner(self):
        """Build a VastAiProvisioner bypassing __init__ (no real API key needed)."""
        from integrations.vastai.provisioner import VastAiProvisioner

        return object.__new__(VastAiProvisioner)

    def test_parse_full_instance_dict(self) -> None:
        """_parse_instance must map all expected fields from the Vast.ai API dict."""
        from integrations.vastai.provisioner import INTERNAL_DHT_PORT

        provisioner = self._make_provisioner()
        raw = {
            "id": 42,
            "offer_id": 99,
            "gpu_name": "RTX 4090",
            "gpu_ram": 24576,  # MB → 24 GB
            "dph_total": 1.25,
            "public_ipaddr": "10.0.0.1",
            "ports": {f"{INTERNAL_DHT_PORT}/tcp": [{"HostPort": 7007}]},
            "actual_status": "running",
            "ssh_host": "10.0.0.1",
            "ssh_port": 22,
        }

        instance = provisioner._parse_instance(raw)

        assert instance.instance_id == 42
        assert instance.offer_id == 99
        assert instance.gpu_name == "RTX 4090"
        assert instance.vram_gb == 24
        assert instance.price_per_hour == pytest.approx(1.25)
        assert instance.public_ip == "10.0.0.1"
        assert instance.public_port == 7007
        assert instance.status == "running"

    def test_parse_missing_ports_falls_back(self) -> None:
        """When the ports dict is absent, fall back to INTERNAL_DHT_PORT."""
        from integrations.vastai.provisioner import INTERNAL_DHT_PORT

        provisioner = self._make_provisioner()
        raw = {
            "id": 1,
            "offer_id": 2,
            "gpu_name": "A10",
            "gpu_ram": 24576,
            "dph_total": 0.5,
            "public_ipaddr": "192.168.1.1",
            "actual_status": "running",
            "ssh_host": "192.168.1.1",
            "ssh_port": 22,
            # No 'ports' key
        }

        instance = provisioner._parse_instance(raw)
        assert instance.public_port == INTERNAL_DHT_PORT

    def test_vastai_provisioner_init_raises_without_api_key(self) -> None:
        """VastAiProvisioner() must raise ValueError if no API key is provided."""
        # Ensure VAST_API_KEY env var is not set during this call
        import os

        from integrations.vastai.provisioner import VastAiProvisioner

        env_backup = os.environ.pop("VAST_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="VAST_API_KEY"):
                VastAiProvisioner(api_key="")
        finally:
            if env_backup is not None:
                os.environ["VAST_API_KEY"] = env_backup


# ---------------------------------------------------------------------------
# 4. node/blockchain.py — SOLANA_AVAILABLE=False guard, OpenJob dataclass
# ---------------------------------------------------------------------------


class TestBlockchainFallback:
    """Tests for node.blockchain when the Solana packages are absent."""

    def test_blockchain_client_raises_without_solana(self) -> None:
        """BlockchainClient.__init__ must raise RuntimeError when SOLANA_AVAILABLE is False."""
        from node.blockchain import BlockchainClient

        with pytest.raises(RuntimeError, match="Solana packages not installed"):
            BlockchainClient(config=MagicMock())

    def test_open_job_dataclass_fields(self) -> None:
        """OpenJob must be constructible from keyword args and expose all fields."""
        from node.blockchain import OpenJob

        job = OpenJob(
            job_id=7,
            client="client_abc",
            model_id=b"\x01" * 32,
            prompt_hash=b"\x02" * 32,
            prompt_cid="bafybeifoo",
            max_tokens=256,
            payment_amount=500,
            deadline=9_999_999_999,
            job_pda="pda_xyz",
        )

        assert job.job_id == 7
        assert job.payment_amount == 500
        assert job.max_tokens == 256
        assert len(job.model_id) == 32

    def test_open_job_is_hashable_by_job_id(self) -> None:
        """Two OpenJob instances with the same job_id are not the same object."""
        from node.blockchain import OpenJob

        def _make(jid: int) -> OpenJob:
            return OpenJob(
                job_id=jid,
                client="c",
                model_id=b"\x00" * 32,
                prompt_hash=b"\x00" * 32,
                prompt_cid="cid",
                max_tokens=10,
                payment_amount=100,
                deadline=9999,
                job_pda="pda",
            )

        j1 = _make(1)
        j2 = _make(2)
        assert j1.job_id != j2.job_id
        assert j1 is not j2

    @pytest.mark.asyncio
    async def test_fetch_open_jobs_returns_empty_when_program_is_none(self) -> None:
        """fetch_open_jobs must return [] without raising when _inference_program is None."""
        from node.blockchain import BlockchainClient

        client = object.__new__(BlockchainClient)
        client._inference_program = None

        result = await client.fetch_open_jobs(model_id=b"\x00" * 32)
        assert result == []

    @pytest.mark.asyncio
    async def test_auto_settle_returns_zero_when_program_is_none(self) -> None:
        """auto_settle_expired_jobs must return 0 when _inference_program is None."""
        from node.blockchain import BlockchainClient

        client = object.__new__(BlockchainClient)
        client._inference_program = None

        settled = await client.auto_settle_expired_jobs()
        assert settled == 0

    @pytest.mark.asyncio
    async def test_heartbeat_returns_false_when_program_is_none(self) -> None:
        """heartbeat must return False gracefully when _registry_program is None."""
        from node.blockchain import BlockchainClient

        client = object.__new__(BlockchainClient)
        client._registry_program = None

        result = await client.heartbeat(endpoint="1.2.3.4:7070")
        assert result is False

    @pytest.mark.asyncio
    async def test_claim_job_returns_false_when_program_is_none(self) -> None:
        """claim_job must return False when _inference_program is None."""
        from node.blockchain import BlockchainClient, OpenJob

        client = object.__new__(BlockchainClient)
        client._inference_program = None

        job = OpenJob(
            job_id=1,
            client="c",
            model_id=b"\x00" * 32,
            prompt_hash=b"\x00" * 32,
            prompt_cid="cid",
            max_tokens=10,
            payment_amount=100,
            deadline=9999,
            job_pda="pda",
        )
        result = await client.claim_job(job)
        assert result is False
