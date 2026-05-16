"""Tests that validate the devnet config file structure."""

import json
from pathlib import Path


def test_devnet_config_exists():
    assert Path("config.devnet.json").exists()


def test_devnet_config_has_required_keys():
    config = json.loads(Path("config.devnet.json").read_text())
    required = {"model_name", "rpc_url", "inference_market_program", "compute_registry_program"}
    assert required.issubset(config.keys())


def test_devnet_config_rpc_is_devnet():
    config = json.loads(Path("config.devnet.json").read_text())
    assert "devnet" in config["rpc_url"]


def test_devnet_config_program_ids_are_base58():
    config = json.loads(Path("config.devnet.json").read_text())
    for key in ("inference_market_program", "compute_registry_program", "governance_program"):
        val = config[key]
        assert len(val) >= 32, f"{key} too short"
        assert " " not in val, f"{key} contains spaces"
