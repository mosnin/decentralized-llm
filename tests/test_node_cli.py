"""Tests for scripts/node_cli.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import yaml

# ─────────────────────────── helpers ───────────────────────────────────────


def _write_minimal_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "rpc_url": "https://api.devnet.solana.com",
        "wallet_path": str(path.parent / "id.json"),
        "compute_registry_program": "8KpR2mT6uLqVwNzS4eBfY9oA3cJ7iGxH1nD5sW0qF2M",
        "inference_market_program": "5YQyZqXkJHy6V3JMxKqXyLqfP9V2A3j8Rk7mN4oD1eW",
        "listen_host": "0.0.0.0",
        "listen_port": 7070,
        "public_host": "203.0.113.10",
        "model_name": "meta-llama/Llama-3.2-3B",
        "num_shards": 4,
        "shard_index": 0,
        "gpu_count": 2,
        "vram_gb": 24,
        "supported_models": ["meta-llama/Llama-3.2-3B"],
        "lighthouse_api_key": "test-key",
        "max_concurrent_jobs": 4,
        "job_poll_interval_seconds": 2.0,
        "stake_amount": 1000,
    }
    with path.open("w") as fh:
        yaml.dump(config, fh)
    # Write a minimal wallet keypair JSON (list of 64 ints) alongside the config
    wallet_path = path.parent / "id.json"
    wallet_path.write_text(str(list(range(64))))


# ─────────────────────────── test_cli_help_exits_zero ───────────────────────


class TestCliHelpExitsZero:
    """Running --help should exit with code 0."""

    def test_cli_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "-m", "scripts.node_cli", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "node_cli" in result.stdout.lower() or "subcommand" in result.stdout.lower()

    def test_subcommand_help_exits_zero(self):
        for subcmd in ("setup", "register", "start", "status", "withdraw"):
            result = subprocess.run(
                [sys.executable, "-m", "scripts.node_cli", subcmd, "--help"],
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).parent.parent),
            )
            assert result.returncode == 0, f"{subcmd} --help failed:\n{result.stderr}"

    def test_config_show_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "-m", "scripts.node_cli", "config", "show", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"


# ─────────────────────────── test_setup_creates_config_dir ─────────────────


class TestSetupCreatesConfigDir:
    """setup subcommand creates the config directory and file."""

    def test_setup_creates_config_dir(self, tmp_path):
        config_file = tmp_path / "config.yaml"

        from scripts.node_cli import build_parser, cmd_setup

        parser = build_parser()
        args = parser.parse_args(["setup", "--config-path", str(config_file)])

        # Mock getpass so the test doesn't block waiting for stdin
        with patch("getpass.getpass", return_value="test-lighthouse-key"):
            cmd_setup(args)

        assert config_file.exists(), "Config file was not created"
        with config_file.open() as fh:
            data = yaml.safe_load(fh)

        assert isinstance(data, dict)
        assert "rpc_url" in data
        assert "wallet_path" in data
        assert "lighthouse_api_key" in data
        assert data["lighthouse_api_key"] == "test-lighthouse-key"

    def test_setup_skips_existing_config(self, tmp_path, capsys):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("rpc_url: https://existing.example.com\n")

        from scripts.node_cli import build_parser, cmd_setup

        parser = build_parser()
        args = parser.parse_args(["setup", "--config-path", str(config_file)])

        with patch("getpass.getpass", return_value=""):
            cmd_setup(args)

        captured = capsys.readouterr()
        assert "already exists" in captured.out
        # Original content must be unchanged
        assert "existing.example.com" in config_file.read_text()

    def test_setup_creates_nested_dir(self, tmp_path):
        config_file = tmp_path / "a" / "b" / "c" / "config.yaml"
        from scripts.node_cli import build_parser, cmd_setup

        parser = build_parser()
        args = parser.parse_args(["setup", "--config-path", str(config_file)])

        with patch("getpass.getpass", return_value=""):
            cmd_setup(args)

        assert config_file.exists()


# ─────────────────────────── test_status_shows_fields ──────────────────────


class TestStatusShowsFields:
    """status subcommand prints the expected fields."""

    def _make_node_record(self):
        record = MagicMock()
        record.endpoint = "203.0.113.10:7070"
        record.stake = 1000
        record.reputation = 99
        record.jobs_completed = 42
        record.jobs_disputed = 1
        record.earnings_claimable = 500
        return record

    def test_status_shows_fields_mocked_rpc(self, tmp_path, capsys):
        """Mock Solana packages and verify output contains all required fields."""
        config_file = tmp_path / "config.yaml"
        _write_minimal_config(config_file)

        node_record = self._make_node_record()

        # pubkey must support bytes() conversion (used in find_program_address seed)
        fake_pubkey = b"\x01" * 32

        mock_keypair = MagicMock()
        mock_keypair.pubkey.return_value = fake_pubkey

        mock_pubkey_cls = MagicMock()
        mock_pubkey_cls.from_string.return_value = MagicMock()
        mock_pubkey_cls.find_program_address.return_value = (MagicMock(), 0)

        mock_account_info = MagicMock()
        mock_account_info.data = b"\x00" * 64

        mock_resp = MagicMock()
        mock_resp.value = mock_account_info

        mock_rpc = AsyncMock()
        mock_rpc.__aenter__ = AsyncMock(return_value=mock_rpc)
        mock_rpc.__aexit__ = AsyncMock(return_value=False)
        mock_rpc.get_account_info = AsyncMock(return_value=mock_resp)

        mock_async_client_cls = MagicMock(return_value=mock_rpc)

        mock_program = AsyncMock()
        mock_program.account = {"NodeInfo": AsyncMock()}
        mock_program.account["NodeInfo"].fetch = AsyncMock(return_value=node_record)

        mock_provider_cls = MagicMock()
        mock_wallet_cls = MagicMock()

        async def fake_program_at(pubkey, provider):
            return mock_program

        solana_mod = MagicMock()
        solana_rpc_mod = MagicMock()
        solana_rpc_async_mod = MagicMock()
        solana_rpc_async_mod.AsyncClient = mock_async_client_cls
        solders_mod = MagicMock()
        solders_keypair_mod = MagicMock()
        solders_keypair_mod.Keypair = MagicMock(return_value=mock_keypair)
        solders_keypair_mod.Keypair.from_json = MagicMock(return_value=mock_keypair)
        solders_pubkey_mod = MagicMock()
        solders_pubkey_mod.Pubkey = mock_pubkey_cls
        anchorpy_mod = MagicMock()
        anchorpy_mod.Program = MagicMock()
        anchorpy_mod.Program.at = fake_program_at
        anchorpy_mod.Provider = mock_provider_cls
        anchorpy_mod.Wallet = mock_wallet_cls
        anchorpy_provider_mod = MagicMock()
        anchorpy_provider_mod.DEFAULT_OPTIONS = {}

        sys_modules_patch = {
            "solana": solana_mod,
            "solana.rpc": solana_rpc_mod,
            "solana.rpc.async_api": solana_rpc_async_mod,
            "solders": solders_mod,
            "solders.keypair": solders_keypair_mod,
            "solders.pubkey": solders_pubkey_mod,
            "anchorpy": anchorpy_mod,
            "anchorpy.provider": anchorpy_provider_mod,
        }

        with patch.dict(sys.modules, sys_modules_patch):
            import importlib

            import scripts.node_cli as cli_mod

            importlib.reload(cli_mod)

            parser = cli_mod.build_parser()
            args = parser.parse_args(["status", "--config-path", str(config_file)])
            cli_mod.cmd_status(args)

        captured = capsys.readouterr()
        output = captured.out + captured.err

        assert "endpoint" in output
        assert "stake" in output
        assert "reputation" in output

    def test_status_without_solana_packages_shows_config_fields(self, tmp_path, capsys):
        """When Solana packages are absent, status shows config-derived fields."""
        config_file = tmp_path / "config.yaml"
        _write_minimal_config(config_file)

        # Remove Solana modules so the import guard fails
        blocked = {
            "solana": None,
            "solana.rpc": None,
            "solana.rpc.async_api": None,
            "solders": None,
            "solders.keypair": None,
            "solders.pubkey": None,
        }

        with patch.dict(sys.modules, blocked):
            import importlib

            import scripts.node_cli as cli_mod

            importlib.reload(cli_mod)

            parser = cli_mod.build_parser()
            args = parser.parse_args(["status", "--config-path", str(config_file)])
            cli_mod.cmd_status(args)

        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert "endpoint" in output
        assert "stake" in output


# ─────────────────────────── test_register_dry_run ─────────────────────────


class TestRegisterDryRun:
    """register without --execute prints anchor command but does NOT call blockchain."""

    def test_register_dry_run_prints_anchor_command(self, tmp_path, capsys):
        config_file = tmp_path / "config.yaml"
        _write_minimal_config(config_file)

        import importlib

        import scripts.node_cli as cli_mod

        importlib.reload(cli_mod)

        parser = cli_mod.build_parser()
        args = parser.parse_args(["register", "--config-path", str(config_file)])
        assert not args.execute

        # Ensure BlockchainClient is never instantiated during dry run
        with patch.object(cli_mod, "BlockchainClient") as mock_bc:
            cli_mod.cmd_register(args)
            mock_bc.assert_not_called()

        captured = capsys.readouterr()
        output = captured.out

        assert "anchor invoke" in output
        assert "register_node" in output
        assert "dry run" in output.lower() or "--execute" in output

    def test_register_dry_run_shows_params(self, tmp_path, capsys):
        config_file = tmp_path / "config.yaml"
        _write_minimal_config(config_file)

        from scripts.node_cli import build_parser, cmd_register

        parser = build_parser()
        args = parser.parse_args(["register", "--config-path", str(config_file)])

        cmd_register(args)

        captured = capsys.readouterr()
        output = captured.out

        assert "endpoint" in output.lower()
        assert "gpu" in output.lower()
        assert "vram" in output.lower()
        assert "model" in output.lower()

    def test_register_execute_calls_blockchain(self, tmp_path):
        """With --execute, BlockchainClient.register_node must be called."""
        config_file = tmp_path / "config.yaml"
        _write_minimal_config(config_file)

        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.register_node = AsyncMock(return_value=True)
        mock_client.close = AsyncMock()

        mock_bc_cls = MagicMock(return_value=mock_client)
        mock_node_config_cls = MagicMock()
        mock_node_config_instance = MagicMock()
        mock_node_config_cls.return_value = mock_node_config_instance

        import importlib

        import scripts.node_cli as cli_mod

        importlib.reload(cli_mod)

        parser = cli_mod.build_parser()
        args = parser.parse_args(["register", "--config-path", str(config_file), "--execute"])
        assert args.execute

        with (
            patch.object(cli_mod, "BlockchainClient", mock_bc_cls),
            patch.object(cli_mod, "NodeConfig", mock_node_config_cls),
        ):
            cli_mod.cmd_register(args)

        mock_client.register_node.assert_called_once()


# ─────────────────────────── test_config_show ──────────────────────────────


class TestConfigShow:
    def test_config_show_prints_yaml(self, tmp_path, capsys):
        config_file = tmp_path / "config.yaml"
        _write_minimal_config(config_file)

        from scripts.node_cli import build_parser, cmd_config_show

        parser = build_parser()
        args = parser.parse_args(["config", "show", "--config-path", str(config_file)])
        cmd_config_show(args)

        captured = capsys.readouterr()
        output = captured.out
        assert "rpc_url" in output
        assert "wallet_path" in output

    def test_config_show_redacts_api_key(self, tmp_path, capsys):
        config_file = tmp_path / "config.yaml"
        _write_minimal_config(config_file)

        from scripts.node_cli import build_parser, cmd_config_show

        parser = build_parser()
        args = parser.parse_args(["config", "show", "--config-path", str(config_file)])
        cmd_config_show(args)

        captured = capsys.readouterr()
        # The actual key value should be redacted
        assert "test-key" not in captured.out
        assert "REDACTED" in captured.out
