"""Tests for governance list/vote CLI subcommands in scripts/node_cli.py."""

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
    wallet_path = path.parent / "id.json"
    wallet_path.write_text(str(list(range(64))))


_REPO_ROOT = str(Path(__file__).parent.parent)


# ─────────────────────────── help exits zero ───────────────────────────────


class TestGovernanceHelpExitsZero:
    def test_governance_list_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "-m", "scripts.node_cli", "governance", "list", "--help"],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "list" in result.stdout.lower() or "proposal" in result.stdout.lower()

    def test_governance_vote_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "-m", "scripts.node_cli", "governance", "vote", "--help"],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "vote" in result.stdout.lower() or "proposal" in result.stdout.lower()


# ─────────────────────────── dry-run mode ──────────────────────────────────


class TestGovernanceDryRun:
    def test_governance_list_dry_run_prints_anchor_command(self, tmp_path, capsys):
        config_file = tmp_path / "config.yaml"
        _write_minimal_config(config_file)

        import importlib

        import scripts.node_cli as cli_mod

        importlib.reload(cli_mod)

        parser = cli_mod.build_parser()
        args = parser.parse_args(["governance", "list", "--config-path", str(config_file)])
        assert not args.execute

        cli_mod.cmd_governance_list(args)

        captured = capsys.readouterr()
        output = captured.out
        assert "anchor invoke" in output
        assert "get_proposals" in output
        assert "--execute" in output or "dry run" in output.lower()

    def test_governance_vote_dry_run_prints_anchor_command(self, tmp_path, capsys):
        config_file = tmp_path / "config.yaml"
        _write_minimal_config(config_file)

        import importlib

        import scripts.node_cli as cli_mod

        importlib.reload(cli_mod)

        parser = cli_mod.build_parser()
        args = parser.parse_args(
            [
                "governance",
                "vote",
                "--config-path",
                str(config_file),
                "--dry-run",
                "42",
                "for",
            ]
        )
        assert args.dry_run
        assert args.proposal_id == 42
        assert args.choice == "for"

        cli_mod.cmd_governance_vote(args)

        captured = capsys.readouterr()
        output = captured.out
        assert "42" in output
        assert "for" in output.lower()
        assert "dry" in output.lower() or "dry-run" in output.lower()


# ─────────────────────────── validation ────────────────────────────────────


class TestGovernanceVoteValidatesChoice:
    def test_governance_vote_validates_choice(self, tmp_path, capsys):
        config_file = tmp_path / "config.yaml"
        _write_minimal_config(config_file)

        import importlib

        import scripts.node_cli as cli_mod

        importlib.reload(cli_mod)

        parser = cli_mod.build_parser()
        args = parser.parse_args(
            ["governance", "vote", "--config-path", str(config_file), "1", "invalid"]
        )

        import pytest

        with pytest.raises(SystemExit) as exc_info:
            cli_mod.cmd_governance_vote(args)

        assert exc_info.value.code != 0

        captured = capsys.readouterr()
        assert "error" in (captured.out + captured.err).lower()
        assert "invalid" in captured.err


# ─────────────────────────── execute mode ──────────────────────────────────


class TestGovernanceListExecuteCallsClient:
    def test_governance_list_execute_calls_client(self, tmp_path, capsys):
        config_file = tmp_path / "config.yaml"
        _write_minimal_config(config_file)

        fake_proposals = [
            {
                "id": 1,
                "title": "Increase max nodes",
                "status": "Active",
                "votes_for": 1500,
                "votes_against": 300,
                "voting_ends_at": 1700000000,
            },
            {
                "id": 2,
                "title": "Reduce stake requirement",
                "status": "Active",
                "votes_for": 800,
                "votes_against": 200,
                "voting_ends_at": 1700100000,
            },
        ]

        mock_client = AsyncMock()
        mock_client.get_governance_proposals = AsyncMock(return_value=fake_proposals)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_client_cls = MagicMock(return_value=mock_client)

        import importlib

        import scripts.node_cli as cli_mod

        importlib.reload(cli_mod)

        parser = cli_mod.build_parser()
        args = parser.parse_args(
            ["governance", "list", "--config-path", str(config_file), "--execute"]
        )
        assert args.execute

        with patch("client.python.DecentralizedLLMClient", mock_client_cls):
            cli_mod.cmd_governance_list(args)

        mock_client.get_governance_proposals.assert_called_once()

        captured = capsys.readouterr()
        output = captured.out
        assert "Increase max nodes" in output
        assert "Reduce stake requirement" in output
        assert "1500" in output
        assert "Active" in output
