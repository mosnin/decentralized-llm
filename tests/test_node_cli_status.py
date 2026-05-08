"""Tests for `node status` and `node earnings` CLI subcommands."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

# ─────────────────────────── helpers ───────────────────────────────────────

_REPO_ROOT = str(Path(__file__).parent.parent)

_DASHBOARD_PAYLOAD = {
    "timestamp": 1_700_000_000.0,
    "uptime_seconds": 13_335.0,
    "version": "0.1.0",
    "inference": {
        "total_requests": 42,
        "failed_requests": 0,
        "success_rate": 1.0,
        "avg_latency_ms": 120.5,
        "p95_latency_ms": 250.0,
        "total_tokens": 84000,
    },
    "network": {
        "active_nodes": 5,
        "jobs_24h": 42,
        "success_rate_24h": 1.0,
        "total_stake_lamports": 1_234_567,
    },
    "health": {
        "status": "ok",
        "blockchain": "ok",
    },
}


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
        "gateway_url": "http://localhost:8080",
    }
    with path.open("w") as fh:
        yaml.dump(config, fh)
    wallet_path = path.parent / "id.json"
    wallet_path.write_text(str(list(range(64))))


def _make_mock_urlopen(payload: dict):
    """Return a context-manager mock that yields a readable HTTP response."""

    def _urlopen(url, timeout=None):
        body = json.dumps(payload).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    return _urlopen


# ─────────────────────────── status tests ──────────────────────────────────


class TestNodeStatusCommand:
    def test_status_command_runs(self, tmp_path, capsys):
        """node_cli.py status exits 0 with mocked dashboard data."""
        config_file = tmp_path / "config.yaml"
        _write_minimal_config(config_file)

        import scripts.node_cli as cli_mod

        importlib.reload(cli_mod)

        parser = cli_mod.build_parser()
        args = parser.parse_args(["status", "--config-path", str(config_file)])

        with patch.object(cli_mod, "_fetch_dashboard", return_value=_DASHBOARD_PAYLOAD):
            cli_mod.cmd_node_status(args)

        captured = capsys.readouterr()
        output = captured.out
        assert "Node Status" in output
        assert "Node ID:" in output
        assert "Status:" in output
        assert "Uptime:" in output
        assert "Version:" in output
        assert "RPC:" in output
        assert "Jobs Today:" in output
        assert "Queue Depth:" in output

    def test_status_running_when_gateway_available(self, tmp_path, capsys):
        """Status shows 'running' when the dashboard API responds."""
        config_file = tmp_path / "config.yaml"
        _write_minimal_config(config_file)

        import scripts.node_cli as cli_mod

        importlib.reload(cli_mod)

        parser = cli_mod.build_parser()
        args = parser.parse_args(["status", "--config-path", str(config_file)])

        with patch.object(cli_mod, "_fetch_dashboard", return_value=_DASHBOARD_PAYLOAD):
            cli_mod.cmd_node_status(args)

        captured = capsys.readouterr()
        assert "running" in captured.out

    def test_status_stopped_when_gateway_unavailable(self, tmp_path, capsys):
        """Status shows 'stopped' when the gateway API is unavailable."""
        config_file = tmp_path / "config.yaml"
        _write_minimal_config(config_file)

        import scripts.node_cli as cli_mod

        importlib.reload(cli_mod)

        parser = cli_mod.build_parser()
        args = parser.parse_args(["status", "--config-path", str(config_file)])

        with patch.object(cli_mod, "_fetch_dashboard", return_value=None):
            cli_mod.cmd_node_status(args)

        captured = capsys.readouterr()
        assert "stopped" in captured.out
        assert "N/A" in captured.out

    def test_status_shows_uptime_formatted(self, tmp_path, capsys):
        """Uptime should be formatted as hours/minutes/seconds."""
        config_file = tmp_path / "config.yaml"
        _write_minimal_config(config_file)

        import scripts.node_cli as cli_mod

        importlib.reload(cli_mod)

        parser = cli_mod.build_parser()
        args = parser.parse_args(["status", "--config-path", str(config_file)])

        with patch.object(cli_mod, "_fetch_dashboard", return_value=_DASHBOARD_PAYLOAD):
            cli_mod.cmd_node_status(args)

        captured = capsys.readouterr()
        # 13335s = 3h 42m 15s
        assert "3h" in captured.out
        assert "42m" in captured.out

    def test_status_help_exits_zero(self):
        """node_cli.py status --help should exit with code 0."""
        result = subprocess.run(
            [sys.executable, "-m", "scripts.node_cli", "status", "--help"],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"


# ─────────────────────────── earnings tests ────────────────────────────────


class TestNodeEarningsCommand:
    def test_earnings_command_runs(self, tmp_path, capsys):
        """node_cli.py earnings exits 0 with mocked dashboard data."""
        config_file = tmp_path / "config.yaml"
        _write_minimal_config(config_file)

        import scripts.node_cli as cli_mod

        importlib.reload(cli_mod)

        parser = cli_mod.build_parser()
        args = parser.parse_args(["earnings", "--config-path", str(config_file)])

        with patch.object(cli_mod, "_fetch_dashboard", return_value=_DASHBOARD_PAYLOAD):
            cli_mod.cmd_node_earnings(args)

        captured = capsys.readouterr()
        output = captured.out
        assert "Earnings Summary" in output
        assert "Jobs Completed:" in output
        assert "Total Earned:" in output
        assert "Avg per Job:" in output
        assert "Top Model:" in output
        assert "Lifetime Totals" in output
        assert "Total Jobs:" in output

    def test_earnings_hours_flag(self, tmp_path, capsys):
        """node_cli.py earnings --hours 12 runs without error."""
        config_file = tmp_path / "config.yaml"
        _write_minimal_config(config_file)

        import scripts.node_cli as cli_mod

        importlib.reload(cli_mod)

        parser = cli_mod.build_parser()
        args = parser.parse_args(["earnings", "--hours", "12", "--config-path", str(config_file)])
        assert args.hours == 12

        with patch.object(cli_mod, "_fetch_dashboard", return_value=_DASHBOARD_PAYLOAD):
            cli_mod.cmd_node_earnings(args)

        captured = capsys.readouterr()
        assert "last 12h" in captured.out

    def test_earnings_default_hours_is_24(self, tmp_path):
        """Default --hours value is 24."""
        config_file = tmp_path / "config.yaml"
        _write_minimal_config(config_file)

        import scripts.node_cli as cli_mod

        importlib.reload(cli_mod)

        parser = cli_mod.build_parser()
        args = parser.parse_args(["earnings", "--config-path", str(config_file)])
        assert args.hours == 24

    def test_earnings_shows_na_when_gateway_unavailable(self, tmp_path, capsys):
        """Earnings shows N/A values when gateway is unavailable."""
        config_file = tmp_path / "config.yaml"
        _write_minimal_config(config_file)

        import scripts.node_cli as cli_mod

        importlib.reload(cli_mod)

        parser = cli_mod.build_parser()
        args = parser.parse_args(["earnings", "--config-path", str(config_file)])

        with patch.object(cli_mod, "_fetch_dashboard", return_value=None):
            cli_mod.cmd_node_earnings(args)

        captured = capsys.readouterr()
        assert "N/A" in captured.out

    def test_earnings_lamports_format(self, tmp_path, capsys):
        """Total Earned should include 'lamports' in output when data is available."""
        config_file = tmp_path / "config.yaml"
        _write_minimal_config(config_file)

        import scripts.node_cli as cli_mod

        importlib.reload(cli_mod)

        parser = cli_mod.build_parser()
        args = parser.parse_args(["earnings", "--config-path", str(config_file)])

        with patch.object(cli_mod, "_fetch_dashboard", return_value=_DASHBOARD_PAYLOAD):
            cli_mod.cmd_node_earnings(args)

        captured = capsys.readouterr()
        assert "lamports" in captured.out

    def test_earnings_help_exits_zero(self):
        """node_cli.py earnings --help should exit with code 0."""
        result = subprocess.run(
            [sys.executable, "-m", "scripts.node_cli", "earnings", "--help"],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"


# ─────────────────────────── urllib mock tests ─────────────────────────────


class TestFetchDashboardUrllib:
    """Test that _fetch_dashboard uses urllib and handles errors gracefully."""

    def test_fetch_dashboard_parses_response(self):
        """_fetch_dashboard returns parsed dict when urlopen succeeds."""
        import scripts.node_cli as cli_mod

        importlib.reload(cli_mod)

        mock_urlopen = _make_mock_urlopen(_DASHBOARD_PAYLOAD)

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            result = cli_mod._fetch_dashboard("http://localhost:8080")

        assert result is not None
        assert result["version"] == "0.1.0"
        assert result["health"]["blockchain"] == "ok"

    def test_fetch_dashboard_returns_none_on_error(self):
        """_fetch_dashboard returns None when urlopen raises."""
        import scripts.node_cli as cli_mod

        importlib.reload(cli_mod)

        import urllib.error

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            result = cli_mod._fetch_dashboard("http://localhost:8080")

        assert result is None

    def test_status_command_uses_urllib(self, tmp_path, capsys):
        """status command fetches from gateway via urllib when not mocked."""
        config_file = tmp_path / "config.yaml"
        _write_minimal_config(config_file)

        import scripts.node_cli as cli_mod

        importlib.reload(cli_mod)

        parser = cli_mod.build_parser()
        args = parser.parse_args(["status", "--config-path", str(config_file)])

        mock_urlopen = _make_mock_urlopen(_DASHBOARD_PAYLOAD)

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            cli_mod.cmd_node_status(args)

        captured = capsys.readouterr()
        assert "Node Status" in captured.out
        assert "running" in captured.out

    def test_earnings_command_uses_urllib(self, tmp_path, capsys):
        """earnings command fetches from gateway via urllib when not mocked."""
        config_file = tmp_path / "config.yaml"
        _write_minimal_config(config_file)

        import scripts.node_cli as cli_mod

        importlib.reload(cli_mod)

        parser = cli_mod.build_parser()
        args = parser.parse_args(["earnings", "--config-path", str(config_file)])

        mock_urlopen = _make_mock_urlopen(_DASHBOARD_PAYLOAD)

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            cli_mod.cmd_node_earnings(args)

        captured = capsys.readouterr()
        assert "Earnings Summary" in captured.out
        assert "lamports" in captured.out
