"""
Tests for the new blockchain operations in node/blockchain.py and scripts/node_cli.py.

All tests mock the Solana RPC – no real network traffic is made.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers to build a fake Solana / anchorpy module tree so we can import
# blockchain.py without having the real packages installed.
# ---------------------------------------------------------------------------


def _make_solana_stubs():
    """Return a dict of module stubs to inject into sys.modules."""
    pubkey_cls = MagicMock(name="Pubkey")
    pubkey_cls.find_program_address = MagicMock(return_value=(MagicMock(name="pda"), 255))
    pubkey_cls.from_string = MagicMock(return_value=MagicMock(name="prog_id"))

    keypair_cls = MagicMock(name="Keypair")
    fake_kp = MagicMock(name="keypair_instance")
    fake_pubkey = MagicMock(name="pubkey_instance")
    fake_kp.pubkey.return_value = fake_pubkey
    keypair_cls.from_json = MagicMock(return_value=fake_kp)

    wallet_cls = MagicMock(name="Wallet")
    fake_wallet = MagicMock(name="wallet_instance")
    fake_wallet.public_key = fake_pubkey
    wallet_cls.return_value = fake_wallet

    provider_cls = MagicMock(name="Provider")

    program_cls = MagicMock(name="Program")
    program_cls.at = AsyncMock(return_value=MagicMock(name="program_instance"))

    async_client_cls = MagicMock(name="AsyncClient")
    fake_rpc = AsyncMock(name="rpc_instance")
    async_client_cls.return_value = fake_rpc

    anchorpy_mod = ModuleType("anchorpy")
    anchorpy_mod.Program = program_cls
    anchorpy_mod.Provider = provider_cls
    anchorpy_mod.Wallet = wallet_cls

    solana_mod = ModuleType("solana")
    solana_rpc_mod = ModuleType("solana.rpc")
    solana_async_mod = ModuleType("solana.rpc.async_api")
    solana_async_mod.AsyncClient = async_client_cls

    solders_mod = ModuleType("solders")
    solders_kp_mod = ModuleType("solders.keypair")
    solders_kp_mod.Keypair = keypair_cls
    solders_pk_mod = ModuleType("solders.pubkey")
    solders_pk_mod.Pubkey = pubkey_cls

    stubs = {
        "anchorpy": anchorpy_mod,
        "solana": solana_mod,
        "solana.rpc": solana_rpc_mod,
        "solana.rpc.async_api": solana_async_mod,
        "solders": solders_mod,
        "solders.keypair": solders_kp_mod,
        "solders.pubkey": solders_pk_mod,
    }
    return stubs, {
        "pubkey_cls": pubkey_cls,
        "keypair_cls": keypair_cls,
        "wallet_cls": wallet_cls,
        "fake_wallet": fake_wallet,
        "fake_pubkey": fake_pubkey,
        "provider_cls": provider_cls,
        "program_cls": program_cls,
        "async_client_cls": async_client_cls,
        "fake_rpc": fake_rpc,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def bc_module():
    """Import node.blockchain with Solana packages stubbed out."""
    stubs, _ = _make_solana_stubs()
    with patch.dict(sys.modules, stubs):
        import importlib

        import node.blockchain as mod

        importlib.reload(mod)
        yield mod


@pytest.fixture()
def fake_config():
    cfg = MagicMock(name="NodeConfig")
    cfg.wallet_path = "/tmp/fake_wallet.json"
    cfg.rpc_url = "https://api.devnet.solana.com"
    cfg.inference_market_program = "5YQyZqXkJHy6V3JMxKqXyLqfP9V2A3j8Rk7mN4oD1eW"
    cfg.compute_registry_program = "8KpR2mT6uLqVwNzS4eBfY9oA3cJ7iGxH1nD5sW0qF2M"
    cfg.governance_program = "3CvE7tX9rMwPfBgY2nKjH6oL4sQ8uZaD5mR1iW0eN9T"
    return cfg


def _build_client(bc_module, fake_config):
    """Return a BlockchainClient with mocked internals pre-attached."""
    client = bc_module.BlockchainClient.__new__(bc_module.BlockchainClient)
    client.config = fake_config

    fake_pubkey = MagicMock(name="pubkey")
    fake_wallet = MagicMock(name="wallet")
    fake_wallet.public_key = fake_pubkey

    fake_rpc = AsyncMock(name="rpc")
    fake_rpc.get_balance = AsyncMock(return_value=MagicMock(value=5_000_000_000))
    fake_rpc.close = AsyncMock()

    fake_inference_prog = AsyncMock(name="inference_program")
    fake_inference_prog.program_id = MagicMock(name="inf_prog_id")
    fake_inference_prog.rpc = {}
    fake_inference_prog.context = MagicMock(return_value=MagicMock())

    fake_registry_prog = AsyncMock(name="registry_program")
    fake_registry_prog.program_id = MagicMock(name="reg_prog_id")
    fake_registry_prog.rpc = {}
    fake_registry_prog.context = MagicMock(return_value=MagicMock())

    client._wallet = fake_wallet
    client._client = fake_rpc
    client._inference_program = fake_inference_prog
    client._registry_program = fake_registry_prog

    return client, fake_wallet, fake_pubkey, fake_rpc, fake_inference_prog, fake_registry_prog


# ---------------------------------------------------------------------------
# get_earnings
# ---------------------------------------------------------------------------


class TestGetEarnings:
    def test_get_earnings_returns_dict_with_required_keys(self, bc_module, fake_config):
        import asyncio

        client, _, _, _, _, registry_prog = _build_client(bc_module, fake_config)

        node_record = MagicMock()
        node_record.staked_amount = 2_000_000_000  # 2 SOL
        node_record.jobs_completed = 10

        registry_prog.account = {"NodeRecord": AsyncMock()}
        registry_prog.account["NodeRecord"].fetch = AsyncMock(return_value=node_record)

        result = asyncio.run(client.get_earnings())

        assert "available_lamports" in result
        assert "total_earned_lamports" in result
        assert "pending_lamports" in result
        assert isinstance(result["available_lamports"], int)
        assert isinstance(result["total_earned_lamports"], int)
        assert isinstance(result["pending_lamports"], int)

    def test_get_earnings_raises_blockchain_error_on_rpc_failure(self, bc_module, fake_config):
        import asyncio

        client, _, _, _, _, registry_prog = _build_client(bc_module, fake_config)

        registry_prog.account = {"NodeRecord": AsyncMock()}
        registry_prog.account["NodeRecord"].fetch = AsyncMock(
            side_effect=RuntimeError("RPC timeout")
        )

        with pytest.raises(bc_module.BlockchainError):
            asyncio.run(client.get_earnings())


# ---------------------------------------------------------------------------
# withdraw_earnings
# ---------------------------------------------------------------------------


class TestWithdrawEarnings:
    def _setup_earnings(self, client, registry_prog, available: int):
        node_record = MagicMock()
        node_record.staked_amount = available
        node_record.jobs_completed = 5
        registry_prog.account = {"NodeRecord": AsyncMock()}
        registry_prog.account["NodeRecord"].fetch = AsyncMock(return_value=node_record)

    def test_withdraw_earnings_returns_tx_signature(self, bc_module, fake_config):
        client, _, _, rpc, inf_prog, reg_prog = _build_client(bc_module, fake_config)

        self._setup_earnings(client, reg_prog, 1_000_000_000)

        fake_tx_sig = "4xQY7ZdPt8w2K9MvNfLjR5oGbX1cEsUzHnCpDqYeAi3BkVWmT6rSu0hJFgOIyN"
        withdraw_rpc = AsyncMock(return_value=fake_tx_sig)
        inf_prog.rpc["withdraw"] = withdraw_rpc
        rpc.get_balance = AsyncMock(return_value=MagicMock(value=4_000_000_000))

        import asyncio

        result = asyncio.run(client.withdraw_earnings(500_000_000))

        assert result["tx_signature"] == fake_tx_sig
        assert result["amount_sol"] == pytest.approx(0.5)
        assert "new_balance_sol" in result
        withdraw_rpc.assert_called_once()

    def test_withdraw_all_uses_full_balance(self, bc_module, fake_config):
        client, _, _, rpc, inf_prog, reg_prog = _build_client(bc_module, fake_config)

        available = 2_500_000_000
        self._setup_earnings(client, reg_prog, available)

        captured_args = {}

        async def fake_withdraw(amount, ctx):
            captured_args["amount"] = amount
            return "tx_all"

        inf_prog.rpc["withdraw"] = fake_withdraw
        rpc.get_balance = AsyncMock(return_value=MagicMock(value=2_500_000_000))

        import asyncio

        result = asyncio.run(
            client.withdraw_earnings(None)  # None → withdraw all
        )

        assert captured_args["amount"] == available
        assert result["tx_signature"] == "tx_all"

    def test_withdraw_raises_insufficient_funds(self, bc_module, fake_config):
        client, _, _, _, inf_prog, reg_prog = _build_client(bc_module, fake_config)

        self._setup_earnings(client, reg_prog, 100_000)  # only 100k lamports available

        import asyncio

        with pytest.raises(bc_module.InsufficientFundsError):
            asyncio.run(
                client.withdraw_earnings(999_999_999)  # way more than available
            )

    def test_withdraw_dry_run_returns_estimate(self, bc_module, fake_config, tmp_path, capsys):
        """The --dry-run CLI flag prints estimate without calling withdraw_earnings."""
        import importlib

        config_file = tmp_path / "config.yaml"
        import yaml

        config_data = {
            "rpc_url": "https://api.devnet.solana.com",
            "wallet_path": str(tmp_path / "id.json"),
            "compute_registry_program": "8KpR2mT6uLqVwNzS4eBfY9oA3cJ7iGxH1nD5sW0qF2M",
            "inference_market_program": "5YQyZqXkJHy6V3JMxKqXyLqfP9V2A3j8Rk7mN4oD1eW",
        }
        with config_file.open("w") as fh:
            yaml.dump(config_data, fh)
        (tmp_path / "id.json").write_text(str(list(range(64))))

        # Build a mock BlockchainClient
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.close = AsyncMock()
        mock_client.get_earnings = AsyncMock(
            return_value={
                "available_lamports": 1_000_000_000,
                "total_earned_lamports": 1_000_000_000,
                "pending_lamports": 0,
            }
        )
        mock_client.withdraw_earnings = AsyncMock()  # must NOT be called

        mock_bc_cls = MagicMock(return_value=mock_client)
        mock_node_config_cls = MagicMock()
        mock_node_config_instance = MagicMock()
        mock_node_config_cls.return_value = mock_node_config_instance

        stubs, _ = _make_solana_stubs()
        with patch.dict(sys.modules, stubs):
            import scripts.node_cli as cli_mod

            importlib.reload(cli_mod)

            with (
                patch.object(cli_mod, "BlockchainClient", mock_bc_cls),
                patch.object(cli_mod, "NodeConfig", mock_node_config_cls),
            ):
                parser = cli_mod.build_parser()
                args = parser.parse_args(
                    ["withdraw", "--config-path", str(config_file), "--dry-run"]
                )
                cli_mod.cmd_withdraw(args)

        captured = capsys.readouterr()
        assert "dry run" in captured.out.lower() or "estimated" in captured.out.lower()
        mock_client.withdraw_earnings.assert_not_called()


# ---------------------------------------------------------------------------
# cast_governance_vote
# ---------------------------------------------------------------------------


class TestCastGovernanceVote:
    def _setup_gov_program(self, bc_module, fake_config, client, rpc):
        """Attach a mock governance program to the client via Program.at."""
        gov_program = AsyncMock(name="gov_program")
        gov_program.program_id = MagicMock(name="gov_prog_id")
        gov_program.rpc = {}
        gov_program.context = MagicMock(return_value=MagicMock())

        proposal = MagicMock()
        proposal.id = 1
        proposal.title = "Test Proposal"
        proposal.status = "Active"
        proposal.voting_ends_at = 9_999_999_999

        proposal_account_cls = AsyncMock()
        proposal_account_cls.fetch = AsyncMock(return_value=proposal)

        vote_record = MagicMock()
        vote_record.has_voted = False

        vote_record_account_cls = AsyncMock()
        vote_record_account_cls.fetch = AsyncMock(return_value=vote_record)

        gov_program.account = {
            "Proposal": proposal_account_cls,
            "VoteRecord": vote_record_account_cls,
        }

        return gov_program, proposal, vote_record

    def _patch_gov_program(self, gov_program):
        """
        Patch sys.modules["anchorpy"].Program.at to return gov_program.
        Returns a context manager that restores the original on exit.
        """
        anchorpy_stub = sys.modules.get("anchorpy")
        if anchorpy_stub is not None:
            anchorpy_stub.Program.at = AsyncMock(return_value=gov_program)

    def test_cast_vote_yes_returns_signature(self, bc_module, fake_config):
        import asyncio

        client, _, _, rpc, _, _ = _build_client(bc_module, fake_config)
        gov_program, _, _ = self._setup_gov_program(bc_module, fake_config, client, rpc)

        fake_sig = "5zAbCdEf"
        cast_vote_rpc = AsyncMock(return_value=fake_sig)
        gov_program.rpc["cast_vote"] = cast_vote_rpc
        self._patch_gov_program(gov_program)

        result = asyncio.run(client.cast_governance_vote(1, True))

        assert result == fake_sig
        cast_vote_rpc.assert_called_once()

    def test_cast_vote_no_returns_signature(self, bc_module, fake_config):
        import asyncio

        client, _, _, rpc, _, _ = _build_client(bc_module, fake_config)
        gov_program, _, _ = self._setup_gov_program(bc_module, fake_config, client, rpc)

        fake_sig = "6xYzWvUt"
        cast_vote_rpc = AsyncMock(return_value=fake_sig)
        gov_program.rpc["cast_vote"] = cast_vote_rpc
        self._patch_gov_program(gov_program)

        result = asyncio.run(client.cast_governance_vote(1, False))

        assert result == fake_sig

    def test_cast_vote_raises_proposal_not_found(self, bc_module, fake_config):
        import asyncio

        client, _, _, rpc, _, _ = _build_client(bc_module, fake_config)
        gov_program, _, _ = self._setup_gov_program(bc_module, fake_config, client, rpc)

        # Make the proposal fetch raise so ProposalNotFoundError is triggered
        gov_program.account["Proposal"].fetch = AsyncMock(
            side_effect=RuntimeError("Account not found")
        )
        self._patch_gov_program(gov_program)

        with pytest.raises(bc_module.ProposalNotFoundError):
            asyncio.run(client.cast_governance_vote(999, True))

    def test_cast_vote_raises_already_voted(self, bc_module, fake_config):
        import asyncio

        client, _, _, rpc, _, _ = _build_client(bc_module, fake_config)
        gov_program, _, vote_record = self._setup_gov_program(bc_module, fake_config, client, rpc)

        # Simulate voter has already voted
        vote_record.has_voted = True
        gov_program.account["VoteRecord"].fetch = AsyncMock(return_value=vote_record)
        self._patch_gov_program(gov_program)

        with pytest.raises(bc_module.AlreadyVotedError):
            asyncio.run(client.cast_governance_vote(1, True))


# ---------------------------------------------------------------------------
# get_governance_proposals
# ---------------------------------------------------------------------------


class TestGetGovernanceProposals:
    def _patch_gov_program(self, gov_program):
        anchorpy_stub = sys.modules.get("anchorpy")
        if anchorpy_stub is not None:
            anchorpy_stub.Program.at = AsyncMock(return_value=gov_program)

    def test_get_proposals_returns_list(self, bc_module, fake_config):
        import asyncio

        client, _, _, rpc, _, _ = _build_client(bc_module, fake_config)

        p1 = MagicMock()
        p1.account.id = 0
        p1.account.title = "Proposal Alpha"
        p1.account.description_cid = "QmAlpha"
        p1.account.votes_for = 100
        p1.account.votes_against = 20
        p1.account.status = "Active"
        p1.account.voting_ends_at = 9_000_000_000

        p2 = MagicMock()
        p2.account.id = 1
        p2.account.title = "Proposal Beta"
        p2.account.description_cid = "QmBeta"
        p2.account.votes_for = 50
        p2.account.votes_against = 10
        p2.account.status = "Active"
        p2.account.voting_ends_at = 9_000_000_001

        gov_program = AsyncMock(name="gov_program")
        gov_program.program_id = MagicMock()
        gov_program.account = {"Proposal": AsyncMock()}
        gov_program.account["Proposal"].all = AsyncMock(return_value=[p1, p2])
        self._patch_gov_program(gov_program)

        proposals = asyncio.run(client.get_governance_proposals())

        assert isinstance(proposals, list)
        assert len(proposals) == 2
        required_keys = {"id", "title", "description", "yes_votes", "no_votes", "status", "ends_at"}
        for p in proposals:
            assert required_keys == required_keys & set(p.keys()), (
                f"Missing keys in proposal dict: {required_keys - set(p.keys())}"
            )

    def test_get_proposals_returns_empty_list_when_no_proposals(self, bc_module, fake_config):
        import asyncio

        client, _, _, rpc, _, _ = _build_client(bc_module, fake_config)

        gov_program = AsyncMock(name="gov_program")
        gov_program.program_id = MagicMock()
        gov_program.account = {"Proposal": AsyncMock()}
        gov_program.account["Proposal"].all = AsyncMock(return_value=[])
        self._patch_gov_program(gov_program)

        proposals = asyncio.run(client.get_governance_proposals())

        assert proposals == []


# ---------------------------------------------------------------------------
# CLI integration tests for withdraw and governance vote
# ---------------------------------------------------------------------------


class TestCmdWithdrawCLI:
    """Test the CLI cmd_withdraw function against the real BlockchainClient mock."""

    def _write_config(self, tmp_path):
        import yaml

        config_file = tmp_path / "config.yaml"
        config_data = {
            "rpc_url": "https://api.devnet.solana.com",
            "wallet_path": str(tmp_path / "id.json"),
            "compute_registry_program": "8KpR2mT6uLqVwNzS4eBfY9oA3cJ7iGxH1nD5sW0qF2M",
            "inference_market_program": "5YQyZqXkJHy6V3JMxKqXyLqfP9V2A3j8Rk7mN4oD1eW",
        }
        with config_file.open("w") as fh:
            yaml.dump(config_data, fh)
        (tmp_path / "id.json").write_text(str(list(range(64))))
        return config_file

    def test_withdraw_calls_blockchain_and_prints_tx(self, tmp_path, capsys):
        import importlib

        config_file = self._write_config(tmp_path)

        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.close = AsyncMock()
        mock_client.withdraw_earnings = AsyncMock(
            return_value={
                "tx_signature": "AbCdEfGh1234",
                "amount_sol": 1.0,
                "new_balance_sol": 4.0,
            }
        )

        mock_bc_cls = MagicMock(return_value=mock_client)
        mock_node_config_cls = MagicMock(return_value=MagicMock())

        stubs, _ = _make_solana_stubs()
        with patch.dict(sys.modules, stubs):
            import scripts.node_cli as cli_mod

            importlib.reload(cli_mod)

            with (
                patch.object(cli_mod, "BlockchainClient", mock_bc_cls),
                patch.object(cli_mod, "NodeConfig", mock_node_config_cls),
            ):
                parser = cli_mod.build_parser()
                args = parser.parse_args(["withdraw", "--config-path", str(config_file)])
                cli_mod.cmd_withdraw(args)

        captured = capsys.readouterr()
        assert "AbCdEfGh1234" in captured.out
        mock_client.withdraw_earnings.assert_called_once_with(None)


class TestCmdGovernanceVoteCLI:
    """Test the CLI cmd_governance_vote function."""

    def _write_config(self, tmp_path):
        import yaml

        config_file = tmp_path / "config.yaml"
        config_data = {
            "rpc_url": "https://api.devnet.solana.com",
            "wallet_path": str(tmp_path / "id.json"),
            "compute_registry_program": "8KpR2mT6uLqVwNzS4eBfY9oA3cJ7iGxH1nD5sW0qF2M",
            "inference_market_program": "5YQyZqXkJHy6V3JMxKqXyLqfP9V2A3j8Rk7mN4oD1eW",
        }
        with config_file.open("w") as fh:
            yaml.dump(config_data, fh)
        (tmp_path / "id.json").write_text(str(list(range(64))))
        return config_file

    def test_governance_vote_for_calls_cast_vote(self, tmp_path, capsys):
        import importlib

        config_file = self._write_config(tmp_path)

        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.close = AsyncMock()
        mock_client.cast_governance_vote = AsyncMock(return_value="VoteTxSig123")

        mock_bc_cls = MagicMock(return_value=mock_client)
        mock_node_config_cls = MagicMock(return_value=MagicMock())

        stubs, _ = _make_solana_stubs()
        with patch.dict(sys.modules, stubs):
            import scripts.node_cli as cli_mod

            importlib.reload(cli_mod)

            with (
                patch.object(cli_mod, "BlockchainClient", mock_bc_cls),
                patch.object(cli_mod, "NodeConfig", mock_node_config_cls),
            ):
                parser = cli_mod.build_parser()
                args = parser.parse_args(
                    ["governance", "vote", "42", "for", "--config-path", str(config_file)]
                )
                cli_mod.cmd_governance_vote(args)

        captured = capsys.readouterr()
        assert "VoteTxSig123" in captured.out
        mock_client.cast_governance_vote.assert_called_once_with(42, True)

    def test_governance_vote_dry_run_prints_without_transacting(self, tmp_path, capsys):
        import importlib

        config_file = self._write_config(tmp_path)

        mock_bc_cls = MagicMock()
        mock_node_config_cls = MagicMock(return_value=MagicMock())

        stubs, _ = _make_solana_stubs()
        with patch.dict(sys.modules, stubs):
            import scripts.node_cli as cli_mod

            importlib.reload(cli_mod)

            with (
                patch.object(cli_mod, "BlockchainClient", mock_bc_cls),
                patch.object(cli_mod, "NodeConfig", mock_node_config_cls),
            ):
                parser = cli_mod.build_parser()
                args = parser.parse_args(
                    [
                        "governance",
                        "vote",
                        "7",
                        "against",
                        "--dry-run",
                        "--config-path",
                        str(config_file),
                    ]
                )
                cli_mod.cmd_governance_vote(args)

        captured = capsys.readouterr()
        assert "dry run" in captured.out.lower()
        # BlockchainClient must never be instantiated on a dry run
        mock_bc_cls.assert_not_called()
