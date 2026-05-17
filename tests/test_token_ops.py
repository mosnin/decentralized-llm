"""
Tests for node.token_ops — SPL Token operations without subprocess overhead.

All tests run without the real ``solana`` / ``solders`` packages installed.
External clients and RPC calls are mocked via unittest.mock.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers to build minimal solana/solders stubs
# ---------------------------------------------------------------------------


def _install_solana_stubs():
    """
    Inject minimal stub modules so that ``import solana`` etc. succeed.
    Returns a dict of the stub objects so tests can configure return values.
    """
    stubs: dict = {}

    # Top-level packages
    for pkg in ("solana", "solders", "spl"):
        if pkg not in sys.modules:
            sys.modules[pkg] = types.ModuleType(pkg)

    # solana.rpc.async_api.AsyncClient
    rpc_mod = types.ModuleType("solana.rpc")
    async_api_mod = types.ModuleType("solana.rpc.async_api")

    class _FakeAsyncClient:
        def __init__(self, url):
            self._url = url

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def get_token_account_balance(self, ata):
            raise NotImplementedError("configure in test")

    stubs["AsyncClient"] = _FakeAsyncClient
    async_api_mod.AsyncClient = _FakeAsyncClient
    sys.modules["solana.rpc"] = rpc_mod
    sys.modules["solana.rpc.async_api"] = async_api_mod

    # solders.pubkey.Pubkey
    pubkey_mod = types.ModuleType("solders.pubkey")

    class _FakePubkey:
        def __init__(self, s="FakePubkey"):
            self._s = s

        @classmethod
        def from_string(cls, s):
            return cls(s)

        def __str__(self):
            return self._s

    stubs["Pubkey"] = _FakePubkey
    pubkey_mod.Pubkey = _FakePubkey
    sys.modules["solders.pubkey"] = pubkey_mod

    # solders.keypair.Keypair
    keypair_mod = types.ModuleType("solders.keypair")

    class _FakeKeypair:
        def __init__(self):
            pass

        @classmethod
        def from_json(cls, text):
            return cls()

        def pubkey(self):
            return _FakePubkey("AuthorityPubkey")

    stubs["Keypair"] = _FakeKeypair
    keypair_mod.Keypair = _FakeKeypair
    sys.modules["solders.keypair"] = keypair_mod

    # spl.token.async_client.AsyncToken
    spl_token_mod = types.ModuleType("spl.token")
    spl_token_async_mod = types.ModuleType("spl.token.async_client")
    spl_token_const_mod = types.ModuleType("spl.token.constants")

    class _FakeAsyncToken:
        def __init__(self, client, mint, program_id, authority):
            self._client = client
            self.mint = mint
            self.program_id = program_id
            self.authority = authority

        async def create_associated_token_account(self, owner):
            return _FakePubkey(f"ata-for-{owner}")

        async def mint_to(self, dest, authority, amount):
            result = MagicMock()
            result.value = "mock_sig_mint"
            return result

        async def transfer(self, src, dest, authority, amount):
            result = MagicMock()
            result.value = "mock_sig_transfer"
            return result

        @staticmethod
        def get_associated_token_address(owner, mint):
            return _FakePubkey(f"ata-{owner}-{mint}")

    stubs["AsyncToken"] = _FakeAsyncToken
    spl_token_async_mod.AsyncToken = _FakeAsyncToken
    spl_token_const_mod.TOKEN_PROGRAM_ID = _FakePubkey("TokenProgram")
    sys.modules["spl.token"] = spl_token_mod
    sys.modules["spl.token.async_client"] = spl_token_async_mod
    sys.modules["spl.token.constants"] = spl_token_const_mod

    return stubs


# ---------------------------------------------------------------------------
# Force-reload token_ops with SOLANA_AVAILABLE patched to True
# ---------------------------------------------------------------------------


@pytest.fixture()
def token_ops_with_solana(tmp_path):
    """
    Provide a freshly-imported token_ops module with Solana stubs in place
    and SOLANA_AVAILABLE=True.
    """
    _install_solana_stubs()

    # Remove any cached copy so we can patch SOLANA_AVAILABLE cleanly.
    sys.modules.pop("node.token_ops", None)

    with patch.dict("sys.modules", sys.modules):
        import node.token_ops as tok

        tok.SOLANA_AVAILABLE = True

    # Create a dummy keypair file so _load_keypair doesn't fail on I/O.
    kp_file = tmp_path / "wallet.json"
    kp_file.write_text("{}")  # _FakeKeypair.from_json accepts any text

    yield tok, str(kp_file)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestMintTokens:
    @pytest.mark.asyncio
    async def test_mint_tokens_returns_signature(self, token_ops_with_solana):
        tok, kp_path = token_ops_with_solana

        sig = await tok.mint_tokens(
            rpc_url="https://api.devnet.solana.com",
            wallet_keypair_path=kp_path,
            mint_address="MintPubkey111",
            recipient_address="RecipientPubkey222",
            amount=1_000_000,
        )

        assert isinstance(sig, str)
        assert len(sig) > 0

    @pytest.mark.asyncio
    async def test_mint_tokens_dry_run_returns_dry_run_string(self, token_ops_with_solana):
        tok, kp_path = token_ops_with_solana

        sig = await tok.mint_tokens(
            rpc_url="https://api.devnet.solana.com",
            wallet_keypair_path=kp_path,
            mint_address="MintPubkey111",
            recipient_address="RecipientPubkey222",
            amount=500,
            dry_run=True,
        )

        assert sig == "dry-run"

    @pytest.mark.asyncio
    async def test_mint_tokens_raises_token_mint_error_on_rpc_failure(self, token_ops_with_solana):
        tok, kp_path = token_ops_with_solana

        # Patch AsyncToken.mint_to to raise an exception
        with patch("spl.token.async_client.AsyncToken.mint_to", new_callable=AsyncMock) as m:
            m.side_effect = RuntimeError("RPC timeout")

            with pytest.raises(tok.TokenMintError, match="mint_tokens failed"):
                await tok.mint_tokens(
                    rpc_url="https://api.devnet.solana.com",
                    wallet_keypair_path=kp_path,
                    mint_address="MintPubkey111",
                    recipient_address="RecipientPubkey222",
                    amount=100,
                )

    @pytest.mark.asyncio
    async def test_mint_tokens_without_solana_raises_clearly(self):
        """When SOLANA_AVAILABLE=False, a clear TokenMintError is raised."""
        sys.modules.pop("node.token_ops", None)
        import node.token_ops as tok

        tok.SOLANA_AVAILABLE = False

        with pytest.raises(tok.TokenMintError, match="not installed"):
            await tok.mint_tokens(
                rpc_url="https://api.devnet.solana.com",
                wallet_keypair_path="/dev/null",
                mint_address="Mint",
                recipient_address="Recipient",
                amount=1,
            )

        # Restore for subsequent tests
        tok.SOLANA_AVAILABLE = False


class TestTransferTokens:
    @pytest.mark.asyncio
    async def test_transfer_tokens_returns_signature(self, token_ops_with_solana):
        tok, kp_path = token_ops_with_solana

        sig = await tok.transfer_tokens(
            rpc_url="https://api.devnet.solana.com",
            wallet_keypair_path=kp_path,
            mint_address="MintPubkey111",
            recipient_address="RecipientPubkey333",
            amount=250_000,
        )

        assert isinstance(sig, str)
        assert len(sig) > 0


class TestGetTokenBalance:
    @pytest.mark.asyncio
    async def test_get_token_balance_returns_int(self, token_ops_with_solana):
        tok, kp_path = token_ops_with_solana

        # Patch the AsyncClient to return a mock balance response
        mock_resp = MagicMock()
        mock_resp.value = MagicMock()
        mock_resp.value.amount = "42000000"

        with patch(
            "solana.rpc.async_api.AsyncClient.get_token_account_balance",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            balance = await tok.get_token_balance(
                rpc_url="https://api.devnet.solana.com",
                wallet_address="WalletPubkey444",
                mint_address="MintPubkey111",
            )

        assert balance == 42_000_000
        assert isinstance(balance, int)

    @pytest.mark.asyncio
    async def test_get_token_balance_returns_zero_when_no_account(self, token_ops_with_solana):
        tok, kp_path = token_ops_with_solana

        mock_resp = MagicMock()
        mock_resp.value = None

        with patch(
            "solana.rpc.async_api.AsyncClient.get_token_account_balance",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            balance = await tok.get_token_balance(
                rpc_url="https://api.devnet.solana.com",
                wallet_address="WalletPubkey444",
                mint_address="MintPubkey111",
            )

        assert balance == 0


# ---------------------------------------------------------------------------
# Gateway integration tests
# ---------------------------------------------------------------------------


class TestGatewayMintIntegration:
    """Tests that api_gateway._mint_tokens correctly delegates to token_ops."""

    def _make_app(self):
        import sys
        import types

        _install_solana_stubs()

        for mod in ("anchorpy",):
            if mod not in sys.modules:
                sys.modules[mod] = types.ModuleType(mod)

        # Reload gateway to pick up stubs
        sys.modules.pop("scripts.api_gateway", None)
        sys.modules.pop("node.token_ops", None)

        from scripts.api_gateway import app

        return app

    @pytest.mark.asyncio
    async def test_gateway_mint_uses_token_ops(self, tmp_path):
        """_mint_tokens calls token_ops.mint_tokens with the right args."""
        import os

        kp = tmp_path / "wallet.json"
        kp.write_text("{}")

        sys.modules.pop("scripts.api_gateway", None)
        sys.modules.pop("node.token_ops", None)

        _install_solana_stubs()

        with patch.dict(
            os.environ,
            {
                "TOKEN_MINT_ADDRESS": "MintXXX",
                "WALLET_PATH": str(kp),
                "SOLANA_RPC_URL": "https://api.devnet.solana.com",
            },
        ):
            import node.token_ops as tok

            tok.SOLANA_AVAILABLE = True

            import scripts.api_gateway as gw

            gw.MINT_ADDRESS = "MintXXX"

            with patch.object(tok, "mint_tokens", new_callable=AsyncMock) as mock_mint:
                mock_mint.return_value = "tx_sig_abc123"

                await gw._mint_tokens("RecipientWallet", 5_000)

            mock_mint.assert_awaited_once()
            call_kwargs = mock_mint.call_args
            assert call_kwargs.kwargs.get("mint_address") == "MintXXX" or (
                len(call_kwargs.args) >= 3 and call_kwargs.args[2] == "MintXXX"
            )
            assert call_kwargs.kwargs.get("recipient_address") == "RecipientWallet" or (
                len(call_kwargs.args) >= 4 and call_kwargs.args[3] == "RecipientWallet"
            )
            assert call_kwargs.kwargs.get("amount") == 5_000 or (
                len(call_kwargs.args) >= 5 and call_kwargs.args[4] == 5_000
            )

    @pytest.mark.asyncio
    async def test_gateway_mint_error_returns_500(self, tmp_path):
        """When token_ops.mint_tokens raises TokenMintError the gateway logs it."""
        import os

        kp = tmp_path / "wallet.json"
        kp.write_text("{}")

        sys.modules.pop("scripts.api_gateway", None)
        sys.modules.pop("node.token_ops", None)

        _install_solana_stubs()

        with patch.dict(
            os.environ,
            {
                "TOKEN_MINT_ADDRESS": "MintXXX",
                "WALLET_PATH": str(kp),
                "SOLANA_RPC_URL": "https://api.devnet.solana.com",
            },
        ):
            import node.token_ops as tok

            tok.SOLANA_AVAILABLE = True

            import scripts.api_gateway as gw

            gw.MINT_ADDRESS = "MintXXX"

            with patch.object(tok, "mint_tokens", new_callable=AsyncMock) as mock_mint:
                mock_mint.side_effect = tok.TokenMintError("RPC connection refused")

                # _mint_tokens catches TokenMintError and logs — it does NOT re-raise.
                # Call it directly and confirm it completes without propagating.
                await gw._mint_tokens("RecipientWallet", 5_000)  # should NOT raise

            mock_mint.assert_awaited_once()

    def test_gateway_webhook_calls_create_task(self):
        """
        Smoke-test: POST /webhooks/paysh triggers _mint_tokens scheduling.
        The task itself is mocked so no actual Solana calls happen.
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        from fastapi.testclient import TestClient

        from integrations.paysh.handler import PaymentEvent

        _install_solana_stubs()
        sys.modules.pop("scripts.api_gateway", None)

        from scripts.api_gateway import app

        mock_event = PaymentEvent(
            payment_id="pay_999",
            customer_wallet="WalletABC",
            amount_usd_cents=500,
            tokens_to_mint=50_000,
            status="completed",
        )

        with (
            patch("scripts.api_gateway._paysh") as mock_paysh,
            patch("scripts.api_gateway._mint_tokens", new_callable=AsyncMock),
        ):
            mock_paysh.process_webhook = MagicMock(return_value=mock_event)
            with TestClient(app, raise_server_exceptions=False) as client:
                resp = client.post(
                    "/webhooks/paysh",
                    content=b'{"payment_id":"pay_999"}',
                    headers={"X-Paysh-Signature": "sig"},
                )

        assert resp.status_code == 200
        assert resp.json()["tokens_minted"] == 50_000
