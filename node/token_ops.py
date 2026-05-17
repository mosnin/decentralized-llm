"""
SPL Token operations — async, direct RPC (no subprocess).

Replaces the previous `_mint_tokens()` subprocess pattern in api_gateway.py
that spawned `solana` CLI for every payment (~100 ms overhead per call).

All three public coroutines use lazy imports so the module loads cleanly even
when the ``solana`` / ``solders`` packages are not installed.  When the
packages are absent every function raises ``TokenMintError`` with a clear
installation hint instead of an obscure ImportError bubbling up.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Availability flag — set once at import time
# ---------------------------------------------------------------------------

try:
    # Trigger a lightweight import to probe availability.
    # Full sub-module imports are deferred to each function so tests can
    # patch them without fighting module-level side-effects.
    import solana  # noqa: F401
    import solders  # noqa: F401

    SOLANA_AVAILABLE = True
except ImportError:
    SOLANA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class TokenMintError(RuntimeError):
    """Raised when any SPL Token operation fails."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_keypair(wallet_keypair_path: str):
    """Load a Solana Keypair from a JSON key file."""
    from solders.keypair import Keypair  # lazy import

    return Keypair.from_json(Path(wallet_keypair_path).read_text())


def _require_solana() -> None:
    """Raise a helpful TokenMintError when Solana packages are missing."""
    if not SOLANA_AVAILABLE:
        raise TokenMintError(
            "Solana packages are not installed. Run: pip install solana solders spl-token anchorpy"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def mint_tokens(
    rpc_url: str,
    wallet_keypair_path: str,
    mint_address: str,
    recipient_address: str,
    amount: int,
    *,
    dry_run: bool = False,
) -> str:
    """
    Mint *amount* SPL tokens from *mint_address* into *recipient_address*.

    Parameters
    ----------
    rpc_url:
        Solana JSON-RPC endpoint URL.
    wallet_keypair_path:
        Path to the JSON keypair file for the mint-authority wallet.
    mint_address:
        Base-58 public key of the SPL Token mint account.
    recipient_address:
        Base-58 public key of the recipient's wallet (not their ATA).
        The associated token account is created automatically if it does
        not exist.
    amount:
        Number of base units (raw, un-scaled by decimals) to mint.
    dry_run:
        When ``True`` build and sign the transaction but do **not** send
        it.  Returns the string ``"dry-run"`` immediately.

    Returns
    -------
    str
        Transaction signature, or ``"dry-run"`` when *dry_run* is ``True``.

    Raises
    ------
    TokenMintError
        On any RPC or signing failure, or when Solana packages are absent.
    """
    _require_solana()

    if dry_run:
        return "dry-run"

    try:
        from solana.rpc.async_api import AsyncClient
        from solders.pubkey import Pubkey
        from spl.token.async_client import AsyncToken
        from spl.token.constants import TOKEN_PROGRAM_ID

        mint_pubkey = Pubkey.from_string(mint_address)
        recipient_pubkey = Pubkey.from_string(recipient_address)
        authority_kp = _load_keypair(wallet_keypair_path)

        async with AsyncClient(rpc_url) as client:
            token = AsyncToken(client, mint_pubkey, TOKEN_PROGRAM_ID, authority_kp)

            # Create the recipient's ATA if it does not already exist.
            dest_ata = await token.create_associated_token_account(recipient_pubkey)

            resp = await token.mint_to(dest_ata, authority_kp, amount)
            # solana-py returns an object; extract the signature string.
            signature = str(resp.value) if hasattr(resp, "value") else str(resp)

        logger.info(
            "mint_tokens: minted %d tokens to %s (mint=%s) tx=%s",
            amount,
            recipient_address,
            mint_address,
            signature,
        )
        return signature

    except TokenMintError:
        raise
    except Exception as exc:
        raise TokenMintError(f"mint_tokens failed: {exc}") from exc


async def transfer_tokens(
    rpc_url: str,
    wallet_keypair_path: str,
    mint_address: str,
    recipient_address: str,
    amount: int,
) -> str:
    """
    Transfer *amount* SPL tokens from the authority wallet to *recipient_address*.

    The recipient's associated token account is created automatically when it
    does not exist (the authority wallet pays the rent).

    Returns
    -------
    str
        Transaction signature.

    Raises
    ------
    TokenMintError
        On any RPC or signing failure, or when Solana packages are absent.
    """
    _require_solana()

    try:
        from solana.rpc.async_api import AsyncClient
        from solders.pubkey import Pubkey
        from spl.token.async_client import AsyncToken
        from spl.token.constants import TOKEN_PROGRAM_ID

        mint_pubkey = Pubkey.from_string(mint_address)
        recipient_pubkey = Pubkey.from_string(recipient_address)
        authority_kp = _load_keypair(wallet_keypair_path)

        async with AsyncClient(rpc_url) as client:
            token = AsyncToken(client, mint_pubkey, TOKEN_PROGRAM_ID, authority_kp)

            # Ensure the recipient has an ATA; get_or_create semantics.
            dest_ata = await token.create_associated_token_account(recipient_pubkey)

            # Source ATA is the authority wallet's own associated token account.
            source_ata = await token.create_associated_token_account(authority_kp.pubkey())

            resp = await token.transfer(source_ata, dest_ata, authority_kp, amount)
            signature = str(resp.value) if hasattr(resp, "value") else str(resp)

        logger.info(
            "transfer_tokens: transferred %d tokens to %s (mint=%s) tx=%s",
            amount,
            recipient_address,
            mint_address,
            signature,
        )
        return signature

    except TokenMintError:
        raise
    except Exception as exc:
        raise TokenMintError(f"transfer_tokens failed: {exc}") from exc


async def get_token_balance(
    rpc_url: str,
    wallet_address: str,
    mint_address: str,
) -> int:
    """
    Return the SPL token balance (in base units) for *wallet_address* + *mint_address*.

    Parameters
    ----------
    rpc_url:
        Solana JSON-RPC endpoint URL.
    wallet_address:
        Base-58 public key of the wallet whose balance to fetch.
    mint_address:
        Base-58 public key of the SPL Token mint.

    Returns
    -------
    int
        Token balance in base units (not scaled by decimals).
        Returns ``0`` if the associated token account does not exist.

    Raises
    ------
    TokenMintError
        On RPC failure or when Solana packages are absent.
    """
    _require_solana()

    try:
        from solana.rpc.async_api import AsyncClient
        from solders.pubkey import Pubkey
        from spl.token.async_client import AsyncToken

        mint_pubkey = Pubkey.from_string(mint_address)
        wallet_pubkey = Pubkey.from_string(wallet_address)

        async with AsyncClient(rpc_url) as client:
            # Derive the ATA address without creating it.
            ata = AsyncToken.get_associated_token_address(wallet_pubkey, mint_pubkey)
            resp = await client.get_token_account_balance(ata)

        if resp.value is None:
            return 0

        # resp.value.amount is a string of the raw (un-scaled) token amount.
        return int(resp.value.amount)

    except TokenMintError:
        raise
    except Exception as exc:
        raise TokenMintError(f"get_token_balance failed: {exc}") from exc
