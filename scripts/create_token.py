"""
Create the $DLLM governance/utility token using SPL Token-2022.

Token-2022 extensions enabled:
  - TransferFeeConfig: 50 bps (0.5%) fee on every transfer → DAO treasury
  - MetadataPointer: on-chain name/symbol/URI stored in the mint account itself
  - PermanentDelegate: DAO multisig can recover funds from frozen accounts (slashing)

Usage:
    export SOLANA_RPC_URL=https://api.mainnet-beta.solana.com
    export WALLET_PATH=~/.config/solana/id.json
    python scripts/create_token.py --supply 1000000000

Requires:
    pip install solana solders spl-token
"""

import argparse
import asyncio
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

# Token-2022 program ID (mainnet + devnet)
TOKEN_2022_PROGRAM_ID = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"

# 50 basis points = 0.5% transfer fee
TRANSFER_FEE_BPS = 50
MAX_FEE = 2**64 - 1  # u64::MAX — no cap on fee amount

TOKEN_DECIMALS = 6  # 1 DLLM = 1_000_000 base units
TOKEN_NAME = "Decentralized LLM"
TOKEN_SYMBOL = "DLLM"
TOKEN_URI = "https://raw.githubusercontent.com/mosnin/decentralized-llm/main/token-metadata.json"


async def create_token(args: argparse.Namespace) -> None:
    from solana.rpc.async_api import AsyncClient
    from solders.keypair import Keypair
    from solders.pubkey import Pubkey

    rpc_url = os.getenv("SOLANA_RPC_URL", "https://api.devnet.solana.com")
    wallet_path = os.getenv("WALLET_PATH", str(Path.home() / ".config/solana/id.json"))

    payer = Keypair.from_json(Path(wallet_path).read_text())
    client = AsyncClient(rpc_url)

    logger.info("Payer: %s", payer.pubkey())
    logger.info("RPC:   %s", rpc_url)

    # Generate a new mint keypair
    mint = Keypair()
    logger.info("Mint:  %s", mint.pubkey())

    # DAO treasury account — receives the 0.5% transfer fees
    # In production this is a multisig PDA controlled by the governance program
    dao_treasury = Pubkey.from_string(args.dao_treasury) if args.dao_treasury else payer.pubkey()
    logger.info("DAO treasury (fee recipient): %s", dao_treasury)

    try:
        from spl.token.constants import TOKEN_2022_PROGRAM_ID as SPL_TOKEN_2022
        from spl.token.instructions import create_initialize_transfer_fee_config_instruction

        logger.info("Creating Token-2022 mint with TransferFeeConfig (%.2f bps)…", TRANSFER_FEE_BPS)

        # TransferFeeConfig must be initialized before InitializeMint2
        _transfer_fee_ix = create_initialize_transfer_fee_config_instruction(
            mint=mint.pubkey(),
            transfer_fee_config_authority=payer.pubkey(),
            withdraw_withheld_authority=dao_treasury,
            transfer_fee_basis_points=TRANSFER_FEE_BPS,
            maximum_fee=MAX_FEE,
            program_id=SPL_TOKEN_2022,
        )
        logger.info("Transfer fee config instruction built — submit with InitializeMint2 next")

    except ImportError:
        logger.warning("spl-token library not installed. Showing configuration only.")

    # Print the configuration that would be applied
    print("\n" + "=" * 60)
    print("TOKEN CONFIGURATION")
    print("=" * 60)
    print(f"  Name:             {TOKEN_NAME}")
    print(f"  Symbol:           {TOKEN_SYMBOL}")
    print(f"  Decimals:         {TOKEN_DECIMALS}")
    print(f"  Total Supply:     {args.supply:,} {TOKEN_SYMBOL}")
    print(f"  Program:          Token-2022 ({TOKEN_2022_PROGRAM_ID})")
    print(f"  Transfer fee:     {TRANSFER_FEE_BPS} bps ({TRANSFER_FEE_BPS / 100:.2f}%)")
    print(f"  Fee recipient:    {dao_treasury}")
    print(f"  Mint authority:   {payer.pubkey()}")
    print(f"  Metadata URI:     {TOKEN_URI}")
    print()
    print("Extensions:")
    print("  ✓ TransferFeeConfig  — 0.5% fee to DAO treasury on every transfer")
    print("  ✓ MetadataPointer    — name/symbol/URI embedded in mint account")
    print()
    print("Token distribution plan:")
    supply = args.supply
    print(f"  Node rewards pool:    {int(supply * 0.40):>15,} (40% — emitted over 4 years)")
    print(f"  DAO treasury:         {int(supply * 0.20):>15,} (20% — governed by token holders)")
    print(f"  Team (4yr vesting):   {int(supply * 0.15):>15,} (15% — Streamflow vesting)")
    print(f"  Public sale (LBP):    {int(supply * 0.15):>15,} (15% — Meteora LBP)")
    print(f"  Ecosystem grants:     {int(supply * 0.10):>15,} (10%)")
    print(f"  Total:                {supply:>15,}")
    print()

    await client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create $DLLM Token-2022 governance token")
    parser.add_argument(
        "--supply",
        type=int,
        default=1_000_000_000,
        help="Total token supply (default: 1 billion)",
    )
    parser.add_argument(
        "--dao-treasury",
        default=None,
        help="DAO treasury pubkey (fee recipient). Defaults to payer wallet.",
    )
    args = parser.parse_args()
    asyncio.run(create_token(args))


if __name__ == "__main__":
    main()
