"""
CLI tool for decentralized-LLM node operators.

Usage:
    python -m scripts.node_cli setup              # Generate Solana keypair and config file
    python -m scripts.node_cli register           # Register node on-chain
    python -m scripts.node_cli start              # Start the node server
    python -m scripts.node_cli status             # Show node status (registration, jobs, earnings)
    python -m scripts.node_cli earnings           # Show earnings summary
    python -m scripts.node_cli withdraw           # Withdraw earnings to wallet
    python -m scripts.node_cli config show        # Show current config as YAML
    python -m scripts.node_cli governance list    # List active governance proposals
    python -m scripts.node_cli governance vote <proposal_id> <for|against|abstain>
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import logging
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path.home() / ".decentralized-llm" / "config.yaml"

DEFAULT_CONFIG = {
    "rpc_url": "https://api.mainnet-beta.solana.com",
    "wallet_path": str(Path.home() / ".config" / "solana" / "id.json"),
    "compute_registry_program": "8KpR2mT6uLqVwNzS4eBfY9oA3cJ7iGxH1nD5sW0qF2M",
    "inference_market_program": "5YQyZqXkJHy6V3JMxKqXyLqfP9V2A3j8Rk7mN4oD1eW",
    "listen_host": "0.0.0.0",
    "listen_port": 7070,
    "public_host": None,
    "model_name": "meta-llama/Llama-3.2-3B",
    "num_shards": 4,
    "shard_index": 0,
    "gpu_count": 1,
    "vram_gb": 24,
    "supported_models": ["meta-llama/Llama-3.2-3B"],
    "lighthouse_api_key": "",
    "max_concurrent_jobs": 4,
    "job_poll_interval_seconds": 2.0,
    "stake_amount": 0,
}

# Optional imports – kept at module scope so tests can patch them.
try:
    from node.blockchain import BlockchainClient
    from node.config import NodeConfig
except ImportError:  # pragma: no cover
    BlockchainClient = None  # type: ignore[assignment,misc]
    NodeConfig = None  # type: ignore[assignment,misc]


# ─────────────────────────── helpers ───────────────────────────────────────


def _config_path_from_args(args: argparse.Namespace) -> Path:
    return Path(args.config_path) if args.config_path else DEFAULT_CONFIG_PATH


def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        print(
            f"Config file not found: {config_path}\nRun `python -m scripts.node_cli setup` first.",
            file=sys.stderr,
        )
        sys.exit(1)
    with config_path.open() as fh:
        return yaml.safe_load(fh) or {}


def _write_config(config_path: Path, config: dict) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as fh:
        yaml.dump(config, fh, default_flow_style=False, sort_keys=True)


# ─────────────────────────── subcommands ───────────────────────────────────


def cmd_setup(args: argparse.Namespace) -> None:
    """Create ~/.decentralized-llm/config.yaml with defaults."""
    config_path = _config_path_from_args(args)

    if config_path.exists():
        print(f"Config already exists at {config_path}")
        print("Delete it first if you want to re-initialise.")
        return

    print("=== Decentralized LLM Node Setup ===\n")

    # Step 1: keypair instructions
    print("Step 1 – Generate a Solana keypair (skip if you already have one):\n")
    print("    solana-keygen new --outfile ~/.config/solana/id.json\n")
    print(
        "    This creates your node identity. Keep the seed phrase safe!\n"
        "    The public key printed by solana-keygen is your node's on-chain identity.\n"
    )

    # Step 2: Lighthouse API key
    print("Step 2 – Lighthouse API key (for IPFS/Filecoin result storage):")
    print("    Get your key at https://files.lighthouse.storage/\n")
    try:
        lighthouse_key = getpass.getpass(
            "    Enter LIGHTHOUSE_API_KEY (hidden, press Enter to skip): "
        )
    except (EOFError, KeyboardInterrupt):
        lighthouse_key = ""

    # Step 3: write config
    config = dict(DEFAULT_CONFIG)
    config["lighthouse_api_key"] = lighthouse_key.strip()

    _write_config(config_path, config)

    print(f"\nConfig written to {config_path}")
    print("Edit it to set your wallet_path, rpc_url, model_name, etc.")
    print("\nNext step: python -m scripts.node_cli register")


def cmd_register(args: argparse.Namespace) -> None:
    """Register this node in the on-chain compute-registry."""
    config_path = _config_path_from_args(args)
    config = _load_config(config_path)

    endpoint = (
        f"{config.get('public_host') or config.get('listen_host', '0.0.0.0')}"
        f":{config.get('listen_port', 7070)}"
    )
    gpu_count = config.get("gpu_count", 1)
    vram_gb = config.get("vram_gb", 24)
    models = config.get("supported_models", [config.get("model_name", "")])
    stake_amount = config.get("stake_amount", 0)
    registry_program = config.get("compute_registry_program", "")
    wallet_path = config.get("wallet_path", "~/.config/solana/id.json")

    print("=== Node Registration Parameters ===\n")
    print(f"  Endpoint      : {endpoint}")
    print(f"  GPU count     : {gpu_count}")
    print(f"  VRAM (GB)     : {vram_gb}")
    print(f"  Models        : {', '.join(models)}")
    print(f"  Stake amount  : {stake_amount}")
    print(f"  Registry prog : {registry_program}")
    print()

    model_args = " ".join(f'"{m}"' for m in models)
    print("Anchor CLI command (dry-run preview):\n")
    print(
        f"    anchor invoke {registry_program} register_node \\\n"
        f'        --endpoint "{endpoint}" \\\n'
        f"        --vram-gb {vram_gb} \\\n"
        f"        --gpu-count {gpu_count} \\\n"
        f"        --model-ids {model_args} \\\n"
        f"        --stake-amount {stake_amount} \\\n"
        f"        --provider.wallet {wallet_path}"
    )
    print()

    if not args.execute:
        print("(Dry run – pass --execute to actually send the transaction.)")
        return

    # Live execution path
    import hashlib

    if BlockchainClient is None or NodeConfig is None:
        print(
            "node.blockchain / node.config not importable. Make sure the package is installed.",
            file=sys.stderr,
        )
        sys.exit(1)

    node_config = NodeConfig()
    for key, val in config.items():
        if hasattr(node_config, key):
            setattr(node_config, key, val)

    client = BlockchainClient(node_config)

    async def _run() -> None:
        await client.connect()
        model_ids = [hashlib.sha256(m.encode()).digest() for m in models]
        success = await client.register_node(
            endpoint=endpoint,
            vram_gb=vram_gb,
            gpu_count=gpu_count,
            model_ids=model_ids,
            stake_amount=stake_amount,
        )
        await client.close()
        if success:
            print("Node registered successfully.")
        else:
            print("Registration failed (already registered, or check logs).", file=sys.stderr)
            sys.exit(1)

    asyncio.run(_run())


def cmd_start(args: argparse.Namespace) -> None:
    """Start the node server."""
    import signal

    config_path = _config_path_from_args(args)
    config = _load_config(config_path)

    if NodeConfig is None:  # pragma: no cover
        print("node.config not importable.", file=sys.stderr)
        sys.exit(1)

    from node.server import Node

    node_config = NodeConfig()
    for key, val in config.items():
        if hasattr(node_config, key):
            setattr(node_config, key, val)

    node = Node(node_config)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(node.stop()))

    try:
        loop.run_until_complete(node.start())
    finally:
        loop.close()


def cmd_status(args: argparse.Namespace) -> None:
    """Fetch and print this node's on-chain status."""
    config_path = _config_path_from_args(args)
    config = _load_config(config_path)

    rpc_url = config.get("rpc_url", "https://api.mainnet-beta.solana.com")
    wallet_path = config.get("wallet_path", str(Path.home() / ".config/solana/id.json"))
    registry_program = config.get("compute_registry_program", "")

    print(f"Connecting to RPC: {rpc_url}")
    print(f"Wallet           : {wallet_path}")
    print(f"Registry program : {registry_program}\n")

    try:
        from solana.rpc.async_api import AsyncClient
        from solders.keypair import Keypair
        from solders.pubkey import Pubkey

        solana_available = True
    except ImportError:
        solana_available = False

    if not solana_available:
        print(
            "Solana packages not installed. Install with:\n"
            "  pip install anchorpy solders solana\n\n"
            "Showing config-derived status only:\n",
            file=sys.stderr,
        )
        endpoint = (
            f"{config.get('public_host') or config.get('listen_host', '0.0.0.0')}"
            f":{config.get('listen_port', 7070)}"
        )
        _print_status(
            endpoint=endpoint,
            stake=config.get("stake_amount", 0),
            reputation=None,
            jobs_completed=None,
            jobs_disputed=None,
            earnings_claimable=None,
        )
        return

    async def _run() -> None:
        kp_json = Path(wallet_path).read_text()
        keypair = Keypair.from_json(kp_json)
        pubkey = keypair.pubkey()
        print(f"Node pubkey      : {pubkey}\n")

        async with AsyncClient(rpc_url) as client:
            # PDA seed: ["node", operator_pubkey]
            node_pda, _ = Pubkey.find_program_address(
                [b"node", bytes(pubkey)],
                Pubkey.from_string(registry_program),
            )

            try:
                resp = await client.get_account_info(node_pda)
                account_data = resp.value
            except Exception as exc:
                print(f"RPC call failed: {exc}", file=sys.stderr)
                account_data = None

        if account_data is None or account_data.data is None:
            print("Node is NOT registered on-chain (no account found at PDA).")
            return

        raw = bytes(account_data.data)
        try:
            from anchorpy import Program, Provider, Wallet
            from anchorpy.provider import DEFAULT_OPTIONS

            kp_json_reload = Path(wallet_path).read_text()
            kp_reload = Keypair.from_json(kp_json_reload)
            wallet = Wallet(kp_reload)
            async with AsyncClient(rpc_url) as rpc:
                provider = Provider(rpc, wallet, DEFAULT_OPTIONS)
                program = await Program.at(Pubkey.from_string(registry_program), provider)
                record = await program.account["NodeInfo"].fetch(node_pda)
                _print_status(
                    endpoint=str(record.endpoint),
                    stake=int(record.stake),
                    reputation=int(record.reputation),
                    jobs_completed=int(record.jobs_completed),
                    jobs_disputed=int(record.jobs_disputed),
                    earnings_claimable=int(record.earnings_claimable),
                )
        except Exception as exc:
            print(f"Could not decode account (anchorpy error: {exc}).")
            print(f"Raw data (hex): {raw[:64].hex()}…")

    asyncio.run(_run())


def _print_status(
    endpoint: str,
    stake,
    reputation,
    jobs_completed,
    jobs_disputed,
    earnings_claimable,
) -> None:
    def _fmt(val) -> str:
        return str(val) if val is not None else "N/A"

    print("─" * 40)
    print("Node Status")
    print("─" * 40)
    print(f"  endpoint           : {endpoint}")
    print(f"  stake              : {_fmt(stake)}")
    print(f"  reputation         : {_fmt(reputation)}")
    print(f"  jobs_completed     : {_fmt(jobs_completed)}")
    print(f"  jobs_disputed      : {_fmt(jobs_disputed)}")
    print(f"  earnings_claimable : {_fmt(earnings_claimable)}")
    print("─" * 40)


def cmd_withdraw(args: argparse.Namespace) -> None:
    """Withdraw claimable earnings to the node wallet."""
    config_path = _config_path_from_args(args)
    config = _load_config(config_path)

    registry_program = config.get("compute_registry_program", "")
    wallet_path = config.get("wallet_path", str(Path.home() / ".config/solana/id.json"))

    print("=== Withdraw Earnings ===\n")
    print("Anchor CLI command:\n")
    print(
        f"    anchor invoke {registry_program} withdraw_earnings \\\n"
        f"        --provider.wallet {wallet_path}"
    )
    print(
        "\nNote: Run with --execute flag (not yet implemented) to send the transaction.\n"
        "For now, use the Anchor CLI command above."
    )


def cmd_config_show(args: argparse.Namespace) -> None:
    """Print current config as YAML."""
    config_path = _config_path_from_args(args)
    config = _load_config(config_path)
    display = dict(config)
    for key in ("lighthouse_api_key", "paysh_api_key", "paysh_webhook_secret"):
        if display.get(key):
            display[key] = "***REDACTED***"
    print(f"# Config: {config_path}\n")
    print(yaml.dump(display, default_flow_style=False, sort_keys=True), end="")


# ─────────────────────────── governance subcommands ────────────────────────


def cmd_governance_list(args: argparse.Namespace) -> None:
    """List active governance proposals."""
    config_path = _config_path_from_args(args)
    config = _load_config(config_path)

    governance_program = "3CvE7tX9rMwPfBgY2nKjH6oL4sQ8uZaD5mR1iW0eN9T"
    wallet_path = config.get("wallet_path", str(Path.home() / ".config/solana/id.json"))
    rpc_url = config.get("rpc_url", "https://api.mainnet-beta.solana.com")

    print("Anchor CLI command:\n")
    print(
        f"    anchor invoke {governance_program} get_proposals \\\n"
        f"        --provider.wallet {wallet_path} \\\n"
        f"        --provider.cluster {rpc_url}"
    )
    print()

    if not args.execute:
        print("(Dry run – pass --execute to fetch proposals via the client SDK.)")
        return

    from client.python import DecentralizedLLMClient

    async def _run() -> None:
        async with DecentralizedLLMClient(
            wallet_path=wallet_path,
            rpc_url=rpc_url,
        ) as client:
            proposals = await client.get_governance_proposals()

        if not proposals:
            print("No active proposals found.")
            return

        col_w = [6, 40, 12, 12, 12, 20]
        header = (
            f"{'ID':<{col_w[0]}}  "
            f"{'Title':<{col_w[1]}}  "
            f"{'Status':<{col_w[2]}}  "
            f"{'For':>{col_w[3]}}  "
            f"{'Against':>{col_w[4]}}  "
            f"{'Ends':<{col_w[5]}}"
        )
        sep = "  ".join("─" * w for w in col_w)
        print(header)
        print(sep)
        for p in proposals:
            print(
                f"{str(p['id']):<{col_w[0]}}  "
                f"{str(p['title']):<{col_w[1]}}  "
                f"{str(p['status']):<{col_w[2]}}  "
                f"{str(p['votes_for']):>{col_w[3]}}  "
                f"{str(p['votes_against']):>{col_w[4]}}  "
                f"{str(p['voting_ends_at']):<{col_w[5]}}"
            )

    asyncio.run(_run())


def cmd_governance_vote(args: argparse.Namespace) -> None:
    """Cast a vote on a governance proposal."""
    config_path = _config_path_from_args(args)
    config = _load_config(config_path)

    valid_choices = {"for", "against", "abstain"}
    if args.choice not in valid_choices:
        print(
            f"Error: choice must be one of {sorted(valid_choices)}, got '{args.choice}'",
            file=sys.stderr,
        )
        sys.exit(1)

    governance_program = "3CvE7tX9rMwPfBgY2nKjH6oL4sQ8uZaD5mR1iW0eN9T"
    wallet_path = config.get("wallet_path", str(Path.home() / ".config/solana/id.json"))
    rpc_url = config.get("rpc_url", "https://api.mainnet-beta.solana.com")

    print("Anchor CLI command:\n")
    print(
        f"    anchor invoke {governance_program} cast_vote \\\n"
        f"        --proposal-id {args.proposal_id} \\\n"
        f"        --choice {args.choice} \\\n"
        f"        --provider.wallet {wallet_path} \\\n"
        f"        --provider.cluster {rpc_url}"
    )
    print()

    if not args.execute:
        print("(Dry run – pass --execute to cast the vote via the client SDK.)")
        return

    from client.python import DecentralizedLLMClient

    async def _run() -> None:
        async with DecentralizedLLMClient(
            wallet_path=wallet_path,
            rpc_url=rpc_url,
        ) as client:
            await client.vote(args.proposal_id, args.choice)
        print(f"Vote '{args.choice}' cast on proposal {args.proposal_id}.")

    asyncio.run(_run())


# ─────────────────────────── argument parser ───────────────────────────────

_CONFIG_PATH_HELP = f"Path to config YAML (default: {DEFAULT_CONFIG_PATH})"


def _add_config_path(p: argparse.ArgumentParser) -> None:
    """Add --config-path to a (sub)parser."""
    p.add_argument("--config-path", metavar="PATH", default=None, help=_CONFIG_PATH_HELP)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="node_cli",
        description="Decentralized LLM node operator CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="subcommand", metavar="SUBCOMMAND")
    subparsers.required = True

    # ── setup ──────────────────────────────────────────────────────────────
    sp_setup = subparsers.add_parser(
        "setup",
        help="Generate Solana keypair instructions and create config file",
        description="Initialise the node configuration at ~/.decentralized-llm/config.yaml.",
    )
    _add_config_path(sp_setup)
    sp_setup.set_defaults(func=cmd_setup)

    # ── register ───────────────────────────────────────────────────────────
    sp_reg = subparsers.add_parser(
        "register",
        help="Register this node in the on-chain compute-registry",
        description=(
            "Prints the Anchor CLI command for on-chain registration. "
            "Pass --execute to send the transaction directly."
        ),
    )
    _add_config_path(sp_reg)
    sp_reg.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Actually send the registration transaction (requires Solana packages)",
    )
    sp_reg.set_defaults(func=cmd_register)

    # ── start ──────────────────────────────────────────────────────────────
    sp_start = subparsers.add_parser(
        "start",
        help="Start the node server",
        description="Load config, create a Node instance, and run the job-processing loop.",
    )
    _add_config_path(sp_start)
    sp_start.set_defaults(func=cmd_start)

    # ── status ─────────────────────────────────────────────────────────────
    sp_status = subparsers.add_parser(
        "status",
        help="Show node status (registration, jobs, earnings)",
        description=(
            "Connect to the Solana RPC, fetch the node's on-chain record, "
            "and print endpoint, stake, reputation, jobs_completed, "
            "jobs_disputed, and earnings_claimable."
        ),
    )
    _add_config_path(sp_status)
    sp_status.set_defaults(func=cmd_status)

    # ── withdraw ───────────────────────────────────────────────────────────
    sp_withdraw = subparsers.add_parser(
        "withdraw",
        help="Withdraw earnings to wallet",
        description="Print (or execute) the Anchor CLI command to withdraw claimable earnings.",
    )
    _add_config_path(sp_withdraw)
    sp_withdraw.set_defaults(func=cmd_withdraw)

    # ── config ─────────────────────────────────────────────────────────────
    sp_config = subparsers.add_parser(
        "config",
        help="Show or edit config",
        description="Show or edit the node configuration.",
    )
    _add_config_path(sp_config)
    config_sub = sp_config.add_subparsers(dest="config_action", metavar="ACTION")
    config_sub.required = True

    sp_config_show = config_sub.add_parser(
        "show",
        help="Print current config as YAML",
    )
    _add_config_path(sp_config_show)
    sp_config_show.set_defaults(func=cmd_config_show)

    # ── governance ─────────────────────────────────────────────────────────
    sp_gov = subparsers.add_parser(
        "governance",
        help="Participate in DAO governance",
        description="List proposals and cast votes in the decentralized governance system.",
    )
    _add_config_path(sp_gov)
    gov_sub = sp_gov.add_subparsers(dest="governance_action", metavar="ACTION")
    gov_sub.required = True

    # governance list
    sp_gov_list = gov_sub.add_parser(
        "list",
        help="List active governance proposals",
        description=(
            "Print the Anchor CLI command for fetching proposals. "
            "Pass --execute to fetch and display them as a table."
        ),
    )
    _add_config_path(sp_gov_list)
    sp_gov_list.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Actually fetch proposals via the client SDK",
    )
    sp_gov_list.set_defaults(func=cmd_governance_list)

    # governance vote
    sp_gov_vote = gov_sub.add_parser(
        "vote",
        help="Cast a vote on a governance proposal",
        description=(
            "Print the Anchor CLI command for voting. "
            "Pass --execute to send the vote transaction via the client SDK."
        ),
    )
    _add_config_path(sp_gov_vote)
    sp_gov_vote.add_argument("proposal_id", type=int, help="Proposal ID (integer)")
    sp_gov_vote.add_argument(
        "choice",
        type=str,
        help="Vote choice: for, against, or abstain",
    )
    sp_gov_vote.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Actually cast the vote via the client SDK",
    )
    sp_gov_vote.set_defaults(func=cmd_governance_vote)

    return parser


# ─────────────────────────── entrypoint ────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
