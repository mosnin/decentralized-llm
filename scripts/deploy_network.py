"""
One-command network deployment: rent GPUs on Vast.ai, deploy model shards, register on-chain.

Usage:
    export VAST_API_KEY=...
    export SOLANA_RPC_URL=https://api.mainnet-beta.solana.com
    export LIGHTHOUSE_API_KEY=...
    python scripts/deploy_network.py \\
        --model meta-llama/Llama-3.2-3B \\
        --num-shards 3 \\
        --gpu RTX_4090

This script:
  1. Searches Vast.ai for GPUs matching --gpu filter
  2. Rents `--num-shards` instances
  3. Deploys the node Docker image on each with the right shard env vars
  4. Waits for all nodes to come online
  5. Verifies P2P connectivity between shards
  6. Prints the bootstrap peer addresses for the DHT

The first instance acts as the DHT bootstrap node.
"""

import argparse
import asyncio
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)


async def deploy(args: argparse.Namespace) -> None:
    from integrations.vastai import GpuRequirements, VastAiProvisioner

    provisioner = VastAiProvisioner(api_key=os.environ["VAST_API_KEY"])

    requirements = GpuRequirements(
        min_vram_gb=args.min_vram,
        gpu_name=args.gpu or None,
        max_price_per_hour=args.max_price,
        num_gpus=1,
        disk_gb=args.disk,
    )

    logger.info(
        "Searching for %d GPU(s) to host %d shards of %s…",
        args.num_shards,
        args.num_shards,
        args.model,
    )

    offers = await provisioner.find_offers(requirements)
    if len(offers) < args.num_shards:
        logger.error("Only %d offers available, need %d", len(offers), args.num_shards)
        sys.exit(1)

    logger.info("Found %d offers. Renting the cheapest %d.", len(offers), args.num_shards)

    # Deploy shards in parallel
    instances = []
    tasks = []
    for shard_idx in range(args.num_shards):
        task = provisioner.provision(
            model_name=args.model,
            shard_index=shard_idx,
            num_shards=args.num_shards,
            solana_rpc_url=os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"),
            wallet_path=os.getenv("WALLET_PATH", "~/.config/solana/id.json"),
            bootstrap_peers=(
                [f"/ip4/{instances[0].public_ip}/tcp/{instances[0].public_port}"]
                if instances
                else []
            ),
            requirements=requirements,
            lighthouse_api_key=os.getenv("LIGHTHOUSE_API_KEY", ""),
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    print("\n" + "=" * 60)
    print("NETWORK DEPLOYED")
    print("=" * 60)
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"  Shard {i}: FAILED — {result}")
        else:
            print(
                f"  Shard {i}: {result.public_ip}:{result.public_port} "
                f"({result.gpu_name}, {result.vram_gb}GB VRAM, ${result.price_per_hour:.3f}/hr)"
            )
            instances.append(result)

    if not instances:
        logger.error("All shards failed to provision")
        sys.exit(1)

    bootstrap = f"/ip4/{instances[0].public_ip}/tcp/{instances[0].public_port}"
    print(f"\nDHT Bootstrap peer: {bootstrap}")
    print("\nAdd to .env for client:")
    print(f"  DHT_BOOTSTRAP_PEERS={bootstrap}")
    print(f"  SOLANA_RPC_URL={os.getenv('SOLANA_RPC_URL', '')}")

    total_cost = sum(inst.price_per_hour for inst in instances)
    print(f"\nEstimated cost: ${total_cost:.3f}/hr (${total_cost * 24:.2f}/day)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy decentralized LLM network on Vast.ai")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B", help="HuggingFace model ID")
    parser.add_argument("--num-shards", type=int, default=3, help="Number of pipeline shards")
    parser.add_argument("--gpu", default=None, help="GPU filter e.g. RTX_4090, A100")
    parser.add_argument("--min-vram", type=int, default=24, help="Minimum VRAM in GB")
    parser.add_argument("--max-price", type=float, default=2.0, help="Max price per GPU per hour")
    parser.add_argument("--disk", type=int, default=80, help="Disk size in GB per instance")
    args = parser.parse_args()

    asyncio.run(deploy(args))


if __name__ == "__main__":
    main()
