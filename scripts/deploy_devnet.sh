#!/usr/bin/env bash
set -euo pipefail

# Deploy all three Anchor programs to Solana devnet
# Usage: ./scripts/deploy_devnet.sh [--keypair <path>]

CLUSTER="devnet"
RPC="https://api.devnet.solana.com"
KEYPAIR="${SOLANA_KEYPAIR:-$HOME/.config/solana/id.json}"

while [[ $# -gt 0 ]]; do
  case $1 in
    --keypair) KEYPAIR="$2"; shift 2;;
    *) echo "Unknown flag: $1"; exit 1;;
  esac
done

echo "Deploying to Solana ${CLUSTER} with wallet: ${KEYPAIR}"

# Ensure wallet has enough SOL for deployment
BALANCE=$(solana balance "$KEYPAIR" --url "$RPC" | awk '{print $1}')
echo "Wallet balance: ${BALANCE} SOL"

# Build programs
echo "Building Anchor programs..."
anchor build

# Deploy each program
echo "Deploying inference-market..."
anchor deploy --program-name inference_market --provider.cluster "$CLUSTER" --provider.wallet "$KEYPAIR"

echo "Deploying compute-registry..."
anchor deploy --program-name compute_registry --provider.cluster "$CLUSTER" --provider.wallet "$KEYPAIR"

echo "Deploying governance..."
anchor deploy --program-name governance --provider.cluster "$CLUSTER" --provider.wallet "$KEYPAIR"

echo "Deployment complete!"
echo "Update Anchor.toml [programs.devnet] with the new program IDs if they changed."
