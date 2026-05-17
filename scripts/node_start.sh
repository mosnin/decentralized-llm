#!/usr/bin/env bash
set -euo pipefail
# Start a compute node connected to Solana devnet.
# Usage: ./scripts/node_start.sh [--config config.devnet.json]

CONFIG="${1:-config.devnet.json}"

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: Config file not found: $CONFIG"
  echo "Copy config.devnet.json and fill in your wallet path and API keys"
  exit 1
fi

echo "Starting decentralized LLM node with config: $CONFIG"

# Check Solana packages
python -c "import anchorpy, solana, solders" 2>/dev/null || {
  echo "Installing Solana Python packages..."
  pip install anchorpy solana solders
}

exec python -m scripts.node_cli start --config-path "$CONFIG"
