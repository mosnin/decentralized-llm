#!/usr/bin/env bash
# docker_entrypoint.sh — wait for Solana RPC then start the node.
#
# Environment variables used:
#   SOLANA_RPC_URL   — HTTP(S) endpoint for the Solana cluster
#   CONFIG_PATH      — path to node config YAML (default: /app/config/config.yaml)

set -euo pipefail

SOLANA_RPC_URL="${SOLANA_RPC_URL:-https://api.devnet.solana.com}"
CONFIG_PATH="${CONFIG_PATH:-/app/config/config.yaml}"

echo "[entrypoint] Waiting for Solana RPC at ${SOLANA_RPC_URL} ..."

MAX_RETRIES=30
SLEEP_SECONDS=2
attempt=0

until curl -sf -o /dev/null "${SOLANA_RPC_URL}" 2>/dev/null; do
    attempt=$((attempt + 1))
    if [ "${attempt}" -ge "${MAX_RETRIES}" ]; then
        echo "[entrypoint] ERROR: Solana RPC unreachable after ${MAX_RETRIES} attempts. Exiting." >&2
        exit 1
    fi
    echo "[entrypoint] Attempt ${attempt}/${MAX_RETRIES} — RPC not ready, retrying in ${SLEEP_SECONDS}s ..."
    sleep "${SLEEP_SECONDS}"
done

echo "[entrypoint] Solana RPC is reachable. Starting node ..."

exec python -m scripts.node_cli start --config-path "${CONFIG_PATH}"
