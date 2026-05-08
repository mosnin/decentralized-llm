#!/usr/bin/env bash
# docker/entrypoint.sh — container entrypoint for the node image.
# Delegates to scripts/docker_entrypoint.sh which handles RPC readiness checks.

set -euo pipefail

exec /app/scripts/docker_entrypoint.sh "$@"
