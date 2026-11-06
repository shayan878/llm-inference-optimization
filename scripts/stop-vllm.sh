#!/usr/bin/env bash
set -Eeuo pipefail

CONTAINER_NAME="${1:-vllm-qwen3-4b}"
if ! command -v docker >/dev/null 2>&1; then
  echo "Docker CLI was not found." >&2
  exit 2
fi

docker stop "$CONTAINER_NAME"
