#!/usr/bin/env bash
set -Eeuo pipefail

CONFIG_FILE="${1:-configs/docker/local.env}"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Configuration file not found: $CONFIG_FILE" >&2
  echo "Copy configs/docker/qwen3-4b.example.env to configs/docker/local.env and edit it." >&2
  exit 2
fi

# The config file is a trusted shell-assignment file maintained by the operator.
# shellcheck disable=SC1090
set -a
source "$CONFIG_FILE"
set +a

required=(VLLM_IMAGE CONTAINER_NAME MODEL_PATH SERVED_MODEL_NAME HOST_PORT CONTAINER_PORT DTYPE MAX_MODEL_LEN GPU_MEMORY_UTILIZATION CACHE_VOLUME)
for key in "${required[@]}"; do
  if [[ -z "${!key:-}" ]]; then
    echo "Missing required setting: $key" >&2
    exit 2
  fi
done

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "MODEL_PATH must exist and be a directory: $MODEL_PATH" >&2
  exit 2
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker CLI was not found." >&2
  exit 2
fi

if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "A container named '$CONTAINER_NAME' already exists. Stop/remove it first." >&2
  exit 2
fi

args=(
  run --detach
  --name "$CONTAINER_NAME"
  --gpus all
  --ipc=host
  --publish "${HOST_PORT}:${CONTAINER_PORT}"
  --volume "${MODEL_PATH}:/model:ro"
  --volume "${CACHE_VOLUME}:/root/.cache/vllm"
  --restart unless-stopped
)

if [[ -n "${HF_TOKEN:-}" ]]; then
  args+=(--env HF_TOKEN)
fi

args+=(
  "$VLLM_IMAGE"
  --model /model
  --served-model-name "$SERVED_MODEL_NAME"
  --dtype "$DTYPE"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --host 0.0.0.0
  --port "$CONTAINER_PORT"
)

echo "Starting ${CONTAINER_NAME} with image ${VLLM_IMAGE}..."
docker "${args[@]}"
echo "Server endpoint: http://localhost:${HOST_PORT}/v1"
echo "Logs: docker logs -f ${CONTAINER_NAME}"
