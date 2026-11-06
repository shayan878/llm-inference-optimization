# Docker deployment

The repository includes a small launcher for a local model directory. It is a starting point for controlled experiments, not a hardened production deployment.

## Why the example looks different from the original internship command

The internship notes used an image name written as `vllm/vllm:0.13.0` and included broad privileges such as `SYS_NICE`, `SYS_ADMIN`, and `seccomp=unconfined`. Current vLLM Docker documentation identifies the official OpenAI-compatible image as `vllm/vllm-openai`. This repository's example uses `vllm/vllm-openai:v0.13.0` as a version-pinned starting point; confirm that the chosen tag exists and supports your hardware/model before use.

The example uses `--gpus all` and `--ipc=host`, which vLLM documents for GPU access and shared memory. It does not add `SYS_ADMIN` or disable security profiles by default. The model directory is mounted read-only.

## Configuration

```bash
cp configs/docker/qwen3-4b.example.env configs/docker/local.env
$EDITOR configs/docker/local.env
./scripts/start-vllm.sh configs/docker/local.env
```

Important settings:

- `MODEL_PATH`: absolute path to a directory containing the model files.
- `DTYPE`: numeric precision for model execution. Verify hardware/backend support before selecting FP8 or other reduced-precision modes.
- `MAX_MODEL_LEN`: maximum supported sequence length. Increasing it can raise KV-cache requirements.
- `GPU_MEMORY_UTILIZATION`: vLLM's per-instance memory budget fraction. It is not GPU partitioning or a hard isolation boundary.
- `--ipc=host`: shares the host IPC namespace so the container has access to shared memory; review the implications for your environment.

## Operations

```bash
docker ps
docker logs -f vllm-qwen3-4b
curl http://localhost:8000/v1/models
./scripts/stop-vllm.sh vllm-qwen3-4b
```

For a production service, add authentication, TLS termination, network restrictions, resource quotas, health checks, log retention, metrics, and a deployment orchestrator appropriate to your environment. Do not expose an unauthenticated inference endpoint to an untrusted network.

## Official references

- [vLLM: Using Docker](https://docs.vllm.ai/en/latest/deployment/docker/)
- [vLLM: CLI reference](https://docs.vllm.ai/en/latest/cli/index.html)
- [vLLM: Optimisation and tuning](https://docs.vllm.ai/en/latest/configuration/optimization/)
