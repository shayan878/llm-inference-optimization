# Troubleshooting notes

## Container cannot see the GPU

Check `nvidia-smi` on the host, verify the NVIDIA Container Toolkit installation, and run a minimal CUDA container with `--gpus all`. The host driver, container CUDA/PyTorch build, GPU compute capability, and vLLM release must be compatible.

## Model fails to start under a lower memory budget

A per-instance memory budget that is too low can leave insufficient space for model weights, runtime allocations, profiling, and KV cache. Start with the documented default for the chosen release, inspect the vLLM startup error, then change one variable at a time. A threshold observed for one model is not portable to another.

## Quantisation configuration mismatch

Check the model configuration and quantisation metadata, then compare it with the explicit vLLM `--quantization` setting and supported methods for that release. Avoid forcing `--quantization` without knowing what the model files declare. Also verify the GPU architecture and kernel/backend compatibility. The Gemma 3 error in the historical notes only establishes a config mismatch, not its root cause.

## Server responds but the benchmark reports no tokens/sec

The benchmark requires a `usage.completion_tokens` field in the streamed response. Confirm that the installed vLLM version supports `stream_options.include_usage` for the selected endpoint. TTFT and total latency may still be recorded if usage is omitted; the tool intentionally does not guess a token count from stream chunks.

## Port already in use or container name already exists

Change `HOST_PORT` or `CONTAINER_NAME` in your local environment file. Inspect existing containers with `docker ps -a` before removing anything.
