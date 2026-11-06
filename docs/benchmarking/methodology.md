# Benchmark methodology

## Metrics

- **TTFT (time to first token):** elapsed wall-clock time from the start of the HTTP request until the first non-empty text delta is received. It includes connection/request overhead and server prefill time as observed by this client.
- **End-to-end latency:** elapsed wall-clock time until the streaming response completes.
- **Generation time:** end-to-end latency minus TTFT. This is a client-side approximation of the generation interval.
- **Completion tokens:** the `usage.completion_tokens` value returned by the API, when available.
- **Tokens per second:** API-reported completion tokens divided by the generation interval. The tool leaves this metric empty when usage is not present; it does not treat SSE events or text chunks as tokens.

The API stream can include an extra final usage chunk when `stream_options.include_usage` is supported. Confirm that your vLLM version returns it before comparing token throughput across environments.

## Run protocol

1. Record the exact host/GPU/software stack using the reproducibility template below.
2. Start the server and wait until it is ready. Do not include model loading in the timed inference request unless startup latency is the metric being studied.
3. Use the same prompt, output-token limit, temperature, model revision, and request concurrency for each configuration.
4. Run warm-up requests separately; do not include them in summary statistics.
5. Capture at least 10 measured runs for exploratory comparisons. For publishable conclusions, increase repetitions and report distribution/variance rather than only an average.
6. Store every run, not just the best result. The benchmark script writes `runs.csv` and `summary.json`.
7. Change one parameter at a time where practical.

## Example

```bash
vllm-benchmark \
  --base-url http://localhost:8000 \
  --model local-qwen3-4b \
  --prompt "Summarise the role of the KV cache in autoregressive inference." \
  --max-tokens 128 \
  --warmups 2 \
  --runs 20 \
  --output-dir results/qwen3-4b-context-1024
```

The utility tests a single request at a time. It is not a concurrency/load generator. Add a separately controlled concurrency harness before using it to claim saturation throughput or production capacity.

## Reproducibility record template

| Field | Value to record |
|---|---|
| Host | CPU, RAM, OS/kernel |
| GPU | Exact SKU, VRAM, count, power limit, clock state |
| Driver / CUDA | NVIDIA driver, CUDA runtime/toolkit |
| Framework | Container digest, vLLM version, PyTorch version |
| Model | Repository/path, immutable revision, architecture |
| Weights | dtype, quantisation method, quantisation metadata |
| Server flags | Full command/config, including all defaults that matter |
| Workload | Prompt source, prompt token count, requested and actual output tokens |
| Load | Concurrency, request arrival pattern, number of repetitions |
| Measurement | Warm-up count, client machine, network path, metric definitions |
| Results | Every run, mean/median/stdev, percentiles, failures |

## Important interpretation caveats

- A change in `max-model-len` affects the maximum supported context and can affect KV-cache allocation; it is not a direct knob for reducing model-weight memory alone.
- `gpu-memory-utilization` is a vLLM per-instance memory budget. Multiple instances can each be configured with a budget, but this is not a hard partition and does not guarantee isolation or prevent aggregate allocation pressure.
- `--swap-space` is not a general-purpose extension of VRAM for all model weights. Confirm what is being offloaded and measure the actual workload before drawing conclusions.
- Quantisation support depends on the model's quantisation metadata, vLLM version, backend, and GPU architecture. A config mismatch should not be diagnosed as a hardware limitation without a separate compatibility check.
