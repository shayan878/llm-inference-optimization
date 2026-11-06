# Historical benchmark results

These measurements were copied from the completed HPC internship issue, edited into a consistent format, and retained as a historical record. They are **not a fresh benchmark run**. The source notes date the work to approximately seven months before its original capture; detailed environment identifiers and exact run counts were not preserved for most experiments.

## Hardware recorded in the notes

- NVIDIA Quadro RTX 6000
- 24,576 MiB reported total GPU memory
- GPU memory measurements reported in MiB
- vLLM container tag noted as `0.13.0`

The notes do not contain NVIDIA driver version, CUDA runtime/build, exact PyTorch build, model revision hashes, GPU clocks/power limits, CPU/RAM details, workload concurrency, or full prompts for every test.

## Single-instance memory-budget experiments

| ID | dtype | max model length | GPU budget | swap | TTFT | TTLT | Reported tok/s* | Used / total GPU memory | Outcome |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline | auto | 4096 | not recorded | not recorded | 1262 ms | 1542 ms | 32 | 23,239 / 24,576 MiB | Served |
| Reduced context | auto | 2048 | 0.80 | not recorded | 1277 ms | 1546 ms | 32 | 20,839 / 24,576 MiB | Served |
| FP8 experiment | float8 | 1024 | 0.50 | 2 GiB | 2060 ms | 2332 ms | 21 | 12,765 / 24,576 MiB | Served |
| FP16 experiment | float16 | 1024 | 0.45 | 2 GiB | 1652 ms | 1910 ms | 26 | 11,543 / 24,576 MiB | Served |
| Low-budget attempt | not recorded | 1024 | 0.25 | not recorded | — | — | — | — | Failed to serve |
| Low-budget attempt | not recorded | 1024 | 0.35 | not recorded | — | — | — | — | Failed to serve |
| Lowest reported successful budget | not recorded | 1024 | 0.42 | not recorded | 1515 ms | 1801 ms | 27 | 10,823 / 24,576 MiB | Served |

*The original shell script incremented a counter for each non-empty SSE `data:` line and labelled the counter as tokens. A single SSE event can contain a text fragment representing zero, one, or multiple tokenizer tokens. As a result, the historical reported throughput values are not validated token/s measurements. They are transcribed faithfully and should not be used as a rigorous performance comparison until rerun with API usage data or a tokenizer-aware harness.*

## Two instances on one GPU

| Instance | Configured memory budget | TTFT | TTLT | Reported tok/s* | GPU memory reported |
|---|---:|---:|---:|---:|---:|
| Model 1 | 0.42 | 1920 ms | 2191 ms | 22 | 10,820 MiB |
| Model 2 | 0.50 | 2293 ms | 2565 ms | 19 | 12,762 MiB |
| **Combined** | **0.92 configured budgets** | — | — | — | **23,585 MiB / 24,576 MiB** |

The note reports that both instances could be deployed concurrently. The combined observed allocation leaves little headroom, and the experiment does not provide a concurrency workload, isolation test, failure-injection test, or long-duration stability test. Treat it as a proof-of-concept, not a production recommendation.

## Model-specific runs

### Qwen3 4B

The notes describe the figures below as averages over 10 executions after applying container flags including FP16, a 1024-token maximum model length, and a 0.30 GPU-memory budget.

| Metric | Historical value |
|---|---:|
| TTFT | 170 ms |
| TTLT | 5208 ms |
| Generation interval | 5038 ms |
| Reported throughput* | 36 tok/s |
| GPU memory | 11,543 / 24,576 MiB |

The original prompt/output token counts and raw ten-run samples were not retained. The mean therefore cannot be independently checked.

### Gemma 3

The attempt failed with a message indicating a mismatch between the model's `compressed-tensors` quantisation metadata and the quantisation argument seen by the server. The available log is insufficient to establish whether the root cause was model packaging, explicit CLI configuration, vLLM version, or hardware/backend compatibility. The repository deliberately does not claim that the GPU alone caused the error.

### Qwen3 8B

| Run | Swap setting | TTFT | TTLT | Generation interval | Reported tok/s* | GPU memory |
|---|---:|---:|---:|---:|---:|---:|
| Initial | 4 GiB (as implied by notes) | 1310 ms | 9268 ms | 7958 ms | 25 | 20,393 MiB |
| Swap removed | 0 GiB | 1001 ms | 8633 ms | 7632 ms | 26 | 20,393 MiB |

The source notes observed no memory change and only a small change in reported throughput after removing the swap setting for this particular test. This single comparison is not enough to generalise about CPU offload or swapping.

## Conclusions retained from the internship

1. Reducing the context limit and GPU memory budget changed observed allocation, but memory savings did not always correspond to faster inference.
2. Lower memory budgets eventually prevented the tested model from starting; the 0.42 success point is specific to that model, environment, and flag set.
3. Two instances were observed on one GPU when memory budgets were configured, with higher per-instance latency in the co-located test.
4. Qwen3 4B and Qwen3 8B were served with the recorded configurations; the Gemma 3 attempt encountered a quantisation configuration error.

## What must be repeated before drawing stronger conclusions

- Repeat all settings with a tokenizer-valid completion-token count and run-level samples.
- Use an identical prompt and output-token limit across configurations.
- Include warm-up runs and distinguish cold-start from steady-state measurements.
- Measure multiple concurrency levels and report request throughput, p50/p95 latency, and GPU memory over time.
- Record exact software/model versions and hardware telemetry.
