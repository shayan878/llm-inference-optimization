# vLLM Inference Lab

**A practical study of GPU memory budgeting, model serving, and reproducible LLM inference benchmarks with vLLM.**

[![CI](https://github.com/shayan878/vllm-inference-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/shayan878/vllm-inference-lab/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![vLLM](https://img.shields.io/badge/Serving-vLLM-orange.svg)](https://docs.vllm.ai/)

This repository turns an HPC internship experiment into a reusable engineering reference. It documents how vLLM was configured on a **24 GiB NVIDIA Quadro RTX 6000**, how GPU-memory budgets affected the ability to run one or two model servers, and how to benchmark an OpenAI-compatible streaming endpoint.

> **Evidence note:** The historical measurements below are transcribed from the original experiment notes. They are not independently reproduced. The original monitoring script counted streaming response chunks as tokens; therefore, its historical `tokens/sec` values are preserved as *reported values*, not treated as verified token throughput. The benchmark utility in this repository uses the server's `usage.completion_tokens` field when available and does not invent a token count when it is absent.

## What this project covers

- **vLLM serving:** containerised model serving with explicit dtype, context-length, cache, and memory-budget settings.
- **GPU memory experiments:** compare memory use and latency while changing `max-model-len` and `gpu-memory-utilization`.
- **Co-location:** document a two-instance experiment on one 24 GiB GPU.
- **Benchmarking:** stream requests against a vLLM OpenAI-compatible API and report TTFT, end-to-end latency, and token throughput only when a real completion-token count is returned.
- **Distributed inference concepts:** explain tensor, pipeline, data, and expert parallelism, plus the role of interconnects and Ray.

## Historical experiment snapshot

The original environment used a 24 GiB Quadro RTX 6000 and vLLM in Docker. The table is a compact transcription; full context and caveats are in [`docs/benchmarking/historical-results.md`](docs/benchmarking/historical-results.md) and the raw machine-readable data is in [`data/historical_results.csv`](data/historical_results.csv).

| Experiment | Relevant settings | TTFT | TTLT | Reported throughput | GPU memory |
|---|---|---:|---:|---:|---:|
| Baseline | `max-model-len=4096`, `dtype=auto` | 1262 ms | 1542 ms | 32 reported tok/s | 23,239 MiB |
| Reduced context | `max-model-len=2048`, GPU budget 0.80 | 1277 ms | 1546 ms | 32 reported tok/s | 20,839 MiB |
| FP8 / low budget | `max-model-len=1024`, budget 0.50, eager | 2060 ms | 2332 ms | 21 reported tok/s | 12,765 MiB |
| FP16 / low budget | `max-model-len=1024`, budget 0.45, eager | 1652 ms | 1910 ms | 26 reported tok/s | 11,543 MiB |
| Qwen3 4B | FP16, budget 0.30; reported mean of 10 runs | 170 ms | 5208 ms | 36 reported tok/s | 11,543 MiB |

**Two-server observation:** two vLLM instances were reported to fit on the same GPU with per-instance budgets of 0.42 and 0.50. The recorded memory figures were 10,820 MiB and 12,762 MiB respectively (23,585 MiB combined). Individual latency increased in that co-located run. See the detailed table for the model names and missing metadata caveats.

### What the results suggest — and what they do not

- Lowering the configured GPU-memory budget reduced the memory observed for an individual instance in these runs, but also reduced available execution headroom and was associated with lower reported throughput in several tests.
- A budget of 0.25 and 0.35 failed to start the tested model; 0.42 was the lowest reported successful budget in that particular experiment. This is **not** a universal threshold.
- Two instances were reported to coexist, but this does not establish isolation, fairness, or production readiness. `gpu-memory-utilization` is a per-vLLM-instance memory budget, not a hardware-enforced GPU partition.
- Historical `tokens/sec` needs to be re-measured with actual completion-token usage before making apples-to-apples performance claims.

## Architecture at a glance

```text
Client / benchmark runner
          |
          | HTTP streaming (OpenAI-compatible API)
          v
   vLLM API server (container)
          |
          +---- Model weights
          +---- GPU execution / CUDA kernels
          +---- KV cache (context + concurrency dependent)
          |
          v
     NVIDIA GPU memory
```

For multi-GPU inference, tensor parallelism shards computations/weights within layers; pipeline parallelism distributes model layers across devices or nodes; data parallelism runs replicas to serve more independent requests; expert parallelism distributes MoE experts. See [`docs/architecture/parallelism.md`](docs/architecture/parallelism.md). Current vLLM documentation describes both tensor and pipeline parallel inference; the older internship note that vLLM did not support pipeline parallelism is outdated.

## Quick start

### 1. Prerequisites

- Linux host with a supported NVIDIA GPU and a working NVIDIA driver.
- Docker Engine and NVIDIA Container Toolkit configured so containers can access the GPU.
- A local model directory readable by the Docker daemon, or a Hugging Face model ID with the required access token.
- Python 3.10+ for the benchmark client.

Check the GPU is visible before starting:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

The CUDA image tag above is only a diagnostic example; choose a CUDA runtime compatible with your installed driver.

### 2. Configure and start vLLM

```bash
cp configs/docker/qwen3-4b.example.env configs/docker/local.env
# Edit MODEL_PATH and any settings in configs/docker/local.env.
chmod +x scripts/start-vllm.sh
./scripts/start-vllm.sh configs/docker/local.env
```

The example pins the vLLM container image tag to `v0.13.0` to reflect the internship environment. For a new deployment, deliberately select and pin a vLLM release compatible with your GPU, CUDA stack, model architecture, and quantisation format; do not blindly use `latest`.

The launcher intentionally does **not** disable Docker's seccomp/AppArmor protections or grant `SYS_ADMIN`. Do not copy those broad privileges from the original command unless a documented requirement justifies them.

### 3. Run the benchmark

In a second terminal, install the client and run it against the server:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

python -m vllm_inference_lab.benchmark \
  --base-url http://localhost:8000 \
  --model local-qwen3-4b \
  --prompt "Explain GPU memory management for LLM inference." \
  --max-tokens 128 \
  --warmups 1 \
  --runs 10 \
  --output-dir results/local-qwen3-4b
```

The `--model` value must match `SERVED_MODEL_NAME` in your environment file. The benchmark writes a per-run CSV and a summary JSON. Token throughput is left blank when the API does not return a completion-token count.

Stop the server with:

```bash
./scripts/stop-vllm.sh
```

## Repository structure

```text
vllm-inference-lab/
├── .github/                 # CI, bug/experiment issue forms, PR template
├── configs/docker/          # Example, non-secret deployment configuration
├── data/                    # Historical measurements (CSV)
├── docs/
│   ├── github-publishing.md  # Metadata and launch checklist
│   ├── architecture/        # Parallelism and distributed inference
│   ├── benchmarking/        # Methodology and historical results
│   └── deployment/          # Docker deployment guide
├── scripts/                 # Container start/stop helpers
├── src/vllm_inference_lab/  # Benchmark client
├── tests/                   # Unit tests
├── CONTRIBUTING.md
├── CITATION.cff
├── LICENSE
└── README.md
```

## Reproducibility checklist

Before publishing a new performance comparison, record: exact GPU SKU and count; driver and CUDA versions; vLLM image digest; PyTorch version; model ID/revision and quantisation; dtype; all vLLM engine flags; prompt dataset and prompt/output token lengths; concurrency and request count; warm-up policy; run-level results; GPU memory and utilisation; and whether throughput comes from API usage counts or an explicitly documented tokenizer.

A starting template is provided in [`docs/benchmarking/methodology.md`](docs/benchmarking/methodology.md). GitHub metadata and an ethical discovery checklist are in [`docs/github-publishing.md`](docs/github-publishing.md).

## Limitations and next steps

This is an engineering study and reproducibility scaffold, not a production serving stack. The supplied notes do not include enough environment and workload metadata to reproduce every result exactly. Future work should add controlled concurrency sweeps, multiple prompt/output lengths, GPU power/clock telemetry, repeated runs for every configuration, confidence intervals, and comparisons across vLLM versions and hardware.

## References

- [vLLM documentation](https://docs.vllm.ai/en/latest/)
- [vLLM Docker deployment](https://docs.vllm.ai/en/latest/deployment/docker/)
- [vLLM optimisation and tuning](https://docs.vllm.ai/en/latest/configuration/optimization/)
- [vLLM parallelism and scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
- [OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/)

## Acknowledgements

Based on HPC internship experiments involving Shayan Daneshyar, Bagher Hosseini, and Ahmadreza Anami. The repository author is responsible for the adaptation and documentation published here. Add any required institutional or team attribution before public release.
