# Results schema

`data/historical_results.csv` keeps the original notes in a machine-readable form. Fields intentionally remain blank when the source did not preserve a value; they are not inferred from defaults.

| Column | Meaning |
|---|---|
| `experiment_id` | Stable identifier for a source experiment |
| `experiment` | Human-readable experiment label |
| `model` | Model identifier, or `not-recorded` |
| `dtype` | Requested execution dtype |
| `max_model_len` | Configured maximum sequence length |
| `gpu_memory_utilization` | Configured vLLM instance memory budget |
| `swap_space_gib` | Configured swap/offload value when recorded |
| `ttft_ms` | Historical TTFT value |
| `ttlt_ms` | Historical time to last response value |
| `reported_generation_time_ms` | Historical reported generation interval |
| `reported_throughput_tokens_per_sec` | Original reported value; not validated token throughput |
| `gpu_memory_used_mib` / `gpu_memory_total_mib` | Historical GPU memory readings |
| `status` | `served` or `failed` |
| `repeat_count` | Number of repetitions when source recorded it |
| `notes` | Interpretation and source caveats |
