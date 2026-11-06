# Reproducibility and responsible reporting

A benchmark result is only as useful as its environment and workload description. Preserve the exact container digest and model revision, all startup arguments, full raw run-level data, prompt/output lengths, concurrency, warm-up procedure, and GPU telemetry. Include unsuccessful configurations, rather than publishing only the fastest successful run.

Do not present the historical values in this repository as a controlled benchmark comparison. They are retained to make the internship work transparent and provide a baseline for a fresh experiment. The original monitoring script used stream events as a proxy for tokens; this repository fixes that issue in the new benchmark client by using API-reported usage only.
