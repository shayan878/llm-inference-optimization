"""Benchmark an OpenAI-compatible vLLM chat-completions streaming endpoint."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from vllm_inference_lab.metrics import (
    calculate_tokens_per_second,
    extract_completion_tokens,
    extract_text_delta,
    parse_sse_data,
)


@dataclass
class RunResult:
    run_index: int
    ttft_ms: float | None
    total_latency_ms: float
    generation_time_ms: float | None
    completion_tokens: int | None
    tokens_per_second: float | None
    content_chunks: int
    generated_characters: int
    finish_reason: str | None
    request_id: str | None


def _now_ms(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000


def run_once(
    client: httpx.Client,
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    run_index: int,
) -> RunResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start_ns = time.perf_counter_ns()
    first_text_ms: float | None = None
    completion_tokens: int | None = None
    content_chunks = 0
    generated_characters = 0
    finish_reason: str | None = None
    request_id: str | None = None

    with client.stream("POST", endpoint, json=payload) as response:
        response.raise_for_status()
        request_id = response.headers.get("x-request-id") or response.headers.get("request-id")
        for line in response.iter_lines():
            if not line:
                continue
            try:
                chunk = parse_sse_data(line)
            except (json.JSONDecodeError, ValueError) as exc:
                raise RuntimeError(
                    f"Could not parse streaming response line: {line[:200]!r}"
                ) from exc
            if chunk is None:
                continue

            if chunk.get("error"):
                raise RuntimeError(f"Server returned an error: {chunk['error']}")

            text_delta = extract_text_delta(chunk)
            if text_delta:
                if first_text_ms is None:
                    first_text_ms = _now_ms(start_ns)
                content_chunks += 1
                generated_characters += len(text_delta)

            reported_tokens = extract_completion_tokens(chunk)
            if reported_tokens is not None:
                completion_tokens = reported_tokens

            choices = chunk.get("choices") or []
            if choices and choices[0].get("finish_reason"):
                finish_reason = str(choices[0]["finish_reason"])

    total_latency_ms = _now_ms(start_ns)
    generation_time_ms = (
        total_latency_ms - first_text_ms if first_text_ms is not None else None
    )
    tokens_per_second = calculate_tokens_per_second(
        completion_tokens, first_text_ms, total_latency_ms
    )
    return RunResult(
        run_index=run_index,
        ttft_ms=round(first_text_ms, 3) if first_text_ms is not None else None,
        total_latency_ms=round(total_latency_ms, 3),
        generation_time_ms=round(generation_time_ms, 3) if generation_time_ms is not None else None,
        completion_tokens=completion_tokens,
        tokens_per_second=round(tokens_per_second, 3) if tokens_per_second is not None else None,
        content_chunks=content_chunks,
        generated_characters=generated_characters,
        finish_reason=finish_reason,
        request_id=request_id,
    )


def _aggregate(results: list[RunResult], field: str) -> dict[str, float | None]:
    values = [
        float(getattr(result, field))
        for result in results
        if getattr(result, field) is not None
    ]
    if not values:
        return {"mean": None, "median": None, "stdev": None, "min": None, "max": None}
    return {
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "stdev": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def write_results(
    output_dir: Path,
    metadata: dict[str, Any],
    warmup_results: list[RunResult],
    results: list[RunResult],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]

    with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "metadata": metadata,
        "warmup_runs": [asdict(item) for item in warmup_results],
        "run_count": len(results),
        "metrics": {
            "ttft_ms": _aggregate(results, "ttft_ms"),
            "total_latency_ms": _aggregate(results, "total_latency_ms"),
            "generation_time_ms": _aggregate(results, "generation_time_ms"),
            "completion_tokens": _aggregate(results, "completion_tokens"),
            "tokens_per_second": _aggregate(results, "tokens_per_second"),
            "content_chunks": _aggregate(results, "content_chunks"),
        },
        "interpretation": (
            "tokens_per_second uses API-reported completion_tokens only. "
            "content_chunks is an event/chunk count, not a tokenizer-derived token count."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure TTFT, latency, and API-reported token throughput "
            "from a streaming chat endpoint."
        )
    )
    parser.add_argument("--base-url", default="http://localhost:8000", help="Server base URL")
    parser.add_argument("--model", required=True, help="Served model name")
    parser.add_argument("--prompt", default="Explain how GPU memory affects LLM inference.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/benchmark"))
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable for optional API key",
    )
    parser.add_argument(
        "--no-verify-tls",
        action="store_true",
        help="Disable TLS verification (not recommended)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.runs < 1 or args.warmups < 0 or args.max_tokens < 1:
        print("runs and max-tokens must be positive; warmups cannot be negative", file=sys.stderr)
        return 2
    if args.timeout <= 0:
        print("timeout must be greater than zero", file=sys.stderr)
        return 2

    endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"
    headers: dict[str, str] = {}
    api_key = os.environ.get(args.api_key_env)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "requested_runs": args.runs,
        "warmups": args.warmups,
        "timeout_seconds": args.timeout,
        "tls_verification": not args.no_verify_tls,
    }
    timeout = httpx.Timeout(args.timeout, connect=min(args.timeout, 15.0))

    try:
        with httpx.Client(
            timeout=timeout,
            headers=headers,
            verify=not args.no_verify_tls,
        ) as client:
            warmups: list[RunResult] = []
            for index in range(1, args.warmups + 1):
                print(f"Warm-up {index}/{args.warmups}...", flush=True)
                warmups.append(
                    run_once(client, endpoint, args.model, args.prompt, args.max_tokens,
                             args.temperature, -index)
                )
            results: list[RunResult] = []
            for index in range(1, args.runs + 1):
                print(f"Run {index}/{args.runs}...", flush=True)
                result = run_once(client, endpoint, args.model, args.prompt, args.max_tokens,
                                  args.temperature, index)
                results.append(result)
                tok_s = (
                    f"{result.tokens_per_second:.2f} tok/s"
                    if result.tokens_per_second is not None
                    else "token usage unavailable"
                )
                ttft = f"{result.ttft_ms:.1f} ms" if result.ttft_ms is not None else "n/a"
                print(f"  TTFT={ttft}; total={result.total_latency_ms:.1f} ms; {tok_s}")

        write_results(args.output_dir, metadata, warmups, results)
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1

    print(f"\nResults saved to: {args.output_dir.resolve()}")
    print(
        "Note: token throughput is based on API-reported completion_tokens, "
        "not stream chunk counts."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
