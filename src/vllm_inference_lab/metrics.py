"""Small, testable helpers for interpreting streaming benchmark data."""

from __future__ import annotations

from typing import Any


def extract_text_delta(payload: dict[str, Any]) -> str:
    """Extract text from an OpenAI-compatible chat-completions stream chunk."""
    choices = payload.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    content = delta.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Some compatible APIs represent multimodal content as typed blocks.
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    return ""


def extract_completion_tokens(payload: dict[str, Any]) -> int | None:
    """Return a real completion-token count when the server includes usage."""
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    count = usage.get("completion_tokens")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        return None
    return count


def calculate_tokens_per_second(
    completion_tokens: int | None,
    ttft_ms: float | None,
    total_latency_ms: float | None,
) -> float | None:
    """Calculate generation throughput only from API-reported token usage."""
    if completion_tokens is None or completion_tokens <= 0:
        return None
    if ttft_ms is None or total_latency_ms is None:
        return None
    generation_ms = total_latency_ms - ttft_ms
    if generation_ms <= 0:
        return None
    return completion_tokens * 1000.0 / generation_ms


def parse_sse_data(line: str) -> dict[str, Any] | None:
    """Parse a Server-Sent Events data line; return None for non-data/terminal lines."""
    if not line.startswith("data:"):
        return None
    raw = line[len("data:") :].strip()
    if not raw or raw == "[DONE]":
        return None
    import json

    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Expected each SSE data payload to be a JSON object")
    return payload
