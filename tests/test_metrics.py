from vllm_inference_lab.metrics import (
    calculate_tokens_per_second,
    extract_completion_tokens,
    extract_text_delta,
    parse_sse_data,
)


def test_extract_text_delta_from_string():
    payload = {"choices": [{"delta": {"content": "hello"}}]}
    assert extract_text_delta(payload) == "hello"


def test_extract_text_delta_from_multimodal_blocks():
    payload = {"choices": [{"delta": {"content": [{"type": "text", "text": "hi"}]}}]}
    assert extract_text_delta(payload) == "hi"


def test_extract_completion_tokens_only_when_reported():
    assert extract_completion_tokens({"usage": {"completion_tokens": 42}}) == 42
    assert extract_completion_tokens({"usage": {"completion_tokens": "42"}}) is None
    assert extract_completion_tokens({}) is None


def test_calculate_tokens_per_second_uses_generation_window():
    assert calculate_tokens_per_second(100, 200.0, 1200.0) == 100.0


def test_calculate_tokens_per_second_does_not_guess_missing_usage():
    assert calculate_tokens_per_second(None, 200.0, 1200.0) is None
    assert calculate_tokens_per_second(10, 500.0, 400.0) is None


def test_parse_sse_data():
    assert parse_sse_data('data: {"id":"x","choices":[]}') == {"id": "x", "choices": []}
    assert parse_sse_data("data: [DONE]") is None
    assert parse_sse_data(": keep-alive") is None


def test_run_once_uses_streamed_api_usage():
    import httpx

    from vllm_inference_lab.benchmark import run_once

    stream = b"""data: {"id":"req-1","choices":[{"delta":{"content":"hello"},"finish_reason":null}]}

 data: {"id":"req-1","choices":[{"delta":{},"finish_reason":"stop"}]}

 data: {"id":"req-1","choices":[],"usage":{"completion_tokens":2}}

 data: [DONE]

""".replace(b"\n data:", b"\ndata:")

    def handler(request):
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream", "x-request-id": "req-1"},
            content=stream,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = run_once(
            client,
            "http://test.local/v1/chat/completions",
            "test-model",
            "hi",
            16,
            0.0,
            1,
        )

    assert result.completion_tokens == 2
    assert result.tokens_per_second is not None
    assert result.content_chunks == 1
    assert result.generated_characters == 5
    assert result.finish_reason == "stop"
    assert result.request_id == "req-1"


def test_run_once_does_not_infer_tokens_from_stream_chunks():
    import httpx

    from vllm_inference_lab.benchmark import run_once

    stream = (
        b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n\n'
        b'data: [DONE]\n\n'
    )

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = run_once(
            client,
            "http://test.local/v1/chat/completions",
            "test-model",
            "hi",
            16,
            0.0,
            1,
        )

    assert result.content_chunks == 1
    assert result.completion_tokens is None
    assert result.tokens_per_second is None
