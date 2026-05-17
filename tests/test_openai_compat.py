"""
Tests for node.openai_compat — OpenAI wire-format compatibility layer.

All tests are pure Python; no network calls are made.
"""

from __future__ import annotations

import json
import time

import pytest  # noqa: F401

# ─────────────────────────── OpenAIErrorCode ─────────────────────────────────


def test_error_code_enum_members():
    from node.openai_compat import OpenAIErrorCode

    expected = {
        "invalid_request_error",
        "authentication_error",
        "permission_error",
        "not_found_error",
        "rate_limit_error",
        "api_error",
        "overloaded_error",
    }
    actual = {member.value for member in OpenAIErrorCode}
    assert actual == expected


def test_error_code_is_str_subclass():
    from node.openai_compat import OpenAIErrorCode

    assert isinstance(OpenAIErrorCode.api_error, str)
    assert OpenAIErrorCode.api_error == "api_error"


# ─────────────────────────── openai_error_response ───────────────────────────


def test_error_response_status_code():
    from node.openai_compat import openai_error_response

    resp = openai_error_response(400, "invalid_request_error", "Bad input")
    assert resp.status_code == 400


def test_error_response_envelope_shape():
    from node.openai_compat import openai_error_response

    resp = openai_error_response(401, "authentication_error", "No key")
    body = json.loads(resp.body)
    assert "error" in body
    err = body["error"]
    assert set(err.keys()) == {"message", "type", "param", "code"}


def test_error_response_null_defaults():
    from node.openai_compat import openai_error_response

    resp = openai_error_response(429, "rate_limit_error", "Too many requests")
    body = json.loads(resp.body)
    assert body["error"]["param"] is None
    assert body["error"]["code"] is None


def test_error_response_optional_fields():
    from node.openai_compat import openai_error_response

    resp = openai_error_response(
        400,
        "invalid_request_error",
        "Bad param",
        param="temperature",
        code="param_out_of_range",
    )
    body = json.loads(resp.body)
    assert body["error"]["param"] == "temperature"
    assert body["error"]["code"] == "param_out_of_range"


def test_error_response_message_and_type():
    from node.openai_compat import openai_error_response

    resp = openai_error_response(500, "api_error", "Oops")
    body = json.loads(resp.body)
    assert body["error"]["message"] == "Oops"
    assert body["error"]["type"] == "api_error"


# ─────────────────────────── count_tokens ────────────────────────────────────


def test_count_tokens_empty_string():
    from node.openai_compat import count_tokens

    assert count_tokens("") == 0


def test_count_tokens_short_text():
    from node.openai_compat import count_tokens

    # "Hello" → 5 chars → 5 // 4 = 1, minimum 1
    assert count_tokens("Hello") == 1


def test_count_tokens_longer_text():
    from node.openai_compat import count_tokens

    text = "a" * 400  # 400 chars → 100 tokens
    assert count_tokens(text) == 100


def test_count_tokens_model_param_ignored():
    from node.openai_compat import count_tokens

    text = "The quick brown fox"
    # model should not change the result
    assert count_tokens(text, "gpt-4") == count_tokens(text, "gpt-3.5-turbo")


# ─────────────────────────── build_usage ─────────────────────────────────────


def test_build_usage_keys():
    from node.openai_compat import build_usage

    usage = build_usage("hello world", "yes indeed")
    assert set(usage.keys()) == {"prompt_tokens", "completion_tokens", "total_tokens"}


def test_build_usage_total_is_sum():
    from node.openai_compat import build_usage

    usage = build_usage("prompt text here", "completion text")
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_build_usage_empty_strings():
    from node.openai_compat import build_usage

    usage = build_usage("", "")
    assert usage["prompt_tokens"] == 0
    assert usage["completion_tokens"] == 0
    assert usage["total_tokens"] == 0


# ─────────────────────────── normalize_model_name ────────────────────────────


def test_normalize_gpt4():
    from node.openai_compat import normalize_model_name

    assert normalize_model_name("gpt-4") == "meta-llama/Llama-3.2-70B"


def test_normalize_gpt4_turbo():
    from node.openai_compat import normalize_model_name

    assert normalize_model_name("gpt-4-turbo") == "meta-llama/Llama-3.2-70B"


def test_normalize_gpt35_turbo():
    from node.openai_compat import normalize_model_name

    assert normalize_model_name("gpt-3.5-turbo") == "meta-llama/Llama-3.2-3B"


def test_normalize_gpt4o():
    from node.openai_compat import normalize_model_name

    assert normalize_model_name("gpt-4o") == "meta-llama/Llama-3.2-8B"


def test_normalize_gpt4o_mini():
    from node.openai_compat import normalize_model_name

    assert normalize_model_name("gpt-4o-mini") == "meta-llama/Llama-3.2-8B"


def test_normalize_passthrough():
    from node.openai_compat import normalize_model_name

    assert normalize_model_name("meta-llama/Llama-3.2-70B") == "meta-llama/Llama-3.2-70B"


def test_normalize_unknown_passthrough():
    from node.openai_compat import normalize_model_name

    assert normalize_model_name("some-custom-model-v1") == "some-custom-model-v1"


# ─────────────────────────── stream_chunk ────────────────────────────────────


def test_stream_chunk_sse_prefix():
    from node.openai_compat import stream_chunk

    chunk = stream_chunk("Hello", "meta-llama/Llama-3.2-8B")
    assert chunk.startswith("data: ")


def test_stream_chunk_sse_suffix():
    from node.openai_compat import stream_chunk

    chunk = stream_chunk("Hello", "meta-llama/Llama-3.2-8B")
    assert chunk.endswith("\n\n")


def test_stream_chunk_json_shape():
    from node.openai_compat import stream_chunk

    chunk = stream_chunk("Hello", "gpt-4o")
    payload = json.loads(chunk.removeprefix("data: ").strip())
    assert payload["object"] == "chat.completion.chunk"
    assert "choices" in payload
    choice = payload["choices"][0]
    assert choice["index"] == 0
    assert "delta" in choice
    assert choice["delta"]["content"] == "Hello"


def test_stream_chunk_finish_reason_none_by_default():
    from node.openai_compat import stream_chunk

    chunk = stream_chunk("token", "mymodel")
    payload = json.loads(chunk.removeprefix("data: ").strip())
    assert payload["choices"][0]["finish_reason"] is None


def test_stream_chunk_finish_reason_stop():
    from node.openai_compat import stream_chunk

    chunk = stream_chunk("", "mymodel", finish_reason="stop")
    payload = json.loads(chunk.removeprefix("data: ").strip())
    assert payload["choices"][0]["finish_reason"] == "stop"


def test_stream_chunk_custom_completion_id():
    from node.openai_compat import stream_chunk

    cid = "chatcmpl-testid123"
    chunk = stream_chunk("Hi", "mymodel", completion_id=cid)
    payload = json.loads(chunk.removeprefix("data: ").strip())
    assert payload["id"] == cid


def test_stream_chunk_has_created_timestamp():
    from node.openai_compat import stream_chunk

    before = int(time.time()) - 1
    chunk = stream_chunk("Hi", "mymodel")
    after = int(time.time()) + 1
    payload = json.loads(chunk.removeprefix("data: ").strip())
    assert before <= payload["created"] <= after


# ─────────────────────────── stream_done ─────────────────────────────────────


def test_stream_done_value():
    from node.openai_compat import stream_done

    assert stream_done() == "data: [DONE]\n\n"


# ─────────────────────────── format_chat_completion ──────────────────────────


def test_format_chat_completion_keys():
    from node.openai_compat import format_chat_completion

    resp = format_chat_completion("Answer here", "meta-llama/Llama-3.2-8B", "Question here")
    assert set(resp.keys()) == {"id", "object", "created", "model", "choices", "usage"}


def test_format_chat_completion_object_type():
    from node.openai_compat import format_chat_completion

    resp = format_chat_completion("Answer", "mymodel", "Prompt")
    assert resp["object"] == "chat.completion"


def test_format_chat_completion_choice_structure():
    from node.openai_compat import format_chat_completion

    resp = format_chat_completion("The answer", "mymodel", "The question")
    choice = resp["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "The answer"
    assert choice["finish_reason"] == "stop"


def test_format_chat_completion_custom_finish_reason():
    from node.openai_compat import format_chat_completion

    resp = format_chat_completion("...", "mymodel", "prompt", finish_reason="length")
    assert resp["choices"][0]["finish_reason"] == "length"


def test_format_chat_completion_usage_populated():
    from node.openai_compat import format_chat_completion

    prompt = "a" * 40  # 10 tokens
    content = "b" * 80  # 20 tokens
    resp = format_chat_completion(content, "mymodel", prompt)
    usage = resp["usage"]
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 20
    assert usage["total_tokens"] == 30


def test_format_chat_completion_id_prefix():
    from node.openai_compat import format_chat_completion

    resp = format_chat_completion("hi", "mymodel", "prompt")
    assert resp["id"].startswith("chatcmpl-")


# ─────────────────────────── format_completion ───────────────────────────────


def test_format_completion_keys():
    from node.openai_compat import format_completion

    resp = format_completion("some text", "mymodel", "the prompt")
    assert set(resp.keys()) == {"id", "object", "created", "model", "choices", "usage"}


def test_format_completion_object_type():
    from node.openai_compat import format_completion

    resp = format_completion("text", "mymodel", "prompt")
    assert resp["object"] == "text_completion"


def test_format_completion_choice_structure():
    from node.openai_compat import format_completion

    resp = format_completion("Generated text", "mymodel", "prompt")
    choice = resp["choices"][0]
    assert choice["index"] == 0
    assert choice["text"] == "Generated text"
    assert choice["finish_reason"] == "stop"
    assert choice["logprobs"] is None


def test_format_completion_id_prefix():
    from node.openai_compat import format_completion

    resp = format_completion("hi", "mymodel", "prompt")
    assert resp["id"].startswith("cmpl-")


# ─────────────────────────── OpenAICompatMiddleware ──────────────────────────


@pytest.fixture()
def test_app():
    """Minimal FastAPI app with OpenAICompatMiddleware attached."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from node.openai_compat import OpenAICompatMiddleware

    app = FastAPI()
    app.add_middleware(OpenAICompatMiddleware)

    @app.get("/ok")
    def ok():
        return {"status": "ok"}

    @app.get("/boom")
    def boom():
        raise RuntimeError("kaboom")

    return TestClient(app, raise_server_exceptions=False)


def test_middleware_adds_request_id_header(test_app):
    resp = test_app.get("/ok")
    assert "x-request-id" in resp.headers


def test_middleware_adds_openai_version_header(test_app):
    resp = test_app.get("/ok")
    assert resp.headers.get("openai-version") == "2020-10-01"


def test_middleware_options_returns_204(test_app):
    resp = test_app.options("/ok")
    assert resp.status_code == 204


def test_middleware_options_cors_headers(test_app):
    resp = test_app.options("/ok")
    assert "access-control-allow-origin" in resp.headers


def test_middleware_500_wrapped_as_openai_error(test_app):
    resp = test_app.get("/boom")
    assert resp.status_code == 500
    body = resp.json()
    assert "error" in body
    err = body["error"]
    assert err["type"] == "api_error"
    assert err["param"] is None
    assert err["code"] is None


def test_middleware_propagates_request_id_from_header(test_app):
    resp = test_app.get("/ok", headers={"X-Request-Id": "my-custom-id-42"})
    assert resp.headers.get("x-request-id") == "my-custom-id-42"
