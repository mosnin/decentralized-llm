"""
OpenAI wire-format compatibility layer.

Provides error formatting, token counting, model name normalization,
SSE streaming helpers, and response builders so the API gateway is a
true drop-in replacement for the OpenAI REST API.
"""

from __future__ import annotations

import json
import time
import uuid
from enum import StrEnum
from typing import Any

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# ─────────────────────────── error codes ─────────────────────────────────────


class OpenAIErrorCode(StrEnum):
    """Standard OpenAI API error type strings."""

    invalid_request_error = "invalid_request_error"
    authentication_error = "authentication_error"
    permission_error = "permission_error"
    not_found_error = "not_found_error"
    rate_limit_error = "rate_limit_error"
    api_error = "api_error"
    overloaded_error = "overloaded_error"


# ─────────────────────────── error response ───────────────────────────────────


def openai_error_response(
    status_code: int,
    error_type: str,
    message: str,
    param: str | None = None,
    code: str | None = None,
) -> JSONResponse:
    """Return a JSONResponse with the exact OpenAI error envelope format.

    Example output::

        {"error": {"message": "...", "type": "...", "param": null, "code": null}}
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        },
    )


# ─────────────────────────── token counting ───────────────────────────────────

_CHARS_PER_TOKEN = 4  # approximate for English text


def count_tokens(text: str, model: str = "") -> int:  # noqa: ARG001
    """Approximate token count using character heuristics (~4 chars/token).

    Does not require tiktoken.  ``model`` is accepted for API compatibility
    but currently ignored — the same heuristic is applied for all models.
    """
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def build_usage(prompt_text: str, completion_text: str, model: str = "") -> dict[str, int]:
    """Return a usage dict compatible with the OpenAI response schema."""
    prompt_tokens = count_tokens(prompt_text, model)
    completion_tokens = count_tokens(completion_text, model)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


# ─────────────────────────── model name mapping ───────────────────────────────

_MODEL_MAP: dict[str, str] = {
    "gpt-4": "meta-llama/Llama-3.2-70B",
    "gpt-4-turbo": "meta-llama/Llama-3.2-70B",
    "gpt-3.5-turbo": "meta-llama/Llama-3.2-3B",
    "gpt-4o": "meta-llama/Llama-3.2-8B",
    "gpt-4o-mini": "meta-llama/Llama-3.2-8B",
}


def normalize_model_name(requested_model: str) -> str:
    """Map OpenAI model aliases to our network model identifiers.

    Unknown names are passed through unchanged so callers can use
    native model IDs directly.
    """
    return _MODEL_MAP.get(requested_model, requested_model)


# ─────────────────────────── SSE helpers ─────────────────────────────────────


def _new_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def stream_chunk(
    content: str,
    model: str,
    finish_reason: str | None = None,
    completion_id: str | None = None,
) -> str:
    """Format a single SSE data line for a chat completion chunk.

    Returns a string ready to be written to an SSE stream, including the
    trailing ``\\n\\n`` separator.
    """
    cid = completion_id or _new_completion_id()
    delta: dict[str, Any] = {}
    if content:
        delta["content"] = content

    payload = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "delta": delta,
                "index": 0,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload)}\n\n"


def stream_done() -> str:
    """Return the terminal SSE line that signals end-of-stream."""
    return "data: [DONE]\n\n"


# ─────────────────────────── response formatters ─────────────────────────────


def format_chat_completion(
    content: str,
    model: str,
    prompt_text: str,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    """Build a full non-streaming chat completion response dict.

    Matches the OpenAI ``/v1/chat/completions`` response schema including
    ``id``, ``object``, ``created``, ``model``, ``choices``, and ``usage``.
    """
    return {
        "id": _new_completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": build_usage(prompt_text, content, model),
    }


def format_completion(
    content: str,
    model: str,
    prompt_text: str,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    """Build a full non-streaming text completion response dict.

    Matches the OpenAI ``/v1/completions`` response schema.
    """
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:24]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "text": content,
                "index": 0,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
        "usage": build_usage(prompt_text, content, model),
    }


# ─────────────────────────── ASGI middleware ─────────────────────────────────


class OpenAICompatMiddleware(BaseHTTPMiddleware):
    """FastAPI/Starlette middleware that adds OpenAI wire-format compatibility.

    Responsibilities:

    * Adds ``X-Request-Id`` response header (generated if absent from request).
    * Handles ``OPTIONS`` preflight requests with a 204 response.
    * Converts HTTP 500 responses to the OpenAI error envelope format.
    * Adds the ``openai-version: 2020-10-01`` response header.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # ── preflight ──────────────────────────────────────────────────────
        if request.method == "OPTIONS":
            return Response(
                status_code=204,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                    "Access-Control-Allow-Headers": "*",
                    "Access-Control-Max-Age": "86400",
                },
            )

        # ── request id ────────────────────────────────────────────────────
        request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex[:24]

        # ── call handler, converting unhandled exceptions to OpenAI errors ─
        try:
            response = await call_next(request)
        except Exception:
            response = openai_error_response(
                status_code=500,
                error_type=OpenAIErrorCode.api_error,
                message="An internal server error occurred.",
            )
            response.headers["X-Request-Id"] = request_id
            response.headers["openai-version"] = "2020-10-01"
            return response

        # ── wrap 500s that did not raise (e.g. explicit HTTPException 500) ─
        if response.status_code == 500:
            response = openai_error_response(
                status_code=500,
                error_type=OpenAIErrorCode.api_error,
                message="An internal server error occurred.",
            )

        # ── inject standard headers ───────────────────────────────────────
        response.headers["X-Request-Id"] = request_id
        response.headers["openai-version"] = "2020-10-01"

        return response
