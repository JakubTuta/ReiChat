"""`POST /v1/chat/completions` — an OpenAI-compatible endpoint so any tool
that already speaks OpenAI's wire format (editor plugins, the `openai` SDK,
agent frameworks) can point its base URL at this server with zero custom
integration work, the same way Ollama's own `/v1/chat/completions` does.

A plain async `django.views.View` rather than a DRF `APIView`: DRF has no
documented async request-handler support (nothing in its own docs or
Context7 confirms `async def get/post` is dispatched correctly), and this
endpoint needs a real async generator for SSE streaming — the same
`StreamingHttpResponse`-with-async-iterator pattern Django's own docs use,
confirmed via Context7 rather than assumed. Deliberately outside DRF, so
`csrf_exempt` is applied explicitly (DRF's `APIView` does this itself; a
plain `View` does not).
"""

from __future__ import annotations

import json
import logging
import time
import typing
import uuid

from asgiref.sync import sync_to_async
from channels.db import database_sync_to_async
from django.http import HttpRequest, HttpResponse, JsonResponse, StreamingHttpResponse
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt

from container.ContainerManager import ContainerManager

from .. import functions
from ..functions import ModelUnavailableError
from ..rag import embeddings
from .parsing import OpenAICompatError, ParsedRequest, parse_chat_completion_request

logger = logging.getLogger(__name__)


def _error_response(status: int, message: str, *, param: str | None = None, code: str | None = None) -> JsonResponse:
    error_type = "invalid_request_error" if status < 500 else "server_error"
    return JsonResponse(
        {"error": {"message": message, "type": error_type, "param": param, "code": code}}, status=status
    )


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _chunk_frame(completion_id: str, created: int, model_label: str, delta: dict, finish_reason: str | None) -> dict:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_label,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


async def _stream_chunks(
    parsed: ParsedRequest, completion_id: str, created: int, model_label: str
) -> typing.AsyncGenerator[bytes, None]:
    yield _sse(_chunk_frame(completion_id, created, model_label, {"role": "assistant"}, None))

    try:
        async for chunk in functions.astream_bot_response(
            parsed["ai_model"],
            parsed["parameters"],
            parsed["message"],
            parsed["image"],
            parsed["history"],
            parsed["system_prompt"],
            parsed["generation_params"],
        ):
            yield _sse(_chunk_frame(completion_id, created, model_label, {"content": chunk}, None))
    except ModelUnavailableError as e:
        # Headers (200, text/event-stream) are already committed by the time
        # a StreamingHttpResponse starts yielding, so a mid-stream failure
        # can only be surfaced in-band — the same constraint every other
        # OpenAI-compatible local server (llama.cpp, vLLM) works under.
        yield _sse({"error": {"message": str(e), "type": "server_error", "param": None, "code": None}})
        yield b"data: [DONE]\n\n"
        return
    except Exception:
        logger.exception("Unexpected error while streaming an OpenAI-compatible chat completion")
        yield _sse(
            {"error": {"message": "An unexpected error occurred.", "type": "server_error", "param": None, "code": None}}
        )
        yield b"data: [DONE]\n\n"
        return

    yield _sse(_chunk_frame(completion_id, created, model_label, {}, "stop"))
    yield b"data: [DONE]\n\n"


async def _build_full_response(
    parsed: ParsedRequest, completion_id: str, created: int, model_label: str
) -> JsonResponse:
    stats: functions.GenerationStats = {}
    full_text = ""

    try:
        async for chunk in functions.astream_bot_response(
            parsed["ai_model"],
            parsed["parameters"],
            parsed["message"],
            parsed["image"],
            parsed["history"],
            parsed["system_prompt"],
            parsed["generation_params"],
            stats,
        ):
            full_text += chunk
    except ModelUnavailableError as e:
        return _error_response(503, str(e))
    except Exception:
        logger.exception("Unexpected error in an OpenAI-compatible chat completion")
        return _error_response(500, "An unexpected error occurred.")

    prompt_tokens = stats.get("prompt_tokens", 0)
    completion_tokens = stats.get("completion_tokens", 0)

    return JsonResponse(
        {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_label,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": full_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    )


class ModelsView(View):
    """`GET /v1/models` — most OpenAI clients list models before anything
    else and refuse to continue without it. Lists every installed model as
    "name:tag", the id /v1/chat/completions and /v1/embeddings accept."""

    def get(self, request: HttpRequest) -> HttpResponse:
        # Lists the user's own Ollama too, which doesn't need Docker at all.
        containers = ContainerManager().get_available_containers()

        data = [
            {
                "id": f"{c['environment']['model']}:{c['environment']['parameters']}",
                "object": "model",
                "created": 0,
                "owned_by": "reichat",
            }
            for c in containers
            if c["environment"].get("model") and c["environment"].get("parameters")
        ]
        return JsonResponse({"object": "list", "data": data})


MAX_EMBEDDING_INPUTS = 256


@method_decorator(csrf_exempt, name="dispatch")
class EmbeddingsView(View):
    """`POST /v1/embeddings` against an installed embedding model, e.g.
    nomic-embed-text:latest — for tools that do their own RAG."""

    def post(self, request: HttpRequest) -> HttpResponse:
        try:
            body = json.loads(request.body)
        except json.JSONDecodeError:
            return _error_response(400, "Request body must be valid JSON.")

        if not isinstance(body, dict):
            return _error_response(400, "Request body must be a JSON object.")

        model_field = body.get("model")
        if not isinstance(model_field, str) or not model_field.strip():
            return _error_response(400, "model is required.", param="model")

        raw_input = body.get("input")
        inputs = [raw_input] if isinstance(raw_input, str) else raw_input
        if (
            not isinstance(inputs, list)
            or not inputs
            or not all(isinstance(i, str) for i in inputs)
        ):
            return _error_response(400, "input must be a string or a non-empty array of strings.", param="input")
        if len(inputs) > MAX_EMBEDDING_INPUTS:
            return _error_response(400, f"At most {MAX_EMBEDDING_INPUTS} inputs per request.", param="input")

        model_name, _, parameters = model_field.partition(":")
        try:
            client = embeddings.get_embeddings_client(model_name, parameters or "latest")
            vectors = client.embed_documents(inputs)
        except ModelUnavailableError as e:
            return _error_response(503, str(e))
        except Exception:
            logger.exception("Unexpected error while computing embeddings")
            return _error_response(500, "An unexpected error occurred.")

        return JsonResponse(
            {
                "object": "list",
                "data": [
                    {"object": "embedding", "index": i, "embedding": vector} for i, vector in enumerate(vectors)
                ],
                "model": model_field,
                # Ollama's embed API doesn't report token counts.
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            }
        )


@method_decorator(csrf_exempt, name="dispatch")
class ChatCompletionsView(View):
    async def post(self, request: HttpRequest) -> HttpResponse:
        try:
            body = json.loads(request.body)
        except json.JSONDecodeError:
            return _error_response(400, "Request body must be valid JSON.")

        try:
            parsed = await database_sync_to_async(parse_chat_completion_request)(body)
        except OpenAICompatError as e:
            return _error_response(e.status, e.message, param=e.param, code=e.code)

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        model_label = f"{parsed['ai_model'].model}:{parsed['parameters']}"

        if not parsed["stream"]:
            return await _build_full_response(parsed, completion_id, created, model_label)

        # Checked up front so the common failure mode (container not running
        # yet) gets a real 503 with a JSON body instead of a 200 whose SSE
        # stream immediately reports an error — see _stream_chunks' docstring
        # note on why an error *after* headers are sent can't do the same.
        try:
            await sync_to_async(functions.get_ollama_url, thread_sensitive=False)(
                parsed["ai_model"].model, parsed["parameters"]
            )
        except ModelUnavailableError as e:
            return _error_response(503, str(e))

        response = StreamingHttpResponse(
            _stream_chunks(parsed, completion_id, created, model_label), content_type="text/event-stream"
        )
        response["Cache-Control"] = "no-cache"
        return response
