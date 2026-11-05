"""An OpenAI-compatible API, so any tool that already speaks OpenAI's wire
format (editor plugins, the `openai` SDK, agent frameworks) can point its base
URL at this server with no custom integration, the way Ollama's own `/v1`
endpoints work.

Request bodies are parsed by hand instead of through FastAPI: these clients
often send no Content-Type, which FastAPI would refuse, and every failure
needs OpenAI's error shape rather than the app's own `{"error": "..."}`.
"""

import json
import logging
import time
import typing
import uuid
from collections.abc import AsyncIterator

import httpx
import ollama
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from app.containers import service
from app.db.session import SessionLocal
from app.llm import generation, runtime
from app.llm.runtime import ModelUnavailableError
from app.openai_compat.parsing import OpenAICompatError, ParsedRequest, parse_chat_completion_request
from app.rag import embeddings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1")

MAX_EMBEDDING_INPUTS = 256
UNEXPECTED_ERROR_MESSAGE = "An unexpected error occurred."
SSE_DONE = b"data: [DONE]\n\n"


def error_body(
    message: str, *, error_type: str = "server_error", param: str | None = None, code: str | None = None
) -> dict[str, typing.Any]:
    return {"error": {"message": message, "type": error_type, "param": param, "code": code}}


def error_response(
    status: int, message: str, *, param: str | None = None, code: str | None = None
) -> JSONResponse:
    error_type = "invalid_request_error" if status < 500 else "server_error"
    return JSONResponse(
        error_body(message, error_type=error_type, param=param, code=code), status_code=status
    )


def rejection(error: OpenAICompatError) -> JSONResponse:
    return error_response(error.status, error.message, param=error.param, code=error.code)


async def read_json(request: Request) -> object:
    try:
        # ValueError covers both bad JSON and a body that isn't valid UTF-8.
        return json.loads(await request.body())
    except ValueError as e:
        raise OpenAICompatError(400, "Request body must be valid JSON.") from e


def sse(payload: dict[str, typing.Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def chunk_frame(
    completion_id: str, created: int, model_label: str, delta: dict[str, str], finish_reason: str | None
) -> dict[str, typing.Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_label,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def unreachable_message(error: Exception) -> str:
    if isinstance(error, ollama.ResponseError):
        return f"Ollama rejected the request: {error}"
    return generation.UNREACHABLE_MESSAGE


def bot_stream(
    parsed: ParsedRequest, stats: generation.GenerationStats | None = None
) -> AsyncIterator[str | generation.ThinkingDelta]:
    return generation.astream_bot_response(
        parsed["ai_model"].model,
        parsed["parameters"],
        parsed["message"],
        parsed["image"],
        typing.cast("list[dict[str, str]]", parsed["history"]),
        parsed["system_prompt"],
        parsed["generation_params"],
        stats,
    )


async def stream_chunks(
    parsed: ParsedRequest, completion_id: str, created: int, model_label: str
) -> AsyncIterator[bytes]:
    yield sse(chunk_frame(completion_id, created, model_label, {"role": "assistant"}, None))

    try:
        async for chunk in bot_stream(parsed):
            if isinstance(chunk, str):
                yield sse(chunk_frame(completion_id, created, model_label, {"content": chunk}, None))
    except ModelUnavailableError as e:
        # The 200 and its headers are already on the wire once streaming has
        # started, so a failure now can only be reported in-band — the same
        # constraint every other OpenAI-compatible local server works under.
        yield sse(error_body(str(e)))
        yield SSE_DONE
        return
    except Exception:
        logger.exception("Unexpected error while streaming an OpenAI-compatible chat completion")
        yield sse(error_body(UNEXPECTED_ERROR_MESSAGE))
        yield SSE_DONE
        return

    yield sse(chunk_frame(completion_id, created, model_label, {}, "stop"))
    yield SSE_DONE


async def full_response(
    parsed: ParsedRequest, completion_id: str, created: int, model_label: str
) -> Response:
    stats = generation.GenerationStats()
    full_text = ""

    try:
        async for chunk in bot_stream(parsed, stats):
            if isinstance(chunk, str):
                full_text += chunk
    except ModelUnavailableError as e:
        return error_response(503, str(e))
    except Exception:
        logger.exception("Unexpected error in an OpenAI-compatible chat completion")
        return error_response(500, UNEXPECTED_ERROR_MESSAGE)

    prompt_tokens = stats.get("prompt_tokens", 0)
    completion_tokens = stats.get("completion_tokens", 0)

    return JSONResponse(
        {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_label,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": full_text}, "finish_reason": "stop"}
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    )


@router.post("/chat/completions")
async def create_chat_completion(request: Request) -> Response:
    try:
        body = await read_json(request)
        # Closed before streaming starts: a stream can run for minutes and
        # must not hold a database connection the whole time.
        async with SessionLocal() as session:
            parsed = await parse_chat_completion_request(body, session)
    except OpenAICompatError as e:
        return rejection(e)
    except Exception:
        logger.exception("Unexpected error reading an OpenAI-compatible chat completion request")
        return error_response(500, UNEXPECTED_ERROR_MESSAGE)

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model_label = f"{parsed['ai_model'].model}:{parsed['parameters']}"

    if not parsed["stream"]:
        return await full_response(parsed, completion_id, created, model_label)

    # Checked up front so the common failure (the model isn't running yet) gets
    # a real 503 with a JSON body, not a 200 whose stream opens with an error.
    try:
        if await runtime.get_ollama_url(parsed["ai_model"].model, parsed["parameters"]) is None:
            raise ModelUnavailableError(generation.NOT_RUNNING_MESSAGE)
    except ModelUnavailableError as e:
        return error_response(503, str(e))
    except Exception:
        logger.exception("Unexpected error looking up the model for an OpenAI-compatible chat completion")
        return error_response(500, UNEXPECTED_ERROR_MESSAGE)

    return StreamingResponse(
        stream_chunks(parsed, completion_id, created, model_label),
        # No charset: Starlette would append one to a text/* media_type.
        headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
    )


@router.get("/models")
async def list_models() -> Response:
    """Most OpenAI clients list models before anything else and refuse to
    continue without it. Lists every installed model as "name:tag", the id
    the other two endpoints accept."""
    try:
        containers = await service.get_available_containers()
    except Exception:
        logger.exception("Could not list models for /v1/models")
        return error_response(500, UNEXPECTED_ERROR_MESSAGE)

    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": f"{c['environment']['model']}:{c['environment']['parameters']}",
                    "object": "model",
                    "created": 0,
                    "owned_by": "reichat",
                }
                for c in containers
                if c["environment"].get("model") and c["environment"].get("parameters")
            ],
        }
    )


def parse_embeddings_request(body: object) -> tuple[str, list[str]]:
    if not isinstance(body, dict):
        raise OpenAICompatError(400, "Request body must be a JSON object.")

    model_field = body.get("model")
    if not isinstance(model_field, str) or not model_field.strip():
        raise OpenAICompatError(400, "model is required.", param="model")

    raw_input = body.get("input")
    inputs = [raw_input] if isinstance(raw_input, str) else raw_input
    if not isinstance(inputs, list) or not inputs or not all(isinstance(i, str) for i in inputs):
        raise OpenAICompatError(400, "input must be a string or a non-empty array of strings.", param="input")
    if len(inputs) > MAX_EMBEDDING_INPUTS:
        raise OpenAICompatError(400, f"At most {MAX_EMBEDDING_INPUTS} inputs per request.", param="input")

    return model_field, inputs


@router.post("/embeddings")
async def create_embeddings(request: Request) -> Response:
    """Against an installed embedding model, e.g. nomic-embed-text:latest —
    for tools that do their own RAG."""
    try:
        model_field, inputs = parse_embeddings_request(await read_json(request))
    except OpenAICompatError as e:
        return rejection(e)

    model_name, _, parameters = model_field.partition(":")
    try:
        client = await embeddings.get_embeddings_client(model_name, parameters or "latest")
        vectors = await client.aembed_documents(inputs)
    except ModelUnavailableError as e:
        return error_response(503, str(e))
    except (httpx.HTTPError, ollama.ResponseError) as e:
        logger.warning("Embedding request to Ollama failed: %s", e)
        return error_response(503, unreachable_message(e))
    except Exception:
        logger.exception("Unexpected error while computing embeddings")
        return error_response(500, UNEXPECTED_ERROR_MESSAGE)

    return JSONResponse(
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
