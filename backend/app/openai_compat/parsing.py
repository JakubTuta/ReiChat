"""Turns an OpenAI `/v1/chat/completions` request body into the shape
`llm.generation.astream_bot_response` already expects. Deliberately a strict
subset of the real API — verified against the OpenAI OpenAPI spec via
Context7 rather than guessed: `tools`/`functions` (this endpoint has no
tool-calling loop — use the WebSocket chat for that), multi-choice (`n>1`),
and non-text `response_format` are all rejected outright rather than
silently ignored, since silently dropping a param a client relied on would
produce a response that looks fine but isn't what was asked for.
"""

from __future__ import annotations

import typing

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import models

_VALID_ROLES = {"system", "user", "assistant"}

# Mirrors MAX_MESSAGE_LENGTH in chat/socket.py — kept as its own
# constant rather than imported from there, since this endpoint's request
# shape and validation are otherwise independent of the WebSocket protocol.
MAX_MESSAGE_LENGTH = 10_000


class OpenAICompatError(Exception):
    """Carries enough to build an OpenAI-shaped `{"error": {...}}` body."""

    def __init__(
        self, status: int, message: str, *, param: str | None = None, code: str | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.param = param
        self.code = code


class ParsedTurn(typing.TypedDict):
    role: str
    content: str
    image: str


class ParsedRequest(typing.TypedDict):
    ai_model: models.AIModel
    parameters: str
    system_prompt: str | None
    message: str
    image: str
    history: list[ParsedTurn]
    stream: bool
    generation_params: dict[str, float | int]


def _extract_text_and_image(content: object, index: int) -> tuple[str, str]:
    if isinstance(content, str):
        return content, ""

    if not isinstance(content, list):
        raise OpenAICompatError(
            400, f"messages[{index}].content must be a string or an array of content parts.", param="messages"
        )

    text_parts: list[str] = []
    image = ""

    for part in content:
        if not isinstance(part, dict):
            raise OpenAICompatError(
                400, f"messages[{index}].content parts must be objects.", param="messages"
            )

        part_type = part.get("type")

        if part_type == "text":
            text = part.get("text")
            if not isinstance(text, str):
                raise OpenAICompatError(
                    400,
                    f"messages[{index}] has a text content part with a non-string 'text'.",
                    param="messages",
                )
            text_parts.append(text)
        elif part_type == "image_url":
            image_url = part.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else None
            if not isinstance(url, str) or not url:
                raise OpenAICompatError(
                    400, f"messages[{index}] has an image_url part missing image_url.url.", param="messages"
                )
            if not image:
                # Only one image per turn reaches Ollama (see llm/generation.py map_history)
                # — first part wins, same as if a caller sent several.
                image = url
        else:
            raise OpenAICompatError(
                400,
                f"messages[{index}] has an unsupported content part type: {part_type!r}.",
                param="messages",
            )

    return "".join(text_parts), image


def _parse_generation_params(body: dict[str, typing.Any]) -> dict[str, float | int]:
    generation_params: dict[str, float | int] = {}

    for key in ("temperature", "top_p"):
        value = body.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise OpenAICompatError(400, f"{key} must be a number.", param=key)
        generation_params[key] = float(value)

    seed = body.get("seed")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise OpenAICompatError(400, "seed must be an integer.", param="seed")
        generation_params["seed"] = seed

    return generation_params


async def parse_chat_completion_request(body: object, session: AsyncSession) -> ParsedRequest:
    if not isinstance(body, dict):
        raise OpenAICompatError(400, "Request body must be a JSON object.")

    model_field = body.get("model")
    if not isinstance(model_field, str) or not model_field.strip():
        raise OpenAICompatError(400, "model is required.", param="model")

    model_name, _, parameters = model_field.partition(":")
    parameters = parameters or "latest"

    ai_model = (
        await session.execute(select(models.AIModel).where(models.AIModel.model == model_name))
    ).scalar_one_or_none()
    if ai_model is None:
        raise OpenAICompatError(
            404, f"The model `{model_field}` does not exist.", param="model", code="model_not_found"
        )

    if body.get("tools") or body.get("functions"):
        raise OpenAICompatError(
            400, "Tool/function calling is not supported by this endpoint.", param="tools"
        )

    response_format = body.get("response_format")
    if isinstance(response_format, dict) and response_format.get("type") not in (None, "text"):
        raise OpenAICompatError(
            400, "Only the default text response_format is supported.", param="response_format"
        )

    n = body.get("n")
    if n is not None and n != 1:
        raise OpenAICompatError(400, "Only n=1 is supported.", param="n")

    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        raise OpenAICompatError(400, "stream must be a boolean.", param="stream")

    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise OpenAICompatError(400, "messages is required and must be a non-empty array.", param="messages")

    system_parts: list[str] = []
    turns: list[ParsedTurn] = []

    for i, raw in enumerate(raw_messages):
        if not isinstance(raw, dict):
            raise OpenAICompatError(400, f"messages[{i}] must be an object.", param="messages")

        role = raw.get("role")
        if role not in _VALID_ROLES:
            raise OpenAICompatError(
                400,
                f"messages[{i}].role must be one of {sorted(_VALID_ROLES)} (tool-call messages "
                "aren't supported by this endpoint).",
                param="messages",
            )

        text, image = _extract_text_and_image(raw.get("content"), i)

        if len(text) > MAX_MESSAGE_LENGTH:
            raise OpenAICompatError(
                400,
                f"messages[{i}] exceeds the maximum length of {MAX_MESSAGE_LENGTH} characters.",
                param="messages",
            )

        if role == "system":
            if text:
                system_parts.append(text)
            continue

        turns.append({"role": role, "content": text, "image": image})

    if not turns or turns[-1]["role"] != "user":
        raise OpenAICompatError(400, "The last message must have role 'user'.", param="messages")

    *history, last = turns

    return {
        "ai_model": ai_model,
        "parameters": parameters,
        "system_prompt": "\n\n".join(system_parts) or None,
        "message": last["content"],
        "image": last["image"],
        "history": history,
        "stream": stream,
        "generation_params": _parse_generation_params(body),
    }
