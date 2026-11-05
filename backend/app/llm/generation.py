"""Talking to the model: streaming a reply, running the tool-call loop,
structured output, and the small generations around a chat (a title, a summary
of older messages). Which Ollama to talk to comes from llm/runtime.py.
"""

import asyncio
import json
import logging
import typing

import httpx
import ollama
import pydantic
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage, message_chunk_to_message
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama

from app.chat import context_window
from app.db import models
from app.llm import runtime
from app.llm.runtime import ModelUnavailableError
from app.mcp_integration import client as mcp_client
from app.tools import web_search as web_search_tool
from app.tools.builtin import BUILTIN_TOOLS, BUILTIN_TOOLS_BY_NAME

logger = logging.getLogger(__name__)

NOT_RUNNING_MESSAGE = (
    "Model container isn't running or is still pulling. "
    "Check the Models page and try again once it shows 'Running'."
)
UNREACHABLE_MESSAGE = "Could not reach the model — it may still be starting up. Try again in a moment."


class ThinkingDelta(typing.NamedTuple):
    """A piece of a thinking model's reasoning, streamed separately from
    the answer itself — yielded alongside plain `str` answer chunks."""

    text: str


class GenerationParams(typing.TypedDict, total=False):
    """Per-chat ChatOllama overrides. A key's absence (or a None value) means
    "use Ollama's own default" — these are genuinely optional, not settings
    with a meaningful zero value, so they're only ever passed to ChatOllama
    when actually set (see _chat_ollama_kwargs).
    """

    temperature: float | None
    num_ctx: int | None
    top_p: float | None
    seed: int | None


def _chat_ollama_kwargs(params: GenerationParams | None) -> dict[str, typing.Any]:
    if not params:
        return {}

    return {k: v for k, v in params.items() if v is not None}


_OLLAMA_HTTPX_TIMEOUT = httpx.Timeout(connect=15.0, read=120.0, write=15.0, pool=15.0)


def _build_chat_ollama(base_url: str, full_model_string: str, **kwargs: typing.Any) -> ChatOllama:
    return ChatOllama(
        model=full_model_string,
        base_url=base_url,
        client_kwargs={"timeout": _OLLAMA_HTTPX_TIMEOUT},
        **kwargs,
    )


class GenerationStats(typing.TypedDict, total=False):
    prompt_tokens: int
    completion_tokens: int
    tokens_per_second: float
    # Filled in by the caller (chat/socket.py), not here — context capacity is
    # a chat setting (generation_params["num_ctx"]), not something Ollama's
    # response carries.
    context_used: int
    context_limit: int


def _record_generation_stats(stats: GenerationStats | None, response_metadata: dict[str, typing.Any]) -> None:
    # Only Ollama's final streamed chunk (done=True) carries these — every
    # other chunk's response_metadata is empty/partial, hence the .get()s
    # rather than assuming they're always present.
    if stats is None:
        return

    if (prompt_tokens := response_metadata.get("prompt_eval_count")) is not None:
        stats["prompt_tokens"] = prompt_tokens

    eval_count = response_metadata.get("eval_count")
    if eval_count is not None:
        stats["completion_tokens"] = eval_count

    eval_duration = response_metadata.get("eval_duration")  # nanoseconds
    if eval_count and eval_duration:
        stats["tokens_per_second"] = eval_count / (eval_duration / 1_000_000_000)


async def _require_ollama_url(model_name: str, parameters: str) -> tuple[str, str]:
    url_info = await runtime.get_ollama_url(model_name, parameters)
    if url_info is None:
        raise ModelUnavailableError(NOT_RUNNING_MESSAGE)
    return url_info


async def astream_bot_response(
    model_name: str,
    parameters: str,
    message: str,
    image: str,
    history: list[dict[str, str]],
    system_prompt: str | None = None,
    generation_params: GenerationParams | None = None,
    stats: GenerationStats | None = None,
    reasoning: bool | str | None = None,
) -> typing.AsyncGenerator[str | ThinkingDelta]:
    """Yields answer text as `str`, and — only when `reasoning` is set —
    the model's reasoning as ThinkingDelta."""
    base_url, full_model_string = await _require_ollama_url(model_name, parameters)

    new_message = _map_history([{"role": "user", "content": message, "image": image}])[0]
    messages: list[typing.Any] = _map_history(history) + [new_message]

    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}, *messages]

    llm = _build_chat_ollama(
        base_url, full_model_string, reasoning=reasoning, **_chat_ollama_kwargs(generation_params)
    )

    # `stats` is a mutable out-param rather than a return value: an async
    # generator consumed with `async for` (as the caller does) has no way to
    # surface a value after the last `yield` the way a plain `return` does
    # on a sync generator's StopIteration — the caller passes an empty dict
    # in and reads it once the loop finishes.
    try:
        async for chunk in llm.astream(messages):
            if thinking := chunk.additional_kwargs.get("reasoning_content"):
                yield ThinkingDelta(thinking)
            if text := chunk.text:
                yield text
            if chunk.response_metadata:
                _record_generation_stats(stats, chunk.response_metadata)
    except httpx.HTTPError as e:
        raise ModelUnavailableError(UNREACHABLE_MESSAGE) from e
    except ollama.ResponseError as e:
        raise ModelUnavailableError(f"Ollama rejected the request: {e}") from e


class ToolCallInfo(typing.TypedDict):
    name: str
    args: dict[str, typing.Any]
    result: str


class ToolApprovalRequest(typing.TypedDict):
    name: str
    server: str
    args: dict[str, typing.Any]


# Asked before an MCP tool runs on a server that isn't auto-approved;
# resolves True to run it.
ToolApprover = typing.Callable[[ToolApprovalRequest], typing.Awaitable[bool]]

# A tool's output goes straight back into the prompt; an unbounded one (a
# web page, a directory listing) would blow through the context window.
MAX_TOOL_RESULT_CHARS = 8000
TOOL_DECLINED_RESULT = "The user declined to run this tool. Don't retry it; answer without it."


MAX_TOOL_ITERATIONS = 5


async def astream_tool_response(
    model_name: str,
    parameters: str,
    message: str,
    image: str,
    history: list[dict[str, str]],
    system_prompt: str | None = None,
    generation_params: GenerationParams | None = None,
    stats: GenerationStats | None = None,
    mcp_servers: list[models.MCPServer] | None = None,
    reasoning: bool | str | None = None,
    approve: ToolApprover | None = None,
) -> typing.AsyncGenerator[str | ThinkingDelta | ToolCallInfo]:
    """Like astream_bot_response, but binds the built-in tool set (plus any
    enabled MCP servers' tools) and runs the tool-call loop: ask the model,
    execute whatever tools it asks for, feed the results back, repeat until
    it answers without requesting one (or MAX_TOOL_ITERATIONS is hit).

    Every turn is streamed, and its text goes out as it arrives — a turn
    that turns out to be the final answer must never be thrown away and
    generated a second time (that used to double every tools-enabled reply
    and showed nothing until the first, discarded pass finished).

    Yields plain `str` chunks of text and `ToolCallInfo` dicts for each tool
    call made along the way, in call order — callers distinguish them with
    isinstance().
    """
    base_url, full_model_string = await _require_ollama_url(model_name, parameters)

    new_message = _map_history([{"role": "user", "content": message, "image": image}])[0]
    lc_messages: list[typing.Any] = _map_history(history) + [new_message]

    if system_prompt:
        lc_messages = [{"role": "system", "content": system_prompt}, *lc_messages]

    # Listing tools from every enabled MCP server means one connection
    # (subprocess spawn or HTTP round-trip) per server, done once up front
    # and in parallel — list_server_tools never raises, a server that's
    # down just contributes no tools rather than blocking this turn.
    mcp_lookup: dict[str, tuple[models.MCPServer, str]] = {}
    tool_specs: list[typing.Any] = list(BUILTIN_TOOLS)
    if await web_search_tool.available():
        tool_specs.append(web_search_tool.web_search)

    if mcp_servers:
        tool_lists = await asyncio.gather(*(mcp_client.list_server_tools(s) for s in mcp_servers))
        registry = mcp_client.build_tool_registry(list(zip(mcp_servers, tool_lists, strict=True)))
        tool_specs += registry.tool_specs
        mcp_lookup = registry.lookup

    llm = _build_chat_ollama(
        base_url, full_model_string, reasoning=reasoning, **_chat_ollama_kwargs(generation_params)
    )
    # langchain types the bound model's stream loosely; what it yields here is AIMessageChunk.
    llm_with_tools: typing.Any = llm.bind_tools(tool_specs)

    try:
        for _ in range(MAX_TOOL_ITERATIONS):
            gathered: AIMessageChunk | None = None

            async for chunk in llm_with_tools.astream(lc_messages):
                gathered = chunk if gathered is None else gathered + chunk
                if thinking := chunk.additional_kwargs.get("reasoning_content"):
                    yield ThinkingDelta(thinking)
                if text := chunk.text:
                    yield text
                if chunk.response_metadata:
                    _record_generation_stats(stats, chunk.response_metadata)

            if gathered is None or not gathered.tool_calls:
                return

            response = typing.cast(AIMessage, message_chunk_to_message(gathered))
            lc_messages.append(response)

            for tool_call in response.tool_calls:
                result = await _run_tool_call(tool_call["name"], tool_call["args"], mcp_lookup, approve)

                yield {
                    "name": tool_call["name"],
                    "args": tool_call["args"],
                    "result": result,
                }
                lc_messages.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=tool_call["id"],
                        name=tool_call["name"],
                    )
                )
    except httpx.HTTPError as e:
        raise ModelUnavailableError(UNREACHABLE_MESSAGE) from e
    except ollama.ResponseError as e:
        raise ModelUnavailableError(f"Ollama rejected the request: {e}") from e

    yield (
        "I wasn't able to finish that after several tool calls — try rephrasing "
        "the question or turning tools off for this message."
    )


def _truncate_tool_result(result: str) -> str:
    if len(result) <= MAX_TOOL_RESULT_CHARS:
        return result
    return f"{result[:MAX_TOOL_RESULT_CHARS]}\n[…output cut off after {MAX_TOOL_RESULT_CHARS} characters]"


async def _run_tool_call(
    name: str,
    args: dict[str, typing.Any],
    mcp_lookup: dict[str, tuple[models.MCPServer, str]],
    approve: ToolApprover | None = None,
) -> str:
    if name in BUILTIN_TOOLS_BY_NAME:
        return str(await BUILTIN_TOOLS_BY_NAME[name].ainvoke(args))

    if name == web_search_tool.web_search.name:
        return _truncate_tool_result(str(await web_search_tool.web_search.ainvoke(args)))

    if name in mcp_lookup:
        server, original_name = mcp_lookup[name]

        # No approver (a caller with no one to ask) means no unapproved
        # side effects either — refuse rather than run unattended.
        if not server.auto_approve:
            request: ToolApprovalRequest = {"name": original_name, "server": server.name, "args": args}
            if approve is None or not await approve(request):
                return TOOL_DECLINED_RESULT

        try:
            return _truncate_tool_result(await mcp_client.call_server_tool(server, original_name, args))
        except Exception as e:
            logger.warning(
                "MCP tool call failed for %s on server %r: %s",
                original_name,
                server.name,
                e,
            )
            return f"Error: MCP tool call failed: {e}"

    return f"Error: unknown tool {name!r}"


async def astream_structured_bot_response(
    model_name: str,
    parameters: str,
    message: str,
    image: str,
    history: list[dict[str, str]],
    structured_output: list[dict[str, str | None]],
    system_prompt: str | None = None,
    generation_params: GenerationParams | None = None,
    stats: GenerationStats | None = None,
) -> str | None:
    base_url, full_model_string = await _require_ollama_url(model_name, parameters)

    output_model = _build_structured_output_model(structured_output)
    json_schema = output_model.model_json_schema()
    parser = JsonOutputParser(pydantic_object=output_model)

    schema_instructions = f"""You must respond with JSON that matches the following schema:
```json
{json.dumps(json_schema, indent=2)}
```
Your response should be valid JSON only, with no other text or explanation."""

    # Ollama models reliably attend to only one system message, so a
    # persona's instructions and the schema instructions are combined into
    # one rather than sent as two separate system messages.
    system_message = f"{system_prompt}\n\n{schema_instructions}" if system_prompt else schema_instructions

    new_message = _map_history([{"role": "user", "content": message, "image": image}])[0]
    system_msg = {"role": "system", "content": system_message}
    messages: list[typing.Any] = _map_history(history) + [system_msg, new_message]

    # The schema itself, not just "json": Ollama compiles it into a grammar
    # that constrains decoding, so the reply can't come back with missing
    # fields or the wrong types. The instructions above stay because models
    # still fill fields better when told what each one is for.
    llm = _build_chat_ollama(
        base_url,
        full_model_string,
        format=json_schema,
        **_chat_ollama_kwargs(generation_params),
    )

    try:
        response = await llm.ainvoke(messages)
    except httpx.HTTPError as e:
        raise ModelUnavailableError(UNREACHABLE_MESSAGE) from e
    except ollama.ResponseError as e:
        raise ModelUnavailableError(f"Ollama rejected the request: {e}") from e

    if response.response_metadata:
        _record_generation_stats(stats, response.response_metadata)

    full_response = response.text

    try:
        parsed_dict = parser.parse(full_response)
    except OutputParserException as e:
        logger.warning("Structured output was not valid JSON: %s", e)
        return None

    try:
        validated = output_model.model_validate(parsed_dict)
    except pydantic.ValidationError as e:
        logger.warning("Structured output did not match the requested schema: %s", e)
        return None

    json_string = json.dumps(validated.model_dump(), indent=4)
    return f"```json\n{json_string}\n```"


def _map_history(history: list[dict[str, str]]) -> list[dict[str, typing.Any]]:
    """Build LangChain-style message dicts. Images must be structured content
    parts (`[{"type": "image_url", "image_url": "data:..."}]`) — ChatOllama
    only reads images from there. A top-level "images" key (the old shape)
    is not a recognized message field and is silently dropped, which made
    the vision feature a no-op regardless of what the user attached.
    """
    mapped: list[dict[str, typing.Any]] = []

    for message in history:
        content: typing.Any = message.get("content", "")
        image = message.get("image", "")

        if image:
            content = [
                {"type": "text", "text": content},
                {"type": "image_url", "image_url": image},
            ]

        mapped.append({"role": message.get("role", ""), "content": content})

    return mapped


_SIMPLE_TYPE_ANNOTATIONS: dict[str, type] = {
    "string": str,
    "number": float,
    "bool": bool,
    "date": str,
}


def _build_structured_output_model(
    structured_output: list[dict[str, str | None]],
) -> type[pydantic.BaseModel]:
    """Turn the user's field list into a real Pydantic model, so the parsed
    response can actually be validated against it — `JsonOutputParser`
    accepts a `pydantic_object` but never validates against it itself, it
    only uses it to render format instructions.
    """
    fields: dict[str, tuple[type, typing.Any]] = {}

    for field_spec in structured_output:
        field_name = str(field_spec["field"])
        field_type = str(field_spec["type"])
        description = field_spec.get("description") or None

        if field_type in _SIMPLE_TYPE_ANNOTATIONS:
            annotation: type = _SIMPLE_TYPE_ANNOTATIONS[field_type]
        elif field_type == "array":
            array_type = str(field_spec.get("arrayType") or "string")
            if array_type not in _SIMPLE_TYPE_ANNOTATIONS:
                array_type = "string"
            annotation = list[_SIMPLE_TYPE_ANNOTATIONS[array_type]]  # type: ignore[valid-type]
        else:
            annotation = str

        fields[field_name] = (annotation, pydantic.Field(description=description))

    output_model: type[pydantic.BaseModel] = pydantic.create_model(  # type: ignore[call-overload]
        "StructuredOutput", **fields
    )
    return output_model


async def asummarize_conversation(
    model_name: str,
    parameters: str,
    previous_summary: str,
    messages: list[dict[str, typing.Any]],
    num_ctx: int,
) -> str:
    base_url, full_model_string = await _require_ollama_url(model_name, parameters)
    prompt = context_window.build_summary_prompt(previous_summary, messages, num_ctx)
    llm = _build_chat_ollama(base_url, full_model_string, num_ctx=num_ctx, temperature=0.2)

    try:
        response = await llm.ainvoke([{"role": "user", "content": prompt}])
    except httpx.HTTPError as e:
        raise ModelUnavailableError(UNREACHABLE_MESSAGE) from e
    except ollama.ResponseError as e:
        raise ModelUnavailableError(f"Ollama rejected the request: {e}") from e

    return response.text.strip()


TITLE_PROMPT = (
    "Write a short title (3 to 6 words) for a conversation that starts like this. "
    "Use the conversation's language. Reply with the title only — no quotes, no punctuation at the end."
)
MAX_TITLE_LENGTH = 60


def clean_title(raw: str) -> str:
    title = raw.strip().splitlines()[0] if raw.strip() else ""
    title = title.strip().strip("\"'`*#").strip()
    title = title.removeprefix("Title:").strip().rstrip(".")
    return title[:MAX_TITLE_LENGTH].strip()


async def agenerate_title(
    model_name: str, parameters: str, user_prompt: str, answer: str, reasoning: bool | str | None
) -> str:
    url_info = await runtime.get_ollama_url(model_name, parameters)
    if url_info is None:
        return ""

    base_url, full_model_string = url_info
    # Thinking would cost far more than the title itself; ask models that
    # can switch it off to do so (reasoning=False), leave the rest alone.
    llm = _build_chat_ollama(
        base_url,
        full_model_string,
        temperature=0.3,
        num_predict=24,
        reasoning=False if reasoning is not None else None,
    )
    excerpt = f"User: {user_prompt[:1000]}\n\nAssistant: {answer[:1000]}"
    response = await llm.ainvoke([{"role": "user", "content": f"{TITLE_PROMPT}\n\n{excerpt}"}])

    return clean_title(response.text)
