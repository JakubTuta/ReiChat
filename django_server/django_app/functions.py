import asyncio
import json
import logging
import typing

import httpx
import ollama
import pydantic
from asgiref.sync import sync_to_async
from django.db import transaction
from django.db.models.manager import BaseManager
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessageChunk, ToolMessage, message_chunk_to_message
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama

from container.ContainerManager import LABEL_GPU, ContainerManager

from . import context_window, models, ollama_api, progress_state
from .mcp_integration import client as mcp_client
from .tools.builtin import BUILTIN_TOOLS, BUILTIN_TOOLS_BY_NAME

logger = logging.getLogger(__name__)


class ModelUnavailableError(Exception):
    pass


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


def _build_chat_ollama(
    base_url: str, full_model_string: str, **kwargs: typing.Any
) -> ChatOllama:
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
    # Filled in by the caller (consumers.py), not here — context capacity is
    # a chat setting (generation_params["num_ctx"]), not something Ollama's
    # response carries.
    context_used: int
    context_limit: int


def _record_generation_stats(
    stats: GenerationStats | None, response_metadata: dict
) -> None:
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


async def astream_bot_response(
    model: models.AIModel,
    parameters: str,
    message: str,
    image: str,
    history: list[dict[str, str]],
    system_prompt: str | None = None,
    generation_params: GenerationParams | None = None,
    stats: GenerationStats | None = None,
) -> typing.AsyncGenerator[str, None]:
    # get_ollama_url talks to the Docker SDK, which is blocking I/O — run it
    # off the event loop so one chat's container lookup can't stall every
    # other concurrent chat's ASGI worker.
    url_info = await sync_to_async(get_ollama_url, thread_sensitive=False)(
        model.model, parameters
    )
    if url_info is None:
        raise ModelUnavailableError(
            "Model container isn't running or is still pulling. "
            "Check the Models page and try again once it shows 'Running'."
        )

    base_url, full_model_string = url_info

    new_message = _map_history([{"role": "user", "content": message, "image": image}])[
        0
    ]
    messages = _map_history(history) + [new_message]

    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}, *messages]

    llm = _build_chat_ollama(
        base_url, full_model_string, **_chat_ollama_kwargs(generation_params)
    )

    # `stats` is a mutable out-param rather than a return value: an async
    # generator consumed with `async for` (as the caller does) has no way to
    # surface a value after the last `yield` the way a plain `return` does
    # on a sync generator's StopIteration — the caller passes an empty dict
    # in and reads it once the loop finishes.
    try:
        async for chunk in llm.astream(messages):
            yield chunk.text
            if chunk.response_metadata:
                _record_generation_stats(stats, chunk.response_metadata)
    except httpx.HTTPError as e:
        raise ModelUnavailableError(
            "Could not reach the model — it may still be starting up. Try again in a moment."
        ) from e
    except ollama.ResponseError as e:
        raise ModelUnavailableError(f"Ollama rejected the request: {e}") from e


class ToolCallInfo(typing.TypedDict):
    name: str
    args: dict[str, typing.Any]
    result: str


MAX_TOOL_ITERATIONS = 5


async def astream_tool_response(
    model: models.AIModel,
    parameters: str,
    message: str,
    image: str,
    history: list[dict[str, str]],
    system_prompt: str | None = None,
    generation_params: GenerationParams | None = None,
    stats: GenerationStats | None = None,
    mcp_servers: list[models.MCPServer] | None = None,
) -> typing.AsyncGenerator[str | ToolCallInfo, None]:
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
    url_info = await sync_to_async(get_ollama_url, thread_sensitive=False)(
        model.model, parameters
    )
    if url_info is None:
        raise ModelUnavailableError(
            "Model container isn't running or is still pulling. "
            "Check the Models page and try again once it shows 'Running'."
        )

    base_url, full_model_string = url_info

    new_message = _map_history([{"role": "user", "content": message, "image": image}])[
        0
    ]
    lc_messages: list = _map_history(history) + [new_message]

    if system_prompt:
        lc_messages = [{"role": "system", "content": system_prompt}, *lc_messages]

    # Listing tools from every enabled MCP server means one connection
    # (subprocess spawn or HTTP round-trip) per server, done once up front
    # and in parallel — list_server_tools never raises, a server that's
    # down just contributes no tools rather than blocking this turn.
    mcp_lookup: dict[str, tuple[models.MCPServer, str]] = {}
    tool_specs: list[typing.Any] = list(BUILTIN_TOOLS)

    if mcp_servers:
        tool_lists = await asyncio.gather(
            *(mcp_client.list_server_tools(s) for s in mcp_servers)
        )
        registry = mcp_client.build_tool_registry(
            list(zip(mcp_servers, tool_lists, strict=True))
        )
        tool_specs += registry.tool_specs
        mcp_lookup = registry.lookup

    llm = _build_chat_ollama(
        base_url, full_model_string, **_chat_ollama_kwargs(generation_params)
    )
    llm_with_tools = llm.bind_tools(tool_specs)

    try:
        for _ in range(MAX_TOOL_ITERATIONS):
            gathered: AIMessageChunk | None = None

            async for chunk in llm_with_tools.astream(lc_messages):
                gathered = chunk if gathered is None else gathered + chunk
                if text := chunk.text:
                    yield text
                if chunk.response_metadata:
                    _record_generation_stats(stats, chunk.response_metadata)

            if gathered is None or not gathered.tool_calls:
                return

            response = message_chunk_to_message(gathered)
            lc_messages.append(response)

            for tool_call in response.tool_calls:
                result = await _run_tool_call(
                    tool_call["name"], tool_call["args"], mcp_lookup
                )

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
        raise ModelUnavailableError(
            "Could not reach the model — it may still be starting up. Try again in a moment."
        ) from e
    except ollama.ResponseError as e:
        raise ModelUnavailableError(f"Ollama rejected the request: {e}") from e

    yield (
        "I wasn't able to finish that after several tool calls — try rephrasing "
        "the question or turning tools off for this message."
    )


async def _run_tool_call(
    name: str,
    args: dict[str, typing.Any],
    mcp_lookup: dict[str, tuple[models.MCPServer, str]],
) -> str:
    if name in BUILTIN_TOOLS_BY_NAME:
        return await sync_to_async(
            BUILTIN_TOOLS_BY_NAME[name].invoke, thread_sensitive=False
        )(args)

    if name in mcp_lookup:
        server, original_name = mcp_lookup[name]
        try:
            return await mcp_client.call_server_tool(server, original_name, args)
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
    model: models.AIModel,
    parameters: str,
    message: str,
    image: str,
    history: list[dict[str, str]],
    structured_output: list[dict[str, str | str | None]],
    system_prompt: str | None = None,
    generation_params: GenerationParams | None = None,
    stats: GenerationStats | None = None,
) -> str | None:
    url_info = await sync_to_async(get_ollama_url, thread_sensitive=False)(
        model.model, parameters
    )
    if url_info is None:
        raise ModelUnavailableError(
            "Model container isn't running or is still pulling. "
            "Check the Models page and try again once it shows 'Running'."
        )

    base_url, full_model_string = url_info

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
    system_message = (
        f"{system_prompt}\n\n{schema_instructions}"
        if system_prompt
        else schema_instructions
    )

    new_message = _map_history([{"role": "user", "content": message, "image": image}])[
        0
    ]
    system_msg = {"role": "system", "content": system_message}
    messages = _map_history(history) + [system_msg, new_message]

    llm = _build_chat_ollama(
        base_url,
        full_model_string,
        format="json",
        **_chat_ollama_kwargs(generation_params),
    )

    try:
        response = await llm.ainvoke(messages)
    except httpx.HTTPError as e:
        raise ModelUnavailableError(
            "Could not reach the model — it may still be starting up. Try again in a moment."
        ) from e
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


def _map_history(
    history: list[dict[str, str]],
) -> list[dict[str, str | list[dict[str, str]]]]:
    """Build LangChain-style message dicts. Images must be structured content
    parts (`[{"type": "image_url", "image_url": "data:..."}]`) — ChatOllama
    only reads images from there. A top-level "images" key (the old shape)
    is not a recognized message field and is silently dropped, which made
    the vision feature a no-op regardless of what the user attached.
    """
    mapped: list[dict[str, str | list[dict[str, str]]]] = []

    for message in history:
        content: str | list[dict[str, str]] = message.get("content", "")
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
    structured_output: list[dict[str, str | str | None]],
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

    return pydantic.create_model("StructuredOutput", **fields)  # type: ignore[call-overload]


def get_ollama_url(model_name: str, parameters: str) -> tuple[str, str] | None:
    """(base_url, "model:tag") for a running container that holds its model,
    None if the container isn't running. Raises ModelUnavailableError for the
    "running but not usable yet" cases, each with its own next step."""
    container_manager = ContainerManager()
    container_name = ContainerManager.container_name(model_name, parameters)

    if progress_state.is_in_progress(container_name):
        raise ModelUnavailableError(
            "This model is still downloading. Wait for it to finish on the Models page, then try again."
        )

    container = container_manager.get_container(container_name)
    if container is None or container.status != "running":
        return None

    base_url = container_manager.base_url_for(container)
    if base_url is None:
        return None

    # None (Ollama not answering yet) falls through: the request itself then
    # fails with the "may still be starting up" message.
    if container_manager.check_model_installed(container) is False:
        raise ModelUnavailableError(
            "This model's download didn't finish. Open the Models page and press 'Resume download'."
        )

    return base_url, f"{model_name}:{parameters}"


def deserialize_messages(messages) -> list[dict[str, typing.Any]]:
    # "id" rides along for context_window bookkeeping (which message a
    # summary covers up to); _map_history ignores it when building the
    # actual LLM payload.
    return [
        {
            "id": message.id,
            "role": str(message.role),
            "content": str(message.content),
            "image": str(message.image) if message.image else "",
        }
        for message in messages
    ]


def get_model_context_length(model_name: str, parameters: str) -> int | None:
    """The model's trained maximum context, read from its own Ollama
    container — None when that can't be determined (container down, older
    Ollama), in which case the caller just doesn't cap."""
    try:
        url_info = get_ollama_url(model_name, parameters)
    except ModelUnavailableError:
        return None
    if url_info is None:
        return None

    info = ollama_api.show_model(*url_info)
    return ollama_api.context_length(info) if info else None


def get_gpu_share(model_name: str, parameters: str) -> float | None:
    """Fraction (0–1) of the loaded model that sits in GPU memory, per
    /api/ps. Below 1 means part of it spilled to the CPU and every token is
    slower — usually because a bigger memory (num_ctx) grew the KV cache
    past what the GPU holds. None when there's no GPU in play at all, so
    CPU-only machines are never nagged about it."""
    manager = ContainerManager()
    container = manager.get_container(ContainerManager.container_name(model_name, parameters))
    if container is None or container.status != "running":
        return None
    if container.attrs.get("Config", {}).get("Labels", {}).get(LABEL_GPU) != "true":
        return None

    base_url = manager.base_url_for(container)
    running = ollama_api.running_models(base_url) if base_url else None
    entry = next((m for m in running or [] if m["name"] == f"{model_name}:{parameters}"), None)
    if entry is None or not entry["size"]:
        return None

    return round(entry["size_vram"] / entry["size"], 2)


async def asummarize_conversation(
    model: models.AIModel,
    parameters: str,
    previous_summary: str,
    messages: list[dict[str, typing.Any]],
    num_ctx: int,
) -> str:
    url_info = await sync_to_async(get_ollama_url, thread_sensitive=False)(model.model, parameters)
    if url_info is None:
        raise ModelUnavailableError(
            "Model container isn't running or is still pulling. "
            "Check the Models page and try again once it shows 'Running'."
        )

    base_url, full_model_string = url_info
    prompt = context_window.build_summary_prompt(previous_summary, messages, num_ctx)
    llm = _build_chat_ollama(base_url, full_model_string, num_ctx=num_ctx, temperature=0.2)

    try:
        response = await llm.ainvoke([{"role": "user", "content": prompt}])
    except httpx.HTTPError as e:
        raise ModelUnavailableError(
            "Could not reach the model — it may still be starting up. Try again in a moment."
        ) from e
    except ollama.ResponseError as e:
        raise ModelUnavailableError(f"Ollama rejected the request: {e}") from e

    return response.text.strip()


def save_memory_summary(chat_history: models.ChatHistory, summary: str, until_message_id: int) -> None:
    chat_history.memory_summary = summary
    chat_history.memory_summary_until_id = until_message_id
    chat_history.save(update_fields=["memory_summary", "memory_summary_until"])


def save_seed_summary(chat_history: models.ChatHistory, summary: str) -> None:
    chat_history.seed_summary = summary
    chat_history.save(update_fields=["seed_summary"])


def get_seed_source_history(chat_history: models.ChatHistory) -> list[dict[str, typing.Any]]:
    if chat_history.seed_source_id is None:
        return []
    source = models.ChatHistory.objects.filter(id=chat_history.seed_source_id).first()
    return deserialize_messages(get_messages_for_chat(source)) if source else []


MIN_SEARCH_QUERY_LENGTH = 2
SEARCH_RESULT_LIMIT = 50
SEARCH_SNIPPET_CONTEXT_CHARS = 60


def _build_search_snippet(content: str, query: str) -> str:
    lower_content = content.lower()
    match_index = lower_content.find(query.lower())

    if match_index == -1:
        # Shouldn't happen given the icontains filter that gets us here, but
        # fail into a plain truncation rather than an IndexError.
        return content[: SEARCH_SNIPPET_CONTEXT_CHARS * 2]

    start = max(0, match_index - SEARCH_SNIPPET_CONTEXT_CHARS)
    end = min(len(content), match_index + len(query) + SEARCH_SNIPPET_CONTEXT_CHARS)

    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(content) else ""

    return f"{prefix}{content[start:end]}{suffix}"


def search_messages(ai_model: models.AIModel, query: str) -> list[dict[str, str]]:
    """Searches every message across every chat for this model — including
    inactive branches, not just each chat's currently active path, since
    finding something said down an old branch is still useful. Capped and
    ordered newest-first rather than paginated: this is a quick-jump search
    box, not a full chat browser.
    """
    query = query.strip()

    if len(query) < MIN_SEARCH_QUERY_LENGTH:
        return []

    matches = (
        models.ChatMessage.objects.filter(
            chat__ai_model=ai_model, content__icontains=query
        )
        .select_related("chat")
        .order_by("-created_at")[:SEARCH_RESULT_LIMIT]
    )

    return [
        {
            "chat_id": str(message.chat_id),
            "chat_title": message.chat.title,
            "role": message.role,
            "snippet": _build_search_snippet(message.content, query),
        }
        for message in matches
    ]


def get_chats(
    model: models.AIModel, sort: bool = False
) -> BaseManager[models.ChatHistory]:
    chats: BaseManager[models.ChatHistory] = models.ChatHistory.objects.filter(
        ai_model=model
    )

    if sort:
        chats = chats.order_by("-last_update_time")

    return chats


def get_chat_history(
    model: models.AIModel | None, chat_id: str
) -> models.ChatHistory | None:
    if model is None:
        return None

    return get_chats(model).filter(id=chat_id).first()


def get_generation_params(chat_history: models.ChatHistory) -> GenerationParams:
    # Plain columns on the already-fetched row — no extra query, unlike
    # get_persona_system_prompt's `.persona` access.
    return GenerationParams(
        temperature=chat_history.temperature,
        num_ctx=chat_history.num_ctx,
        top_p=chat_history.top_p,
        seed=chat_history.seed,
    )


def get_persona_system_prompt(chat_history: models.ChatHistory) -> str | None:
    # `persona_id` (the raw FK column) is always loaded with the row; only
    # `.persona` (the related object) triggers a query, and only worth
    # paying when a persona is actually set.
    if not chat_history.persona_id:
        return None

    return chat_history.persona.system_prompt


def get_messages_for_chat(chat_history: models.ChatHistory) -> list[models.ChatMessage]:
    """Root-to-leaf path of the currently active branch — everything the
    user sees and everything sent to the model as history. Walking up via
    `parent` from `active_leaf`, rather than a flat created_at-ordered scan,
    is what makes regenerate/edit-resend/branch-switch safe: an old branch's
    messages are never deleted, just left off this path.
    """
    return _load_active_path(chat_history)[0]


def _load_active_path(
    chat_history: models.ChatHistory,
) -> tuple[list[models.ChatMessage], dict[int | None, list[int]]]:
    """(active path, parent id -> child ids oldest first) in two queries,
    however long the chat — following `.parent` one hop at a time used to
    cost a query per message, plus one more per message for its siblings.
    The tree pass reads only ids (no content, no base64 images); full rows
    are fetched just for the messages actually on the path.
    """
    parent_of: dict[int, int | None] = {}
    children_of: dict[int | None, list[int]] = {}

    tree_rows = (
        models.ChatMessage.objects.filter(chat=chat_history)
        .order_by("created_at", "id")
        .values_list("id", "parent_id")
    )
    for message_id, parent_id in tree_rows:
        parent_of[message_id] = parent_id
        children_of.setdefault(parent_id, []).append(message_id)

    path_ids: list[int] = []
    node_id = chat_history.active_leaf_id
    while node_id is not None and node_id in parent_of:
        path_ids.append(node_id)
        node_id = parent_of[node_id]
    path_ids.reverse()

    rows = models.ChatMessage.objects.in_bulk(path_ids)

    return [rows[i] for i in path_ids], children_of


def get_branch_siblings(message: models.ChatMessage) -> list[models.ChatMessage]:
    """All messages competing for the same slot as `message` — same parent,
    same chat — oldest first. What a branch switcher cycles through."""
    return list(
        models.ChatMessage.objects.filter(
            chat=message.chat_id, parent=message.parent_id
        ).order_by("created_at", "id")
    )


def get_branch_leaf(message: models.ChatMessage) -> models.ChatMessage:
    """Descends to the tip of whatever was most recently built on top of
    `message` — where switching to its branch should land."""
    node = message
    child = node.children.order_by("-created_at", "-id").first()

    while child is not None:
        node = child
        child = node.children.order_by("-created_at", "-id").first()

    return node


def serialize_active_path_for_display(chat_history: models.ChatHistory) -> list[dict[str, str | int]]:
    """Like deserialize_messages, but for the REST payloads the frontend
    renders from — adds branch position so a message with siblings can show
    a "◀ 2/3 ▶" switcher. Not used for LLM history, which has no use for it.
    """
    path, children_of = _load_active_path(chat_history)
    result: list[dict[str, str | int]] = []

    for message in path:
        sibling_ids = children_of.get(message.parent_id, [message.id])
        result.append(
            {
                "role": str(message.role),
                "content": str(message.content),
                "image": str(message.image) if message.image else "",
                "sibling_count": len(sibling_ids),
                "sibling_index": sibling_ids.index(message.id),
            }
        )

    return result


def add_messages_to_history(
    chat_history: models.ChatHistory,
    user_content: str,
    assistant_content: str,
    image: str = "",
) -> None:
    with transaction.atomic():
        parent = chat_history.active_leaf
        user_message = models.ChatMessage.objects.create(
            chat=chat_history,
            parent=parent,
            role="user",
            content=user_content,
            image=image,
        )
        assistant_message = models.ChatMessage.objects.create(
            chat=chat_history,
            parent=user_message,
            role="assistant",
            content=assistant_content,
        )
        chat_history.active_leaf = assistant_message
        chat_history.save(update_fields=["active_leaf", "last_update_time"])


def commit_new_assistant_branch(
    chat_history: models.ChatHistory,
    parent_user_message: models.ChatMessage,
    assistant_content: str,
) -> None:
    """Regenerate: adds a new sibling reply under the same user message
    instead of touching the old one. Only ever called once a new response
    has actually been generated — a failed or cancelled regenerate simply
    never calls this, so the old branch is untouched and still active_leaf.
    """
    with transaction.atomic():
        new_assistant = models.ChatMessage.objects.create(
            chat=chat_history,
            parent=parent_user_message,
            role="assistant",
            content=assistant_content,
        )
        chat_history.active_leaf = new_assistant
        chat_history.save(update_fields=["active_leaf", "last_update_time"])


def commit_new_exchange_branch(
    chat_history: models.ChatHistory,
    parent_message: models.ChatMessage | None,
    user_content: str,
    assistant_content: str,
    image: str = "",
) -> None:
    """Edit-resend: branches a new user+assistant pair under the same parent
    the edited message had, leaving the old branch — and everything after
    it — exactly where it was. Only called once a new response actually
    exists, for the same reason as commit_new_assistant_branch.
    """
    with transaction.atomic():
        new_user = models.ChatMessage.objects.create(
            chat=chat_history,
            parent=parent_message,
            role="user",
            content=user_content,
            image=image,
        )
        new_assistant = models.ChatMessage.objects.create(
            chat=chat_history,
            parent=new_user,
            role="assistant",
            content=assistant_content,
        )
        chat_history.active_leaf = new_assistant
        chat_history.save(update_fields=["active_leaf", "last_update_time"])


def prepare_edit_resend(
    chat_history: models.ChatHistory,
    index: int,
) -> tuple[models.ChatMessage | None, list[dict[str, str]]] | None:
    """Returns (parent to branch the new exchange from, history before it)
    for the user message at `index` in the active path — an index rather
    than a message id because the frontend only ever tracks messages by
    their position in the array it already has loaded, and that array's
    order matches the active path by construction (see ChatCard.vue). The
    edited message's own old content/image aren't returned: the caller
    already has the new ones from the client, which is the whole point of
    an edit. None if the index is out of range or doesn't land on a user
    message.
    """
    path = get_messages_for_chat(chat_history)

    if index < 0 or index >= len(path) or path[index].role != "user":
        return None

    history = deserialize_messages(path[:index])

    return path[index].parent, history


def prepare_regenerate(
    chat_history: models.ChatHistory,
) -> tuple[models.ChatMessage, str, str, list[dict[str, str]]] | None:
    """Returns (user message to branch a new reply from, its content, its
    image, history before that turn) for the active path's most recent
    exchange, or None if it doesn't end in one (nothing to regenerate — e.g.
    empty chat, or the last turn is still an un-answered user message).
    """
    path = get_messages_for_chat(chat_history)

    if len(path) < 2 or path[-1].role != "assistant" or path[-2].role != "user":
        return None

    last_user = path[-2]
    history = deserialize_messages(path[:-2])

    return (
        last_user,
        str(last_user.content),
        str(last_user.image) if last_user.image else "",
        history,
    )


def switch_branch(
    chat_history: models.ChatHistory,
    index: int,
    sibling_index: int,
) -> list[dict[str, str | int]] | None:
    """Moves the active path's branch point at `index` to a different
    sibling, jumping active_leaf to the tip of whatever was built on that
    sibling. Returns the new active path for display, or None if `index` or
    `sibling_index` don't resolve to a real message.
    """
    path = get_messages_for_chat(chat_history)

    if index < 0 or index >= len(path):
        return None

    siblings = get_branch_siblings(path[index])

    if sibling_index < 0 or sibling_index >= len(siblings):
        return None

    chat_history.active_leaf = get_branch_leaf(siblings[sibling_index])
    chat_history.save(update_fields=["active_leaf"])

    return serialize_active_path_for_display(chat_history)


def get_ai_model(value: str) -> models.AIModel | None:
    return models.AIModel.objects.filter(model=value).first()
