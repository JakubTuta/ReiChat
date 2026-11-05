"""The model-call helpers in llm/generation.py, with ChatOllama replaced by a
fake that plays back scripted chunks — see tests/chat/test_socket.py for the
end-to-end wiring of these into a chat."""

import json
from unittest.mock import AsyncMock

import httpx
import ollama
import pytest
from langchain_core.messages import AIMessageChunk

from app.db import models
from app.llm import generation, runtime
from app.llm.runtime import ModelUnavailableError

MODEL = "llama3.1-tool-fn-test"


@pytest.fixture(autouse=True)
def model_is_running(monkeypatch):
    monkeypatch.setattr(
        runtime,
        "get_ollama_url",
        AsyncMock(side_effect=lambda name, params: ("http://fake", f"{name}:{params}")),
    )


@pytest.fixture(autouse=True)
def web_search_off(monkeypatch):
    # The real check asks Docker whether the SearXNG container runs.
    monkeypatch.setattr(generation.web_search_tool, "available", AsyncMock(return_value=False))


def test_chat_ollama_kwargs_passes_through_only_set_params():
    assert generation._chat_ollama_kwargs(
        {"temperature": 0.7, "num_ctx": None, "top_p": 0.9, "seed": None}
    ) == {
        "temperature": 0.7,
        "top_p": 0.9,
    }


def test_record_generation_stats_computes_tokens_per_second():
    stats: generation.GenerationStats = {}

    generation._record_generation_stats(
        stats, {"prompt_eval_count": 10, "eval_count": 50, "eval_duration": 2_000_000_000}
    )

    assert stats == {"prompt_tokens": 10, "completion_tokens": 50, "tokens_per_second": 25.0}


def _text_turn(*texts: str) -> list[AIMessageChunk]:
    return [AIMessageChunk(content=text) for text in texts]


def _tool_turn(*tool_calls: dict) -> list[AIMessageChunk]:
    return [
        AIMessageChunk(
            content="",
            tool_call_chunks=[
                {"name": call["name"], "args": json.dumps(call["args"]), "id": call["id"], "index": i}
                for i, call in enumerate(tool_calls)
            ],
        )
    ]


class _FakeToolLLM:
    """Stands in for ChatOllama — bind_tools returns self, and each astream()
    call plays back the next turn's chunks. Counts calls so tests can assert
    no turn is ever generated twice.
    """

    def __init__(self, turns: list[list[AIMessageChunk]]):
        self._turns = list(turns)
        self.stream_calls = 0
        self.seen_messages: list[list] = []

    def bind_tools(self, tools):
        return self

    async def astream(self, messages):
        self.stream_calls += 1
        self.seen_messages.append(list(messages))
        for chunk in self._turns.pop(0):
            yield chunk


@pytest.fixture
def fake_llm(monkeypatch):
    def install(turns):
        llm = _FakeToolLLM(turns)
        monkeypatch.setattr(generation, "ChatOllama", lambda **kwargs: llm)
        return llm

    return install


async def collect(generator):
    return [chunk async for chunk in generator]


async def test_astream_tool_response_executes_a_real_tool_then_streams_final_answer(fake_llm):
    fake_llm(
        [
            _tool_turn({"name": "calculator", "args": {"expression": "2+2"}, "id": "call_1"}),
            # Second turn, now with the tool result in context — answers.
            _text_turn("The ", "answer is 4."),
        ]
    )

    chunks = await collect(generation.astream_tool_response(MODEL, "8b", "what is 2+2", "", []))

    # calculator is the real built-in tool, not mocked — this proves the
    # loop actually calls BUILTIN_TOOLS_BY_NAME and feeds a real result back.
    assert chunks[0] == {"name": "calculator", "args": {"expression": "2+2"}, "result": "4"}
    assert chunks[1:] == ["The ", "answer is 4."]


async def test_astream_tool_response_gives_up_after_max_iterations(fake_llm):
    # Always asks for another tool call — never terminates on its own.
    fake_llm(
        [
            _tool_turn({"name": "current_datetime", "args": {}, "id": f"call_{i}"})
            for i in range(generation.MAX_TOOL_ITERATIONS)
        ]
    )

    chunks = await collect(generation.astream_tool_response(MODEL, "8b", "hi", "", []))

    tool_call_chunks = [c for c in chunks if isinstance(c, dict)]
    text_chunks = [c for c in chunks if isinstance(c, str)]
    assert len(tool_call_chunks) == generation.MAX_TOOL_ITERATIONS
    assert len(text_chunks) == 1
    assert "wasn't able to finish" in text_chunks[0]


def _mcp_server(name="Local", **fields) -> models.MCPServer:
    return models.MCPServer(name=name, transport=models.MCPServer.TRANSPORT_STDIO, command="x", **fields)


@pytest.fixture
def mcp(monkeypatch):
    calls = []

    async def list_tools(server):
        return [{"name": "search", "description": "Search", "input_schema": {"type": "object"}}]

    async def call_tool(server, tool_name, arguments):
        calls.append((server, tool_name, arguments))
        return f"results for {arguments.get('q')}"

    monkeypatch.setattr(generation.mcp_client, "list_server_tools", list_tools)
    monkeypatch.setattr(generation.mcp_client, "call_server_tool", call_tool)
    return calls


async def test_astream_tool_response_routes_mcp_tool_calls(fake_llm, mcp):
    server = _mcp_server(auto_approve=True)
    fake_llm(
        [_tool_turn({"name": "mcp_Local_search", "args": {"q": "life"}, "id": "call_1"}), _text_turn("done")]
    )

    chunks = await collect(
        generation.astream_tool_response(MODEL, "8b", "search for life", "", [], mcp_servers=[server])
    )

    assert chunks[0] == {"name": "mcp_Local_search", "args": {"q": "life"}, "result": "results for life"}
    assert chunks[1:] == ["done"]
    assert mcp == [(server, "search", {"q": "life"})]


async def _run_one_mcp_call(fake_llm, server, approve, mcp):
    fake_llm(
        [
            _tool_turn({"name": f"mcp_{server.name}_search", "args": {"q": "x"}, "id": "c1"}),
            _text_turn("done"),
        ]
    )
    return await collect(
        generation.astream_tool_response(MODEL, "8b", "hi", "", [], mcp_servers=[server], approve=approve)
    )


async def test_untrusted_mcp_tool_waits_for_the_users_approval(fake_llm, mcp):
    asked = []

    async def approve(request):
        asked.append(request)
        return True

    chunks = await _run_one_mcp_call(fake_llm, _mcp_server("Untrusted"), approve, mcp)

    assert asked == [{"name": "search", "server": "Untrusted", "args": {"q": "x"}}]
    assert [call[2] for call in mcp] == [{"q": "x"}]
    assert chunks[0]["result"] == "results for x"


async def test_denied_mcp_tool_never_runs(fake_llm, mcp):
    async def deny(request):
        return False

    chunks = await _run_one_mcp_call(fake_llm, _mcp_server("Untrusted"), deny, mcp)

    assert mcp == []
    assert chunks[0]["result"] == generation.TOOL_DECLINED_RESULT


async def test_untrusted_mcp_tool_without_anyone_to_ask_is_refused(fake_llm, mcp):
    chunks = await _run_one_mcp_call(fake_llm, _mcp_server("Untrusted"), None, mcp)

    assert mcp == []
    assert chunks[0]["result"] == generation.TOOL_DECLINED_RESULT


async def test_huge_tool_output_is_cut_down_before_reaching_the_model(fake_llm, monkeypatch):
    async def list_tools(server):
        return [{"name": "search", "description": "Search", "input_schema": {"type": "object"}}]

    async def call_tool(server, tool_name, arguments):
        return "x" * 50_000

    monkeypatch.setattr(generation.mcp_client, "list_server_tools", list_tools)
    monkeypatch.setattr(generation.mcp_client, "call_server_tool", call_tool)
    fake_llm([_tool_turn({"name": "mcp_Local_search", "args": {}, "id": "c1"}), _text_turn("done")])

    chunks = await collect(
        generation.astream_tool_response(
            MODEL, "8b", "hi", "", [], mcp_servers=[_mcp_server(auto_approve=True)]
        )
    )

    assert len(chunks[0]["result"]) < generation.MAX_TOOL_RESULT_CHARS + 100
    assert chunks[0]["result"].endswith("characters]")


async def test_web_search_is_offered_and_run_only_when_turned_on(fake_llm, monkeypatch):
    monkeypatch.setattr(generation.web_search_tool, "available", AsyncMock(return_value=True))
    bound: list = []

    class Tool:
        name = "web_search"

        @staticmethod
        async def ainvoke(args):
            return f"results for {args['query']}"

    monkeypatch.setattr(generation.web_search_tool, "web_search", Tool)
    llm = fake_llm(
        [_tool_turn({"name": "web_search", "args": {"query": "ollama news"}, "id": "c1"}), _text_turn("done")]
    )
    original_bind = llm.bind_tools

    def bind(tools):
        bound.extend(tools)
        return original_bind(tools)

    llm.bind_tools = bind

    chunks = await collect(generation.astream_tool_response(MODEL, "8b", "news?", "", []))

    assert Tool in bound
    assert chunks[0] == {
        "name": "web_search",
        "args": {"query": "ollama news"},
        "result": "results for ollama news",
    }


async def test_astream_bot_response_yields_thinking_separately(monkeypatch):
    built_with: dict = {}

    class ThinkingLLM:
        async def astream(self, messages):
            yield AIMessageChunk(content="", additional_kwargs={"reasoning_content": "Let me add. "})
            yield AIMessageChunk(content="5")

    def build(**kwargs):
        built_with.update(kwargs)
        return ThinkingLLM()

    monkeypatch.setattr(generation, "ChatOllama", build)

    chunks = await collect(generation.astream_bot_response(MODEL, "8b", "2+3?", "", [], reasoning=True))

    assert chunks == [generation.ThinkingDelta("Let me add. "), "5"]
    assert built_with["reasoning"] is True
    assert built_with["base_url"] == "http://fake"
    assert built_with["model"] == f"{MODEL}:8b"


async def test_the_history_system_prompt_and_message_are_sent_in_order(monkeypatch):
    sent = []

    class RecordingLLM:
        async def astream(self, messages):
            sent.extend(messages)
            yield AIMessageChunk(content="ok")

    monkeypatch.setattr(generation, "ChatOllama", lambda **kwargs: RecordingLLM())

    await collect(
        generation.astream_bot_response(
            MODEL,
            "8b",
            "now",
            "",
            [{"role": "user", "content": "before"}, {"role": "assistant", "content": "answer"}],
            system_prompt="Be brief.",
        )
    )

    assert sent == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "before"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "now"},
    ]


async def test_an_attached_image_is_sent_as_a_structured_content_part(monkeypatch):
    sent = []

    class RecordingLLM:
        async def astream(self, messages):
            sent.extend(messages)
            yield AIMessageChunk(content="ok")

    monkeypatch.setattr(generation, "ChatOllama", lambda **kwargs: RecordingLLM())

    await collect(
        generation.astream_bot_response(MODEL, "8b", "what is this?", "data:image/png;base64,AAA", [])
    )

    assert sent == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": "data:image/png;base64,AAA"},
            ],
        }
    ]


async def test_a_model_that_is_not_running_says_how_to_start_it(monkeypatch):
    monkeypatch.setattr(runtime, "get_ollama_url", AsyncMock(return_value=None))

    with pytest.raises(ModelUnavailableError, match="Check the Models page"):
        await collect(generation.astream_bot_response(MODEL, "8b", "hi", "", []))


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (httpx.ConnectError("refused"), "Could not reach the model"),
        (
            ollama.ResponseError("model requires more memory", 500),
            "Ollama rejected the request: model requires more memory",
        ),
    ],
)
async def test_a_failing_ollama_becomes_a_readable_error(monkeypatch, failure, message):
    class FailingLLM:
        def bind_tools(self, tools):
            return self

        async def astream(self, messages):
            raise failure
            yield  # pragma: no cover

        async def ainvoke(self, messages):
            raise failure

    monkeypatch.setattr(generation, "ChatOllama", lambda **kwargs: FailingLLM())

    for call in (
        generation.astream_bot_response(MODEL, "8b", "hi", "", []),
        generation.astream_tool_response(MODEL, "8b", "hi", "", []),
    ):
        with pytest.raises(ModelUnavailableError, match=message):
            await collect(call)
    with pytest.raises(ModelUnavailableError, match=message):
        await generation.astream_structured_bot_response(
            MODEL, "8b", "hi", "", [], [{"field": "a", "type": "string"}]
        )
    with pytest.raises(ModelUnavailableError, match=message):
        await generation.asummarize_conversation(MODEL, "8b", "", [], 4096)


# Structured output


class _JsonLLM:
    def __init__(self, reply: str, metadata=None):
        self.reply = reply
        self.metadata = metadata or {}
        self.sent: list = []

    async def ainvoke(self, messages):
        self.sent = list(messages)
        return AIMessageChunk(content=self.reply, response_metadata=self.metadata)


@pytest.fixture
def json_llm(monkeypatch):
    built_with: dict = {}

    def install(reply, metadata=None):
        llm = _JsonLLM(reply, metadata)

        def build(**kwargs):
            built_with.update(kwargs)
            return llm

        monkeypatch.setattr(generation, "ChatOllama", build)
        return llm, built_with

    return install


FIELDS = [{"field": "city", "type": "string"}, {"field": "population", "type": "number"}]


async def test_structured_output_constrains_decoding_with_the_schema(json_llm):
    _, built_with = json_llm('{"city": "Paris", "population": 2.1}')

    result = await generation.astream_structured_bot_response(
        MODEL, "8b", "Capital of France?", "", [], FIELDS
    )

    assert isinstance(built_with["format"], dict)
    assert set(built_with["format"]["properties"]) == {"city", "population"}
    assert built_with["format"]["required"] == ["city", "population"]
    assert '"city": "Paris"' in result


async def test_structured_output_that_is_not_json_is_none(json_llm):
    json_llm("I am sorry, I cannot do that.")

    assert await generation.astream_structured_bot_response(MODEL, "8b", "hi", "", [], FIELDS) is None


def test_structured_output_field_types():
    model = generation._build_structured_output_model(
        [
            {"field": "name", "type": "string", "description": "Who"},
            {"field": "age", "type": "number"},
            {"field": "ok", "type": "bool"},
            {"field": "tags", "type": "array", "arrayType": "string"},
            {"field": "scores", "type": "array", "arrayType": "number"},
            {"field": "odd", "type": "array", "arrayType": "mystery"},
            {"field": "when", "type": "date"},
            {"field": "unknown", "type": "mystery"},
        ]
    )

    properties = model.model_json_schema()["properties"]
    assert properties["name"]["type"] == "string"
    assert properties["name"]["description"] == "Who"
    assert properties["age"]["type"] == "number"
    assert properties["ok"]["type"] == "boolean"
    assert properties["tags"]["items"]["type"] == "string"
    assert properties["scores"]["items"]["type"] == "number"
    assert properties["odd"]["items"]["type"] == "string"
    assert properties["when"]["type"] == "string"
    assert properties["unknown"]["type"] == "string"


# Titles and summaries


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('"Planning a Trip to Rome."', "Planning a Trip to Rome"),
        ("Title: Sourdough basics\nSome extra line", "Sourdough basics"),
        ("x" * 100, "x" * 60),
        ("", ""),
        ("   \n  ", ""),
    ],
)
def test_clean_title(raw, expected):
    assert generation.clean_title(raw) == expected


async def test_a_title_is_generated_with_thinking_switched_off_for_models_that_think(monkeypatch):
    built_with: dict = {}

    class TitleLLM:
        async def ainvoke(self, messages):
            built_with["prompt"] = messages[0]["content"]
            return AIMessageChunk(content='"Trip to Rome."')

    def build(**kwargs):
        built_with.update(kwargs)
        return TitleLLM()

    monkeypatch.setattr(generation, "ChatOllama", build)

    title = await generation.agenerate_title(MODEL, "8b", "Plan a trip to Rome", "Sure, here is a plan", True)

    assert title == "Trip to Rome"
    assert built_with["reasoning"] is False
    assert built_with["num_predict"] == 24
    assert "User: Plan a trip to Rome" in built_with["prompt"]
    assert "Assistant: Sure, here is a plan" in built_with["prompt"]


async def test_a_summary_is_requested_with_the_transcript_and_returned_trimmed(monkeypatch):
    seen: dict = {}

    class SummaryLLM:
        async def ainvoke(self, messages):
            seen["prompt"] = messages[0]["content"]
            return AIMessageChunk(content="  They talked about Rome.  \n")

    def build(**kwargs):
        seen.update(kwargs)
        return SummaryLLM()

    monkeypatch.setattr(generation, "ChatOllama", build)

    summary = await generation.asummarize_conversation(
        MODEL, "8b", "Earlier: hello.", [{"role": "user", "content": "Tell me about Rome"}], 4096
    )

    assert summary == "They talked about Rome."
    assert "Earlier: hello." in seen["prompt"]
    assert "User: Tell me about Rome" in seen["prompt"]
    assert seen["num_ctx"] == 4096
    assert seen["temperature"] == 0.2
