"""Pure-logic helpers in functions.py that don't need a channel layer,
Ollama, or Docker — see test_consumers.py for the end-to-end wiring of these
into the actual generation call.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessageChunk

from django_app import functions, models

pytestmark = pytest.mark.django_db


def test_chat_ollama_kwargs_omits_unset_params():
    assert functions._chat_ollama_kwargs(None) == {}
    assert functions._chat_ollama_kwargs({}) == {}
    assert functions._chat_ollama_kwargs(
        {"temperature": None, "num_ctx": None, "top_p": None, "seed": None}
    ) == {}


def test_chat_ollama_kwargs_passes_through_only_set_params():
    assert functions._chat_ollama_kwargs(
        {"temperature": 0.7, "num_ctx": None, "top_p": 0.9, "seed": None}
    ) == {"temperature": 0.7, "top_p": 0.9}


def test_get_generation_params_reads_chat_history_fields():
    ai_model = models.AIModel.objects.create(name="llama3.1", model="llama3.1-gp-test", index=1)
    chat = models.ChatHistory.objects.create(
        ai_model=ai_model, title="t", temperature=0.5, num_ctx=4096, top_p=None, seed=1
    )

    assert functions.get_generation_params(chat) == {
        "temperature": 0.5,
        "num_ctx": 4096,
        "top_p": None,
        "seed": 1,
    }


def test_record_generation_stats_computes_tokens_per_second():
    stats: functions.GenerationStats = {}

    functions._record_generation_stats(
        stats,
        {"prompt_eval_count": 10, "eval_count": 50, "eval_duration": 2_000_000_000},
    )

    assert stats == {"prompt_tokens": 10, "completion_tokens": 50, "tokens_per_second": 25.0}


def test_record_generation_stats_ignores_none_out_param():
    # Intermediate (done=False) streamed chunks call this too — must be a
    # harmless no-op when the caller didn't ask for stats.
    functions._record_generation_stats(None, {"prompt_eval_count": 10, "eval_count": 50})


def test_record_generation_stats_skips_tokens_per_second_without_duration():
    stats: functions.GenerationStats = {}

    functions._record_generation_stats(stats, {"prompt_eval_count": 10, "eval_count": 50})

    assert stats == {"prompt_tokens": 10, "completion_tokens": 50}
    assert "tokens_per_second" not in stats


def test_record_generation_stats_from_intermediate_chunk_is_a_noop():
    # Non-final streamed chunks carry no eval_count/prompt_eval_count at
    # all — must not fabricate zeros.
    stats: functions.GenerationStats = {}

    functions._record_generation_stats(stats, {"logprobs": None})

    assert stats == {}


def test_build_search_snippet_centers_the_match_with_ellipses():
    content = "a" * 100 + "NEEDLE" + "b" * 100

    snippet = functions._build_search_snippet(content, "needle")

    assert "…" in snippet
    assert snippet.count("…") == 2
    assert "needle" in snippet.lower()
    assert len(snippet) < len(content)


def test_build_search_snippet_no_leading_ellipsis_when_match_is_near_the_start():
    content = "NEEDLE" + "b" * 200

    snippet = functions._build_search_snippet(content, "needle")

    assert snippet.startswith("NEEDLE")
    assert not snippet.startswith("…")
    assert snippet.endswith("…")


def test_search_messages_requires_minimum_query_length():
    ai_model = models.AIModel.objects.create(name="llama3.1", model="llama3.1-search-test", index=1)
    chat = models.ChatHistory.objects.create(ai_model=ai_model, title="t")
    models.ChatMessage.objects.create(chat=chat, role="user", content="a")

    assert functions.search_messages(ai_model, "a") == []
    assert functions.search_messages(ai_model, "") == []
    assert functions.search_messages(ai_model, "  ") == []


def test_search_messages_finds_matches_case_insensitively_across_chats():
    ai_model = models.AIModel.objects.create(name="llama3.1", model="llama3.1-search-test2", index=1)
    chat1 = models.ChatHistory.objects.create(ai_model=ai_model, title="Trip planning")
    chat2 = models.ChatHistory.objects.create(ai_model=ai_model, title="Recipe ideas")
    models.ChatMessage.objects.create(chat=chat1, role="user", content="Where should I go in Portugal?")
    models.ChatMessage.objects.create(chat=chat2, role="assistant", content="Try a PORTUGAL-style tart.")
    models.ChatMessage.objects.create(chat=chat2, role="user", content="Something unrelated")

    results = functions.search_messages(ai_model, "portugal")

    assert len(results) == 2
    chat_titles = {r["chat_title"] for r in results}
    assert chat_titles == {"Trip planning", "Recipe ideas"}
    for result in results:
        assert "portugal" in result["snippet"].lower()


def test_search_messages_scoped_to_the_given_model_only():
    model_a = models.AIModel.objects.create(name="a", model="llama3.1-search-scope-a", index=1)
    model_b = models.AIModel.objects.create(name="b", model="llama3.1-search-scope-b", index=1)
    chat_a = models.ChatHistory.objects.create(ai_model=model_a, title="Chat A")
    chat_b = models.ChatHistory.objects.create(ai_model=model_b, title="Chat B")
    models.ChatMessage.objects.create(chat=chat_a, role="user", content="shared keyword here")
    models.ChatMessage.objects.create(chat=chat_b, role="user", content="shared keyword here too")

    results = functions.search_messages(model_a, "shared keyword")

    assert len(results) == 1
    assert results[0]["chat_title"] == "Chat A"


def test_search_messages_includes_inactive_branches():
    ai_model = models.AIModel.objects.create(name="llama3.1", model="llama3.1-search-branch", index=1)
    chat = models.ChatHistory.objects.create(ai_model=ai_model, title="Branchy chat")
    root = models.ChatMessage.objects.create(chat=chat, role="user", content="Hi")
    models.ChatMessage.objects.create(chat=chat, parent=root, role="assistant", content="original findme reply")
    active = models.ChatMessage.objects.create(chat=chat, parent=root, role="assistant", content="regenerated reply")
    chat.active_leaf = active
    chat.save(update_fields=["active_leaf"])

    results = functions.search_messages(ai_model, "findme")

    assert len(results) == 1
    assert "findme" in results[0]["snippet"].lower()


# --- astream_tool_response ---------------------------------------------------


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

    def bind_tools(self, tools):
        return self

    async def astream(self, messages):
        self.stream_calls += 1
        for chunk in self._turns.pop(0):
            yield chunk


@pytest.fixture
def tool_test_model() -> models.AIModel:
    models.AIModel.objects.filter(model="llama3.1-tool-fn-test").delete()
    return models.AIModel.objects.create(
        name="t", model="llama3.1-tool-fn-test", can_use_tools=True, index=1
    )


@pytest.fixture
def mcp_test_server() -> models.MCPServer:
    models.MCPServer.objects.filter(name="Local").delete()
    return models.MCPServer.objects.create(name="Local", transport=models.MCPServer.TRANSPORT_STDIO, command="x")


async def test_astream_tool_response_executes_a_real_tool_then_streams_final_answer(tool_test_model, monkeypatch):
    ai_model = tool_test_model
    monkeypatch.setattr(
        functions, "get_ollama_url", lambda model_name, parameters: ("http://fake", f"{model_name}:{parameters}")
    )

    fake_llm = _FakeToolLLM([
        _tool_turn({"name": "calculator", "args": {"expression": "2+2"}, "id": "call_1"}),
        # Second turn, now with the tool result in context — answers.
        _text_turn("The ", "answer is 4."),
    ])
    monkeypatch.setattr(functions, "ChatOllama", lambda **kwargs: fake_llm)

    chunks = [chunk async for chunk in functions.astream_tool_response(ai_model, "8b", "what is 2+2", "", [])]

    # calculator is the real built-in tool, not mocked — this proves the
    # loop actually calls BUILTIN_TOOLS_BY_NAME and feeds a real result back.
    assert chunks[0] == {"name": "calculator", "args": {"expression": "2+2"}, "result": "4"}
    assert chunks[1:] == ["The ", "answer is 4."]


async def test_astream_tool_response_unknown_tool_name_reports_error_result(tool_test_model, monkeypatch):
    ai_model = tool_test_model
    monkeypatch.setattr(
        functions, "get_ollama_url", lambda model_name, parameters: ("http://fake", f"{model_name}:{parameters}")
    )

    fake_llm = _FakeToolLLM([
        _tool_turn({"name": "nonexistent", "args": {}, "id": "call_1"}),
        _text_turn("ok"),
    ])
    monkeypatch.setattr(functions, "ChatOllama", lambda **kwargs: fake_llm)

    chunks = [chunk async for chunk in functions.astream_tool_response(ai_model, "8b", "hi", "", [])]

    assert chunks[0]["name"] == "nonexistent"
    assert "Error" in chunks[0]["result"]


async def test_astream_tool_response_no_tool_calls_streams_immediately(tool_test_model, monkeypatch):
    ai_model = tool_test_model
    monkeypatch.setattr(
        functions, "get_ollama_url", lambda model_name, parameters: ("http://fake", f"{model_name}:{parameters}")
    )

    fake_llm = _FakeToolLLM([_text_turn("Hi ", "there.")])
    monkeypatch.setattr(functions, "ChatOllama", lambda **kwargs: fake_llm)

    chunks = [chunk async for chunk in functions.astream_tool_response(ai_model, "8b", "hi", "", [])]

    assert chunks == ["Hi ", "there."]
    # The answer is streamed from the one and only generation — never
    # decided in one call and regenerated in a second.
    assert fake_llm.stream_calls == 1


async def test_astream_tool_response_gives_up_after_max_iterations(tool_test_model, monkeypatch):
    ai_model = tool_test_model
    monkeypatch.setattr(
        functions, "get_ollama_url", lambda model_name, parameters: ("http://fake", f"{model_name}:{parameters}")
    )

    # Always asks for another tool call — never terminates on its own.
    fake_llm = _FakeToolLLM([
        _tool_turn({"name": "current_datetime", "args": {}, "id": f"call_{i}"})
        for i in range(functions.MAX_TOOL_ITERATIONS)
    ])
    monkeypatch.setattr(functions, "ChatOllama", lambda **kwargs: fake_llm)

    chunks = [chunk async for chunk in functions.astream_tool_response(ai_model, "8b", "hi", "", [])]

    tool_call_chunks = [c for c in chunks if isinstance(c, dict)]
    text_chunks = [c for c in chunks if isinstance(c, str)]
    assert len(tool_call_chunks) == functions.MAX_TOOL_ITERATIONS
    assert len(text_chunks) == 1
    assert "wasn't able to finish" in text_chunks[0]


async def test_astream_tool_response_routes_mcp_tool_calls(tool_test_model, mcp_test_server, monkeypatch):
    ai_model = tool_test_model
    mcp_server = mcp_test_server
    monkeypatch.setattr(
        functions, "get_ollama_url", lambda model_name, parameters: ("http://fake", f"{model_name}:{parameters}")
    )

    async def _fake_list_tools(server):
        return [{"name": "search", "description": "Search", "input_schema": {"type": "object"}}]

    async def _fake_call_tool(server, tool_name, arguments):
        assert server is mcp_server
        assert tool_name == "search"
        return f"results for {arguments['q']}"

    monkeypatch.setattr(functions.mcp_client, "list_server_tools", _fake_list_tools)
    monkeypatch.setattr(functions.mcp_client, "call_server_tool", _fake_call_tool)

    prefixed_name = "mcp_Local_search"
    fake_llm = _FakeToolLLM([
        _tool_turn({"name": prefixed_name, "args": {"q": "life"}, "id": "call_1"}),
        _text_turn("done"),
    ])
    monkeypatch.setattr(functions, "ChatOllama", lambda **kwargs: fake_llm)

    chunks = [
        chunk async for chunk in functions.astream_tool_response(
            ai_model, "8b", "search for life", "", [], mcp_servers=[mcp_server],
        )
    ]

    assert chunks[0] == {"name": prefixed_name, "args": {"q": "life"}, "result": "results for life"}
    assert chunks[1:] == ["done"]


async def test_astream_tool_response_mcp_tool_call_failure_reports_error_result(
    tool_test_model, mcp_test_server, monkeypatch,
):
    ai_model = tool_test_model
    mcp_server = mcp_test_server
    monkeypatch.setattr(
        functions, "get_ollama_url", lambda model_name, parameters: ("http://fake", f"{model_name}:{parameters}")
    )

    async def _fake_list_tools(server):
        return [{"name": "search", "description": "Search", "input_schema": {"type": "object"}}]

    async def _raise_call_tool(server, tool_name, arguments):
        raise ConnectionError("server went away")

    monkeypatch.setattr(functions.mcp_client, "list_server_tools", _fake_list_tools)
    monkeypatch.setattr(functions.mcp_client, "call_server_tool", _raise_call_tool)

    prefixed_name = "mcp_Local_search"
    fake_llm = _FakeToolLLM([
        _tool_turn({"name": prefixed_name, "args": {}, "id": "call_1"}),
        _text_turn("done"),
    ])
    monkeypatch.setattr(functions, "ChatOllama", lambda **kwargs: fake_llm)

    chunks = [
        chunk async for chunk in functions.astream_tool_response(
            ai_model, "8b", "hi", "", [], mcp_servers=[mcp_server],
        )
    ]

    assert "server went away" in chunks[0]["result"]


async def test_astream_tool_response_streams_preamble_text_before_a_tool_call(tool_test_model, monkeypatch):
    monkeypatch.setattr(
        functions, "get_ollama_url", lambda model_name, parameters: ("http://fake", f"{model_name}:{parameters}")
    )

    fake_llm = _FakeToolLLM([
        _text_turn("Let me check. ") + _tool_turn({"name": "calculator", "args": {"expression": "3*3"}, "id": "c1"}),
        _text_turn("It's 9."),
    ])
    monkeypatch.setattr(functions, "ChatOllama", lambda **kwargs: fake_llm)

    chunks = [chunk async for chunk in functions.astream_tool_response(tool_test_model, "8b", "3*3?", "", [])]

    assert chunks == [
        "Let me check. ",
        {"name": "calculator", "args": {"expression": "3*3"}, "result": "9"},
        "It's 9.",
    ]
    assert fake_llm.stream_calls == 2


def test_gpu_share_is_none_for_cpu_only_containers(monkeypatch):
    from unittest.mock import MagicMock

    from container.ContainerManager import ContainerManager

    container = MagicMock(status="running", attrs={"Config": {"Labels": {"reichat.gpu": "false"}}})
    monkeypatch.setattr(ContainerManager, "_client", MagicMock())
    monkeypatch.setattr(ContainerManager, "get_container", lambda self, name: container)

    assert functions.get_gpu_share("llama3.1", "8b") is None


def test_gpu_share_reads_the_loaded_model_from_ps(monkeypatch):
    from unittest.mock import MagicMock

    from container.ContainerManager import ContainerManager

    container = MagicMock(status="running", attrs={"Config": {"Labels": {"reichat.gpu": "true"}}})
    monkeypatch.setattr(ContainerManager, "_client", MagicMock())
    monkeypatch.setattr(ContainerManager, "get_container", lambda self, name: container)
    monkeypatch.setattr(ContainerManager, "base_url_for", lambda self, c: "http://o")
    monkeypatch.setattr(
        functions.ollama_api, "running_models",
        lambda base_url: [{"name": "llama3.1:8b", "size": 1000, "size_vram": 750, "context_length": 8192}],
    )

    assert functions.get_gpu_share("llama3.1", "8b") == 0.75
