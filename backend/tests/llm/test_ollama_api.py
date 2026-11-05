import httpx
import pytest
import respx

from app.llm import ollama_api

BASE = "http://o"


@pytest.fixture(autouse=True)
def _fresh_show_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ollama_api, "_show_cache", {})


def test_context_length_matches_any_architecture_prefix():
    assert ollama_api.context_length({"model_info": {"qwen2.context_length": 32768}}) == 32768
    assert ollama_api.context_length({"model_info": {"general.architecture": "llama"}}) is None
    assert ollama_api.context_length({}) is None


@respx.mock
async def test_show_model_is_cached_per_server_and_tag():
    route = respx.post(f"{BASE}/api/show").respond(
        200, json={"model_info": {"llama.context_length": 8192}, "capabilities": ["completion"]}
    )

    first = await ollama_api.show_model(BASE, "llama3.2:1b")
    second = await ollama_api.show_model(BASE, "llama3.2:1b")

    assert first is second
    assert route.call_count == 1
    assert ollama_api.capabilities(first) == ["completion"]


@respx.mock
async def test_show_model_asks_for_the_model_it_was_given():
    route = respx.post(f"{BASE}/api/show").respond(200, json={})

    await ollama_api.show_model(BASE, "llama3.2:1b")

    assert route.calls.last.request.read() == b'{"model":"llama3.2:1b"}'


@respx.mock
async def test_show_model_returns_none_when_unreachable():
    respx.post(f"{BASE}/api/show").mock(side_effect=httpx.ConnectError("refused"))

    assert await ollama_api.show_model(BASE, "llama3.2:1b") is None


@respx.mock
async def test_show_model_returns_none_on_an_error_status_and_does_not_cache_it():
    route = respx.post(f"{BASE}/api/show").respond(500)

    assert await ollama_api.show_model(BASE, "llama3.2:1b") is None
    route.respond(200, json={"capabilities": ["completion"]})
    assert await ollama_api.show_model(BASE, "llama3.2:1b") == {"capabilities": ["completion"]}


@respx.mock
async def test_running_models_normalizes_ps_output():
    payload = {"models": [{"name": "qwen2.5:0.5b", "size": 1000, "size_vram": 600, "context_length": 8192}]}
    respx.get(f"{BASE}/api/ps").respond(200, json=payload)

    assert await ollama_api.running_models(BASE) == [
        {"name": "qwen2.5:0.5b", "size": 1000, "size_vram": 600, "context_length": 8192, "expires_at": None}
    ]


@respx.mock
async def test_running_models_is_none_when_ollama_cannot_be_asked():
    respx.get(f"{BASE}/api/ps").mock(side_effect=httpx.ConnectError("refused"))

    assert await ollama_api.running_models(BASE) is None


@respx.mock
async def test_installed_model_sizes_maps_name_to_bytes():
    respx.get(f"{BASE}/api/tags").respond(
        200, json={"models": [{"name": "llama3.2:1b", "size": 1300}, {"model": "qwen2.5:0.5b", "size": 400}]}
    )

    assert await ollama_api.installed_model_sizes(BASE) == {"llama3.2:1b": 1300, "qwen2.5:0.5b": 400}


@respx.mock
async def test_installed_model_sizes_is_none_when_the_answer_is_not_the_expected_shape():
    respx.get(f"{BASE}/api/tags").respond(200, json=["not", "an", "object"])

    assert await ollama_api.installed_model_sizes(BASE) is None


@respx.mock
async def test_unload_model_sends_keep_alive_zero():
    route = respx.post(f"{BASE}/api/generate").respond(200, json={})

    assert await ollama_api.unload_model(BASE, "llama3.2:1b") is True
    assert route.calls.last.request.read() == b'{"model":"llama3.2:1b","keep_alive":0}'


@respx.mock
async def test_unload_model_reports_failure():
    respx.post(f"{BASE}/api/generate").respond(404)

    assert await ollama_api.unload_model(BASE, "llama3.2:1b") is False


def test_thinking_controls_reads_levels_and_default():
    info = {
        "capabilities": ["completion", "thinking"],
        "thinking": {"values": ["low", "medium", "high"], "default": "medium"},
    }

    assert ollama_api.thinking_controls(info) == {"values": ["low", "medium", "high"], "default": "medium"}


def test_thinking_controls_none_for_models_that_cannot_think():
    assert ollama_api.thinking_controls({"capabilities": ["completion"]}) is None
    assert (
        ollama_api.thinking_controls({"capabilities": ["thinking"], "thinking": {"values": [False]}}) is None
    )


def test_thinking_controls_assume_on_off_when_metadata_is_missing():
    assert ollama_api.thinking_controls({"capabilities": ["thinking"]}) == {
        "values": [False, True],
        "default": True,
    }
