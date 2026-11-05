import json

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
async def test_installed_model_sizes_maps_name_to_bytes():
    respx.get(f"{BASE}/api/tags").respond(
        200, json={"models": [{"name": "llama3.2:1b", "size": 1300}, {"model": "qwen2.5:0.5b", "size": 400}]}
    )

    assert await ollama_api.installed_model_sizes(BASE) == {"llama3.2:1b": 1300, "qwen2.5:0.5b": 400}


@respx.mock
async def test_unload_model_sends_keep_alive_zero():
    route = respx.post(f"{BASE}/api/generate").respond(200, json={})

    assert await ollama_api.unload_model(BASE, "llama3.2:1b") is True
    assert route.calls.last.request.read() == b'{"model":"llama3.2:1b","keep_alive":0}'


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


@respx.mock
async def test_version_reports_what_ollama_says():
    respx.get(f"{BASE}/api/version").respond(200, json={"version": "0.12.3"})

    assert await ollama_api.version(BASE) == "0.12.3"


@respx.mock
@pytest.mark.parametrize("failure", [httpx.ConnectError("refused"), httpx.Response(200, text="<html>")])
async def test_version_is_none_when_nothing_answers_like_ollama(failure):
    if isinstance(failure, Exception):
        respx.get(f"{BASE}/api/version").mock(side_effect=failure)
    else:
        respx.get(f"{BASE}/api/version").mock(return_value=failure)

    assert await ollama_api.version(BASE) is None


@respx.mock
async def test_wait_until_ready_returns_as_soon_as_ollama_answers():
    respx.get(f"{BASE}/api/version").mock(
        side_effect=[httpx.ConnectError("booting"), httpx.Response(503), httpx.Response(200, json={})]
    )

    assert await ollama_api.wait_until_ready(BASE, max_wait_seconds=5, poll_interval=0) is True


@respx.mock
async def test_wait_until_ready_gives_up_after_the_deadline():
    respx.get(f"{BASE}/api/version").mock(side_effect=httpx.ConnectError("never"))

    assert await ollama_api.wait_until_ready(BASE, max_wait_seconds=0.05, poll_interval=0.01) is False


@respx.mock
async def test_is_model_installed_reads_ollama_tags():
    respx.get(f"{BASE}/api/tags").respond(
        200, json={"models": [{"name": "llama3.1:8b", "model": "llama3.1:8b"}]}
    )

    assert await ollama_api.is_model_installed(BASE, "llama3.1", "8b") is True
    assert await ollama_api.is_model_installed(BASE, "llama3.1", "70b") is False


@respx.mock
async def test_delete_model_sends_the_model_name_in_the_body():
    route = respx.delete(f"{BASE}/api/delete").respond(200)

    assert await ollama_api.delete_model(BASE, "llama3.2", "3b") is True
    assert route.calls.last.request.read() == b'{"model":"llama3.2:3b"}'


def ndjson(*events: dict) -> bytes:
    return b"\n".join(json.dumps(event).encode() for event in events) + b"\n"


@respx.mock
async def test_pull_model_reports_failure_on_stream_error():
    respx.post(f"{BASE}/api/pull").respond(
        200,
        content=ndjson({"status": "pulling manifest"}, {"error": "pull model manifest: file does not exist"}),
    )

    success, error = await ollama_api.pull_model(BASE, "bogus-model", "8b")

    assert success is False
    assert error == "pull model manifest: file does not exist"


@respx.mock
async def test_pull_model_reports_success_and_final_progress():
    respx.post(f"{BASE}/api/pull").respond(
        200,
        content=ndjson(
            {"status": "pulling layer", "digest": "sha256:abc", "total": 100, "completed": 50},
            {"status": "pulling layer", "digest": "sha256:abc", "total": 100, "completed": 100},
            {"status": "success"},
        ),
    )
    calls: list[tuple[int, str]] = []

    success, error = await ollama_api.pull_model(
        BASE, "llama3.1", "8b", progress_cb=lambda pct, detail: calls.append((pct, detail))
    )

    assert success is True
    assert error is None
    assert calls[0] == (50, "Layer abc — 50%")
    assert calls[-1] == (100, "Model ready")
