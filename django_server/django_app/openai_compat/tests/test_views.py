"""ChatCompletionsView over Django's AsyncClient (pytest-django's
`async_client` fixture). `functions.astream_bot_response` is monkeypatched
everywhere here so these tests never touch Ollama or Docker — same approach
as test_consumers.py.
"""

from __future__ import annotations

import json

import pytest
from channels.db import database_sync_to_async

from django_app import functions, models
from django_app.functions import ModelUnavailableError

pytestmark = [pytest.mark.django_db(transaction=True), pytest.mark.asyncio]


@pytest.fixture(autouse=True)
async def _close_leaked_db_connection():
    # database_sync_to_async is thread_sensitive: outside a real ASGI request
    # cycle it funnels every call onto one shared worker thread for the life
    # of the process, and nothing closes the connection that thread opens —
    # same issue documented in test_consumers.py. Left alone, it holds
    # test_chatbot open past teardown and the next --create-db run hangs.
    yield

    from django.db import connections

    await database_sync_to_async(connections.close_all)()


@pytest.fixture
def ai_model():
    models.AIModel.objects.filter(model="llama3.1").delete()
    return models.AIModel.objects.create(name="llama3.1", model="llama3.1", index=1)


URL = "/v1/chat/completions"


async def _post(async_client, body: dict):
    return await async_client.post(URL, data=json.dumps(body), content_type="application/json")


async def test_non_streaming_completion_returns_full_message(ai_model, async_client, monkeypatch):
    async def _fake_astream(model, parameters, message, image, history, system_prompt=None, generation_params=None, stats=None):
        if stats is not None:
            stats["prompt_tokens"] = 5
            stats["completion_tokens"] = 2
        for chunk in ("Hel", "lo"):
            yield chunk

    monkeypatch.setattr(functions, "astream_bot_response", _fake_astream)

    response = await _post(
        async_client, {"model": "llama3.1:8b", "messages": [{"role": "user", "content": "Hi"}]}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "llama3.1:8b"
    assert data["choices"] == [
        {"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}
    ]
    assert data["usage"] == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
    assert data["id"].startswith("chatcmpl-")


async def test_streaming_completion_returns_sse_chunks_and_done(ai_model, async_client, monkeypatch):
    async def _fake_astream(model, parameters, message, image, history, system_prompt=None, generation_params=None, stats=None):
        for chunk in ("Hel", "lo"):
            yield chunk

    monkeypatch.setattr(functions, "astream_bot_response", _fake_astream)
    monkeypatch.setattr(functions, "get_ollama_url", lambda *a, **k: ("http://fake", "llama3.1:8b"))

    response = await _post(
        async_client,
        {"model": "llama3.1:8b", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
    )

    assert response.status_code == 200
    assert response["Content-Type"] == "text/event-stream"

    body = b"".join([chunk async for chunk in response.streaming_content]).decode()
    lines = [line for line in body.split("\n\n") if line]

    assert lines[-1] == "data: [DONE]"

    frames = [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]
    assert frames[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert frames[1]["choices"][0]["delta"] == {"content": "Hel"}
    assert frames[2]["choices"][0]["delta"] == {"content": "lo"}
    assert frames[-1]["choices"][0]["delta"] == {}
    assert frames[-1]["choices"][0]["finish_reason"] == "stop"
    assert all(f["object"] == "chat.completion.chunk" for f in frames)
    assert len({f["id"] for f in frames}) == 1


async def test_unknown_model_returns_404_with_openai_error_shape(ai_model, async_client):
    response = await _post(
        async_client, {"model": "does-not-exist:8b", "messages": [{"role": "user", "content": "Hi"}]}
    )

    assert response.status_code == 404
    data = response.json()
    assert data["error"]["code"] == "model_not_found"
    assert data["error"]["type"] == "invalid_request_error"


async def test_invalid_json_body_returns_400(ai_model, async_client):
    response = await async_client.post(URL, data="not json", content_type="application/json")

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


async def test_missing_messages_returns_400(ai_model, async_client):
    response = await _post(async_client, {"model": "llama3.1:8b"})

    assert response.status_code == 400
    assert response.json()["error"]["param"] == "messages"


async def test_model_unavailable_returns_503_for_non_streaming(ai_model, async_client, monkeypatch):
    async def _fake_astream(*args, **kwargs):
        raise ModelUnavailableError("Model container isn't running.")
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(functions, "astream_bot_response", _fake_astream)

    response = await _post(
        async_client, {"model": "llama3.1:8b", "messages": [{"role": "user", "content": "Hi"}]}
    )

    assert response.status_code == 503
    assert response.json()["error"]["type"] == "server_error"


async def test_model_unavailable_returns_503_before_streaming_starts(ai_model, async_client, monkeypatch):
    def _fake_get_ollama_url(*args, **kwargs):
        raise ModelUnavailableError("Model container isn't running.")

    monkeypatch.setattr(functions, "get_ollama_url", _fake_get_ollama_url)

    response = await _post(
        async_client,
        {"model": "llama3.1:8b", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
    )

    assert response.status_code == 503
    assert response.json()["error"]["type"] == "server_error"


async def test_models_lists_installed_models_as_name_tag(async_client, monkeypatch):
    from container.ContainerManager import ContainerManager

    monkeypatch.setattr(ContainerManager, "is_connected", lambda self: True)
    monkeypatch.setattr(
        ContainerManager, "get_available_containers",
        lambda self: [{"name": "llama3.1_8b", "status": "running", "port": "1", "environment": {"model": "llama3.1", "parameters": "8b"}}],
    )

    response = await async_client.get("/v1/models")

    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [{"id": "llama3.1:8b", "object": "model", "created": 0, "owned_by": "reichat"}],
    }


async def test_embeddings_returns_one_vector_per_input(async_client, monkeypatch):
    from django_app.openai_compat import views

    class _FakeClient:
        def embed_documents(self, texts):
            return [[float(len(t)), 0.0] for t in texts]

    monkeypatch.setattr(views.embeddings, "get_embeddings_client", lambda model, params: _FakeClient())

    response = await async_client.post(
        "/v1/embeddings", data=json.dumps({"model": "nomic-embed-text", "input": ["ab", "abcd"]}),
        content_type="application/json",
    )

    assert response.status_code == 200
    body = response.json()
    assert [d["embedding"] for d in body["data"]] == [[2.0, 0.0], [4.0, 0.0]]
    assert [d["index"] for d in body["data"]] == [0, 1]


async def test_embeddings_reports_a_stopped_model_as_503(async_client, monkeypatch):
    from django_app.openai_compat import views

    def _down(model, params):
        raise ModelUnavailableError("Embedding model nomic-embed-text:latest isn't running.")

    monkeypatch.setattr(views.embeddings, "get_embeddings_client", _down)

    response = await async_client.post(
        "/v1/embeddings", data=json.dumps({"model": "nomic-embed-text", "input": "hi"}),
        content_type="application/json",
    )

    assert response.status_code == 503
    assert "isn't running" in response.json()["error"]["message"]


async def test_embeddings_rejects_bad_input(async_client):
    response = await async_client.post(
        "/v1/embeddings", data=json.dumps({"model": "x", "input": [1, 2]}), content_type="application/json",
    )

    assert response.status_code == 400
