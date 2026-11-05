"""The OpenAI-compatible endpoints. `generation.astream_bot_response` is
replaced throughout so these never touch Ollama or Docker."""

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import ollama
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import models
from app.llm import generation, runtime
from app.llm.runtime import ModelUnavailableError
from app.openai_compat import router as openai_router
from tests.factories import make_model

pytestmark = pytest.mark.usefixtures("db")

CHAT = "/v1/chat/completions"
HI = {"model": "llama3.1:8b", "messages": [{"role": "user", "content": "Hi"}]}


@pytest.fixture
async def ai_model(db: AsyncSession) -> models.AIModel:
    return await make_model(db, "llama3.1", name="llama3.1")


@pytest.fixture(autouse=True)
def model_is_running(monkeypatch: pytest.MonkeyPatch) -> None:
    async def running(*args: Any, **kwargs: Any) -> tuple[str, str]:
        return "http://fake", "llama3.1:8b"

    monkeypatch.setattr(runtime, "get_ollama_url", running)


def streaming(*chunks: str, stats: dict[str, int] | None = None) -> Callable[..., AsyncIterator[str]]:
    async def fake(
        model_name,
        parameters,
        message,
        image,
        history,
        system_prompt=None,
        generation_params=None,
        stats_out=None,
    ) -> AsyncIterator[str]:
        if stats_out is not None and stats:
            stats_out.update(stats)
        for chunk in chunks:
            yield chunk

    return fake


def failing(error: Exception, *, after: tuple[str, ...] = ()) -> Callable[..., AsyncIterator[str]]:
    async def fake(*args: Any, **kwargs: Any) -> AsyncIterator[str]:
        for chunk in after:
            yield chunk
        raise error

    return fake


async def post(client: httpx.AsyncClient, body: dict, path: str = CHAT) -> httpx.Response:
    return await client.post(path, content=json.dumps(body))


def frames_of(response: httpx.Response) -> tuple[list[dict], str]:
    events = [event for event in response.text.split("\n\n") if event]
    return [json.loads(e.removeprefix("data: ")) for e in events[:-1]], events[-1]


async def test_non_streaming_completion_returns_full_message(
    ai_model: models.AIModel, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        generation,
        "astream_bot_response",
        streaming("Hel", "lo", stats={"prompt_tokens": 5, "completion_tokens": 2}),
    )

    response = await post(client, HI)

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "llama3.1:8b"
    assert data["choices"] == [
        {"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}
    ]
    assert data["usage"] == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
    assert data["id"].startswith("chatcmpl-")


async def test_a_body_without_a_content_type_is_still_read(
    ai_model: models.AIModel, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
):
    # Plenty of OpenAI clients send no Content-Type at all.
    monkeypatch.setattr(generation, "astream_bot_response", streaming("ok"))

    response = await client.post(CHAT, content=json.dumps(HI), headers={"Content-Type": ""})

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "ok"


async def test_the_request_is_handed_to_the_model_as_sent(
    ai_model: models.AIModel, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
):
    seen: dict[str, Any] = {}

    async def fake(
        model_name,
        parameters,
        message,
        image,
        history,
        system_prompt=None,
        generation_params=None,
        stats=None,
    ):
        seen.update(
            model_name=model_name,
            parameters=parameters,
            message=message,
            history=history,
            system_prompt=system_prompt,
            generation_params=generation_params,
        )
        yield "ok"

    monkeypatch.setattr(generation, "astream_bot_response", fake)

    await post(
        client,
        {
            "model": "llama3.1:8b",
            "temperature": 0.5,
            "seed": 3,
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "One"},
                {"role": "assistant", "content": "Two"},
                {"role": "user", "content": "Three"},
            ],
        },
    )

    assert seen == {
        "model_name": "llama3.1",
        "parameters": "8b",
        "message": "Three",
        "history": [
            {"role": "user", "content": "One", "image": ""},
            {"role": "assistant", "content": "Two", "image": ""},
        ],
        "system_prompt": "Be brief.",
        "generation_params": {"temperature": 0.5, "seed": 3},
    }


async def test_streaming_completion_returns_sse_chunks_and_done(
    ai_model: models.AIModel, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(generation, "astream_bot_response", streaming("Hel", "lo"))

    response = await post(client, {**HI, "stream": True})

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    assert response.headers["cache-control"] == "no-cache"

    frames, last = frames_of(response)
    assert last == "data: [DONE]"
    assert response.text.endswith("data: [DONE]\n\n")
    assert frames[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert frames[1]["choices"][0]["delta"] == {"content": "Hel"}
    assert frames[2]["choices"][0]["delta"] == {"content": "lo"}
    assert frames[-1]["choices"][0]["delta"] == {}
    assert frames[-1]["choices"][0]["finish_reason"] == "stop"
    assert all(f["object"] == "chat.completion.chunk" for f in frames)
    assert all(f["model"] == "llama3.1:8b" for f in frames)
    assert len({f["id"] for f in frames}) == 1


async def test_unknown_model_returns_404_with_openai_error_shape(
    ai_model: models.AIModel, client: httpx.AsyncClient
):
    response = await post(
        client, {"model": "does-not-exist:8b", "messages": [{"role": "user", "content": "Hi"}]}
    )

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "message": "The model `does-not-exist:8b` does not exist.",
            "type": "invalid_request_error",
            "param": "model",
            "code": "model_not_found",
        }
    }


@pytest.mark.parametrize("raw", [b"not json", b"\xff\xfe\x00"])
async def test_a_body_that_is_not_json_returns_400(client: httpx.AsyncClient, raw: bytes):
    response = await client.post(CHAT, content=raw)

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": "Request body must be valid JSON.",
            "type": "invalid_request_error",
            "param": None,
            "code": None,
        }
    }


async def test_model_unavailable_returns_503_for_non_streaming(
    ai_model: models.AIModel, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        generation, "astream_bot_response", failing(ModelUnavailableError("Model container isn't running."))
    )

    response = await post(client, HI)

    assert response.status_code == 503
    assert response.json()["error"]["type"] == "server_error"
    assert response.json()["error"]["message"] == "Model container isn't running."


async def test_model_unavailable_returns_503_before_streaming_starts(
    ai_model: models.AIModel, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
):
    async def unavailable(*args: Any, **kwargs: Any) -> None:
        raise ModelUnavailableError("Model container isn't running.")

    monkeypatch.setattr(runtime, "get_ollama_url", unavailable)

    response = await post(client, {**HI, "stream": True})

    assert response.status_code == 503
    assert response.headers["content-type"] == "application/json"
    assert response.json()["error"]["type"] == "server_error"


async def test_a_stopped_model_returns_503_before_streaming_starts(
    ai_model: models.AIModel, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
):
    async def stopped(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(runtime, "get_ollama_url", stopped)

    response = await post(client, {**HI, "stream": True})

    assert response.status_code == 503
    assert response.json()["error"]["message"] == generation.NOT_RUNNING_MESSAGE


async def test_a_model_failing_mid_stream_is_reported_in_the_stream(
    ai_model: models.AIModel, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        generation,
        "astream_bot_response",
        failing(ModelUnavailableError("Ollama went away."), after=("Hel",)),
    )

    response = await post(client, {**HI, "stream": True})

    assert response.status_code == 200
    frames, last = frames_of(response)
    assert last == "data: [DONE]"
    assert frames[1]["choices"][0]["delta"] == {"content": "Hel"}
    assert frames[-1] == {
        "error": {"message": "Ollama went away.", "type": "server_error", "param": None, "code": None}
    }
    assert all(f.get("choices", [{}])[0].get("finish_reason") != "stop" for f in frames)


async def test_models_lists_installed_models_as_name_tag(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
):
    async def containers() -> list[dict[str, Any]]:
        return [
            {
                "name": "llama3.1_8b",
                "status": "running",
                "port": "1",
                "environment": {"model": "llama3.1", "parameters": "8b"},
            },
            {"name": "half", "status": "exited", "port": None, "environment": {"model": "half"}},
        ]

    monkeypatch.setattr(openai_router.service, "get_available_containers", containers)

    response = await client.get("/v1/models")

    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [{"id": "llama3.1:8b", "object": "model", "created": 0, "owned_by": "reichat"}],
    }


class FakeEmbeddings:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.error is not None:
            raise self.error
        return [[float(len(t)), 0.0] for t in texts]


def using_embeddings(monkeypatch: pytest.MonkeyPatch, client_object: FakeEmbeddings) -> list[tuple[str, str]]:
    asked: list[tuple[str, str]] = []

    async def get_client(model: str, parameters: str) -> FakeEmbeddings:
        asked.append((model, parameters))
        return client_object

    monkeypatch.setattr(openai_router.embeddings, "get_embeddings_client", get_client)
    return asked


async def test_embeddings_returns_one_vector_per_input(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
):
    using_embeddings(monkeypatch, FakeEmbeddings())

    response = await post(client, {"model": "nomic-embed-text", "input": ["ab", "abcd"]}, "/v1/embeddings")

    assert response.status_code == 200
    body = response.json()
    assert [d["embedding"] for d in body["data"]] == [[2.0, 0.0], [4.0, 0.0]]
    assert [d["index"] for d in body["data"]] == [0, 1]
    assert all(d["object"] == "embedding" for d in body["data"])
    assert body["object"] == "list"
    assert body["model"] == "nomic-embed-text"
    assert body["usage"] == {"prompt_tokens": 0, "total_tokens": 0}


async def test_embeddings_accepts_a_single_string_and_defaults_the_tag_to_latest(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
):
    asked = using_embeddings(monkeypatch, FakeEmbeddings())

    response = await post(client, {"model": "nomic-embed-text", "input": "hi"}, "/v1/embeddings")

    assert [d["embedding"] for d in response.json()["data"]] == [[2.0, 0.0]]
    assert asked == [("nomic-embed-text", "latest")]


async def test_embeddings_reports_a_stopped_model_as_503(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
):
    async def down(model: str, parameters: str) -> None:
        raise ModelUnavailableError("Embedding model nomic-embed-text:latest isn't running.")

    monkeypatch.setattr(openai_router.embeddings, "get_embeddings_client", down)

    response = await post(client, {"model": "nomic-embed-text", "input": "hi"}, "/v1/embeddings")

    assert response.status_code == 503
    assert "isn't running" in response.json()["error"]["message"]


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (httpx.ConnectError("refused"), generation.UNREACHABLE_MESSAGE),
        (
            ollama.ResponseError("model not found", 404),
            "Ollama rejected the request: model not found (status code: 404)",
        ),
    ],
)
async def test_embeddings_reports_an_unreachable_ollama_as_503(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch, error: Exception, message: str
):
    using_embeddings(monkeypatch, FakeEmbeddings(error))

    response = await post(client, {"model": "nomic-embed-text", "input": "hi"}, "/v1/embeddings")

    assert response.status_code == 503
    assert response.json()["error"]["message"] == message


@pytest.mark.parametrize(
    ("body", "param"),
    [
        ({"model": "x", "input": [1, 2]}, "input"),
        ({"model": "x", "input": ["a"] * 257}, "input"),
        ({"input": "hi"}, "model"),
    ],
)
async def test_embeddings_rejects_bad_input(client: httpx.AsyncClient, body: dict, param: str):
    response = await post(client, body, "/v1/embeddings")

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert response.json()["error"]["param"] == param


async def test_v1_is_not_throttled(client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch):
    async def nothing() -> list[dict[str, Any]]:
        return []

    monkeypatch.setattr(openai_router.service, "get_available_containers", nothing)

    statuses = {(await client.get("/v1/models")).status_code for _ in range(125)}

    assert statuses == {200}


async def test_a_page_on_another_origin_cannot_use_the_api(
    ai_model: models.AIModel, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(generation, "astream_bot_response", failing(AssertionError("model was called")))

    response = await client.post(CHAT, content=json.dumps(HI), headers={"Origin": "https://evil.example"})

    assert response.status_code == 403
