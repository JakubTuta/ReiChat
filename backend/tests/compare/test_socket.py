"""The compare WebSocket. Compare is ephemeral (no chat involved), so there is
no persistence to check, just protocol and fan-out behaviour."""

import asyncio
from collections.abc import AsyncIterator, Iterable
from typing import Any

import pytest
from httpx_ws import AsyncWebSocketSession, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import models
from app.llm import generation
from app.llm.runtime import ModelUnavailableError
from tests.conftest import ConnectWs
from tests.factories import make_model

pytestmark = pytest.mark.usefixtures("db")

A = {"model": "compare-a", "parameters": "8b"}
B = {"model": "compare-b", "parameters": "8b"}


@pytest.fixture
async def two_models(db: AsyncSession) -> tuple[models.AIModel, models.AIModel]:
    return await make_model(db, "compare-a", name="A", index=1), await make_model(
        db, "compare-b", name="B", index=2
    )


async def collect_until_done(ws: AsyncWebSocketSession, target_keys: Iterable[str]) -> dict[str, list[dict]]:
    """Compare fans out concurrently, so message order across targets isn't
    guaranteed — collect everything and group by target instead of asserting
    a fixed sequence."""
    keys = list(target_keys)
    by_target: dict[str, list[dict]] = {key: [] for key in keys}
    done: set[str] = set()

    while len(done) < len(keys):
        message = await ws.receive_json(timeout=5)
        by_target[message["target"]].append(message)
        if message.get("done"):
            done.add(message["target"])

    return by_target


async def test_receive_streams_from_two_models_concurrently(
    connect_ws: ConnectWs, two_models: Any, monkeypatch: pytest.MonkeyPatch
):
    async def fake(
        model_name, parameters, message, image, history, system_prompt=None, params=None, stats=None
    ):
        for chunk in (f"{model_name}-1", f"{model_name}-2"):
            yield chunk

    monkeypatch.setattr(generation, "astream_bot_response", fake)

    async with connect_ws("/ws/compare/") as ws:
        await ws.send_json({"prompt": "Hi", "targets": [A, B]})
        by_target = await collect_until_done(ws, ["compare-a:8b", "compare-b:8b"])

    assert [m["message"] for m in by_target["compare-a:8b"] if not m["done"]] == [
        "compare-a-1",
        "compare-a-2",
    ]
    assert [m["message"] for m in by_target["compare-b:8b"] if not m["done"]] == [
        "compare-b-1",
        "compare-b-2",
    ]
    assert by_target["compare-a:8b"][-1] == {"target": "compare-a:8b", "message": "", "done": True}
    assert by_target["compare-b:8b"][-1] == {"target": "compare-b:8b", "message": "", "done": True}


async def test_every_target_gets_the_prompt_and_no_history(
    connect_ws: ConnectWs, two_models: Any, monkeypatch: pytest.MonkeyPatch
):
    seen: list[tuple[Any, ...]] = []

    async def fake(*args: Any, **kwargs: Any) -> AsyncIterator[str]:
        seen.append(args[:7])
        yield "ok"

    monkeypatch.setattr(generation, "astream_bot_response", fake)

    async with connect_ws("/ws/compare/") as ws:
        await ws.send_json({"prompt": "Hi", "targets": [A]})
        await collect_until_done(ws, ["compare-a:8b"])

    assert seen == [("compare-a", "8b", "Hi", "", [], None, None)]


async def test_receive_includes_stats_when_reported(
    connect_ws: ConnectWs, two_models: Any, monkeypatch: pytest.MonkeyPatch
):
    async def fake(
        model_name, parameters, message, image, history, system_prompt=None, params=None, stats=None
    ):
        stats["completion_tokens"] = 5
        yield "hi"

    monkeypatch.setattr(generation, "astream_bot_response", fake)

    async with connect_ws("/ws/compare/") as ws:
        await ws.send_json({"prompt": "Hi", "targets": [A]})
        by_target = await collect_until_done(ws, ["compare-a:8b"])

    assert by_target["compare-a:8b"][-1]["stats"] == {"completion_tokens": 5}


async def test_receive_unknown_model_reports_per_target_error(connect_ws: ConnectWs, two_models: Any):
    async with connect_ws("/ws/compare/") as ws:
        await ws.send_json({"prompt": "Hi", "targets": [{"model": "does-not-exist", "parameters": "8b"}]})

        assert await ws.receive_json(timeout=5) == {
            "target": "does-not-exist:8b",
            "error": "Model not found.",
            "done": True,
        }


async def test_receive_one_model_failing_does_not_block_the_other(
    connect_ws: ConnectWs, two_models: Any, monkeypatch: pytest.MonkeyPatch
):
    async def fake(model_name, *args: Any, **kwargs: Any) -> AsyncIterator[str]:
        if model_name == "compare-a":
            raise ModelUnavailableError("not running")
        yield "ok"

    monkeypatch.setattr(generation, "astream_bot_response", fake)

    async with connect_ws("/ws/compare/") as ws:
        await ws.send_json({"prompt": "Hi", "targets": [A, B]})
        by_target = await collect_until_done(ws, ["compare-a:8b", "compare-b:8b"])

    assert by_target["compare-a:8b"] == [{"target": "compare-a:8b", "error": "not running", "done": True}]
    assert by_target["compare-b:8b"][-1]["done"] is True
    assert any(m.get("message") == "ok" for m in by_target["compare-b:8b"])


async def test_an_unexpected_failure_in_one_target_is_reported_for_that_target(
    connect_ws: ConnectWs, two_models: Any, monkeypatch: pytest.MonkeyPatch
):
    async def fake(model_name, *args: Any, **kwargs: Any) -> AsyncIterator[str]:
        if model_name == "compare-a":
            raise RuntimeError("boom")
        yield "ok"

    monkeypatch.setattr(generation, "astream_bot_response", fake)

    async with connect_ws("/ws/compare/") as ws:
        await ws.send_json({"prompt": "Hi", "targets": [A, B]})
        by_target = await collect_until_done(ws, ["compare-a:8b", "compare-b:8b"])

    assert by_target["compare-a:8b"] == [
        {"target": "compare-a:8b", "error": "An unexpected error occurred. Please try again.", "done": True}
    ]
    assert by_target["compare-b:8b"][-1]["done"] is True


async def test_receive_rejects_empty_prompt(connect_ws: ConnectWs):
    async with connect_ws("/ws/compare/") as ws:
        await ws.send_json({"prompt": "", "targets": [{"model": "x", "parameters": "8b"}]})

        assert await ws.receive_json(timeout=5) == {
            "target": None,
            "error": "Prompt must be a non-empty string.",
            "done": True,
        }


async def test_receive_rejects_a_prompt_over_the_limit(connect_ws: ConnectWs):
    async with connect_ws("/ws/compare/") as ws:
        await ws.send_json({"prompt": "x" * 10_001, "targets": [A]})

        assert await ws.receive_json(timeout=5) == {
            "target": None,
            "error": "Prompt too long. Maximum length is 10000 characters.",
            "done": True,
        }


@pytest.mark.parametrize(
    "targets", [[{"model": f"m{i}", "parameters": "8b"} for i in range(5)], [], "all", None]
)
async def test_receive_rejects_the_wrong_number_of_targets(connect_ws: ConnectWs, targets: object):
    async with connect_ws("/ws/compare/") as ws:
        await ws.send_json({"prompt": "Hi", "targets": targets})
        response = await ws.receive_json(timeout=5)

    assert response["target"] is None
    assert response["error"] == "Provide between 1 and 4 models to compare."


@pytest.mark.parametrize(
    "target", ["compare-a", {"model": "compare-a"}, {"model": 5, "parameters": "8b"}, {"parameters": "8b"}]
)
async def test_receive_rejects_a_target_without_model_and_parameters(connect_ws: ConnectWs, target: object):
    async with connect_ws("/ws/compare/") as ws:
        await ws.send_json({"prompt": "Hi", "targets": [target]})

        assert await ws.receive_json(timeout=5) == {
            "target": None,
            "error": "Each target needs a model and parameters.",
            "done": True,
        }


async def test_receive_invalid_json_reports_error(connect_ws: ConnectWs):
    async with connect_ws("/ws/compare/") as ws:
        await ws.send_text("not json")

        assert (await ws.receive_json(timeout=5))["error"] == "Invalid message format."


async def test_a_message_that_is_not_an_object_is_refused(connect_ws: ConnectWs):
    async with connect_ws("/ws/compare/") as ws:
        await ws.send_text("[1]")

        assert await ws.receive_json(timeout=5) == {
            "target": None,
            "error": "Invalid message format.",
            "done": True,
        }


async def test_a_binary_frame_ends_the_conversation(connect_ws: ConnectWs):
    async with connect_ws("/ws/compare/") as ws:
        await ws.send_bytes(b"\x00")

        with pytest.raises(WebSocketDisconnect) as closed:
            await ws.receive_json(timeout=5)

    assert closed.value.code == 1000


async def test_receive_while_running_is_rejected(
    connect_ws: ConnectWs, two_models: Any, monkeypatch: pytest.MonkeyPatch
):
    started = asyncio.Event()
    release = asyncio.Event()

    async def fake(*args: Any, **kwargs: Any) -> AsyncIterator[str]:
        started.set()
        await release.wait()
        yield "ok"

    monkeypatch.setattr(generation, "astream_bot_response", fake)

    async with connect_ws("/ws/compare/") as ws:
        await ws.send_json({"prompt": "Hi", "targets": [A]})
        await asyncio.wait_for(started.wait(), timeout=5)

        await ws.send_json({"prompt": "Hi again", "targets": [B]})
        response = await ws.receive_json(timeout=5)
        assert response["error"] == "A comparison is already running. Wait for it to finish."

        release.set()
        await collect_until_done(ws, ["compare-a:8b"])


async def test_a_new_comparison_can_start_once_the_last_one_finished(
    connect_ws: ConnectWs, two_models: Any, monkeypatch: pytest.MonkeyPatch
):
    async def fake(model_name, *args: Any, **kwargs: Any) -> AsyncIterator[str]:
        yield f"from {model_name}"

    monkeypatch.setattr(generation, "astream_bot_response", fake)

    async with connect_ws("/ws/compare/") as ws:
        await ws.send_json({"prompt": "One", "targets": [A]})
        await collect_until_done(ws, ["compare-a:8b"])

        await ws.send_json({"prompt": "Two", "targets": [B]})
        by_target = await collect_until_done(ws, ["compare-b:8b"])

    assert by_target["compare-b:8b"][0]["message"] == "from compare-b"


async def test_stop_cancels_in_flight_comparison_for_every_target(
    connect_ws: ConnectWs, two_models: Any, monkeypatch: pytest.MonkeyPatch
):
    started: list[str] = []
    both_started = asyncio.Event()
    release = asyncio.Event()

    async def fake(model_name, *args: Any, **kwargs: Any) -> AsyncIterator[str]:
        started.append(model_name)
        if len(started) == 2:
            both_started.set()
        await release.wait()
        yield "should never be reached"

    monkeypatch.setattr(generation, "astream_bot_response", fake)

    async with connect_ws("/ws/compare/") as ws:
        await ws.send_json({"prompt": "Hi", "targets": [A, B]})
        await asyncio.wait_for(both_started.wait(), timeout=5)

        await ws.send_json({"type": "stop"})
        by_target = await collect_until_done(ws, ["compare-a:8b", "compare-b:8b"])

    assert by_target["compare-a:8b"] == [
        {"target": "compare-a:8b", "message": "", "done": True, "stopped": True}
    ]
    assert by_target["compare-b:8b"] == [
        {"target": "compare-b:8b", "message": "", "done": True, "stopped": True}
    ]


async def test_stop_with_nothing_running_is_a_no_op(connect_ws: ConnectWs):
    async with connect_ws("/ws/compare/") as ws:
        await ws.send_json({"type": "stop"})

        with pytest.raises(TimeoutError):
            await ws.receive_json(timeout=0.2)


async def test_closing_the_tab_cancels_every_running_comparison(
    connect_ws: ConnectWs, two_models: Any, monkeypatch: pytest.MonkeyPatch
):
    cancelled: list[str] = []
    started = asyncio.Event()

    async def fake(model_name, *args: Any, **kwargs: Any) -> AsyncIterator[str]:
        try:
            started.set()
            await asyncio.Event().wait()
            yield "never"
        finally:
            cancelled.append(model_name)

    monkeypatch.setattr(generation, "astream_bot_response", fake)

    async with connect_ws("/ws/compare/") as ws:
        await ws.send_json({"prompt": "Hi", "targets": [A, B]})
        await asyncio.wait_for(started.wait(), timeout=5)
        await asyncio.sleep(0.05)

    for _ in range(100):
        if len(cancelled) == 2:
            break
        await asyncio.sleep(0.05)
    assert sorted(cancelled) == ["compare-a", "compare-b"]
