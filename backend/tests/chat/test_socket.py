"""The chat WebSocket, driven through a real connection to the app.

`generation.astream_bot_response` and friends are replaced throughout so these
never touch Ollama or Docker; they exercise the socket's own protocol
handling, error surfacing and history persistence.
"""

import asyncio
import base64
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest
from httpx_ws import WebSocketDisconnect
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat import history
from app.chat import socket as chat_socket
from app.db import models
from app.llm import generation, runtime
from app.llm.runtime import ModelUnavailableError
from app.rag import retrieval
from tests.conftest import ConnectWs
from tests.factories import make_chat, make_exchange, make_model

pytestmark = pytest.mark.usefixtures("db")

PROMPT = {"message": "Hi", "ai_model": "llama3.1", "ai_model_parameters": "8b"}


def returning(value: Any) -> Callable[..., Any]:
    async def answer(*args: Any, **kwargs: Any) -> Any:
        return value

    return answer


@pytest.fixture(autouse=True)
def no_model_lookups(monkeypatch: pytest.MonkeyPatch) -> None:
    # The real lookups ask the model's Ollama container; there is none here.
    # Tests that care override these.
    monkeypatch.setattr(runtime, "get_model_info", returning(None))
    monkeypatch.setattr(runtime, "get_gpu_share", returning(None))


@pytest.fixture
async def model(db: AsyncSession) -> models.AIModel:
    return await make_model(db, "llama3.1", name="llama3.1")


@pytest.fixture
async def chat(db: AsyncSession, model: models.AIModel) -> models.ChatHistory:
    return await make_chat(db, model, title="Test chat")


@pytest.fixture
async def tools_chat(db: AsyncSession) -> models.ChatHistory:
    tools_model = await make_model(db, "llama3.1-tools-test", name="llama3.1", can_use_tools=True)
    return await make_chat(db, tools_model, title="Tools chat", tools_enabled=True)


def without_context(frame: dict) -> dict:
    """Done frames always carry a context report (covered by its own tests
    below) — strip it where a test is about something else."""
    return {k: v for k, v in frame.items() if k != "context"}


def streaming(*chunks: Any) -> Callable[..., AsyncIterator[Any]]:
    async def fake(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        for chunk in chunks:
            yield chunk

    return fake


def failing(error: Exception) -> Callable[..., AsyncIterator[Any]]:
    async def fake(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        raise error
        yield  # unreachable; keeps this an async generator

    return fake


def hanging_after(first: str, release: asyncio.Event) -> Callable[..., AsyncIterator[str]]:
    async def fake(*args: Any, **kwargs: Any) -> AsyncIterator[str]:
        yield first
        await release.wait()

    return fake


def capturing(captured: dict, stats_values: dict | None = None) -> Callable[..., AsyncIterator[str]]:
    async def fake(
        model_name,
        parameters,
        message,
        image,
        history,
        system_prompt=None,
        generation_params=None,
        stats=None,
        reasoning=None,
    ) -> AsyncIterator[str]:
        captured.update(
            history=history, system_prompt=system_prompt, generation_params=generation_params, message=message
        )
        yield "Hi"
        if stats is not None and stats_values:
            stats.update(stats_values)

    return fake


async def collect(connect_ws: ConnectWs, chat: models.ChatHistory, **fields: Any) -> list[dict]:
    """Sends one prompt and returns every frame up to and including the done one."""
    async with connect_ws(f"/ws/chat/{chat.id}/") as ws:
        await ws.send_json({**PROMPT, **fields})
        frames = []
        while True:
            frame = await ws.receive_json(timeout=5)
            frames.append(frame)
            if frame.get("done"):
                return frames


async def all_messages(db: AsyncSession, chat: models.ChatHistory) -> list[models.ChatMessage]:
    rows = await db.execute(
        select(models.ChatMessage)
        .where(models.ChatMessage.chat_id == chat.id)
        .order_by(models.ChatMessage.created_at, models.ChatMessage.id)
        .execution_options(populate_existing=True)
    )
    return list(rows.scalars())


async def active_path(db: AsyncSession, chat: models.ChatHistory) -> list[tuple[str, str]]:
    await db.refresh(chat)
    return [(m.role, m.content) for m in await history.get_messages_for_chat(db, chat)]


async def long_history(db: AsyncSession, chat: models.ChatHistory, turns: int, words: int = 400) -> list:
    parent = None
    answers = []
    for i in range(turns):
        _, parent = await make_exchange(
            db, chat, f"question {i} " + "word " * words, f"answer {i} " + "word " * words, parent
        )
        answers.append(parent)
    return answers


async def test_receive_streams_response_and_persists_history(
    connect_ws: ConnectWs, chat: models.ChatHistory, db: AsyncSession, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(generation, "astream_bot_response", streaming("Hel", "lo"))

    async with connect_ws(f"/ws/chat/{chat.id}/") as ws:
        await ws.send_json(PROMPT)

        assert await ws.receive_json(timeout=5) == {"message": "Hel", "done": False}
        assert await ws.receive_json(timeout=5) == {"message": "lo", "done": False}
        assert without_context(await ws.receive_json(timeout=5)) == {"message": "Hello", "done": True}

    messages = await all_messages(db, chat)
    assert [(m.role, m.content) for m in messages] == [("user", "Hi"), ("assistant", "Hello")]


async def test_receive_passes_chat_persona_system_prompt_to_astream(
    connect_ws: ConnectWs, chat: models.ChatHistory, db: AsyncSession, monkeypatch: pytest.MonkeyPatch
):
    persona = models.Persona(name="Pirate", system_prompt="Talk like a pirate.")
    db.add(persona)
    await db.commit()
    chat.persona_id = persona.id
    await db.commit()
    captured: dict = {}
    monkeypatch.setattr(generation, "astream_bot_response", capturing(captured))

    await collect(connect_ws, chat, message="Ahoy")

    assert captured["system_prompt"] == "Talk like a pirate."


async def test_receive_passes_chat_generation_params_to_astream(
    connect_ws: ConnectWs, chat: models.ChatHistory, db: AsyncSession, monkeypatch: pytest.MonkeyPatch
):
    chat.temperature = 0.9
    chat.num_ctx = 8192
    chat.top_p = 0.5
    chat.seed = 7
    await db.commit()
    captured: dict = {}
    monkeypatch.setattr(generation, "astream_bot_response", capturing(captured))

    await collect(connect_ws, chat)

    assert captured["generation_params"] == {"temperature": 0.9, "num_ctx": 8192, "top_p": 0.5, "seed": 7}


async def test_done_frame_reports_memory_use_and_keeps_stats_separate(
    connect_ws: ConnectWs, chat: models.ChatHistory, db: AsyncSession, monkeypatch: pytest.MonkeyPatch
):
    chat.num_ctx = 4096
    await db.commit()
    captured: dict = {}
    monkeypatch.setattr(
        generation,
        "astream_bot_response",
        capturing(captured, {"prompt_tokens": 10, "completion_tokens": 20, "tokens_per_second": 50.0}),
    )

    done = (await collect(connect_ws, chat))[-1]

    assert done["stats"] == {"prompt_tokens": 10, "completion_tokens": 20, "tokens_per_second": 50.0}
    assert done["context"]["limit"] == 4096
    assert done["context"]["preset"] == "short"
    assert done["context"]["dropped"] == 0
    assert done["context"]["summarized"] is False
    # Estimated prompt plus the exact completion count.
    assert done["context"]["used"] > 20
    assert captured["generation_params"]["num_ctx"] == 4096


async def test_without_a_chosen_length_the_standard_preset_is_requested_explicitly(
    connect_ws: ConnectWs, chat: models.ChatHistory, monkeypatch: pytest.MonkeyPatch
):
    captured: dict = {}
    monkeypatch.setattr(generation, "astream_bot_response", capturing(captured))

    done = (await collect(connect_ws, chat))[-1]

    assert captured["generation_params"]["num_ctx"] == 8192
    assert done["context"]["preset"] == "standard"


async def test_memory_never_exceeds_what_the_model_supports(
    connect_ws: ConnectWs, chat: models.ChatHistory, db: AsyncSession, monkeypatch: pytest.MonkeyPatch
):
    chat.num_ctx = 32768
    await db.commit()
    monkeypatch.setattr(
        runtime, "get_model_info", returning({"context_length": 2048, "thinking": None, "capabilities": []})
    )
    captured: dict = {}
    monkeypatch.setattr(generation, "astream_bot_response", capturing(captured))

    done = (await collect(connect_ws, chat))[-1]

    assert captured["generation_params"]["num_ctx"] == 2048
    assert done["context"]["model_max"] == 2048


async def test_trim_mode_leaves_out_the_oldest_messages_and_says_how_many(
    connect_ws: ConnectWs, chat: models.ChatHistory, db: AsyncSession, monkeypatch: pytest.MonkeyPatch
):
    chat.num_ctx = 4096
    await db.commit()
    await long_history(db, chat, turns=10)
    captured: dict = {}
    monkeypatch.setattr(generation, "astream_bot_response", capturing(captured))

    done = (await collect(connect_ws, chat))[-1]

    assert done["context"]["dropped"] > 0
    assert len(captured["history"]) == 20 - done["context"]["dropped"]
    assert captured["history"][0]["role"] == "user"
    assert captured["history"][-1]["content"].startswith("answer 9")


async def test_summarize_mode_folds_dropped_messages_into_a_saved_summary(
    connect_ws: ConnectWs, chat: models.ChatHistory, db: AsyncSession, monkeypatch: pytest.MonkeyPatch
):
    chat.num_ctx = 4096
    chat.memory_mode = models.ChatHistory.MEMORY_MODE_SUMMARIZE
    await db.commit()
    await long_history(db, chat, turns=10)
    captured: dict = {}
    folded: list = []

    async def summarize(model_name, parameters, previous_summary, messages, num_ctx):
        folded.extend(messages)
        return "The user asked ten long questions."

    monkeypatch.setattr(generation, "asummarize_conversation", summarize)
    monkeypatch.setattr(generation, "astream_bot_response", capturing(captured))

    frames = await collect(connect_ws, chat)
    done = frames[-1]

    assert any("summarized" in f.get("notice", "") for f in frames)
    assert done["context"]["summarized"] is True
    assert "The user asked ten long questions." in captured["system_prompt"]
    assert folded[0]["content"].startswith("question 0")
    assert len(folded) + len(captured["history"]) == 20

    await db.refresh(chat)
    assert chat.memory_summary == "The user asked ten long questions."
    assert chat.memory_summary_until_id == folded[-1]["id"]


async def test_existing_summary_is_reused_without_summarizing_again(
    connect_ws: ConnectWs, chat: models.ChatHistory, db: AsyncSession, monkeypatch: pytest.MonkeyPatch
):
    chat.num_ctx = 4096
    chat.memory_mode = models.ChatHistory.MEMORY_MODE_SUMMARIZE
    await db.commit()
    first_answer = (await long_history(db, chat, turns=3, words=20))[0]
    chat.memory_summary = "Earlier: greetings."
    chat.memory_summary_until_id = first_answer.id
    await db.commit()

    async def must_not_summarize(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("summarized again")

    captured: dict = {}
    monkeypatch.setattr(generation, "asummarize_conversation", must_not_summarize)
    monkeypatch.setattr(generation, "astream_bot_response", capturing(captured))

    done = (await collect(connect_ws, chat))[-1]

    assert done["context"]["summarized"] is True
    assert "Earlier: greetings." in captured["system_prompt"]
    assert captured["history"][0]["content"].startswith("question 1")


async def test_continued_chat_summarizes_its_source_on_the_first_message(
    connect_ws: ConnectWs,
    chat: models.ChatHistory,
    model: models.AIModel,
    db: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
):
    await long_history(db, chat, turns=2, words=10)
    continued = await make_chat(db, model, title="Continued", seed_source_id=chat.id)

    async def summarize(model_name, parameters, previous_summary, messages, num_ctx):
        return f"Summary of {len(messages)} messages."

    captured: dict = {}
    monkeypatch.setattr(generation, "asummarize_conversation", summarize)
    monkeypatch.setattr(generation, "astream_bot_response", capturing(captured))

    frames = await collect(connect_ws, continued)

    assert any("previous chat" in f.get("notice", "") for f in frames)
    assert "Summary of 4 messages." in captured["system_prompt"]
    await db.refresh(continued)
    assert continued.seed_summary == "Summary of 4 messages."


async def test_receive_omits_stats_key_entirely_when_nothing_was_reported(
    connect_ws: ConnectWs, chat: models.ChatHistory, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(generation, "astream_bot_response", streaming("Hi"))

    frames = await collect(connect_ws, chat)

    assert frames[0] == {"message": "Hi", "done": False}
    assert without_context(frames[1]) == {"message": "Hi", "done": True}
    assert "stats" not in frames[1]


async def test_receive_rejects_invalid_json(connect_ws: ConnectWs, chat: models.ChatHistory):
    async with connect_ws(f"/ws/chat/{chat.id}/") as ws:
        await ws.send_text("not json")

        assert await ws.receive_json(timeout=5) == {"error": "Invalid message format.", "done": True}


@pytest.mark.parametrize("payload", ["[1, 2]", '"hello"', "42", "null"])
async def test_a_message_that_is_not_an_object_is_refused_not_a_crash(
    connect_ws: ConnectWs, chat: models.ChatHistory, payload: str
):
    async with connect_ws(f"/ws/chat/{chat.id}/") as ws:
        await ws.send_text(payload)

        assert await ws.receive_json(timeout=5) == {"error": "Invalid message format.", "done": True}


async def test_a_binary_frame_ends_the_conversation(connect_ws: ConnectWs, chat: models.ChatHistory):
    async with connect_ws(f"/ws/chat/{chat.id}/") as ws:
        await ws.send_bytes(b"\x00\x01")

        with pytest.raises(WebSocketDisconnect) as closed:
            await ws.receive_json(timeout=5)

    assert closed.value.code == 1000


async def test_receive_rejects_missing_required_keys(connect_ws: ConnectWs, chat: models.ChatHistory):
    async with connect_ws(f"/ws/chat/{chat.id}/") as ws:
        await ws.send_json({"message": "hi"})
        response = await ws.receive_json(timeout=5)

    assert response["done"] is True
    assert response["error"] == "The message is missing required fields: ai_model, ai_model_parameters."


@pytest.mark.parametrize(
    ("fields", "error"),
    [
        ({"message": ""}, "Message must be a non-empty string."),
        ({"message": 5}, "Message must be a non-empty string."),
        ({"message": "x" * 10_001}, "Message too long. Maximum length is 10000 characters."),
        ({"ai_model": 5}, "Invalid model or chat history."),
        ({"ai_model_parameters": 8}, "ai_model_parameters must be a string."),
        ({"image": "not-a-data-uri"}, "Invalid image format. Expected a base64 data URI."),
        ({"image": "data:image/png;base64,"}, "Invalid image format. Expected a base64 data URI."),
        ({"image": "data:image/png;base64,@@@"}, "Invalid image data — could not decode base64."),
        ({"structured_output": "nope"}, "structured_output must be a list."),
        ({"structured_output": ["nope"]}, "structured_output[0] must be an object."),
        (
            {"structured_output": [{"type": "string"}]},
            "structured_output[0].field must be a non-empty string.",
        ),
        ({"structured_output": [{"field": "a"}]}, "structured_output[0].type must be a non-empty string."),
        (
            {"structured_output": [{"field": "a", "type": "string", "description": "x" * 501}]},
            "structured_output[0].description must be a string of at most 500 characters.",
        ),
        (
            {"structured_output": [{"field": "a", "type": "array", "arrayType": 3}]},
            "structured_output[0].arrayType must be a string.",
        ),
    ],
)
async def test_a_malformed_prompt_is_refused_in_plain_words(
    connect_ws: ConnectWs, chat: models.ChatHistory, db: AsyncSession, fields: dict, error: str
):
    async with connect_ws(f"/ws/chat/{chat.id}/") as ws:
        await ws.send_json({**PROMPT, **fields})

        assert await ws.receive_json(timeout=5) == {"error": error, "done": True}

    assert await all_messages(db, chat) == []


async def test_an_image_over_the_size_limit_is_refused(connect_ws: ConnectWs, chat: models.ChatHistory):
    oversized = "data:image/png;base64," + base64.b64encode(b"\0" * (5 * 1024 * 1024 + 1)).decode()

    async with connect_ws(f"/ws/chat/{chat.id}/") as ws:
        await ws.send_json({**PROMPT, "image": oversized})

        assert await ws.receive_json(timeout=5) == {
            "error": "Image exceeds the maximum allowed size of 5 MB.",
            "done": True,
        }


async def test_a_small_image_is_passed_on_to_the_model_and_saved(
    connect_ws: ConnectWs, chat: models.ChatHistory, db: AsyncSession, monkeypatch: pytest.MonkeyPatch
):
    image = "data:image/png;base64," + base64.b64encode(b"tiny").decode()
    seen: list[str] = []

    async def fake(model_name, parameters, message, image, *args, **kwargs):
        seen.append(image)
        yield "ok"

    monkeypatch.setattr(generation, "astream_bot_response", fake)

    await collect(connect_ws, chat, image=image)

    assert seen == [image]
    assert (await all_messages(db, chat))[0].image == image


async def test_receive_surfaces_model_unavailable_as_readable_error(
    connect_ws: ConnectWs, chat: models.ChatHistory, db: AsyncSession, monkeypatch: pytest.MonkeyPatch
):
    message = (
        "Model container isn't running or is still pulling. "
        "Check the Models page and try again once it shows 'Running'."
    )
    monkeypatch.setattr(generation, "astream_bot_response", failing(ModelUnavailableError(message)))

    response = (await collect(connect_ws, chat))[-1]

    assert response == {"error": message, "done": True}
    assert await all_messages(db, chat) == []


async def test_failure_during_generation_setup_still_sends_an_error_frame(
    connect_ws: ConnectWs, chat: models.ChatHistory, monkeypatch: pytest.MonkeyPatch
):
    async def broken_retrieval(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("pgvector exploded")

    monkeypatch.setattr(retrieval, "retrieve_context_for_chat", broken_retrieval)

    response = (await collect(connect_ws, chat))[-1]

    assert response == {"error": "An unexpected error occurred. Please try again.", "done": True}


async def test_an_unexpected_failure_while_reading_the_message_leaves_the_socket_usable(
    connect_ws: ConnectWs, chat: models.ChatHistory, monkeypatch: pytest.MonkeyPatch
):
    real = history.get_ai_model
    calls = 0

    async def flaky(session: AsyncSession, value: str) -> models.AIModel | None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("connection reset")
        return await real(session, value)

    monkeypatch.setattr(history, "get_ai_model", flaky)
    monkeypatch.setattr(generation, "astream_bot_response", streaming("Fine"))

    async with connect_ws(f"/ws/chat/{chat.id}/") as ws:
        await ws.send_json(PROMPT)
        assert await ws.receive_json(timeout=5) == {
            "error": "An unexpected error occurred. Please try again.",
            "done": True,
        }

        await ws.send_json(PROMPT)
        assert await ws.receive_json(timeout=5) == {"message": "Fine", "done": False}


async def test_rag_warnings_are_sent_as_notices_before_the_answer(
    connect_ws: ConnectWs, chat: models.ChatHistory, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        retrieval,
        "retrieve_context_for_chat",
        returning(retrieval.RetrievalResult("", [], ["Your documents weren't searched."])),
    )
    monkeypatch.setattr(generation, "astream_bot_response", streaming("Answer"))

    frames = await collect(connect_ws, chat)

    assert frames[0] == {"notice": "Your documents weren't searched.", "done": False}
    assert frames[1] == {"message": "Answer", "done": False}


async def test_citations_ride_along_on_the_done_frame(
    connect_ws: ConnectWs, chat: models.ChatHistory, monkeypatch: pytest.MonkeyPatch
):
    citation = {"document_id": 1, "filename": "notes.pdf", "chunk_index": 2, "page": 3}
    monkeypatch.setattr(
        retrieval,
        "retrieve_context_for_chat",
        returning(retrieval.RetrievalResult("Excerpt", [citation], [])),
    )
    captured: dict = {}
    monkeypatch.setattr(generation, "astream_bot_response", capturing(captured))

    done = (await collect(connect_ws, chat))[-1]

    assert done["citations"] == [citation]
    assert "Excerpt" in captured["system_prompt"]


async def test_receive_unknown_model_reports_invalid_instead_of_500(
    connect_ws: ConnectWs, chat: models.ChatHistory
):
    response = (await collect(connect_ws, chat, ai_model="does-not-exist"))[-1]

    assert response == {"error": "Invalid model or chat history.", "done": True}


async def test_a_chat_that_does_not_exist_is_reported_not_crashed(
    connect_ws: ConnectWs, chat: models.ChatHistory
):
    async with connect_ws(f"/ws/chat/{chat.id + 99}/") as ws:
        await ws.send_json(PROMPT)

        assert await ws.receive_json(timeout=5) == {"error": "Invalid model or chat history.", "done": True}


async def test_a_chat_that_belongs_to_another_model_is_not_reachable(
    connect_ws: ConnectWs, chat: models.ChatHistory, db: AsyncSession
):
    await make_model(db, "other-model")

    async with connect_ws(f"/ws/chat/{chat.id}/") as ws:
        await ws.send_json({**PROMPT, "ai_model": "other-model"})

        assert await ws.receive_json(timeout=5) == {"error": "Invalid model or chat history.", "done": True}


async def test_a_non_numeric_chat_id_is_not_a_chat_socket(connect_ws: ConnectWs):
    with pytest.RaisesGroup(WebSocketDisconnect):
        async with connect_ws("/ws/chat/abc/"):
            pass


async def test_stop_cancels_in_flight_generation_and_persists_partial_response(
    connect_ws: ConnectWs, chat: models.ChatHistory, db: AsyncSession, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(generation, "astream_bot_response", hanging_after("Hel", asyncio.Event()))

    async with connect_ws(f"/ws/chat/{chat.id}/") as ws:
        await ws.send_json(PROMPT)
        assert await ws.receive_json(timeout=5) == {"message": "Hel", "done": False}

        await ws.send_json({"type": "stop"})
        response = await ws.receive_json(timeout=5)

    assert without_context(response) == {"message": "", "done": True, "stopped": True}
    messages = await all_messages(db, chat)
    assert [(m.role, m.content) for m in messages] == [("user", "Hi"), ("assistant", "Hel")]


async def test_stopping_twice_saves_the_partial_reply_once(
    connect_ws: ConnectWs, chat: models.ChatHistory, db: AsyncSession, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(generation, "astream_bot_response", hanging_after("Hel", asyncio.Event()))

    async with connect_ws(f"/ws/chat/{chat.id}/") as ws:
        await ws.send_json(PROMPT)
        await ws.receive_json(timeout=5)

        await ws.send_json({"type": "stop"})
        await ws.send_json({"type": "stop"})
        assert (await ws.receive_json(timeout=5))["stopped"] is True
        with pytest.raises(TimeoutError):
            await ws.receive_json(timeout=0.2)

    assert [m.content for m in await all_messages(db, chat)] == ["Hi", "Hel"]


async def test_stop_with_no_active_generation_is_a_noop(connect_ws: ConnectWs, chat: models.ChatHistory):
    async with connect_ws(f"/ws/chat/{chat.id}/") as ws:
        await ws.send_json({"type": "stop"})

        with pytest.raises(TimeoutError):
            await ws.receive_json(timeout=0.2)


async def test_closing_the_tab_mid_reply_stops_the_model_and_keeps_what_was_written(
    connect_ws: ConnectWs, chat: models.ChatHistory, db: AsyncSession, monkeypatch: pytest.MonkeyPatch
):
    stopped = asyncio.Event()

    async def fake(*args: Any, **kwargs: Any) -> AsyncIterator[str]:
        try:
            yield "Hel"
            await asyncio.Event().wait()
        finally:
            stopped.set()

    monkeypatch.setattr(generation, "astream_bot_response", fake)

    async with connect_ws(f"/ws/chat/{chat.id}/") as ws:
        await ws.send_json(PROMPT)
        assert await ws.receive_json(timeout=5) == {"message": "Hel", "done": False}

    await asyncio.wait_for(stopped.wait(), timeout=5)
    for _ in range(100):
        messages = await all_messages(db, chat)
        if len(messages) == 2:
            break
        await asyncio.sleep(0.05)
    assert [(m.role, m.content) for m in messages] == [("user", "Hi"), ("assistant", "Hel")]


async def test_prompt_while_generating_is_rejected_instead_of_queued(
    connect_ws: ConnectWs, chat: models.ChatHistory, monkeypatch: pytest.MonkeyPatch
):
    release = asyncio.Event()
    monkeypatch.setattr(generation, "astream_bot_response", hanging_after("Hel", release))

    async with connect_ws(f"/ws/chat/{chat.id}/") as ws:
        await ws.send_json(PROMPT)
        assert await ws.receive_json(timeout=5) == {"message": "Hel", "done": False}

        await ws.send_json({**PROMPT, "message": "Again"})
        assert await ws.receive_json(timeout=5) == {
            "error": "A response is already generating. Stop it first.",
            "done": True,
        }

        release.set()
        assert without_context(await ws.receive_json(timeout=5)) == {"message": "Hel", "done": True}


async def test_receive_unknown_message_type_reports_error(connect_ws: ConnectWs, chat: models.ChatHistory):
    async with connect_ws(f"/ws/chat/{chat.id}/") as ws:
        await ws.send_json({"type": "bogus"})
        response = await ws.receive_json(timeout=5)

    assert response == {"error": "Unknown message type: 'bogus'", "done": True}


async def test_a_message_type_that_is_not_text_is_unknown_too(
    connect_ws: ConnectWs, chat: models.ChatHistory
):
    async with connect_ws(f"/ws/chat/{chat.id}/") as ws:
        await ws.send_json({"type": ["prompt"]})
        response = await ws.receive_json(timeout=5)

    assert response["done"] is True
    assert "Unknown message type" in response["error"]


async def send_first_turn(ws: Any, monkeypatch: pytest.MonkeyPatch, reply: str = "Hello") -> None:
    monkeypatch.setattr(generation, "astream_bot_response", streaming(reply))
    await ws.send_json(PROMPT)
    assert await ws.receive_json(timeout=5) == {"message": reply, "done": False}
    assert without_context(await ws.receive_json(timeout=5)) == {"message": reply, "done": True}


async def test_regenerate_replaces_last_assistant_message_and_keeps_user_message(
    connect_ws: ConnectWs, chat: models.ChatHistory, db: AsyncSession, monkeypatch: pytest.MonkeyPatch
):
    async with connect_ws(f"/ws/chat/{chat.id}/") as ws:
        await send_first_turn(ws, monkeypatch)

        async def regenerate(model_name, parameters, message, image, history, *args, **kwargs):
            # The regenerated turn's history must exclude the answer being
            # replaced, or the model would see its own stale reply as context.
            assert history == []
            assert message == "Hi"
            yield "Better hello"

        monkeypatch.setattr(generation, "astream_bot_response", regenerate)

        await ws.send_json({"type": "regenerate", "ai_model": "llama3.1", "ai_model_parameters": "8b"})
        assert await ws.receive_json(timeout=5) == {"message": "Better hello", "done": False}
        assert without_context(await ws.receive_json(timeout=5)) == {"message": "Better hello", "done": True}

    assert await active_path(db, chat) == [("user", "Hi"), ("assistant", "Better hello")]
    # The old reply is a sibling branch, not deleted — regenerate must never
    # destroy data, only add to it.
    assert [(m.role, m.content) for m in await all_messages(db, chat)] == [
        ("user", "Hi"),
        ("assistant", "Hello"),
        ("assistant", "Better hello"),
    ]


async def test_regenerate_with_nothing_to_regenerate_reports_error(
    connect_ws: ConnectWs, chat: models.ChatHistory
):
    async with connect_ws(f"/ws/chat/{chat.id}/") as ws:
        await ws.send_json({"type": "regenerate", "ai_model": "llama3.1", "ai_model_parameters": "8b"})
        response = await ws.receive_json(timeout=5)

    assert response == {"error": "Nothing to regenerate yet — send a message first.", "done": True}


async def test_regenerate_failure_leaves_original_assistant_message_intact(
    connect_ws: ConnectWs, chat: models.ChatHistory, db: AsyncSession, monkeypatch: pytest.MonkeyPatch
):
    async with connect_ws(f"/ws/chat/{chat.id}/") as ws:
        await send_first_turn(ws, monkeypatch)
        monkeypatch.setattr(
            generation,
            "astream_bot_response",
            failing(ModelUnavailableError("Model container isn't running or is still pulling.")),
        )

        await ws.send_json({"type": "regenerate", "ai_model": "llama3.1", "ai_model_parameters": "8b"})
        response = await ws.receive_json(timeout=5)

    assert response["done"] is True
    assert "error" in response
    assert [(m.role, m.content) for m in await all_messages(db, chat)] == [
        ("user", "Hi"),
        ("assistant", "Hello"),
    ]


async def test_regenerate_while_generating_is_rejected(
    connect_ws: ConnectWs, chat: models.ChatHistory, monkeypatch: pytest.MonkeyPatch
):
    release = asyncio.Event()
    monkeypatch.setattr(generation, "astream_bot_response", hanging_after("Hel", release))

    async with connect_ws(f"/ws/chat/{chat.id}/") as ws:
        await ws.send_json(PROMPT)
        assert await ws.receive_json(timeout=5) == {"message": "Hel", "done": False}

        await ws.send_json({"type": "regenerate", "ai_model": "llama3.1", "ai_model_parameters": "8b"})
        assert await ws.receive_json(timeout=5) == {
            "error": "A response is already generating. Stop it first.",
            "done": True,
        }

        release.set()
        assert without_context(await ws.receive_json(timeout=5)) == {"message": "Hel", "done": True}


async def test_edit_resend_replaces_edited_message_and_discards_everything_after_it(
    connect_ws: ConnectWs, chat: models.ChatHistory, db: AsyncSession, monkeypatch: pytest.MonkeyPatch
):
    async with connect_ws(f"/ws/chat/{chat.id}/") as ws:
        await send_first_turn(ws, monkeypatch)

        monkeypatch.setattr(generation, "astream_bot_response", streaming("I'm good"))
        await ws.send_json({**PROMPT, "message": "How are you"})
        assert await ws.receive_json(timeout=5) == {"message": "I'm good", "done": False}
        assert without_context(await ws.receive_json(timeout=5)) == {"message": "I'm good", "done": True}

        async def edit(model_name, parameters, message, image, history, *args, **kwargs):
            # History must stop before the edited message — the model shouldn't
            # see the stale exchange this edit is replacing, or the turn it
            # invalidated.
            assert history == []
            assert message == "Hey there"
            yield "Hi yourself"

        monkeypatch.setattr(generation, "astream_bot_response", edit)

        await ws.send_json(
            {
                "type": "edit_resend",
                "index": 0,
                "message": "Hey there",
                "ai_model": "llama3.1",
                "ai_model_parameters": "8b",
            }
        )
        assert await ws.receive_json(timeout=5) == {"message": "Hi yourself", "done": False}
        assert without_context(await ws.receive_json(timeout=5)) == {"message": "Hi yourself", "done": True}

    assert await active_path(db, chat) == [("user", "Hey there"), ("assistant", "Hi yourself")]
    # The entire original branch survives as an inactive branch — edit-resend
    # must never destroy data either.
    assert {(m.role, m.content) for m in await all_messages(db, chat)} == {
        ("user", "Hi"),
        ("assistant", "Hello"),
        ("user", "How are you"),
        ("assistant", "I'm good"),
        ("user", "Hey there"),
        ("assistant", "Hi yourself"),
    }


@pytest.mark.parametrize("index", [1, 5, -1])
async def test_edit_resend_on_assistant_or_missing_index_reports_error(
    connect_ws: ConnectWs, chat: models.ChatHistory, monkeypatch: pytest.MonkeyPatch, index: int
):
    async with connect_ws(f"/ws/chat/{chat.id}/") as ws:
        await send_first_turn(ws, monkeypatch)

        await ws.send_json(
            {
                "type": "edit_resend",
                "index": index,
                "message": "x",
                "ai_model": "llama3.1",
                "ai_model_parameters": "8b",
            }
        )
        response = await ws.receive_json(timeout=5)

    assert response == {
        "error": "That message can no longer be edited — the conversation may have changed.",
        "done": True,
    }


@pytest.mark.parametrize("index", ["0", 1.5, True, None])
async def test_edit_resend_needs_a_whole_number_index(
    connect_ws: ConnectWs, chat: models.ChatHistory, index: object
):
    async with connect_ws(f"/ws/chat/{chat.id}/") as ws:
        await ws.send_json(
            {
                "type": "edit_resend",
                "index": index,
                "message": "x",
                "ai_model": "llama3.1",
                "ai_model_parameters": "8b",
            }
        )

        assert await ws.receive_json(timeout=5) == {"error": "index must be an integer.", "done": True}


async def test_edit_resend_failure_leaves_original_tail_intact(
    connect_ws: ConnectWs, chat: models.ChatHistory, db: AsyncSession, monkeypatch: pytest.MonkeyPatch
):
    async with connect_ws(f"/ws/chat/{chat.id}/") as ws:
        await send_first_turn(ws, monkeypatch)
        monkeypatch.setattr(
            generation,
            "astream_bot_response",
            failing(ModelUnavailableError("Model container isn't running or is still pulling.")),
        )

        await ws.send_json(
            {
                "type": "edit_resend",
                "index": 0,
                "message": "Hey there",
                "ai_model": "llama3.1",
                "ai_model_parameters": "8b",
            }
        )
        response = await ws.receive_json(timeout=5)

    assert response["done"] is True
    assert "error" in response
    assert [(m.role, m.content) for m in await all_messages(db, chat)] == [
        ("user", "Hi"),
        ("assistant", "Hello"),
    ]


async def test_edit_resend_while_generating_is_rejected(
    connect_ws: ConnectWs, chat: models.ChatHistory, monkeypatch: pytest.MonkeyPatch
):
    release = asyncio.Event()
    monkeypatch.setattr(generation, "astream_bot_response", hanging_after("Hel", release))

    async with connect_ws(f"/ws/chat/{chat.id}/") as ws:
        await ws.send_json(PROMPT)
        assert await ws.receive_json(timeout=5) == {"message": "Hel", "done": False}

        await ws.send_json(
            {
                "type": "edit_resend",
                "index": 0,
                "message": "Hey",
                "ai_model": "llama3.1",
                "ai_model_parameters": "8b",
            }
        )
        assert await ws.receive_json(timeout=5) == {
            "error": "A response is already generating. Stop it first.",
            "done": True,
        }

        release.set()
        assert without_context(await ws.receive_json(timeout=5)) == {"message": "Hel", "done": True}


async def test_edit_resend_missing_required_keys_reports_error(
    connect_ws: ConnectWs, chat: models.ChatHistory
):
    async with connect_ws(f"/ws/chat/{chat.id}/") as ws:
        await ws.send_json({"type": "edit_resend", "index": 0})
        response = await ws.receive_json(timeout=5)

    assert response["done"] is True
    assert (
        response["error"] == "The message is missing required fields: message, ai_model, ai_model_parameters."
    )


async def test_receive_uses_tool_path_when_chat_and_model_both_allow_it(
    connect_ws: ConnectWs, tools_chat: models.ChatHistory, db: AsyncSession, monkeypatch: pytest.MonkeyPatch
):
    call = {"name": "calculator", "args": {"expression": "2+2"}, "result": "4"}
    monkeypatch.setattr(generation, "astream_tool_response", streaming(call, "The answer is 4."))
    monkeypatch.setattr(generation, "astream_bot_response", failing(AssertionError("plain path used")))

    async with connect_ws(f"/ws/chat/{tools_chat.id}/") as ws:
        await ws.send_json({**PROMPT, "message": "What is 2+2?", "ai_model": "llama3.1-tools-test"})

        assert await ws.receive_json(timeout=5) == {"tool_call": call, "done": False}
        assert await ws.receive_json(timeout=5) == {"message": "The answer is 4.", "done": False}
        assert without_context(await ws.receive_json(timeout=5)) == {
            "message": "The answer is 4.",
            "done": True,
        }

    assert [m.content for m in await all_messages(db, tools_chat)] == ["What is 2+2?", "The answer is 4."]


async def test_only_enabled_tool_servers_are_handed_to_the_model(
    connect_ws: ConnectWs, tools_chat: models.ChatHistory, db: AsyncSession, monkeypatch: pytest.MonkeyPatch
):
    db.add_all(
        [
            models.MCPServer(name="On", transport="stdio", command="x", enabled=True),
            models.MCPServer(name="Off", transport="stdio", command="x", enabled=False),
        ]
    )
    await db.commit()
    seen: list[str] = []

    async def fake(
        model_name, parameters, message, image, history, system_prompt, params, stats, mcp_servers, *rest
    ):
        seen.extend(server.name for server in mcp_servers)
        yield "ok"

    monkeypatch.setattr(generation, "astream_tool_response", fake)

    await collect(connect_ws, tools_chat, ai_model="llama3.1-tools-test")

    assert seen == ["On"]


async def test_receive_ignores_tools_enabled_when_model_lacks_tools_capability(
    connect_ws: ConnectWs, chat: models.ChatHistory, db: AsyncSession, monkeypatch: pytest.MonkeyPatch
):
    # The model here has can_use_tools=False — turning tools_enabled on for
    # the chat alone must not be enough.
    chat.tools_enabled = True
    await db.commit()
    monkeypatch.setattr(generation, "astream_bot_response", streaming("plain reply"))
    monkeypatch.setattr(generation, "astream_tool_response", failing(AssertionError("tool path used")))

    frames = await collect(connect_ws, chat)

    assert frames[0] == {"message": "plain reply", "done": False}
    assert without_context(frames[1]) == {"message": "plain reply", "done": True}


async def test_receive_structured_output_takes_priority_over_tools(
    connect_ws: ConnectWs, tools_chat: models.ChatHistory, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(generation, "astream_structured_bot_response", returning('```json\n{"a": 1}\n```'))
    monkeypatch.setattr(generation, "astream_tool_response", failing(AssertionError("tool path used")))

    response = (
        await collect(
            connect_ws,
            tools_chat,
            ai_model="llama3.1-tools-test",
            structured_output=[{"field": "a", "type": "number"}],
        )
    )[-1]

    assert response["done"] is True
    assert response["message"] == '```json\n{"a": 1}\n```'


async def test_an_unparseable_structured_reply_is_reported_and_not_saved(
    connect_ws: ConnectWs, chat: models.ChatHistory, db: AsyncSession, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(generation, "astream_structured_bot_response", returning(None))

    response = (await collect(connect_ws, chat, structured_output=[{"field": "a", "type": "number"}]))[-1]

    assert response == {
        "error": "Unable to parse the structured response. Try again or adjust the schema.",
        "done": True,
    }
    assert await all_messages(db, chat) == []


async def test_an_empty_reply_is_an_error_not_a_saved_blank_message(
    connect_ws: ConnectWs, chat: models.ChatHistory, db: AsyncSession, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(generation, "astream_bot_response", streaming())

    response = (await collect(connect_ws, chat))[-1]

    assert response == {"error": "The model returned an empty response. Try again.", "done": True}
    assert await all_messages(db, chat) == []


async def test_done_frame_reports_how_much_of_the_model_fits_on_the_gpu(
    connect_ws: ConnectWs, chat: models.ChatHistory, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(runtime, "get_gpu_share", returning(0.6))
    monkeypatch.setattr(generation, "astream_bot_response", capturing({}))

    done = (await collect(connect_ws, chat))[-1]

    assert done["context"]["gpu_share"] == 0.6


async def test_thinking_is_streamed_separately_and_saved_with_the_reply(
    connect_ws: ConnectWs, chat: models.ChatHistory, db: AsyncSession, monkeypatch: pytest.MonkeyPatch
):
    chat.think_level = "on"
    await db.commit()
    monkeypatch.setattr(
        runtime,
        "get_model_info",
        returning(
            {
                "context_length": 8192,
                "thinking": {"values": [False, True], "default": True},
                "capabilities": [],
            }
        ),
    )
    received: dict = {}

    async def fake(
        model_name, parameters, message, image, history, system_prompt, params, stats, reasoning=None
    ):
        received["reasoning"] = reasoning
        yield generation.ThinkingDelta("Hmm, ")
        yield generation.ThinkingDelta("easy.")
        yield "4"

    monkeypatch.setattr(generation, "astream_bot_response", fake)

    frames = await collect(connect_ws, chat)

    assert frames[0] == {"thinking": "Hmm, ", "done": False}
    assert frames[1] == {"thinking": "easy.", "done": False}
    assert frames[2] == {"message": "4", "done": False}
    assert frames[-1]["message"] == "4"
    assert received["reasoning"] is True

    reply = (await all_messages(db, chat))[-1]
    assert (reply.content, reply.thinking) == ("4", "Hmm, easy.")


async def test_tool_approval_round_trip_over_the_socket(
    connect_ws: ConnectWs, tools_chat: models.ChatHistory, monkeypatch: pytest.MonkeyPatch
):
    decisions: list[bool] = []

    async def fake(*args: Any, **kwargs: Any) -> AsyncIterator[str]:
        approve = args[-1]
        decisions.append(await approve({"name": "send_email", "server": "Mail", "args": {"to": "a@b.c"}}))
        yield "ok"

    monkeypatch.setattr(generation, "astream_tool_response", fake)

    async with connect_ws(f"/ws/chat/{tools_chat.id}/") as ws:
        await ws.send_json({**PROMPT, "message": "mail it", "ai_model": "llama3.1-tools-test"})

        request = await ws.receive_json(timeout=5)
        assert request["tool_approval"]["name"] == "send_email"
        assert request["tool_approval"]["server"] == "Mail"
        assert request["done"] is False

        await ws.send_json({"type": "tool_approval", "id": request["tool_approval"]["id"], "approved": False})

        assert await ws.receive_json(timeout=5) == {"message": "ok", "done": False}
    assert decisions == [False]


async def test_an_approval_is_granted_only_by_an_explicit_true(
    connect_ws: ConnectWs, tools_chat: models.ChatHistory, monkeypatch: pytest.MonkeyPatch
):
    decisions: list[bool] = []

    async def fake(*args: Any, **kwargs: Any) -> AsyncIterator[str]:
        approve = args[-1]
        decisions.append(await approve({"name": "t", "server": "S", "args": {}}))
        decisions.append(await approve({"name": "t", "server": "S", "args": {}}))
        yield "ok"

    monkeypatch.setattr(generation, "astream_tool_response", fake)

    async with connect_ws(f"/ws/chat/{tools_chat.id}/") as ws:
        await ws.send_json({**PROMPT, "ai_model": "llama3.1-tools-test"})

        first = await ws.receive_json(timeout=5)
        await ws.send_json({"type": "tool_approval", "id": first["tool_approval"]["id"], "approved": "yes"})
        second = await ws.receive_json(timeout=5)
        await ws.send_json({"type": "tool_approval", "id": second["tool_approval"]["id"], "approved": True})
        assert await ws.receive_json(timeout=5) == {"message": "ok", "done": False}

    assert decisions == [False, True]


async def test_an_unanswered_approval_counts_as_denied(
    connect_ws: ConnectWs, tools_chat: models.ChatHistory, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(chat_socket, "TOOL_APPROVAL_TIMEOUT_SECONDS", 0.05)
    decisions: list[bool] = []

    async def fake(*args: Any, **kwargs: Any) -> AsyncIterator[str]:
        decisions.append(await args[-1]({"name": "t", "server": "S", "args": {}}))
        yield "moved on"

    monkeypatch.setattr(generation, "astream_tool_response", fake)

    async with connect_ws(f"/ws/chat/{tools_chat.id}/") as ws:
        await ws.send_json({**PROMPT, "ai_model": "llama3.1-tools-test"})

        assert "tool_approval" in await ws.receive_json(timeout=5)
        assert await ws.receive_json(timeout=5) == {"message": "moved on", "done": False}
    assert decisions == [False]


async def test_an_answer_for_an_unknown_approval_is_ignored(connect_ws: ConnectWs, chat: models.ChatHistory):
    async with connect_ws(f"/ws/chat/{chat.id}/") as ws:
        await ws.send_json({"type": "tool_approval", "id": "nope", "approved": True})

        with pytest.raises(TimeoutError):
            await ws.receive_json(timeout=0.2)


async def test_first_reply_in_an_untitled_chat_gets_a_generated_title(
    connect_ws: ConnectWs, chat: models.ChatHistory, db: AsyncSession, monkeypatch: pytest.MonkeyPatch
):
    chat.title = models.DEFAULT_CHAT_TITLE
    await db.commit()
    monkeypatch.setattr(generation, "agenerate_title", returning("Greeting the assistant"))
    monkeypatch.setattr(generation, "astream_bot_response", capturing({}))

    async with connect_ws(f"/ws/chat/{chat.id}/") as ws:
        await ws.send_json(PROMPT)
        while not (await ws.receive_json(timeout=5)).get("done"):
            pass

        assert await ws.receive_json(timeout=5) == {"chat_title": "Greeting the assistant"}

    await db.refresh(chat)
    assert chat.title == "Greeting the assistant"


async def test_a_failing_title_costs_nothing_but_the_title(
    connect_ws: ConnectWs, chat: models.ChatHistory, db: AsyncSession, monkeypatch: pytest.MonkeyPatch
):
    chat.title = models.DEFAULT_CHAT_TITLE
    await db.commit()

    async def broken(*args: Any) -> str:
        raise RuntimeError("model went away")

    monkeypatch.setattr(generation, "agenerate_title", broken)
    monkeypatch.setattr(generation, "astream_bot_response", capturing({}))

    async with connect_ws(f"/ws/chat/{chat.id}/") as ws:
        await ws.send_json(PROMPT)
        while not (await ws.receive_json(timeout=5)).get("done"):
            pass

        with pytest.raises(TimeoutError):
            await ws.receive_json(timeout=0.3)

    await db.refresh(chat)
    assert chat.title == models.DEFAULT_CHAT_TITLE
    assert len(await all_messages(db, chat)) == 2


async def test_titled_chats_are_not_renamed(
    connect_ws: ConnectWs, chat: models.ChatHistory, monkeypatch: pytest.MonkeyPatch
):
    async def must_not_title(*args: Any) -> str:
        raise AssertionError("titled again")

    monkeypatch.setattr(generation, "agenerate_title", must_not_title)
    monkeypatch.setattr(generation, "astream_bot_response", capturing({}))

    frames = await collect(connect_ws, chat)

    assert frames[-1]["done"] is True


async def test_later_exchanges_are_not_titled(
    connect_ws: ConnectWs, chat: models.ChatHistory, db: AsyncSession, monkeypatch: pytest.MonkeyPatch
):
    chat.title = models.DEFAULT_CHAT_TITLE
    await db.commit()
    await make_exchange(db, chat, "Earlier", "Answer")

    async def must_not_title(*args: Any) -> str:
        raise AssertionError("titled again")

    monkeypatch.setattr(generation, "agenerate_title", must_not_title)
    monkeypatch.setattr(generation, "astream_bot_response", capturing({}))

    assert (await collect(connect_ws, chat))[-1]["done"] is True


async def test_replies_record_which_model_wrote_them(
    connect_ws: ConnectWs, chat: models.ChatHistory, db: AsyncSession, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(generation, "astream_bot_response", capturing({}))

    await collect(connect_ws, chat)

    assert (await all_messages(db, chat))[-1].model_label == "llama3.1:8b"


async def test_tools_follow_what_the_installed_model_reports_over_the_catalog(
    connect_ws: ConnectWs, chat: models.ChatHistory, db: AsyncSession, monkeypatch: pytest.MonkeyPatch
):
    # The catalog said no tools (this fixture's model), but the model says yes.
    chat.tools_enabled = True
    await db.commit()
    monkeypatch.setattr(
        runtime,
        "get_model_info",
        returning({"context_length": 8192, "thinking": None, "capabilities": ["tools"]}),
    )
    used: list[str] = []

    async def fake(*args: Any, **kwargs: Any) -> AsyncIterator[str]:
        used.append("tools")
        yield "ok"

    monkeypatch.setattr(generation, "astream_tool_response", fake)

    await collect(connect_ws, chat)

    assert used == ["tools"]


async def test_two_chats_can_generate_at_the_same_time(
    connect_ws: ConnectWs,
    chat: models.ChatHistory,
    model: models.AIModel,
    db: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
):
    other = await make_chat(db, model, title="Other")
    release = asyncio.Event()
    monkeypatch.setattr(generation, "astream_bot_response", hanging_after("Hel", release))

    async with connect_ws(f"/ws/chat/{chat.id}/") as first, connect_ws(f"/ws/chat/{other.id}/") as second:
        await first.send_json(PROMPT)
        await second.send_json(PROMPT)
        assert await first.receive_json(timeout=5) == {"message": "Hel", "done": False}
        assert await second.receive_json(timeout=5) == {"message": "Hel", "done": False}

        release.set()
        assert (await first.receive_json(timeout=5))["done"] is True
        assert (await second.receive_json(timeout=5))["done"] is True

    assert len(await all_messages(db, chat)) == 2
    assert len(await all_messages(db, other)) == 2
