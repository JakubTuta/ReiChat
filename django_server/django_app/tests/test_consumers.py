"""ChatConsumer over a real (in-memory) channel layer via Channels'
WebsocketCommunicator — verified against the Channels testing docs
(channels.testing.WebsocketCommunicator, database_sync_to_async).

`functions.astream_bot_response` is monkeypatched everywhere here so these
tests never touch Ollama or Docker; they exercise the consumer's own
protocol handling, error surfacing, and history persistence.
"""

from __future__ import annotations

import asyncio

import pytest
from channels.db import database_sync_to_async
from channels.routing import URLRouter
from channels.testing import WebsocketCommunicator
from django.urls import re_path

from django_app import functions, models
from django_app.consumers import ChatConsumer
from django_app.functions import ModelUnavailableError

pytestmark = [pytest.mark.django_db(transaction=True), pytest.mark.asyncio]


def _application():
    return URLRouter([re_path(r"ws/chat/(?P<chat_id>\d+)/$", ChatConsumer.as_asgi())])


@pytest.fixture(autouse=True)
def _in_memory_channel_layer(settings):
    # The real deployment uses channels_redis; these tests run in-process
    # against whatever channel layer settings.CHANNEL_LAYERS points at,
    # and Redis isn't reachable from outside the Docker network here.
    settings.CHANNEL_LAYERS = {"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}}


@pytest.fixture(autouse=True)
async def _close_leaked_db_connection():
    # `database_sync_to_async` is thread_sensitive: outside a real ASGI
    # request cycle it funnels every call onto one shared worker thread for
    # the life of the process, and nothing ever tells Django to close the
    # connection that thread opens — a real production concern too (the
    # Channels docs call out periodic `aclose_old_connections` for exactly
    # this in long-lived consumers). Left alone here, it holds test_chatbot
    # open forever and the next `--create-db` run hangs waiting on DROP
    # DATABASE. One connection total (not one per test), so closing it once
    # per test is cheap and keeps the suite re-runnable.
    yield
    from django.db import connections

    # close_old_connections() only closes connections past CONN_MAX_AGE
    # (600s in settings) or already broken — a young idle one survives it.
    # close_all() is unconditional, which is what a test teardown needs.
    await database_sync_to_async(connections.close_all)()


@pytest.fixture(autouse=True)
def _no_model_context_lookup(monkeypatch):
    # The real lookup asks the model's Ollama container for its maximum
    # context; there's no container here. Tests that care override this.
    monkeypatch.setattr(functions, "get_model_info", lambda model_name, parameters: None)
    monkeypatch.setattr(functions, "get_gpu_share", lambda model_name, parameters: None)


def _without_context(frame: dict) -> dict:
    """Done frames always carry a context report (covered by its own
    tests below) — strip it where a test is about something else."""
    return {k: v for k, v in frame.items() if k != "context"}


@pytest.fixture
def chat():
    # The 0004_seed_model_catalog migration pre-populates a real "llama3.1"
    # row into every fresh test database — clear it so this fixture's own
    # row is the only one and IntegrityError doesn't depend on what the
    # seed catalog happens to contain.
    models.AIModel.objects.filter(model="llama3.1").delete()
    ai_model = models.AIModel.objects.create(name="llama3.1", model="llama3.1", index=1)
    return models.ChatHistory.objects.create(ai_model=ai_model, title="Test chat")


@pytest.fixture
def tools_chat():
    models.AIModel.objects.filter(model="llama3.1-tools-test").delete()
    ai_model = models.AIModel.objects.create(
        name="llama3.1", model="llama3.1-tools-test", can_use_tools=True, index=1
    )
    return models.ChatHistory.objects.create(ai_model=ai_model, title="Tools chat", tools_enabled=True)


async def test_receive_streams_response_and_persists_history(chat, monkeypatch):
    async def _fake_astream(model, parameters, message, image, history, system_prompt=None, generation_params=None, stats=None, reasoning=None):
        for chunk in ("Hel", "lo"):
            yield chunk

    monkeypatch.setattr(functions, "astream_bot_response", _fake_astream)

    communicator = WebsocketCommunicator(_application(), f"/ws/chat/{chat.id}/")
    connected, _ = await communicator.connect()
    assert connected

    await communicator.send_json_to(
        {"message": "Hi", "ai_model": "llama3.1", "ai_model_parameters": "8b"}
    )

    assert await communicator.receive_json_from() == {"message": "Hel", "done": False}
    assert await communicator.receive_json_from() == {"message": "lo", "done": False}
    assert _without_context(await communicator.receive_json_from()) == {"message": "Hello", "done": True}

    await communicator.disconnect()

    messages = await database_sync_to_async(list)(
        models.ChatMessage.objects.filter(chat=chat).order_by("created_at")
    )
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[0].content == "Hi"
    assert messages[1].content == "Hello"


async def test_receive_passes_chat_persona_system_prompt_to_astream(chat, monkeypatch):
    persona = await database_sync_to_async(models.Persona.objects.create)(
        name="Pirate", system_prompt="Talk like a pirate."
    )
    chat.persona = persona
    await database_sync_to_async(chat.save)(update_fields=["persona"])

    received_system_prompt = None

    async def _fake_astream(model, parameters, message, image, history, system_prompt=None, generation_params=None, stats=None, reasoning=None):
        nonlocal received_system_prompt
        received_system_prompt = system_prompt
        yield "Arr"

    monkeypatch.setattr(functions, "astream_bot_response", _fake_astream)

    communicator = WebsocketCommunicator(_application(), f"/ws/chat/{chat.id}/")
    connected, _ = await communicator.connect()
    assert connected

    await communicator.send_json_to(
        {"message": "Ahoy", "ai_model": "llama3.1", "ai_model_parameters": "8b"}
    )

    assert await communicator.receive_json_from() == {"message": "Arr", "done": False}
    assert _without_context(await communicator.receive_json_from()) == {"message": "Arr", "done": True}

    await communicator.disconnect()

    assert received_system_prompt == "Talk like a pirate."


async def test_receive_passes_chat_generation_params_to_astream(chat, monkeypatch):
    chat.temperature = 0.9
    chat.num_ctx = 8192
    chat.top_p = 0.5
    chat.seed = 7
    await database_sync_to_async(chat.save)(update_fields=["temperature", "num_ctx", "top_p", "seed"])

    received_params = None

    async def _fake_astream(model, parameters, message, image, history, system_prompt=None, generation_params=None, stats=None, reasoning=None):
        nonlocal received_params
        received_params = generation_params
        yield "ok"

    monkeypatch.setattr(functions, "astream_bot_response", _fake_astream)

    communicator = WebsocketCommunicator(_application(), f"/ws/chat/{chat.id}/")
    assert (await communicator.connect())[0]

    await communicator.send_json_to(
        {"message": "Hi", "ai_model": "llama3.1", "ai_model_parameters": "8b"}
    )

    assert await communicator.receive_json_from() == {"message": "ok", "done": False}
    assert _without_context(await communicator.receive_json_from()) == {"message": "ok", "done": True}

    await communicator.disconnect()

    assert received_params == {"temperature": 0.9, "num_ctx": 8192, "top_p": 0.5, "seed": 7}


async def _send_and_collect(chat, message: str = "Hi") -> list[dict]:
    communicator = WebsocketCommunicator(_application(), f"/ws/chat/{chat.id}/")
    assert (await communicator.connect())[0]
    await communicator.send_json_to({"message": message, "ai_model": "llama3.1", "ai_model_parameters": "8b"})

    frames = []
    while True:
        frame = await communicator.receive_json_from(timeout=5)
        frames.append(frame)
        if frame.get("done"):
            break

    await communicator.disconnect()
    return frames


def _capturing_astream(captured: dict, stats_values: dict | None = None):
    async def _fake_astream(model, parameters, message, image, history, system_prompt=None, generation_params=None, stats=None, reasoning=None):
        captured.update(history=history, system_prompt=system_prompt, generation_params=generation_params)
        yield "Hi"
        if stats is not None and stats_values:
            stats.update(stats_values)

    return _fake_astream


async def test_done_frame_reports_memory_use_and_keeps_stats_separate(chat, monkeypatch):
    chat.num_ctx = 4096
    await database_sync_to_async(chat.save)(update_fields=["num_ctx"])
    captured: dict = {}
    monkeypatch.setattr(
        functions, "astream_bot_response",
        _capturing_astream(captured, {"prompt_tokens": 10, "completion_tokens": 20, "tokens_per_second": 50.0}),
    )

    done = (await _send_and_collect(chat))[-1]

    assert done["stats"] == {"prompt_tokens": 10, "completion_tokens": 20, "tokens_per_second": 50.0}
    assert done["context"]["limit"] == 4096
    assert done["context"]["preset"] == "short"
    assert done["context"]["dropped"] == 0
    assert done["context"]["summarized"] is False
    # Estimated prompt plus the exact completion count.
    assert done["context"]["used"] > 20
    assert captured["generation_params"]["num_ctx"] == 4096


async def test_without_a_chosen_length_the_standard_preset_is_requested_explicitly(chat, monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(functions, "astream_bot_response", _capturing_astream(captured))

    done = (await _send_and_collect(chat))[-1]

    assert captured["generation_params"]["num_ctx"] == 8192
    assert done["context"]["preset"] == "standard"


async def test_memory_never_exceeds_what_the_model_supports(chat, monkeypatch):
    chat.num_ctx = 32768
    await database_sync_to_async(chat.save)(update_fields=["num_ctx"])
    monkeypatch.setattr(
        functions, "get_model_info", lambda model_name, parameters: {"context_length": 2048, "thinking": None, "capabilities": []}
    )
    captured: dict = {}
    monkeypatch.setattr(functions, "astream_bot_response", _capturing_astream(captured))

    done = (await _send_and_collect(chat))[-1]

    assert captured["generation_params"]["num_ctx"] == 2048
    assert done["context"]["model_max"] == 2048


async def _make_long_history(chat, turns: int, words: int = 400) -> None:
    def _build():
        parent = None
        for i in range(turns):
            user = models.ChatMessage.objects.create(
                chat=chat, parent=parent, role="user", content=f"question {i} " + "word " * words
            )
            parent = models.ChatMessage.objects.create(
                chat=chat, parent=user, role="assistant", content=f"answer {i} " + "word " * words
            )
        chat.active_leaf = parent
        chat.save(update_fields=["active_leaf"])

    await database_sync_to_async(_build)()


async def test_trim_mode_leaves_out_the_oldest_messages_and_says_how_many(chat, monkeypatch):
    chat.num_ctx = 4096
    await database_sync_to_async(chat.save)(update_fields=["num_ctx"])
    await _make_long_history(chat, turns=10)
    captured: dict = {}
    monkeypatch.setattr(functions, "astream_bot_response", _capturing_astream(captured))

    done = (await _send_and_collect(chat))[-1]

    assert done["context"]["dropped"] > 0
    assert len(captured["history"]) == 20 - done["context"]["dropped"]
    assert captured["history"][0]["role"] == "user"
    assert captured["history"][-1]["content"].startswith("answer 9")


async def test_summarize_mode_folds_dropped_messages_into_a_saved_summary(chat, monkeypatch):
    chat.num_ctx = 4096
    chat.memory_mode = models.ChatHistory.MEMORY_MODE_SUMMARIZE
    await database_sync_to_async(chat.save)(update_fields=["num_ctx", "memory_mode"])
    await _make_long_history(chat, turns=10)
    captured: dict = {}
    folded: list = []

    async def _fake_summarize(model, parameters, previous_summary, messages, num_ctx):
        folded.extend(messages)
        return "The user asked ten long questions."

    monkeypatch.setattr(functions, "asummarize_conversation", _fake_summarize)
    monkeypatch.setattr(functions, "astream_bot_response", _capturing_astream(captured))

    frames = await _send_and_collect(chat)
    done = frames[-1]

    assert any("summarized" in f.get("notice", "") for f in frames)
    assert done["context"]["summarized"] is True
    assert "The user asked ten long questions." in captured["system_prompt"]
    assert folded[0]["content"].startswith("question 0")
    assert len(folded) + len(captured["history"]) == 20

    await database_sync_to_async(chat.refresh_from_db)()
    assert chat.memory_summary == "The user asked ten long questions."
    assert chat.memory_summary_until_id == folded[-1]["id"]


async def test_existing_summary_is_reused_without_summarizing_again(chat, monkeypatch):
    chat.num_ctx = 4096
    chat.memory_mode = models.ChatHistory.MEMORY_MODE_SUMMARIZE
    await database_sync_to_async(chat.save)(update_fields=["num_ctx", "memory_mode"])
    await _make_long_history(chat, turns=3, words=20)

    first_answer = await database_sync_to_async(
        lambda: models.ChatMessage.objects.filter(chat=chat, content__startswith="answer 0").get()
    )()
    chat.memory_summary = "Earlier: greetings."
    chat.memory_summary_until = first_answer
    await database_sync_to_async(chat.save)(update_fields=["memory_summary", "memory_summary_until"])

    async def _must_not_summarize(*args, **kwargs):
        raise AssertionError("summarized again")

    captured: dict = {}
    monkeypatch.setattr(functions, "asummarize_conversation", _must_not_summarize)
    monkeypatch.setattr(functions, "astream_bot_response", _capturing_astream(captured))

    done = (await _send_and_collect(chat))[-1]

    assert done["context"]["summarized"] is True
    assert "Earlier: greetings." in captured["system_prompt"]
    assert captured["history"][0]["content"].startswith("question 1")


async def test_continued_chat_summarizes_its_source_on_the_first_message(chat, monkeypatch):
    await _make_long_history(chat, turns=2, words=10)
    continued = await database_sync_to_async(models.ChatHistory.objects.create)(
        ai_model=chat.ai_model, title="Continued", seed_source=chat
    )

    async def _fake_summarize(model, parameters, previous_summary, messages, num_ctx):
        return f"Summary of {len(messages)} messages."

    captured: dict = {}
    monkeypatch.setattr(functions, "asummarize_conversation", _fake_summarize)
    monkeypatch.setattr(functions, "astream_bot_response", _capturing_astream(captured))

    frames = await _send_and_collect(continued)

    assert any("previous chat" in f.get("notice", "") for f in frames)
    assert "Summary of 4 messages." in captured["system_prompt"]
    await database_sync_to_async(continued.refresh_from_db)()
    assert continued.seed_summary == "Summary of 4 messages."


async def test_receive_omits_stats_key_entirely_when_nothing_was_reported(chat, monkeypatch):
    async def _fake_astream(model, parameters, message, image, history, system_prompt=None, generation_params=None, stats=None, reasoning=None):
        yield "Hi"

    monkeypatch.setattr(functions, "astream_bot_response", _fake_astream)

    communicator = WebsocketCommunicator(_application(), f"/ws/chat/{chat.id}/")
    assert (await communicator.connect())[0]

    await communicator.send_json_to(
        {"message": "Hi", "ai_model": "llama3.1", "ai_model_parameters": "8b"}
    )

    assert await communicator.receive_json_from() == {"message": "Hi", "done": False}
    response = await communicator.receive_json_from()

    assert _without_context(response) == {"message": "Hi", "done": True}
    assert "stats" not in response

    await communicator.disconnect()


async def test_receive_rejects_invalid_json(chat):
    communicator = WebsocketCommunicator(_application(), f"/ws/chat/{chat.id}/")
    assert (await communicator.connect())[0]

    await communicator.send_to(text_data="not json")
    response = await communicator.receive_json_from()

    assert response == {"error": "Invalid message format.", "done": True}
    await communicator.disconnect()


async def test_receive_rejects_missing_required_keys(chat):
    communicator = WebsocketCommunicator(_application(), f"/ws/chat/{chat.id}/")
    assert (await communicator.connect())[0]

    await communicator.send_json_to({"message": "hi"})
    response = await communicator.receive_json_from()

    assert response["done"] is True
    assert "error" in response
    await communicator.disconnect()


async def test_receive_surfaces_model_unavailable_as_readable_error(chat, monkeypatch):
    async def _fake_astream(*args, **kwargs):
        if False:  # pragma: no cover - keeps this an async generator
            yield ""
        raise ModelUnavailableError(
            "Model container isn't running or is still pulling. "
            "Check the Models page and try again once it shows 'Running'."
        )

    monkeypatch.setattr(functions, "astream_bot_response", _fake_astream)

    communicator = WebsocketCommunicator(_application(), f"/ws/chat/{chat.id}/")
    assert (await communicator.connect())[0]

    await communicator.send_json_to(
        {"message": "Hi", "ai_model": "llama3.1", "ai_model_parameters": "8b"}
    )
    response = await communicator.receive_json_from()

    assert response["done"] is True
    assert "still pulling" in response["error"]

    messages = await database_sync_to_async(list)(models.ChatMessage.objects.filter(chat=chat))
    assert messages == []

    await communicator.disconnect()


async def test_failure_during_generation_setup_still_sends_an_error_frame(chat, monkeypatch):
    from django_app.rag import retrieval

    def _broken_retrieval(chat_history, query, previous_question=None):
        raise RuntimeError("pgvector exploded")

    monkeypatch.setattr(retrieval, "retrieve_context_for_chat", _broken_retrieval)

    communicator = WebsocketCommunicator(_application(), f"/ws/chat/{chat.id}/")
    assert (await communicator.connect())[0]

    await communicator.send_json_to(
        {"message": "Hi", "ai_model": "llama3.1", "ai_model_parameters": "8b"}
    )
    response = await communicator.receive_json_from()

    assert response == {"error": "An unexpected error occurred. Please try again.", "done": True}
    await communicator.disconnect()


async def test_rag_warnings_are_sent_as_notices_before_the_answer(chat, monkeypatch):
    from django_app.rag import retrieval

    monkeypatch.setattr(
        retrieval,
        "retrieve_context_for_chat",
        lambda chat_history, query, previous_question=None: retrieval.RetrievalResult(
            "", [], ["Your documents weren't searched."]
        ),
    )

    async def _fake_astream(*args, **kwargs):
        yield "Answer"

    monkeypatch.setattr(functions, "astream_bot_response", _fake_astream)

    communicator = WebsocketCommunicator(_application(), f"/ws/chat/{chat.id}/")
    assert (await communicator.connect())[0]

    await communicator.send_json_to(
        {"message": "Hi", "ai_model": "llama3.1", "ai_model_parameters": "8b"}
    )

    assert await communicator.receive_json_from() == {"notice": "Your documents weren't searched.", "done": False}
    assert await communicator.receive_json_from() == {"message": "Answer", "done": False}
    await communicator.disconnect()


async def test_receive_unknown_model_reports_invalid_instead_of_500(chat):
    communicator = WebsocketCommunicator(_application(), f"/ws/chat/{chat.id}/")
    assert (await communicator.connect())[0]

    await communicator.send_json_to(
        {"message": "Hi", "ai_model": "does-not-exist", "ai_model_parameters": "8b"}
    )
    response = await communicator.receive_json_from()

    assert response == {"error": "Invalid model or chat history.", "done": True}
    await communicator.disconnect()


async def test_stop_cancels_in_flight_generation_and_persists_partial_response(chat, monkeypatch):
    hang = asyncio.Event()

    async def _fake_astream(model, parameters, message, image, history, system_prompt=None, generation_params=None, stats=None, reasoning=None):
        yield "Hel"
        await hang.wait()
        yield "lo"  # pragma: no cover - never reached, cancelled first

    monkeypatch.setattr(functions, "astream_bot_response", _fake_astream)

    communicator = WebsocketCommunicator(_application(), f"/ws/chat/{chat.id}/")
    assert (await communicator.connect())[0]

    await communicator.send_json_to(
        {"message": "Hi", "ai_model": "llama3.1", "ai_model_parameters": "8b"}
    )
    assert await communicator.receive_json_from() == {"message": "Hel", "done": False}

    await communicator.send_json_to({"type": "stop"})
    response = await communicator.receive_json_from()

    assert _without_context(response) == {"message": "", "done": True, "stopped": True}

    messages = await database_sync_to_async(list)(
        models.ChatMessage.objects.filter(chat=chat).order_by("created_at")
    )
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[1].content == "Hel"

    await communicator.disconnect()


async def test_stop_with_no_active_generation_is_a_noop(chat):
    communicator = WebsocketCommunicator(_application(), f"/ws/chat/{chat.id}/")
    assert (await communicator.connect())[0]

    await communicator.send_json_to({"type": "stop"})

    assert await communicator.receive_nothing() is True
    await communicator.disconnect()


async def test_prompt_while_generating_is_rejected_instead_of_queued(chat, monkeypatch):
    hang = asyncio.Event()

    async def _fake_astream(model, parameters, message, image, history, system_prompt=None, generation_params=None, stats=None, reasoning=None):
        yield "Hel"
        await hang.wait()

    monkeypatch.setattr(functions, "astream_bot_response", _fake_astream)

    communicator = WebsocketCommunicator(_application(), f"/ws/chat/{chat.id}/")
    assert (await communicator.connect())[0]

    await communicator.send_json_to(
        {"message": "Hi", "ai_model": "llama3.1", "ai_model_parameters": "8b"}
    )
    assert await communicator.receive_json_from() == {"message": "Hel", "done": False}

    await communicator.send_json_to(
        {"message": "Again", "ai_model": "llama3.1", "ai_model_parameters": "8b"}
    )
    response = await communicator.receive_json_from()

    assert response == {"error": "A response is already generating. Stop it first.", "done": True}

    hang.set()
    assert _without_context(await communicator.receive_json_from()) == {"message": "Hel", "done": True}
    await communicator.disconnect()


async def test_receive_unknown_message_type_reports_error(chat):
    communicator = WebsocketCommunicator(_application(), f"/ws/chat/{chat.id}/")
    assert (await communicator.connect())[0]

    await communicator.send_json_to({"type": "bogus"})
    response = await communicator.receive_json_from()

    assert response["done"] is True
    assert "Unknown message type" in response["error"]
    await communicator.disconnect()


async def _send_first_turn(communicator, monkeypatch, reply="Hello"):
    async def _fake_astream(model, parameters, message, image, history, system_prompt=None, generation_params=None, stats=None, reasoning=None):
        yield reply

    monkeypatch.setattr(functions, "astream_bot_response", _fake_astream)
    await communicator.send_json_to(
        {"message": "Hi", "ai_model": "llama3.1", "ai_model_parameters": "8b"}
    )
    assert await communicator.receive_json_from() == {"message": reply, "done": False}
    assert _without_context(await communicator.receive_json_from()) == {"message": reply, "done": True}


def _active_path_and_all_messages(chat):
    chat.refresh_from_db()
    active_path = [(m.role, m.content) for m in functions.get_messages_for_chat(chat)]
    all_messages = [
        (m.role, m.content)
        for m in models.ChatMessage.objects.filter(chat=chat).order_by("created_at")
    ]
    return active_path, all_messages


async def test_regenerate_replaces_last_assistant_message_and_keeps_user_message(chat, monkeypatch):
    communicator = WebsocketCommunicator(_application(), f"/ws/chat/{chat.id}/")
    assert (await communicator.connect())[0]

    await _send_first_turn(communicator, monkeypatch, reply="Hello")

    async def _fake_regenerate(model, parameters, message, image, history, system_prompt=None, generation_params=None, stats=None, reasoning=None):
        # The regenerated turn's history must exclude the answer being
        # replaced, or the model would see its own stale reply as context.
        assert history == []
        assert message == "Hi"
        yield "Better hello"

    monkeypatch.setattr(functions, "astream_bot_response", _fake_regenerate)

    await communicator.send_json_to(
        {"type": "regenerate", "ai_model": "llama3.1", "ai_model_parameters": "8b"}
    )
    assert await communicator.receive_json_from() == {"message": "Better hello", "done": False}
    assert _without_context(await communicator.receive_json_from()) == {"message": "Better hello", "done": True}

    active_path, all_messages = await database_sync_to_async(_active_path_and_all_messages)(chat)

    assert active_path == [("user", "Hi"), ("assistant", "Better hello")]
    # The old reply is a sibling branch, not deleted — regenerate must never
    # destroy data, only add to it.
    assert all_messages == [
        ("user", "Hi"),
        ("assistant", "Hello"),
        ("assistant", "Better hello"),
    ]

    await communicator.disconnect()


async def test_regenerate_with_nothing_to_regenerate_reports_error(chat):
    communicator = WebsocketCommunicator(_application(), f"/ws/chat/{chat.id}/")
    assert (await communicator.connect())[0]

    await communicator.send_json_to(
        {"type": "regenerate", "ai_model": "llama3.1", "ai_model_parameters": "8b"}
    )
    response = await communicator.receive_json_from()

    assert response["done"] is True
    assert "Nothing to regenerate" in response["error"]
    await communicator.disconnect()


async def test_regenerate_failure_leaves_original_assistant_message_intact(chat, monkeypatch):
    communicator = WebsocketCommunicator(_application(), f"/ws/chat/{chat.id}/")
    assert (await communicator.connect())[0]

    await _send_first_turn(communicator, monkeypatch, reply="Hello")

    async def _fake_failing_regenerate(*args, **kwargs):
        if False:  # pragma: no cover - keeps this an async generator
            yield ""
        raise ModelUnavailableError("Model container isn't running or is still pulling.")

    monkeypatch.setattr(functions, "astream_bot_response", _fake_failing_regenerate)

    await communicator.send_json_to(
        {"type": "regenerate", "ai_model": "llama3.1", "ai_model_parameters": "8b"}
    )
    response = await communicator.receive_json_from()
    assert response["done"] is True
    assert "error" in response

    messages = await database_sync_to_async(list)(
        models.ChatMessage.objects.filter(chat=chat).order_by("created_at")
    )
    assert [(m.role, m.content) for m in messages] == [("user", "Hi"), ("assistant", "Hello")]

    await communicator.disconnect()


async def test_regenerate_while_generating_is_rejected(chat, monkeypatch):
    hang = asyncio.Event()

    async def _fake_astream(model, parameters, message, image, history, system_prompt=None, generation_params=None, stats=None, reasoning=None):
        yield "Hel"
        await hang.wait()

    monkeypatch.setattr(functions, "astream_bot_response", _fake_astream)

    communicator = WebsocketCommunicator(_application(), f"/ws/chat/{chat.id}/")
    assert (await communicator.connect())[0]

    await communicator.send_json_to(
        {"message": "Hi", "ai_model": "llama3.1", "ai_model_parameters": "8b"}
    )
    assert await communicator.receive_json_from() == {"message": "Hel", "done": False}

    await communicator.send_json_to(
        {"type": "regenerate", "ai_model": "llama3.1", "ai_model_parameters": "8b"}
    )
    response = await communicator.receive_json_from()
    assert response == {"error": "A response is already generating. Stop it first.", "done": True}

    hang.set()
    assert _without_context(await communicator.receive_json_from()) == {"message": "Hel", "done": True}
    await communicator.disconnect()


async def test_edit_resend_replaces_edited_message_and_discards_everything_after_it(chat, monkeypatch):
    communicator = WebsocketCommunicator(_application(), f"/ws/chat/{chat.id}/")
    assert (await communicator.connect())[0]

    await _send_first_turn(communicator, monkeypatch, reply="Hello")

    async def _fake_second_turn(model, parameters, message, image, history, system_prompt=None, generation_params=None, stats=None, reasoning=None):
        yield "I'm good"

    monkeypatch.setattr(functions, "astream_bot_response", _fake_second_turn)
    await communicator.send_json_to(
        {"message": "How are you", "ai_model": "llama3.1", "ai_model_parameters": "8b"}
    )
    assert await communicator.receive_json_from() == {"message": "I'm good", "done": False}
    assert _without_context(await communicator.receive_json_from()) == {"message": "I'm good", "done": True}

    async def _fake_edit(model, parameters, message, image, history, system_prompt=None, generation_params=None, stats=None, reasoning=None):
        # History must stop before the edited message — the model shouldn't
        # see the stale exchange this edit is replacing, or the turn it
        # invalidated.
        assert history == []
        assert message == "Hey there"
        yield "Hi yourself"

    monkeypatch.setattr(functions, "astream_bot_response", _fake_edit)

    await communicator.send_json_to(
        {
            "type": "edit_resend",
            "index": 0,
            "message": "Hey there",
            "ai_model": "llama3.1",
            "ai_model_parameters": "8b",
        }
    )
    assert await communicator.receive_json_from() == {"message": "Hi yourself", "done": False}
    assert _without_context(await communicator.receive_json_from()) == {"message": "Hi yourself", "done": True}

    active_path, all_messages = await database_sync_to_async(_active_path_and_all_messages)(chat)

    assert active_path == [("user", "Hey there"), ("assistant", "Hi yourself")]
    # The entire original branch (Hi/Hello/How are you/I'm good) survives as
    # an inactive branch — edit-resend must never destroy data either.
    assert set(all_messages) == {
        ("user", "Hi"),
        ("assistant", "Hello"),
        ("user", "How are you"),
        ("assistant", "I'm good"),
        ("user", "Hey there"),
        ("assistant", "Hi yourself"),
    }

    await communicator.disconnect()


async def test_edit_resend_on_assistant_index_reports_error(chat, monkeypatch):
    communicator = WebsocketCommunicator(_application(), f"/ws/chat/{chat.id}/")
    assert (await communicator.connect())[0]
    await _send_first_turn(communicator, monkeypatch, reply="Hello")

    await communicator.send_json_to(
        {
            "type": "edit_resend",
            "index": 1,
            "message": "x",
            "ai_model": "llama3.1",
            "ai_model_parameters": "8b",
        }
    )
    response = await communicator.receive_json_from()

    assert response["done"] is True
    assert "can no longer be edited" in response["error"]
    await communicator.disconnect()


async def test_edit_resend_out_of_range_index_reports_error(chat):
    communicator = WebsocketCommunicator(_application(), f"/ws/chat/{chat.id}/")
    assert (await communicator.connect())[0]

    await communicator.send_json_to(
        {
            "type": "edit_resend",
            "index": 5,
            "message": "x",
            "ai_model": "llama3.1",
            "ai_model_parameters": "8b",
        }
    )
    response = await communicator.receive_json_from()

    assert response["done"] is True
    assert "can no longer be edited" in response["error"]
    await communicator.disconnect()


async def test_edit_resend_failure_leaves_original_tail_intact(chat, monkeypatch):
    communicator = WebsocketCommunicator(_application(), f"/ws/chat/{chat.id}/")
    assert (await communicator.connect())[0]
    await _send_first_turn(communicator, monkeypatch, reply="Hello")

    async def _fake_failing(*args, **kwargs):
        if False:  # pragma: no cover - keeps this an async generator
            yield ""
        raise ModelUnavailableError("Model container isn't running or is still pulling.")

    monkeypatch.setattr(functions, "astream_bot_response", _fake_failing)

    await communicator.send_json_to(
        {
            "type": "edit_resend",
            "index": 0,
            "message": "Hey there",
            "ai_model": "llama3.1",
            "ai_model_parameters": "8b",
        }
    )
    response = await communicator.receive_json_from()
    assert response["done"] is True
    assert "error" in response

    messages = await database_sync_to_async(list)(
        models.ChatMessage.objects.filter(chat=chat).order_by("created_at")
    )
    assert [(m.role, m.content) for m in messages] == [("user", "Hi"), ("assistant", "Hello")]

    await communicator.disconnect()


async def test_edit_resend_while_generating_is_rejected(chat, monkeypatch):
    hang = asyncio.Event()

    async def _fake_astream(model, parameters, message, image, history, system_prompt=None, generation_params=None, stats=None, reasoning=None):
        yield "Hel"
        await hang.wait()

    monkeypatch.setattr(functions, "astream_bot_response", _fake_astream)

    communicator = WebsocketCommunicator(_application(), f"/ws/chat/{chat.id}/")
    assert (await communicator.connect())[0]

    await communicator.send_json_to(
        {"message": "Hi", "ai_model": "llama3.1", "ai_model_parameters": "8b"}
    )
    assert await communicator.receive_json_from() == {"message": "Hel", "done": False}

    await communicator.send_json_to(
        {
            "type": "edit_resend",
            "index": 0,
            "message": "Hey",
            "ai_model": "llama3.1",
            "ai_model_parameters": "8b",
        }
    )
    response = await communicator.receive_json_from()
    assert response == {"error": "A response is already generating. Stop it first.", "done": True}

    hang.set()
    assert _without_context(await communicator.receive_json_from()) == {"message": "Hel", "done": True}
    await communicator.disconnect()


async def test_edit_resend_missing_required_keys_reports_error(chat):
    communicator = WebsocketCommunicator(_application(), f"/ws/chat/{chat.id}/")
    assert (await communicator.connect())[0]

    await communicator.send_json_to({"type": "edit_resend", "index": 0})
    response = await communicator.receive_json_from()

    assert response["done"] is True
    assert "error" in response
    await communicator.disconnect()


async def test_receive_uses_tool_path_when_chat_and_model_both_allow_it(tools_chat, monkeypatch):
    async def _fake_astream_tools(model, parameters, message, image, history, system_prompt=None, generation_params=None, stats=None, mcp_servers=None, reasoning=None, approve=None):
        yield {"name": "calculator", "args": {"expression": "2+2"}, "result": "4"}
        yield "The answer is 4."

    async def _fail_if_called(*args, **kwargs):
        raise AssertionError("astream_bot_response should not be used when tools are enabled")
        yield  # pragma: no cover — unreachable, keeps this an async generator

    monkeypatch.setattr(functions, "astream_tool_response", _fake_astream_tools)
    monkeypatch.setattr(functions, "astream_bot_response", _fail_if_called)

    communicator = WebsocketCommunicator(_application(), f"/ws/chat/{tools_chat.id}/")
    assert (await communicator.connect())[0]

    await communicator.send_json_to(
        {"message": "What is 2+2?", "ai_model": "llama3.1-tools-test", "ai_model_parameters": "8b"}
    )

    assert await communicator.receive_json_from() == {
        "tool_call": {"name": "calculator", "args": {"expression": "2+2"}, "result": "4"},
        "done": False,
    }
    assert await communicator.receive_json_from() == {"message": "The answer is 4.", "done": False}
    assert _without_context(await communicator.receive_json_from()) == {"message": "The answer is 4.", "done": True}

    await communicator.disconnect()

    messages = await database_sync_to_async(list)(
        models.ChatMessage.objects.filter(chat=tools_chat).order_by("created_at")
    )
    assert [m.content for m in messages] == ["What is 2+2?", "The answer is 4."]


async def test_receive_ignores_tools_enabled_when_model_lacks_tools_capability(chat, monkeypatch):
    # `chat` fixture's AIModel has can_use_tools=False by default — turning
    # tools_enabled on for the chat alone must not be enough.
    chat.tools_enabled = True
    await database_sync_to_async(chat.save)(update_fields=["tools_enabled"])

    async def _fake_astream(model, parameters, message, image, history, system_prompt=None, generation_params=None, stats=None, reasoning=None):
        yield "plain reply"

    async def _fail_if_called(*args, **kwargs):
        raise AssertionError("astream_tool_response should not be used — model doesn't support tools")
        yield  # pragma: no cover

    monkeypatch.setattr(functions, "astream_bot_response", _fake_astream)
    monkeypatch.setattr(functions, "astream_tool_response", _fail_if_called)

    communicator = WebsocketCommunicator(_application(), f"/ws/chat/{chat.id}/")
    assert (await communicator.connect())[0]

    await communicator.send_json_to({"message": "Hi", "ai_model": "llama3.1", "ai_model_parameters": "8b"})

    assert await communicator.receive_json_from() == {"message": "plain reply", "done": False}
    assert _without_context(await communicator.receive_json_from()) == {"message": "plain reply", "done": True}

    await communicator.disconnect()


async def test_receive_structured_output_takes_priority_over_tools(tools_chat, monkeypatch):
    async def _fake_structured(*args, **kwargs):
        return '```json\n{"a": 1}\n```'

    async def _fail_if_called(*args, **kwargs):
        raise AssertionError("astream_tool_response should not run when structured_output is set")
        yield  # pragma: no cover

    monkeypatch.setattr(functions, "astream_structured_bot_response", _fake_structured)
    monkeypatch.setattr(functions, "astream_tool_response", _fail_if_called)

    communicator = WebsocketCommunicator(_application(), f"/ws/chat/{tools_chat.id}/")
    assert (await communicator.connect())[0]

    await communicator.send_json_to({
        "message": "Hi",
        "ai_model": "llama3.1-tools-test",
        "ai_model_parameters": "8b",
        "structured_output": [{"field": "a", "type": "number"}],
    })

    response = await communicator.receive_json_from()
    assert response["done"] is True
    assert response["message"] == '```json\n{"a": 1}\n```'

    await communicator.disconnect()


async def test_done_frame_reports_how_much_of_the_model_fits_on_the_gpu(chat, monkeypatch):
    monkeypatch.setattr(functions, "get_gpu_share", lambda model_name, parameters: 0.6)
    monkeypatch.setattr(functions, "astream_bot_response", _capturing_astream({}))

    done = (await _send_and_collect(chat))[-1]

    assert done["context"]["gpu_share"] == 0.6


async def test_thinking_is_streamed_separately_and_saved_with_the_reply(chat, monkeypatch):
    chat.think_level = "on"
    await database_sync_to_async(chat.save)(update_fields=["think_level"])
    monkeypatch.setattr(
        functions, "get_model_info",
        lambda model_name, parameters: {"context_length": 8192, "thinking": {"values": [False, True], "default": True}, "capabilities": []},
    )
    received: dict = {}

    async def _fake_astream(model, parameters, message, image, history, system_prompt=None, generation_params=None, stats=None, reasoning=None):
        received["reasoning"] = reasoning
        yield functions.ThinkingDelta("Hmm, ")
        yield functions.ThinkingDelta("easy.")
        yield "4"

    monkeypatch.setattr(functions, "astream_bot_response", _fake_astream)

    frames = await _send_and_collect(chat)

    assert frames[0] == {"thinking": "Hmm, ", "done": False}
    assert frames[1] == {"thinking": "easy.", "done": False}
    assert frames[2] == {"message": "4", "done": False}
    assert frames[-1]["message"] == "4"
    assert received["reasoning"] is True

    reply = await database_sync_to_async(
        lambda: models.ChatMessage.objects.filter(chat=chat, role="assistant").get()
    )()
    assert reply.content == "4"
    assert reply.thinking == "Hmm, easy."


async def test_tool_approval_round_trip_over_the_socket(tools_chat, monkeypatch):
    decisions = []

    async def _fake_astream_tools(model, parameters, message, image, history, system_prompt=None, generation_params=None, stats=None, mcp_servers=None, reasoning=None, approve=None):
        decisions.append(await approve({"name": "send_email", "server": "Mail", "args": {"to": "a@b.c"}}))
        yield "ok"

    monkeypatch.setattr(functions, "astream_tool_response", _fake_astream_tools)

    communicator = WebsocketCommunicator(_application(), f"/ws/chat/{tools_chat.id}/")
    assert (await communicator.connect())[0]
    await communicator.send_json_to(
        {"message": "mail it", "ai_model": "llama3.1-tools-test", "ai_model_parameters": "8b"}
    )

    request = await communicator.receive_json_from(timeout=5)
    assert request["tool_approval"]["name"] == "send_email"
    assert request["tool_approval"]["server"] == "Mail"
    assert request["done"] is False

    await communicator.send_json_to({"type": "tool_approval", "id": request["tool_approval"]["id"], "approved": False})

    assert await communicator.receive_json_from(timeout=5) == {"message": "ok", "done": False}
    assert decisions == [False]
    await communicator.disconnect()


async def test_first_reply_in_an_untitled_chat_gets_a_generated_title(chat, monkeypatch):
    chat.title = models.DEFAULT_CHAT_TITLE
    await database_sync_to_async(chat.save)(update_fields=["title"])

    async def _fake_title(model, parameters, user_prompt, answer, reasoning):
        return "Greeting the assistant"

    monkeypatch.setattr(functions, "agenerate_title", _fake_title)
    monkeypatch.setattr(functions, "astream_bot_response", _capturing_astream({}))

    communicator = WebsocketCommunicator(_application(), f"/ws/chat/{chat.id}/")
    assert (await communicator.connect())[0]
    await communicator.send_json_to({"message": "Hi", "ai_model": "llama3.1", "ai_model_parameters": "8b"})

    while not (await communicator.receive_json_from(timeout=5)).get("done"):
        pass
    assert await communicator.receive_json_from(timeout=5) == {"chat_title": "Greeting the assistant"}
    await communicator.disconnect()

    await database_sync_to_async(chat.refresh_from_db)()
    assert chat.title == "Greeting the assistant"


async def test_titled_chats_are_not_renamed(chat, monkeypatch):
    async def _must_not_title(*args):
        raise AssertionError("titled again")

    monkeypatch.setattr(functions, "agenerate_title", _must_not_title)
    monkeypatch.setattr(functions, "astream_bot_response", _capturing_astream({}))

    frames = await _send_and_collect(chat)

    assert frames[-1]["done"] is True


async def test_replies_record_which_model_wrote_them(chat, monkeypatch):
    monkeypatch.setattr(functions, "astream_bot_response", _capturing_astream({}))

    await _send_and_collect(chat)

    reply = await database_sync_to_async(
        lambda: models.ChatMessage.objects.filter(chat=chat, role="assistant").get()
    )()
    assert reply.model_label == "llama3.1:8b"


async def test_tools_follow_what_the_installed_model_reports_over_the_catalog(chat, monkeypatch):
    # The catalog said no tools (chat fixture's model), but the model says yes.
    chat.tools_enabled = True
    await database_sync_to_async(chat.save)(update_fields=["tools_enabled"])
    monkeypatch.setattr(
        functions, "get_model_info",
        lambda model_name, parameters: {"context_length": 8192, "thinking": None, "capabilities": ["tools"]},
    )
    used: list[str] = []

    async def _fake_tools(*args, **kwargs):
        used.append("tools")
        yield "ok"

    monkeypatch.setattr(functions, "astream_tool_response", _fake_tools)

    await _send_and_collect(chat)

    assert used == ["tools"]

