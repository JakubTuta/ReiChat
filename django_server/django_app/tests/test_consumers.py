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
    async def _fake_astream(model, parameters, message, image, history, system_prompt=None, generation_params=None, stats=None):
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
    assert await communicator.receive_json_from() == {"message": "Hello", "done": True}

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

    async def _fake_astream(model, parameters, message, image, history, system_prompt=None, generation_params=None, stats=None):
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
    assert await communicator.receive_json_from() == {"message": "Arr", "done": True}

    await communicator.disconnect()

    assert received_system_prompt == "Talk like a pirate."


async def test_receive_passes_chat_generation_params_to_astream(chat, monkeypatch):
    chat.temperature = 0.9
    chat.num_ctx = 8192
    chat.top_p = 0.5
    chat.seed = 7
    await database_sync_to_async(chat.save)(update_fields=["temperature", "num_ctx", "top_p", "seed"])

    received_params = None

    async def _fake_astream(model, parameters, message, image, history, system_prompt=None, generation_params=None, stats=None):
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
    assert await communicator.receive_json_from() == {"message": "ok", "done": True}

    await communicator.disconnect()

    assert received_params == {"temperature": 0.9, "num_ctx": 8192, "top_p": 0.5, "seed": 7}


async def test_receive_includes_generation_stats_with_context_usage_when_num_ctx_set(chat, monkeypatch):
    chat.num_ctx = 4096
    await database_sync_to_async(chat.save)(update_fields=["num_ctx"])

    async def _fake_astream(model, parameters, message, image, history, system_prompt=None, generation_params=None, stats=None):
        yield "Hi"
        # Mirrors what _record_generation_stats would set from Ollama's
        # final (done=True) streamed chunk.
        if stats is not None:
            stats["prompt_tokens"] = 10
            stats["completion_tokens"] = 20
            stats["tokens_per_second"] = 50.0

    monkeypatch.setattr(functions, "astream_bot_response", _fake_astream)

    communicator = WebsocketCommunicator(_application(), f"/ws/chat/{chat.id}/")
    assert (await communicator.connect())[0]

    await communicator.send_json_to(
        {"message": "Hi", "ai_model": "llama3.1", "ai_model_parameters": "8b"}
    )

    assert await communicator.receive_json_from() == {"message": "Hi", "done": False}
    response = await communicator.receive_json_from()

    assert response["message"] == "Hi"
    assert response["done"] is True
    assert response["stats"] == {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "tokens_per_second": 50.0,
        "context_limit": 4096,
        "context_used": 30,
    }

    await communicator.disconnect()


async def test_receive_omits_context_usage_without_num_ctx_override(chat, monkeypatch):
    async def _fake_astream(model, parameters, message, image, history, system_prompt=None, generation_params=None, stats=None):
        yield "Hi"
        if stats is not None:
            stats["prompt_tokens"] = 10
            stats["completion_tokens"] = 20

    monkeypatch.setattr(functions, "astream_bot_response", _fake_astream)

    communicator = WebsocketCommunicator(_application(), f"/ws/chat/{chat.id}/")
    assert (await communicator.connect())[0]

    await communicator.send_json_to(
        {"message": "Hi", "ai_model": "llama3.1", "ai_model_parameters": "8b"}
    )

    assert await communicator.receive_json_from() == {"message": "Hi", "done": False}
    response = await communicator.receive_json_from()

    assert response["stats"] == {"prompt_tokens": 10, "completion_tokens": 20}
    assert "context_used" not in response["stats"]

    await communicator.disconnect()


async def test_receive_omits_stats_key_entirely_when_nothing_was_reported(chat, monkeypatch):
    async def _fake_astream(model, parameters, message, image, history, system_prompt=None, generation_params=None, stats=None):
        yield "Hi"

    monkeypatch.setattr(functions, "astream_bot_response", _fake_astream)

    communicator = WebsocketCommunicator(_application(), f"/ws/chat/{chat.id}/")
    assert (await communicator.connect())[0]

    await communicator.send_json_to(
        {"message": "Hi", "ai_model": "llama3.1", "ai_model_parameters": "8b"}
    )

    assert await communicator.receive_json_from() == {"message": "Hi", "done": False}
    response = await communicator.receive_json_from()

    assert response == {"message": "Hi", "done": True}
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

    def _broken_retrieval(chat_history, query):
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
        lambda chat_history, query: retrieval.RetrievalResult("", [], ["Your documents weren't searched."]),
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

    async def _fake_astream(model, parameters, message, image, history, system_prompt=None, generation_params=None, stats=None):
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

    assert response == {"message": "", "done": True, "stopped": True}

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

    async def _fake_astream(model, parameters, message, image, history, system_prompt=None, generation_params=None, stats=None):
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
    assert await communicator.receive_json_from() == {"message": "Hel", "done": True}
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
    async def _fake_astream(model, parameters, message, image, history, system_prompt=None, generation_params=None, stats=None):
        yield reply

    monkeypatch.setattr(functions, "astream_bot_response", _fake_astream)
    await communicator.send_json_to(
        {"message": "Hi", "ai_model": "llama3.1", "ai_model_parameters": "8b"}
    )
    assert await communicator.receive_json_from() == {"message": reply, "done": False}
    assert await communicator.receive_json_from() == {"message": reply, "done": True}


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

    async def _fake_regenerate(model, parameters, message, image, history, system_prompt=None, generation_params=None, stats=None):
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
    assert await communicator.receive_json_from() == {"message": "Better hello", "done": True}

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

    async def _fake_astream(model, parameters, message, image, history, system_prompt=None, generation_params=None, stats=None):
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
    assert await communicator.receive_json_from() == {"message": "Hel", "done": True}
    await communicator.disconnect()


async def test_edit_resend_replaces_edited_message_and_discards_everything_after_it(chat, monkeypatch):
    communicator = WebsocketCommunicator(_application(), f"/ws/chat/{chat.id}/")
    assert (await communicator.connect())[0]

    await _send_first_turn(communicator, monkeypatch, reply="Hello")

    async def _fake_second_turn(model, parameters, message, image, history, system_prompt=None, generation_params=None, stats=None):
        yield "I'm good"

    monkeypatch.setattr(functions, "astream_bot_response", _fake_second_turn)
    await communicator.send_json_to(
        {"message": "How are you", "ai_model": "llama3.1", "ai_model_parameters": "8b"}
    )
    assert await communicator.receive_json_from() == {"message": "I'm good", "done": False}
    assert await communicator.receive_json_from() == {"message": "I'm good", "done": True}

    async def _fake_edit(model, parameters, message, image, history, system_prompt=None, generation_params=None, stats=None):
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
    assert await communicator.receive_json_from() == {"message": "Hi yourself", "done": True}

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

    async def _fake_astream(model, parameters, message, image, history, system_prompt=None, generation_params=None, stats=None):
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
    assert await communicator.receive_json_from() == {"message": "Hel", "done": True}
    await communicator.disconnect()


async def test_edit_resend_missing_required_keys_reports_error(chat):
    communicator = WebsocketCommunicator(_application(), f"/ws/chat/{chat.id}/")
    assert (await communicator.connect())[0]

    await communicator.send_json_to({"type": "edit_resend", "index": 0})
    response = await communicator.receive_json_from()

    assert response["done"] is True
    assert "error" in response
    await communicator.disconnect()


# --- Tool calling ------------------------------------------------------------


async def test_receive_uses_tool_path_when_chat_and_model_both_allow_it(tools_chat, monkeypatch):
    async def _fake_astream_tools(model, parameters, message, image, history, system_prompt=None, generation_params=None, stats=None, mcp_servers=None):
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
    assert await communicator.receive_json_from() == {"message": "The answer is 4.", "done": True}

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

    async def _fake_astream(model, parameters, message, image, history, system_prompt=None, generation_params=None, stats=None):
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
    assert await communicator.receive_json_from() == {"message": "plain reply", "done": True}

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
