"""The chat REST endpoints: reading a chat's active branch, switching branches,
the chat list and its settings, continuing a chat, and search."""

import datetime
from unittest.mock import AsyncMock

import httpx
import pytest
from sqlalchemy import event, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import models
from app.db.session import engine
from app.llm import runtime
from tests.factories import make_chat, make_exchange, make_message, make_model, minutes_ago

pytestmark = pytest.mark.usefixtures("db")


@pytest.fixture
async def chat(db: AsyncSession) -> models.ChatHistory:
    model = await make_model(db, "llama3.1")
    return await make_chat(db, model, title="Test chat")


@pytest.fixture
async def branching_chat(db: AsyncSession, chat: models.ChatHistory) -> models.ChatHistory:
    """user "Hi" -> assistant "Hello" (inactive) / assistant "Hi again" (active)."""
    user_msg = await make_message(db, chat, "user", "Hi")
    await make_message(db, chat, "assistant", "Hello", user_msg)
    await make_message(db, chat, "assistant", "Hi again", user_msg, activate=True)
    return chat


async def refreshed(db: AsyncSession, chat: models.ChatHistory) -> models.ChatHistory:
    return (
        await db.execute(
            select(models.ChatHistory)
            .where(models.ChatHistory.id == chat.id)
            .execution_options(populate_existing=True)
        )
    ).scalar_one()


async def model_name(db: AsyncSession, chat: models.ChatHistory) -> str:
    return (
        await db.execute(select(models.AIModel.model).where(models.AIModel.id == chat.ai_model_id))
    ).scalar_one()


async def history_url(db: AsyncSession, chat: models.ChatHistory) -> str:
    return f"/chat-history/{await model_name(db, chat)}/{chat.id}"


async def all_chats_url(db: AsyncSession, chat: models.ChatHistory) -> str:
    return f"/all-chats/{await model_name(db, chat)}"


async def count(db: AsyncSession, table) -> int:
    return (await db.execute(select(func.count()).select_from(table))).scalar_one()


async def add_collection(db: AsyncSession, name: str = "Docs") -> models.DocumentCollection:
    embedder = await make_model(db, f"embedder-for-{name}", is_embedding=True)
    collection = models.DocumentCollection(
        name=name, embedding_model_id=embedder.id, embedding_parameters="latest"
    )
    db.add(collection)
    await db.commit()
    return collection


# Reading a chat


async def test_get_chat_history_returns_active_path_with_branch_metadata(
    client: httpx.AsyncClient, db: AsyncSession, branching_chat
):
    response = await client.get(await history_url(db, branching_chat))

    assert response.status_code == 200
    assert response.json() == [
        {
            "role": "user",
            "content": "Hi",
            "image": "",
            "thinking": "",
            "model": "",
            "sibling_count": 1,
            "sibling_index": 0,
        },
        {
            "role": "assistant",
            "content": "Hi again",
            "image": "",
            "thinking": "",
            "model": "",
            "sibling_count": 2,
            "sibling_index": 1,
        },
    ]


async def test_a_chat_that_does_not_exist_reads_as_empty_not_as_an_error(
    client: httpx.AsyncClient, db: AsyncSession, chat
):
    response = await client.get(f"/chat-history/{await model_name(db, chat)}/999999")

    assert response.status_code == 200
    assert response.json() == []


async def test_reading_a_chat_under_the_wrong_model_finds_nothing(
    client: httpx.AsyncClient, db: AsyncSession, chat
):
    await make_model(db, "other")

    assert (await client.get(f"/chat-history/other/{chat.id}")).json() == []


async def test_reading_with_an_unknown_model_is_a_404(client: httpx.AsyncClient):
    response = await client.get("/chat-history/does-not-exist/1")

    assert response.status_code == 404
    assert response.json() == {"error": "AI model not found"}


async def test_a_non_numeric_chat_id_is_a_400_not_a_server_error(
    client: httpx.AsyncClient, db: AsyncSession, chat
):
    response = await client.get(f"/chat-history/{await model_name(db, chat)}/abc")

    assert response.status_code == 400
    assert response.json()["details"][0]["field"] == "chat_id"


async def test_get_chat_history_query_count_does_not_grow_with_chat_length(
    client: httpx.AsyncClient, db: AsyncSession, chat
):
    parent = None
    for i in range(30):
        parent = await make_message(
            db, chat, "user" if i % 2 == 0 else "assistant", f"m{i}", parent, activate=True
        )
    statements = []

    def record(conn, cursor, statement, *args):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", record)
    try:
        response = await client.get(await history_url(db, chat))
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", record)

    assert response.status_code == 200
    assert [m["content"] for m in response.json()] == [f"m{i}" for i in range(30)]
    assert len(statements) <= 6


async def test_routes_accept_model_names_with_slashes(client: httpx.AsyncClient, db: AsyncSession):
    model = await make_model(db, "hf.co/u/repo", is_custom=True)
    chat = await make_chat(db, model)

    assert (await client.get("/all-chats/hf.co/u/repo")).status_code == 200
    assert (await client.get(f"/chat-history/hf.co/u/repo/{chat.id}")).status_code == 200
    assert (await client.get(f"/chat-context/hf.co/u/repo/{chat.id}")).status_code == 200
    assert (await client.get("/search/hf.co/u/repo?q=hello")).status_code == 200


# Switching branches


async def test_patch_switch_branch_moves_active_leaf_to_chosen_sibling(
    client: httpx.AsyncClient, db: AsyncSession, branching_chat
):
    response = await client.patch(
        await history_url(db, branching_chat), json={"index": 1, "sibling_index": 0}
    )

    assert response.status_code == 200
    assert [(m["content"], m["sibling_index"]) for m in response.json()] == [("Hi", 0), ("Hello", 0)]
    leaf = (
        await db.execute(
            select(models.ChatMessage.content).where(
                models.ChatMessage.id == (await refreshed(db, branching_chat)).active_leaf_id
            )
        )
    ).scalar_one()
    assert leaf == "Hello"


async def test_patch_switch_branch_out_of_range_sibling_returns_400(
    client: httpx.AsyncClient, db: AsyncSession, branching_chat
):
    response = await client.patch(
        await history_url(db, branching_chat), json={"index": 1, "sibling_index": 5}
    )

    assert response.status_code == 400
    assert response.json() == {"error": "That branch no longer exists — the conversation may have changed."}


async def test_patch_switch_branch_missing_body_params_returns_400(
    client: httpx.AsyncClient, db: AsyncSession, branching_chat
):
    response = await client.patch(await history_url(db, branching_chat), json={"index": 1})

    assert response.status_code == 400
    assert response.json() == {
        "error": "Missing required body parameters",
        "missing_params": ["sibling_index"],
    }


@pytest.mark.parametrize(
    "body",
    [
        {"index": "1", "sibling_index": 0},
        {"index": True, "sibling_index": 0},
        {"index": 1.5, "sibling_index": 0},
    ],
)
async def test_patch_switch_branch_non_integer_params_returns_400(
    client: httpx.AsyncClient, db: AsyncSession, branching_chat, body
):
    response = await client.patch(await history_url(db, branching_chat), json=body)

    assert response.status_code == 400


async def test_patch_switch_branch_chat_not_found_returns_404(
    client: httpx.AsyncClient, db: AsyncSession, chat
):
    response = await client.patch(
        f"/chat-history/{await model_name(db, chat)}/999999", json={"index": 0, "sibling_index": 0}
    )

    assert response.status_code == 404
    assert response.json() == {"error": "Chat not found"}


async def test_patch_switch_branch_unknown_model_returns_404(client: httpx.AsyncClient, branching_chat):
    response = await client.patch(
        f"/chat-history/does-not-exist/{branching_chat.id}", json={"index": 0, "sibling_index": 0}
    )

    assert response.status_code == 404


# The chat list


async def test_get_all_chats_includes_persona(client: httpx.AsyncClient, db: AsyncSession, chat):
    persona = models.Persona(name="Pirate", system_prompt="Arr.")
    db.add(persona)
    await db.commit()
    chat.persona_id = persona.id
    await db.commit()

    response = await client.get(await all_chats_url(db, chat))

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": chat.id,
            "title": "Test chat",
            "persona": {"id": persona.id, "name": "Pirate"},
            "temperature": None,
            "num_ctx": None,
            "top_p": None,
            "seed": None,
            "active_collections": [],
            "tools_enabled": False,
            "memory_mode": "trim",
            "think_level": "",
            "continued_from": None,
        }
    ]


async def test_get_all_chats_persona_null_when_unset(client: httpx.AsyncClient, db: AsyncSession, chat):
    assert (await client.get(await all_chats_url(db, chat))).json()[0]["persona"] is None


async def test_get_all_chats_includes_tools_enabled(client: httpx.AsyncClient, db: AsyncSession, chat):
    chat.tools_enabled = True
    await db.commit()

    assert (await client.get(await all_chats_url(db, chat))).json()[0]["tools_enabled"] is True


async def test_get_all_chats_includes_active_collections(client: httpx.AsyncClient, db: AsyncSession, chat):
    collection = await add_collection(db, "Manuals")
    await db.execute(
        models.chat_history_active_collection.insert().values(
            chat_history_id=chat.id, document_collection_id=collection.id
        )
    )
    await db.commit()

    body = (await client.get(await all_chats_url(db, chat))).json()

    assert body[0]["active_collections"] == [{"id": collection.id, "name": "Manuals"}]


async def test_a_continued_chat_says_where_it_came_from(client: httpx.AsyncClient, db: AsyncSession, chat):
    continued = await make_chat(
        db, await db.get(models.AIModel, chat.ai_model_id), title="Next", seed_source_id=chat.id
    )

    body = {c["id"]: c for c in (await client.get(await all_chats_url(db, chat))).json()}

    assert body[continued.id]["continued_from"] == {"id": chat.id, "title": "Test chat"}
    assert body[chat.id]["continued_from"] is None


async def test_chats_are_listed_most_recently_active_first(client: httpx.AsyncClient, db: AsyncSession):
    model = await make_model(db, "llama3.1")
    old = await make_chat(db, model, title="old")
    new = await make_chat(db, model, title="new")
    for chat, minutes in ((old, 120), (new, 5)):
        await db.execute(
            update(models.ChatHistory)
            .where(models.ChatHistory.id == chat.id)
            .values(last_update_time=minutes_ago(minutes))
        )
    await db.commit()

    body = (await client.get("/all-chats/llama3.1")).json()

    assert [c["title"] for c in body] == ["new", "old"]


async def test_the_list_only_has_this_models_chats(client: httpx.AsyncClient, db: AsyncSession, chat):
    other = await make_model(db, "other")
    await make_chat(db, other, title="elsewhere")

    assert [c["title"] for c in (await client.get(await all_chats_url(db, chat))).json()] == ["Test chat"]


async def test_the_list_costs_the_same_number_of_queries_for_many_chats(
    client: httpx.AsyncClient, db: AsyncSession
):
    model = await make_model(db, "llama3.1")
    collection = await add_collection(db)
    source = await make_chat(db, model)
    for n in range(10):
        chat = await make_chat(db, model, title=f"chat {n}", seed_source_id=source.id)
        await db.execute(
            models.chat_history_active_collection.insert().values(
                chat_history_id=chat.id, document_collection_id=collection.id
            )
        )
    await db.commit()
    statements = []

    def record(conn, cursor, statement, *args):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", record)
    try:
        response = await client.get("/all-chats/llama3.1")
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", record)

    assert len(response.json()) == 11
    assert len([s for s in statements if s.lstrip().upper().startswith("SELECT")]) <= 4


async def test_a_model_with_no_chats_lists_none(client: httpx.AsyncClient, db: AsyncSession):
    await make_model(db, "llama3.1")

    assert (await client.get("/all-chats/llama3.1")).json() == []


async def test_listing_for_an_unknown_model_is_a_404(client: httpx.AsyncClient):
    assert (await client.get("/all-chats/nothing")).status_code == 404


# Creating and deleting chats


async def test_a_new_chat_gets_the_default_title(client: httpx.AsyncClient, db: AsyncSession):
    await make_model(db, "llama3.1")

    response = await client.post("/all-chats/llama3.1", json={})

    assert response.status_code == 201
    body = response.json()
    assert body["title"] == "New chat"
    assert body["ai_model"]["model"] == "llama3.1"
    assert body["ai_model"]["versions"] == []
    assert set(body) == {"id", "ai_model", "title", "last_update_time"}


async def test_a_new_chat_can_be_created_without_any_body(client: httpx.AsyncClient, db: AsyncSession):
    await make_model(db, "llama3.1")

    response = await client.post("/all-chats/llama3.1")

    assert response.status_code == 201
    assert response.json()["title"] == "New chat"


async def test_a_new_chat_can_be_given_a_title(client: httpx.AsyncClient, db: AsyncSession):
    await make_model(db, "llama3.1")

    response = await client.post("/all-chats/llama3.1", json={"title": "  Trip to Rome "})

    assert response.json()["title"] == "Trip to Rome"
    assert await count(db, models.ChatHistory) == 1


async def test_a_new_chat_carries_its_models_versions(client: httpx.AsyncClient, db: AsyncSession):
    model = await make_model(db, "llama3.1")
    db.add(models.AIModelVersion(ai_model_id=model.id, parameters="8b", size="4.6 GB"))
    await db.commit()

    response = await client.post("/all-chats/llama3.1", json={})

    assert [v["parameters"] for v in response.json()["ai_model"]["versions"]] == ["8b"]


async def test_a_new_chat_cannot_have_a_blank_title(client: httpx.AsyncClient, db: AsyncSession):
    await make_model(db, "llama3.1")

    response = await client.post("/all-chats/llama3.1", json={"title": "   "})

    assert response.status_code == 400
    assert await count(db, models.ChatHistory) == 0


async def test_creating_a_chat_for_an_unknown_model_is_a_404(client: httpx.AsyncClient):
    assert (await client.post("/all-chats/nothing", json={})).status_code == 404


async def test_delete_chat_removes_it_and_its_messages(
    client: httpx.AsyncClient, db: AsyncSession, branching_chat
):
    other = await make_chat(db, await db.get(models.AIModel, branching_chat.ai_model_id), title="keeper")

    response = await client.request(
        "DELETE", await all_chats_url(db, branching_chat), json={"chat_id": branching_chat.id}
    )

    assert response.status_code == 200
    assert response.json() == {"status": "Chat deleted"}
    assert [c for c in (await db.execute(select(models.ChatHistory.title))).scalars()] == ["keeper"]
    assert await count(db, models.ChatMessage) == 0
    assert other.id


async def test_deleting_a_chat_takes_the_collections_made_for_it_and_frees_its_continuations(
    client: httpx.AsyncClient, db: AsyncSession, chat
):
    embedder = await make_model(db, "embedder", is_embedding=True)
    db.add(
        models.DocumentCollection(
            name="Owned", chat_id=chat.id, embedding_model_id=embedder.id, embedding_parameters="x"
        )
    )
    continued = await make_chat(db, await db.get(models.AIModel, chat.ai_model_id), seed_source_id=chat.id)
    await db.commit()

    await client.request("DELETE", await all_chats_url(db, chat), json={"chat_id": chat.id})

    assert await count(db, models.DocumentCollection) == 0
    assert (await refreshed(db, continued)).seed_source_id is None


async def test_delete_chat_that_does_not_exist_is_a_404(client: httpx.AsyncClient, db: AsyncSession, chat):
    response = await client.request("DELETE", await all_chats_url(db, chat), json={"chat_id": 999999})

    assert response.status_code == 404
    assert response.json() == {"error": "Chat not found"}


async def test_a_chat_cannot_be_deleted_through_another_models_url(
    client: httpx.AsyncClient, db: AsyncSession, chat
):
    await make_model(db, "other")

    response = await client.request("DELETE", "/all-chats/other", json={"chat_id": chat.id})

    assert response.status_code == 404
    assert await count(db, models.ChatHistory) == 1


async def test_delete_chat_needs_a_chat_id(client: httpx.AsyncClient, db: AsyncSession, chat):
    response = await client.request("DELETE", await all_chats_url(db, chat), json={})

    assert response.status_code == 400
    assert response.json()["missing_params"] == ["chat_id"]


async def test_delete_chat_accepts_the_id_as_text_like_the_ui_sends_it(
    client: httpx.AsyncClient, db: AsyncSession, chat
):
    response = await client.request("DELETE", await all_chats_url(db, chat), json={"chat_id": str(chat.id)})

    assert response.status_code == 200


# Changing a chat's settings


async def put(client: httpx.AsyncClient, db: AsyncSession, chat: models.ChatHistory, **fields):
    return await client.put(await all_chats_url(db, chat), json={"id": chat.id, **fields})


async def test_put_all_chats_sets_memory_mode(client: httpx.AsyncClient, db: AsyncSession, chat):
    response = await put(client, db, chat, memory_mode="summarize")

    assert response.status_code == 200
    assert response.json() == {"status": "Chat updated"}
    assert (await refreshed(db, chat)).memory_mode == "summarize"


async def test_put_all_chats_rejects_unknown_memory_mode(client: httpx.AsyncClient, db: AsyncSession, chat):
    response = await put(client, db, chat, memory_mode="forever")

    assert response.status_code == 400
    assert (await refreshed(db, chat)).memory_mode == "trim"


async def test_put_all_chats_renames_the_chat(client: httpx.AsyncClient, db: AsyncSession, chat):
    await put(client, db, chat, title="Renamed")

    assert (await refreshed(db, chat)).title == "Renamed"


async def test_a_title_is_stored_as_sent(client: httpx.AsyncClient, db: AsyncSession, chat):
    await put(client, db, chat, title="  spaced  ")

    assert (await refreshed(db, chat)).title == "  spaced  "


async def test_put_all_chats_sets_persona(client: httpx.AsyncClient, db: AsyncSession, chat):
    persona = models.Persona(name="Pirate", system_prompt="Arr.")
    db.add(persona)
    await db.commit()

    response = await put(client, db, chat, persona_id=persona.id)

    assert response.status_code == 200
    stored = await refreshed(db, chat)
    assert stored.persona_id == persona.id
    # Only persona_id was sent — title must be untouched.
    assert stored.title == "Test chat"


async def test_put_all_chats_clears_persona(client: httpx.AsyncClient, db: AsyncSession, chat):
    persona = models.Persona(name="Pirate", system_prompt="Arr.")
    db.add(persona)
    await db.commit()
    chat.persona_id = persona.id
    await db.commit()

    response = await put(client, db, chat, persona_id=None)

    assert response.status_code == 200
    assert (await refreshed(db, chat)).persona_id is None


async def test_put_all_chats_unknown_persona_returns_404(client: httpx.AsyncClient, db: AsyncSession, chat):
    response = await put(client, db, chat, persona_id=999999)

    assert response.status_code == 404
    assert response.json() == {"error": "Persona not found"}


async def test_put_all_chats_no_fields_returns_400(client: httpx.AsyncClient, db: AsyncSession, chat):
    response = await put(client, db, chat)

    assert response.status_code == 400
    assert response.json()["error"].startswith("Nothing to update")


async def test_put_all_chats_needs_an_id(client: httpx.AsyncClient, db: AsyncSession, chat):
    response = await client.put(await all_chats_url(db, chat), json={"title": "x"})

    assert response.status_code == 400
    assert response.json()["missing_params"] == ["id"]


async def test_put_all_chats_for_a_chat_that_does_not_exist_is_a_404(
    client: httpx.AsyncClient, db: AsyncSession, chat
):
    response = await client.put(await all_chats_url(db, chat), json={"id": 999999, "title": "x"})

    assert response.status_code == 404
    assert response.json() == {"error": "Chat not found"}


async def test_put_all_chats_sets_generation_params(client: httpx.AsyncClient, db: AsyncSession, chat):
    response = await put(client, db, chat, temperature=0.8, num_ctx=8192, top_p=0.9, seed=42)

    assert response.status_code == 200
    stored = await refreshed(db, chat)
    assert (stored.temperature, stored.num_ctx, stored.top_p, stored.seed) == (0.8, 8192, 0.9, 42)


async def test_whole_numbers_are_fine_for_the_decimal_params(
    client: httpx.AsyncClient, db: AsyncSession, chat
):
    response = await put(client, db, chat, temperature=1, top_p=1)

    assert response.status_code == 200
    stored = await refreshed(db, chat)
    assert (stored.temperature, stored.top_p) == (1.0, 1.0)


async def test_a_fractional_whole_number_param_is_cut_like_it_always_was(
    client: httpx.AsyncClient, db: AsyncSession, chat
):
    await put(client, db, chat, num_ctx=4096.7, seed=7.9)

    stored = await refreshed(db, chat)
    assert (stored.num_ctx, stored.seed) == (4096, 7)


async def test_put_all_chats_clears_generation_param_with_null(
    client: httpx.AsyncClient, db: AsyncSession, chat
):
    chat.temperature = 0.8
    chat.num_ctx = 4096
    await db.commit()

    response = await put(client, db, chat, temperature=None, num_ctx=None)

    assert response.status_code == 200
    stored = await refreshed(db, chat)
    assert (stored.temperature, stored.num_ctx) == (None, None)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", -0.1),
        ("temperature", 2.1),
        ("top_p", -0.1),
        ("top_p", 1.1),
        ("num_ctx", 0),
        ("num_ctx", 1023),
        ("temperature", "hot"),
        ("temperature", "0.5"),
        ("temperature", True),
        ("num_ctx", True),
        ("num_ctx", "4096"),
        ("seed", False),
        ("seed", 2**40),
    ],
)
async def test_put_all_chats_rejects_invalid_generation_param(
    client: httpx.AsyncClient, db: AsyncSession, chat, field, value
):
    response = await put(client, db, chat, **{field: value})

    assert response.status_code == 400
    assert getattr(await refreshed(db, chat), field) is None


async def test_a_not_a_number_temperature_is_refused(client: httpx.AsyncClient, db: AsyncSession, chat):
    response = await client.put(
        await all_chats_url(db, chat),
        content=f'{{"id": {chat.id}, "temperature": NaN}}',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 400


async def test_put_all_chats_seed_has_no_upper_or_lower_bound_within_the_database_range(
    client: httpx.AsyncClient, db: AsyncSession, chat
):
    assert (await put(client, db, chat, seed=-12345)).status_code == 200
    assert (await refreshed(db, chat)).seed == -12345


async def test_put_all_chats_sets_active_collections(client: httpx.AsyncClient, db: AsyncSession, chat):
    first, second = await add_collection(db, "One"), await add_collection(db, "Two")

    response = await put(client, db, chat, collection_ids=[first.id, second.id])

    assert response.status_code == 200
    listed = (await client.get(await all_chats_url(db, chat))).json()[0]["active_collections"]
    assert sorted(c["name"] for c in listed) == ["One", "Two"]


async def test_put_all_chats_replaces_the_active_collections(
    client: httpx.AsyncClient, db: AsyncSession, chat
):
    first, second = await add_collection(db, "One"), await add_collection(db, "Two")
    await put(client, db, chat, collection_ids=[first.id])

    await put(client, db, chat, collection_ids=[second.id])

    listed = (await client.get(await all_chats_url(db, chat))).json()[0]["active_collections"]
    assert [c["name"] for c in listed] == ["Two"]


async def test_put_all_chats_clears_active_collections_with_empty_list(
    client: httpx.AsyncClient, db: AsyncSession, chat
):
    collection = await add_collection(db)
    await put(client, db, chat, collection_ids=[collection.id])

    response = await put(client, db, chat, collection_ids=[])

    assert response.status_code == 200
    assert (await client.get(await all_chats_url(db, chat))).json()[0]["active_collections"] == []


async def test_the_same_collection_twice_is_activated_once(client: httpx.AsyncClient, db: AsyncSession, chat):
    collection = await add_collection(db)

    response = await put(client, db, chat, collection_ids=[collection.id, collection.id])

    assert response.status_code == 200
    assert len((await client.get(await all_chats_url(db, chat))).json()[0]["active_collections"]) == 1


async def test_put_all_chats_unknown_collection_id_returns_404(
    client: httpx.AsyncClient, db: AsyncSession, chat
):
    response = await put(client, db, chat, collection_ids=[999999])

    assert response.status_code == 404
    assert response.json() == {"error": "One or more collections not found"}


@pytest.mark.parametrize("value", ["1", [True], ["1"], [1.5], 5])
async def test_put_all_chats_collection_ids_must_be_a_list_of_integers(
    client: httpx.AsyncClient, db: AsyncSession, chat, value
):
    assert (await put(client, db, chat, collection_ids=value)).status_code == 400


async def test_put_all_chats_enables_and_disables_tools(client: httpx.AsyncClient, db: AsyncSession, chat):
    assert (await put(client, db, chat, tools_enabled=True)).status_code == 200
    assert (await refreshed(db, chat)).tools_enabled is True

    assert (await put(client, db, chat, tools_enabled=False)).status_code == 200
    assert (await refreshed(db, chat)).tools_enabled is False


@pytest.mark.parametrize("value", ["yes", 1, None])
async def test_put_all_chats_tools_enabled_rejects_non_boolean(
    client: httpx.AsyncClient, db: AsyncSession, chat, value
):
    response = await put(client, db, chat, tools_enabled=value)

    assert response.status_code == 400
    assert (await refreshed(db, chat)).tools_enabled is False


async def test_put_all_chats_sets_think_level(client: httpx.AsyncClient, db: AsyncSession, chat):
    assert (await put(client, db, chat, think_level="high")).status_code == 200
    assert (await refreshed(db, chat)).think_level == "high"


@pytest.mark.parametrize("value", ["x" * 33, 5, None])
async def test_put_all_chats_think_level_must_be_a_short_string(
    client: httpx.AsyncClient, db: AsyncSession, chat, value
):
    response = await put(client, db, chat, think_level=value)

    assert response.status_code == 400
    assert (await refreshed(db, chat)).think_level == ""


async def test_moving_a_chat_to_another_model_keeps_who_wrote_what(
    client: httpx.AsyncClient, db: AsyncSession, chat
):
    other = await make_model(db, "mistral-move-test", index=2)
    user = await make_message(db, chat, "user", "Hi")
    await make_message(db, chat, "assistant", "Hello", user, activate=True, model_label="llama3.1:8b")

    response = await put(client, db, chat, move_to_model="mistral-move-test")

    assert response.status_code == 200
    assert (await refreshed(db, chat)).ai_model_id == other.id
    history = (await client.get(f"/chat-history/mistral-move-test/{chat.id}")).json()
    assert history[1]["model"] == "llama3.1:8b"
    assert (await client.get(f"/chat-history/llama3.1/{chat.id}")).json() == []


async def test_chats_cannot_move_to_an_embedding_model(client: httpx.AsyncClient, db: AsyncSession, chat):
    embedder = await make_model(db, "embedder", is_embedding=True)

    response = await put(client, db, chat, move_to_model=embedder.model)

    assert response.status_code == 400
    assert response.json() == {"error": "Embedding models can't chat — pick a chat model."}
    assert (await refreshed(db, chat)).ai_model_id != embedder.id


async def test_moving_a_chat_to_an_unknown_model_is_a_404(client: httpx.AsyncClient, db: AsyncSession, chat):
    response = await put(client, db, chat, move_to_model="nothing")

    assert response.status_code == 404
    assert response.json() == {"error": "AI model not found"}


async def test_an_update_that_is_refused_changes_nothing_at_all(
    client: httpx.AsyncClient, db: AsyncSession, chat
):
    collection = await add_collection(db)

    response = await put(
        client,
        db,
        chat,
        title="New name",
        tools_enabled=True,
        collection_ids=[collection.id],
        persona_id=999999,
    )

    assert response.status_code == 404
    stored = await refreshed(db, chat)
    assert (stored.title, stored.tools_enabled) == ("Test chat", False)
    assert (await client.get(await all_chats_url(db, chat))).json()[0]["active_collections"] == []


async def test_an_update_with_one_invalid_field_applies_none_of_them(
    client: httpx.AsyncClient, db: AsyncSession, chat
):
    response = await put(client, db, chat, title="New name", temperature=9)

    assert response.status_code == 400
    assert (await refreshed(db, chat)).title == "Test chat"


async def test_changing_settings_does_not_reorder_the_chat_list(
    client: httpx.AsyncClient, db: AsyncSession, chat
):
    old = minutes_ago(90)
    await db.execute(
        update(models.ChatHistory).where(models.ChatHistory.id == chat.id).values(last_update_time=old)
    )
    await db.commit()

    await put(client, db, chat, title="Renamed", temperature=0.5)

    assert abs((await refreshed(db, chat)).last_update_time - old) < datetime.timedelta(seconds=1)


# Continuing a chat


async def test_continue_chat_copies_settings_and_links_the_source(
    client: httpx.AsyncClient, db: AsyncSession, chat
):
    persona = models.Persona(name="Pirate", system_prompt="Arr.")
    db.add(persona)
    await db.commit()
    collection = await add_collection(db)
    chat.persona_id = persona.id
    chat.num_ctx = 16384
    chat.memory_mode = "summarize"
    chat.temperature = 0.4
    chat.tools_enabled = True
    chat.think_level = "high"
    await db.commit()
    await put(client, db, chat, collection_ids=[collection.id])

    response = await client.post(f"/chat-continue/{await model_name(db, chat)}", json={"chat_id": chat.id})

    assert response.status_code == 201
    body = response.json()
    assert body["title"] == "Test chat (continued)"
    continued = (
        await db.execute(select(models.ChatHistory).where(models.ChatHistory.id == body["id"]))
    ).scalar_one()
    assert continued.seed_source_id == chat.id
    assert (continued.persona_id, continued.num_ctx, continued.memory_mode) == (
        persona.id,
        16384,
        "summarize",
    )
    assert (continued.temperature, continued.tools_enabled, continued.think_level) == (0.4, True, "high")
    assert continued.ai_model_id == chat.ai_model_id
    listed = {c["id"]: c for c in (await client.get(await all_chats_url(db, chat))).json()}
    assert [c["name"] for c in listed[continued.id]["active_collections"]] == ["Docs"]


async def test_continuing_does_not_copy_the_messages(client: httpx.AsyncClient, db: AsyncSession, chat):
    await make_exchange(db, chat, "Hi", "Hello")

    body = (
        await client.post(f"/chat-continue/{await model_name(db, chat)}", json={"chat_id": chat.id})
    ).json()

    assert (await client.get(f"/chat-history/llama3.1/{body['id']}")).json() == []
    assert await count(db, models.ChatMessage) == 2


async def test_continuing_a_chat_that_does_not_exist_is_a_404(
    client: httpx.AsyncClient, db: AsyncSession, chat
):
    response = await client.post(f"/chat-continue/{await model_name(db, chat)}", json={"chat_id": 999999})

    assert response.status_code == 404
    assert response.json() == {"error": "Chat not found"}


async def test_continuing_needs_a_chat_id(client: httpx.AsyncClient, db: AsyncSession, chat):
    response = await client.post(f"/chat-continue/{await model_name(db, chat)}", json={})

    assert response.status_code == 400
    assert response.json()["missing_params"] == ["chat_id"]


# From the Compare page


async def test_compare_answer_becomes_a_chat(client: httpx.AsyncClient, db: AsyncSession, chat):
    response = await client.post(
        f"/chat-from-compare/{await model_name(db, chat)}",
        json={"parameters": "8b", "prompt": "Capital of France?", "response": "Paris."},
    )

    assert response.status_code == 201
    new_id = response.json()["id"]
    assert response.json()["title"] == "New chat"
    history = (await client.get(f"/chat-history/llama3.1/{new_id}")).json()
    assert [(m["role"], m["content"], m["model"]) for m in history] == [
        ("user", "Capital of France?", ""),
        ("assistant", "Paris.", "llama3.1:8b"),
    ]


async def test_compare_answer_is_stored_as_written(client: httpx.AsyncClient, db: AsyncSession, chat):
    response = await client.post(
        "/chat-from-compare/llama3.1", json={"parameters": "8b", "prompt": "  Hi\n", "response": " Hello "}
    )

    history = (await client.get(f"/chat-history/llama3.1/{response.json()['id']}")).json()
    assert [m["content"] for m in history] == ["  Hi\n", " Hello "]


@pytest.mark.parametrize("field", ["parameters", "prompt", "response"])
async def test_compare_answer_needs_every_field_to_have_text(
    client: httpx.AsyncClient, db: AsyncSession, chat, field
):
    body = {"parameters": "8b", "prompt": "Hi", "response": "Hello", field: "   "}

    response = await client.post("/chat-from-compare/llama3.1", json=body)

    assert response.status_code == 400
    assert await count(db, models.ChatHistory) == 1


async def test_compare_answer_for_an_unknown_model_is_a_404(client: httpx.AsyncClient):
    response = await client.post(
        "/chat-from-compare/nothing", json={"parameters": "8b", "prompt": "Hi", "response": "Hello"}
    )

    assert response.status_code == 404


# How full the chat's memory is


async def test_chat_context_estimates_memory_use_before_sending(
    client: httpx.AsyncClient, db: AsyncSession, chat, monkeypatch
):
    monkeypatch.setattr(
        runtime,
        "get_model_info",
        AsyncMock(return_value={"context_length": 4096, "thinking": None, "capabilities": []}),
    )
    await make_exchange(db, chat, "word " * 700, "ok")

    response = await client.get(f"/chat-context/{await model_name(db, chat)}/{chat.id}?parameters=8b")

    assert response.status_code == 200
    body = response.json()
    assert body["limit"] == 4096
    assert body["model_max"] == 4096
    assert body["preset"] == "short"
    assert 1000 < body["used"] < 1100
    assert body["presets"]["standard"] == 8192
    assert body["summarized"] is False
    assert body["dropped"] == 0
    assert body["thinking"] is None
    assert body["capabilities"] == []
    runtime.get_model_info.assert_awaited_once_with("llama3.1", "8b")


async def test_chat_context_without_parameters_does_not_ask_the_model(
    client: httpx.AsyncClient, db: AsyncSession, chat, monkeypatch
):
    asked = AsyncMock()
    monkeypatch.setattr(runtime, "get_model_info", asked)

    response = await client.get(f"/chat-context/{await model_name(db, chat)}/{chat.id}")

    body = response.json()
    assert (body["limit"], body["model_max"], body["thinking"], body["capabilities"]) == (
        8192,
        None,
        None,
        None,
    )
    asked.assert_not_awaited()


async def test_chat_context_uses_the_chats_own_context_size(
    client: httpx.AsyncClient, db: AsyncSession, chat
):
    chat.num_ctx = 16384
    await db.commit()

    body = (await client.get(f"/chat-context/{await model_name(db, chat)}/{chat.id}")).json()

    assert (body["limit"], body["preset"]) == (16384, "long")


async def test_chat_context_counts_the_summary_and_hides_what_it_covers(
    client: httpx.AsyncClient, db: AsyncSession, chat
):
    first, _ = await make_exchange(db, chat, "word " * 600, "ok")
    _, reply = await make_exchange(db, chat, "next", "fine", parent=(await _last_message(db, chat)))
    chat.memory_mode = "summarize"
    chat.memory_summary = "They discussed words."
    chat.memory_summary_until_id = reply.id - 1
    await db.commit()

    body = (await client.get(f"/chat-context/{await model_name(db, chat)}/{chat.id}")).json()

    assert body["summarized"] is True
    assert body["used"] < 100
    assert first.id


async def _last_message(db: AsyncSession, chat: models.ChatHistory) -> models.ChatMessage:
    return (
        await db.execute(
            select(models.ChatMessage)
            .where(models.ChatMessage.chat_id == chat.id)
            .order_by(models.ChatMessage.id.desc())
            .limit(1)
        )
    ).scalar_one()


async def test_chat_context_for_a_chat_that_does_not_exist_is_a_404(
    client: httpx.AsyncClient, db: AsyncSession, chat
):
    response = await client.get(f"/chat-context/{await model_name(db, chat)}/999999")

    assert response.status_code == 404
    assert response.json() == {"error": "Chat not found"}


# Search


async def test_get_search_finds_matches_for_the_model(client: httpx.AsyncClient, db: AsyncSession, chat):
    await make_message(db, chat, "user", "tell me about quantum computing")

    response = await client.get(f"/search/{await model_name(db, chat)}?q=quantum")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["chat_title"] == "Test chat"
    assert body[0]["chat_id"] == str(chat.id)
    assert "quantum" in body[0]["snippet"].lower()


async def test_get_search_unknown_model_returns_404(client: httpx.AsyncClient):
    response = await client.get("/search/does-not-exist?q=hi")

    assert response.status_code == 404
    assert response.json() == {"error": "AI model not found"}


async def test_get_search_missing_query_returns_empty_list(client: httpx.AsyncClient, db: AsyncSession, chat):
    response = await client.get(f"/search/{await model_name(db, chat)}")

    assert response.status_code == 200
    assert response.json() == []


async def test_get_search_no_matches_returns_empty_list(client: httpx.AsyncClient, db: AsyncSession, chat):
    await make_message(db, chat, "user", "hello there")

    response = await client.get(f"/search/{await model_name(db, chat)}?q=nonexistent")

    assert response.status_code == 200
    assert response.json() == []
