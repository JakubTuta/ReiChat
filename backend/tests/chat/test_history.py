import datetime

import pytest
from sqlalchemy import event, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat import history
from app.db import models
from app.db.session import engine
from tests.factories import make_chat, make_exchange, make_message, make_model, minutes_ago

pytestmark = pytest.mark.usefixtures("db")


async def fresh(db: AsyncSession, chat: models.ChatHistory) -> models.ChatHistory:
    return (
        await db.execute(
            select(models.ChatHistory)
            .where(models.ChatHistory.id == chat.id)
            .execution_options(populate_existing=True)
        )
    ).scalar_one()


async def age(db: AsyncSession, chat: models.ChatHistory, minutes: int = 60) -> datetime.datetime:
    """Backdates the chat so a later write that touches it is easy to spot."""
    old = minutes_ago(minutes)
    await db.execute(
        update(models.ChatHistory).where(models.ChatHistory.id == chat.id).values(last_update_time=old)
    )
    await db.commit()
    return old


async def messages_of(db: AsyncSession, chat: models.ChatHistory) -> list[models.ChatMessage]:
    return list(
        (
            await db.execute(
                select(models.ChatMessage)
                .where(models.ChatMessage.chat_id == chat.id)
                .order_by(models.ChatMessage.id)
                .execution_options(populate_existing=True)
            )
        ).scalars()
    )


def test_messages_are_serialized_for_the_model_with_their_ids():
    message = models.ChatMessage(id=7, chat_id=1, role="user", content="hi", image="")
    with_image = models.ChatMessage(id=8, chat_id=1, role="user", content="look", image="data:x")

    assert history.deserialize_messages([message, with_image]) == [
        {"id": 7, "role": "user", "content": "hi", "image": ""},
        {"id": 8, "role": "user", "content": "look", "image": "data:x"},
    ]


async def test_a_chat_is_only_found_under_its_own_model(db: AsyncSession):
    mine, other = await make_model(db, "mine"), await make_model(db, "other")
    chat = await make_chat(db, mine)

    assert (await history.get_chat_history(db, mine, chat.id)).id == chat.id
    assert await history.get_chat_history(db, other, chat.id) is None
    assert await history.get_chat_history(db, mine, chat.id + 100) is None
    assert await history.get_chat_history(db, None, chat.id) is None


async def test_chats_are_listed_most_recently_updated_first(db: AsyncSession):
    model = await make_model(db)
    old = await make_chat(db, model, title="old")
    new = await make_chat(db, model, title="new")
    await age(db, old, 120)
    await age(db, new, 5)
    await make_chat(db, await make_model(db, "elsewhere"), title="other model")

    chats = (await db.execute(history.chats_of_model(model.id, newest_first=True))).scalars().all()

    assert [c.title for c in chats] == ["new", "old"]


# The branching tree


async def build_tree(db: AsyncSession):
    """
    root(user) ── a1(assistant, old branch) ── u2 ── a2
              └── a1b(assistant, ACTIVE)
    """
    chat = await make_chat(db)
    root = await make_message(db, chat, "user", "Hi")
    a1 = await make_message(db, chat, "assistant", "first reply", root)
    u2 = await make_message(db, chat, "user", "and then?", a1)
    a2 = await make_message(db, chat, "assistant", "deep reply", u2)
    a1b = await make_message(db, chat, "assistant", "regenerated reply", root, activate=True)
    return chat, root, a1, u2, a2, a1b


async def test_the_active_path_runs_from_the_root_to_the_active_leaf(db: AsyncSession):
    chat, root, _, _, _, a1b = await build_tree(db)

    path = await history.get_messages_for_chat(db, chat)

    assert [m.id for m in path] == [root.id, a1b.id]


async def test_an_old_branch_is_never_on_the_path_but_is_still_stored(db: AsyncSession):
    chat, _, a1, u2, a2, _ = await build_tree(db)

    path_ids = {m.id for m in await history.get_messages_for_chat(db, chat)}

    assert not path_ids & {a1.id, u2.id, a2.id}
    assert len(await messages_of(db, chat)) == 5


async def test_the_whole_path_costs_two_queries_however_long_the_chat(db: AsyncSession):
    chat = await make_chat(db)
    parent = None
    for n in range(20):
        parent = await make_message(
            db, chat, "user" if n % 2 == 0 else "assistant", f"m{n}", parent, activate=True
        )
    statements = []

    def record(conn, cursor, statement, *args):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", record)
    try:
        path = await history.get_messages_for_chat(db, chat)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", record)

    assert len(path) == 20
    assert len(statements) == 2


async def test_siblings_share_a_parent_and_come_oldest_first(db: AsyncSession):
    _, root, a1, _, _, a1b = await build_tree(db)

    siblings = await history.get_branch_siblings(db, a1b)

    assert [m.id for m in siblings] == [a1.id, a1b.id]
    assert [m.id for m in await history.get_branch_siblings(db, root)] == [root.id]


async def test_the_leaf_of_a_branch_is_the_tip_of_the_most_recent_work_on_it(db: AsyncSession):
    chat, root, a1, u2, a2, a1b = await build_tree(db)
    # Two replies under a2: the newer one is where the branch continues.
    await make_message(db, chat, "user", "older follow-up", a2)
    newest = await make_message(db, chat, "user", "newer follow-up", a2)

    assert (await history.get_branch_leaf(db, a1)).id == newest.id
    assert (await history.get_branch_leaf(db, a1b)).id == a1b.id
    assert (await history.get_branch_leaf(db, root)).id == a1b.id  # the newest child of root wins


async def test_the_display_path_shows_each_messages_place_among_its_siblings(db: AsyncSession):
    chat, *_ = await build_tree(db)

    shown = await history.serialize_active_path_for_display(db, chat)

    assert shown == [
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
            "content": "regenerated reply",
            "image": "",
            "thinking": "",
            "model": "",
            "sibling_count": 2,
            "sibling_index": 1,
        },
    ]


# Saving an exchange


async def test_a_finished_exchange_is_saved_under_the_active_leaf(db: AsyncSession):
    chat = await make_chat(db)
    _, previous = await make_exchange(db, chat, "Hi", "Hello")
    await age(db, chat)
    chat = await fresh(db, chat)

    await history.add_messages_to_history(
        db, chat, "How are you?", "Fine.", image="data:x", thinking="hm", model_label="llama3.1:8b"
    )

    stored = await messages_of(db, chat)
    user, assistant = stored[-2], stored[-1]
    assert (user.role, user.content, user.image, user.parent_id) == (
        "user",
        "How are you?",
        "data:x",
        previous.id,
    )
    assert (assistant.role, assistant.content, assistant.thinking, assistant.model_label) == (
        "assistant",
        "Fine.",
        "hm",
        "llama3.1:8b",
    )
    assert assistant.parent_id == user.id
    assert user.created_at < assistant.created_at
    refreshed = await fresh(db, chat)
    assert refreshed.active_leaf_id == assistant.id
    assert chat.active_leaf_id == assistant.id


async def test_saving_an_exchange_moves_the_chat_to_the_top_of_the_list(db: AsyncSession):
    chat = await make_chat(db)
    old = await age(db, chat)

    await history.add_messages_to_history(db, chat, "Hi", "Hello")

    assert (await fresh(db, chat)).last_update_time > old


async def test_a_reply_goes_under_the_leaf_the_conversation_was_built_from(db: AsyncSession):
    # Another request moves the active leaf while a reply is still being
    # generated. The chat was loaded earlier, in a session that has since closed.
    chat = await make_chat(db)
    _, reply = await make_exchange(db, chat, "Hi", "Hello")
    chat = await fresh(db, chat)
    db.expunge(chat)
    other = await make_message(db, chat, "assistant", "elsewhere", None)
    await db.execute(
        update(models.ChatHistory).where(models.ChatHistory.id == chat.id).values(active_leaf_id=other.id)
    )
    await db.commit()

    await history.add_messages_to_history(db, chat, "Next", "Answer")

    user = [m for m in await messages_of(db, chat) if m.content == "Next"][0]
    assert user.parent_id == reply.id


async def test_a_regenerated_reply_becomes_a_sibling_and_the_old_one_stays(db: AsyncSession):
    chat = await make_chat(db)
    user, old_reply = await make_exchange(db, chat, "Hi", "first")
    old = await age(db, chat)
    chat = await fresh(db, chat)

    await history.commit_new_assistant_branch(db, chat, user.id, "second", thinking="t", model_label="m:1")

    new_reply = (await messages_of(db, chat))[-1]
    assert (new_reply.parent_id, new_reply.content, new_reply.thinking, new_reply.model_label) == (
        user.id,
        "second",
        "t",
        "m:1",
    )
    assert any(m.id == old_reply.id for m in await messages_of(db, chat))
    refreshed = await fresh(db, chat)
    assert refreshed.active_leaf_id == new_reply.id
    assert refreshed.last_update_time > old


async def test_an_edited_message_branches_a_new_exchange_beside_the_old_one(db: AsyncSession):
    chat = await make_chat(db)
    root = await make_message(db, chat, "user", "first question")
    old_reply = await make_message(db, chat, "assistant", "first answer", root, activate=True)
    chat = await fresh(db, chat)

    await history.commit_new_exchange_branch(
        db, chat, None, "edited question", "edited answer", image="data:x"
    )

    stored = await messages_of(db, chat)
    new_user, new_assistant = stored[-2], stored[-1]
    assert (new_user.parent_id, new_user.content, new_user.image) == (None, "edited question", "data:x")
    assert new_assistant.parent_id == new_user.id
    assert any(m.id == old_reply.id for m in stored)
    assert (await fresh(db, chat)).active_leaf_id == new_assistant.id


# Edit, regenerate, switch


async def test_edit_resend_names_the_parent_and_the_history_before_the_edited_message(db: AsyncSession):
    chat = await make_chat(db)
    _, first_reply = await make_exchange(db, chat, "one", "uno")
    await make_exchange(db, chat, "two", "dos", first_reply)
    chat = await fresh(db, chat)

    parent_id, before = await history.prepare_edit_resend(db, chat, 2)

    assert parent_id == first_reply.id
    assert [(m["role"], m["content"]) for m in before] == [("user", "one"), ("assistant", "uno")]


async def test_editing_the_first_message_branches_from_nothing(db: AsyncSession):
    chat = await make_chat(db)
    await make_exchange(db, chat, "one", "uno")
    chat = await fresh(db, chat)

    assert await history.prepare_edit_resend(db, chat, 0) == (None, [])


@pytest.mark.parametrize("index", [1, 99])
async def test_edit_resend_only_works_on_a_user_message_that_exists(db: AsyncSession, index):
    chat = await make_chat(db)
    await make_exchange(db, chat, "one", "uno")
    chat = await fresh(db, chat)

    assert await history.prepare_edit_resend(db, chat, index) is None


async def test_regenerate_names_the_last_user_message_and_the_history_before_it(db: AsyncSession):
    chat = await make_chat(db)
    _, first_reply = await make_exchange(db, chat, "one", "uno")
    user, _ = await make_exchange(db, chat, "two", "dos", first_reply)
    chat = await fresh(db, chat)

    user_id, content, image, before = await history.prepare_regenerate(db, chat)

    assert (user_id, content, image) == (user.id, "two", "")
    assert [m["content"] for m in before] == ["one", "uno"]


async def test_there_is_nothing_to_regenerate_in_an_empty_chat(db: AsyncSession):
    assert await history.prepare_regenerate(db, await make_chat(db)) is None


async def test_switching_branches_lands_on_the_tip_of_the_chosen_one(db: AsyncSession):
    chat, _, _, _, a2, _ = await build_tree(db)
    chat = await fresh(db, chat)

    shown = await history.switch_branch(db, chat, index=1, sibling_index=0)

    assert [m["content"] for m in shown] == ["Hi", "first reply", "and then?", "deep reply"]
    assert (await fresh(db, chat)).active_leaf_id == a2.id
    assert chat.active_leaf_id == a2.id


@pytest.mark.parametrize(("index", "sibling_index"), [(5, 0), (1, 99)])
async def test_switching_to_a_branch_that_does_not_exist_changes_nothing(
    db: AsyncSession, index, sibling_index
):
    chat, *_, a1b = await build_tree(db)
    chat = await fresh(db, chat)

    assert await history.switch_branch(db, chat, index, sibling_index) is None
    assert (await fresh(db, chat)).active_leaf_id == a1b.id


# Small writes


async def test_a_generated_title_never_replaces_one_the_user_typed(db: AsyncSession):
    renamed = await make_chat(db, title="My own name")
    untitled = await make_chat(db)

    assert await history.save_title_if_untitled(db, renamed, "Generated") is False
    assert await history.save_title_if_untitled(db, untitled, "Generated") is True

    assert (await fresh(db, renamed)).title == "My own name"
    assert (await fresh(db, untitled)).title == "Generated"
    assert untitled.title == "Generated"
    assert renamed.title == "My own name"


async def test_a_memory_summary_is_remembered_with_the_message_it_covers(db: AsyncSession):
    chat = await make_chat(db)
    message = await make_message(db, chat, "user", "x")

    await history.save_memory_summary(db, chat, "They greeted each other.", message.id)

    stored = await fresh(db, chat)
    assert (stored.memory_summary, stored.memory_summary_until_id) == ("They greeted each other.", message.id)
    assert chat.memory_summary == "They greeted each other."


async def test_a_continued_chat_can_read_the_conversation_it_continues(db: AsyncSession):
    source = await make_chat(db, title="source")
    await make_exchange(db, source, "Where is Rome?", "In Italy.")
    continuation = await make_chat(db, seed_source_id=source.id)

    seed = await history.get_seed_source_history(db, continuation)

    assert [(m["role"], m["content"]) for m in seed] == [
        ("user", "Where is Rome?"),
        ("assistant", "In Italy."),
    ]


async def test_a_chat_whose_source_was_deleted_has_no_seed_history(db: AsyncSession):
    source = await make_chat(db)
    continuation = await make_chat(db, seed_source_id=source.id)
    await db.delete(source)
    await db.commit()

    assert await history.get_seed_source_history(db, await fresh(db, continuation)) == []


# Search


def test_the_snippet_centers_the_match_with_ellipses():
    content = "a" * 100 + "NEEDLE" + "b" * 100

    snippet = history._build_search_snippet(content, "needle")

    assert snippet.count("…") == 2
    assert "needle" in snippet.lower()
    assert len(snippet) < len(content)


async def test_search_requires_a_minimum_query_length(db: AsyncSession):
    model = await make_model(db)
    chat = await make_chat(db, model)
    await make_message(db, chat, "user", "a")

    assert await history.search_messages(db, model, "a") == []
    assert await history.search_messages(db, model, "") == []
    assert await history.search_messages(db, model, "  ") == []


async def test_search_finds_matches_case_insensitively_across_chats(db: AsyncSession):
    model = await make_model(db)
    trip = await make_chat(db, model, title="Trip planning")
    recipes = await make_chat(db, model, title="Recipe ideas")
    await make_message(db, trip, "user", "Where should I go in Portugal?")
    await make_message(db, recipes, "assistant", "Try a PORTUGAL-style tart.")
    await make_message(db, recipes, "user", "Something unrelated")

    results = await history.search_messages(db, model, "portugal")

    assert len(results) == 2
    assert {r["chat_title"] for r in results} == {"Trip planning", "Recipe ideas"}
    assert {r["chat_id"] for r in results} == {str(trip.id), str(recipes.id)}
    assert {r["role"] for r in results} == {"user", "assistant"}
    for result in results:
        assert "portugal" in result["snippet"].lower()


async def test_search_is_scoped_to_the_given_model_only(db: AsyncSession):
    model_a, model_b = await make_model(db, "a"), await make_model(db, "b")
    chat_a, chat_b = (
        await make_chat(db, model_a, title="Chat A"),
        await make_chat(db, model_b, title="Chat B"),
    )
    await make_message(db, chat_a, "user", "shared keyword here")
    await make_message(db, chat_b, "user", "shared keyword here too")

    results = await history.search_messages(db, model_a, "shared keyword")

    assert [r["chat_title"] for r in results] == ["Chat A"]


async def test_search_includes_inactive_branches(db: AsyncSession):
    model = await make_model(db)
    chat = await make_chat(db, model, title="Branchy chat")
    root = await make_message(db, chat, "user", "Hi")
    await make_message(db, chat, "assistant", "original findme reply", root)
    await make_message(db, chat, "assistant", "regenerated reply", root, activate=True)

    results = await history.search_messages(db, model, "findme")

    assert len(results) == 1
    assert "findme" in results[0]["snippet"].lower()


async def test_search_treats_percent_and_underscore_literally(db: AsyncSession):
    model = await make_model(db)
    chat = await make_chat(db, model)
    await make_message(db, chat, "user", "we hit 100% of the target")
    await make_message(db, chat, "user", "plain text about the target")
    await make_message(db, chat, "user", "a_b and axb")

    assert [r["snippet"] for r in await history.search_messages(db, model, "100%")] == [
        "we hit 100% of the target"
    ]
    assert [r["snippet"] for r in await history.search_messages(db, model, "a_b")] == ["a_b and axb"]
    assert await history.search_messages(db, model, "%%") == []
