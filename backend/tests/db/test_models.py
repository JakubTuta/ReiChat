import datetime

import pytest
from sqlalchemy import delete, func, select, text
from sqlalchemy.exc import IntegrityError, InvalidRequestError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db import models

pytestmark = pytest.mark.usefixtures("db")


def make_model(name: str = "tiny", **overrides: object) -> models.AIModel:
    return models.AIModel(name=name, model=name, description=f"{name} model", **overrides)


async def make_chat(session: AsyncSession, ai_model: models.AIModel | None = None) -> models.ChatHistory:
    if ai_model is None:
        ai_model = make_model()
        session.add(ai_model)
        await session.flush()
    chat = models.ChatHistory(ai_model_id=ai_model.id)
    session.add(chat)
    await session.flush()
    return chat


async def test_columns_take_their_python_side_defaults(db: AsyncSession):
    chat = await make_chat(db)
    await db.commit()

    row = (await db.execute(select(models.ChatHistory).where(models.ChatHistory.id == chat.id))).scalar_one()
    assert row.title == "New chat"
    assert row.memory_mode == "trim"
    assert row.tools_enabled is False
    assert row.memory_summary == ""
    assert row.temperature is None
    assert row.last_update_time.tzinfo is not None
    assert (
        datetime.timedelta(0)
        <= datetime.datetime.now(datetime.UTC) - row.last_update_time
        < datetime.timedelta(minutes=1)
    )


async def test_timestamps_inside_one_transaction_stay_distinct(db: AsyncSession):
    chat = await make_chat(db)
    first = models.ChatMessage(chat_id=chat.id, role="user", content="hi")
    db.add(first)
    await db.flush()
    second = models.ChatMessage(chat_id=chat.id, role="assistant", content="hello", parent_id=first.id)
    db.add(second)
    await db.flush()

    assert second.created_at > first.created_at


async def test_deleting_a_catalog_model_takes_its_chats_and_messages_with_it(db: AsyncSession):
    ai_model = make_model()
    db.add(ai_model)
    await db.flush()
    chat = await make_chat(db, ai_model)
    db.add(models.ChatMessage(chat_id=chat.id, role="user", content="hi"))
    db.add(models.AIModelVersion(ai_model_id=ai_model.id, parameters="1b", size="1 GB"))
    await db.commit()

    await db.execute(delete(models.AIModel))
    await db.commit()

    for table in (models.ChatHistory, models.ChatMessage, models.AIModelVersion):
        assert (await db.execute(select(func.count()).select_from(table))).scalar_one() == 0


async def test_deleting_a_persona_keeps_the_chat_and_clears_the_link(db: AsyncSession):
    persona = models.Persona(name="Pirate", system_prompt="Arr")
    db.add(persona)
    await db.flush()
    chat = await make_chat(db)
    chat.persona_id = persona.id
    await db.commit()

    await db.execute(delete(models.Persona))
    await db.commit()
    await db.refresh(chat)

    assert chat.persona_id is None


async def test_chat_and_message_can_point_at_each_other(db: AsyncSession):
    chat = await make_chat(db)
    message = models.ChatMessage(chat_id=chat.id, role="user", content="hi")
    db.add(message)
    await db.flush()
    chat.active_leaf_id = message.id
    chat.memory_summary_until_id = message.id
    await db.commit()

    await db.execute(delete(models.ChatMessage).where(models.ChatMessage.id == message.id))
    await db.commit()
    await db.refresh(chat)

    assert chat.active_leaf_id is None
    assert chat.memory_summary_until_id is None


async def test_deleting_a_message_deletes_its_replies(db: AsyncSession):
    chat = await make_chat(db)
    root = models.ChatMessage(chat_id=chat.id, role="user", content="hi")
    db.add(root)
    await db.flush()
    db.add(models.ChatMessage(chat_id=chat.id, role="assistant", content="hello", parent_id=root.id))
    await db.commit()

    await db.execute(delete(models.ChatMessage).where(models.ChatMessage.id == root.id))
    await db.commit()

    assert (await db.execute(select(func.count()).select_from(models.ChatMessage))).scalar_one() == 0


async def test_deleting_a_chat_sets_continuations_free(db: AsyncSession):
    ai_model = make_model()
    db.add(ai_model)
    await db.flush()
    source = await make_chat(db, ai_model)
    continuation = await make_chat(db, ai_model)
    continuation.seed_source_id = source.id
    await db.commit()

    await db.execute(delete(models.ChatHistory).where(models.ChatHistory.id == source.id))
    await db.commit()
    await db.refresh(continuation)

    assert continuation.seed_source_id is None


async def test_a_model_used_by_a_collection_cannot_be_deleted(db: AsyncSession):
    embedder = make_model("embedder", is_embedding=True)
    db.add(embedder)
    await db.flush()
    db.add(
        models.DocumentCollection(name="Docs", embedding_model_id=embedder.id, embedding_parameters="latest")
    )
    await db.commit()

    with pytest.raises(IntegrityError):
        await db.execute(delete(models.AIModel).where(models.AIModel.id == embedder.id))
    await db.rollback()


async def test_a_chat_can_activate_collections_and_losing_one_unlinks_it(db: AsyncSession):
    embedder = make_model("embedder", is_embedding=True)
    db.add(embedder)
    await db.flush()
    chat = await make_chat(db)
    keep = models.DocumentCollection(
        name="Keep", embedding_model_id=embedder.id, embedding_parameters="latest"
    )
    drop = models.DocumentCollection(
        name="Drop", embedding_model_id=embedder.id, embedding_parameters="latest"
    )
    db.add_all([keep, drop])
    await db.flush()
    await db.execute(
        models.chat_history_active_collection.insert(),
        [
            {"chat_history_id": chat.id, "document_collection_id": keep.id},
            {"chat_history_id": chat.id, "document_collection_id": drop.id},
        ],
    )
    await db.commit()

    await db.execute(delete(models.DocumentCollection).where(models.DocumentCollection.id == drop.id))
    await db.commit()

    loaded = (
        await db.execute(
            select(models.ChatHistory)
            .where(models.ChatHistory.id == chat.id)
            .options(selectinload(models.ChatHistory.active_collections))
        )
    ).scalar_one()
    assert [collection.name for collection in loaded.active_collections] == ["Keep"]


async def test_deleting_a_chat_removes_the_collections_created_for_it(db: AsyncSession):
    embedder = make_model("embedder", is_embedding=True)
    db.add(embedder)
    await db.flush()
    chat = await make_chat(db)
    db.add(
        models.DocumentCollection(
            name="Owned", chat_id=chat.id, embedding_model_id=embedder.id, embedding_parameters="latest"
        )
    )
    await db.commit()

    await db.execute(delete(models.ChatHistory).where(models.ChatHistory.id == chat.id))
    await db.commit()

    assert (await db.execute(select(func.count()).select_from(models.DocumentCollection))).scalar_one() == 0


async def test_model_names_and_version_tags_are_unique(db: AsyncSession):
    ai_model = make_model()
    db.add(ai_model)
    await db.flush()
    model_id = ai_model.id
    db.add(models.AIModelVersion(ai_model_id=model_id, parameters="1b", size="1 GB"))
    await db.commit()

    db.add(make_model())
    with pytest.raises(IntegrityError):
        await db.flush()
    await db.rollback()

    db.add(models.AIModelVersion(ai_model_id=model_id, parameters="1b", size="2 GB"))
    with pytest.raises(IntegrityError):
        await db.flush()
    await db.rollback()


async def test_messages_load_in_the_order_they_were_written(db: AsyncSession):
    chat = await make_chat(db)
    same_instant = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
    for content in ("one", "two", "three"):
        db.add(models.ChatMessage(chat_id=chat.id, role="user", content=content, created_at=same_instant))
    await db.commit()

    loaded = (
        await db.execute(
            select(models.ChatHistory)
            .where(models.ChatHistory.id == chat.id)
            .options(selectinload(models.ChatHistory.messages))
        )
    ).scalar_one()

    assert [message.content for message in loaded.messages] == ["one", "two", "three"]


async def test_a_relationship_that_was_not_eager_loaded_fails_loudly(db: AsyncSession):
    chat = await make_chat(db)
    await db.commit()
    loaded = (
        await db.execute(select(models.ChatHistory).where(models.ChatHistory.id == chat.id))
    ).scalar_one()

    with pytest.raises(InvalidRequestError, match="lazy='raise'"):
        _ = loaded.ai_model


async def test_an_unset_setting_value_is_sql_null_not_json_null(db: AsyncSession):
    db.add(models.AppSetting(key="external_ollama_url", value=None))
    await db.commit()

    row = (await db.execute(text("SELECT value IS NULL AS is_sql_null FROM app_setting"))).one()

    assert row.is_sql_null is True


async def test_a_setting_value_round_trips_as_json(db: AsyncSession):
    db.add(models.AppSetting(key="gpu_share", value={"percent": 50, "tags": ["a"]}))
    await db.commit()

    value = (await db.execute(select(models.AppSetting.value))).scalar_one()

    assert value == {"percent": 50, "tags": ["a"]}


async def test_vectors_round_trip_and_rank_by_cosine_distance(db: AsyncSession):
    embedder = make_model("embedder", is_embedding=True)
    db.add(embedder)
    await db.flush()
    collection = models.DocumentCollection(
        name="Docs", embedding_model_id=embedder.id, embedding_parameters="latest"
    )
    db.add(collection)
    await db.flush()
    document = models.Document(collection_id=collection.id, filename="a.txt")
    db.add(document)
    await db.flush()
    db.add_all(
        [
            models.DocumentChunk(
                document_id=document.id, chunk_index=index, content=content, char_start=0, embedding=vector
            )
            for index, (content, vector) in enumerate(
                [("north", [0.0, 1.0, 0.0]), ("east", [1.0, 0.0, 0.0]), ("northeast", [0.7, 0.7, 0.0])]
            )
        ]
    )
    await db.commit()

    query = [0.1, 1.0, 0.0]
    nearest = (
        await db.execute(
            select(models.DocumentChunk.content, models.DocumentChunk.embedding)
            .order_by(models.DocumentChunk.embedding.cosine_distance(query))
            .limit(2)
        )
    ).all()

    assert [row.content for row in nearest] == ["north", "northeast"]
    assert nearest[0].embedding == pytest.approx([0.0, 1.0, 0.0])


async def test_vectors_of_different_sizes_can_live_in_different_collections(db: AsyncSession):
    embedder = make_model("embedder", is_embedding=True)
    db.add(embedder)
    await db.flush()
    for name, vector in (("small", [1.0, 2.0]), ("large", [1.0, 2.0, 3.0, 4.0])):
        collection = models.DocumentCollection(
            name=name, embedding_model_id=embedder.id, embedding_parameters="latest"
        )
        db.add(collection)
        await db.flush()
        document = models.Document(collection_id=collection.id, filename=f"{name}.txt")
        db.add(document)
        await db.flush()
        db.add(
            models.DocumentChunk(
                document_id=document.id, chunk_index=0, content=name, char_start=0, embedding=vector
            )
        )
    await db.commit()

    assert (await db.execute(select(func.count()).select_from(models.DocumentChunk))).scalar_one() == 2


async def test_ports_and_container_names_are_unique(db: AsyncSession):
    db.add(models.ContainerPort(container_name="a", port=11434))
    await db.commit()

    db.add(models.ContainerPort(container_name="b", port=11434))
    with pytest.raises(IntegrityError):
        await db.flush()
    await db.rollback()
