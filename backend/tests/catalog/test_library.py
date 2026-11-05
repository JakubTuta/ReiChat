import asyncio

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.catalog import library
from app.db import models
from app.db.session import SessionLocal

pytestmark = pytest.mark.usefixtures("db")


async def add_model(db: AsyncSession, name: str = "m", **fields) -> models.AIModel:
    model = models.AIModel(name=name, model=name, description="", index=1, **fields)
    db.add(model)
    await db.commit()
    return model


async def test_a_missing_model_is_created(db: AsyncSession):
    model_id, created = await library.get_or_create_model(
        db, model="hf.co/u/repo", name="repo", description="Mine", is_custom=True
    )
    await db.commit()

    assert created is True
    row = (await db.execute(select(models.AIModel).where(models.AIModel.id == model_id))).scalar_one()
    assert (row.model, row.name, row.description, row.is_custom, row.popularity, row.index) == (
        "hf.co/u/repo",
        "repo",
        "Mine",
        True,
        0,
        0,
    )


async def test_an_existing_model_is_left_exactly_as_it_was(db: AsyncSession):
    existing = await add_model(db, "llama3.2", popularity=5)

    model_id, created = await library.get_or_create_model(
        db, model="llama3.2", name="other", description="other", is_custom=True
    )
    await db.commit()

    assert (model_id, created) == (existing.id, False)
    await db.refresh(existing)
    assert (existing.name, existing.is_custom, existing.popularity) == ("llama3.2", False, 5)


async def test_two_simultaneous_adds_of_the_same_model_make_one(db: AsyncSession):
    async def add() -> bool:
        async with SessionLocal() as session:
            _, created = await library.get_or_create_model(
                session, model="same", name="same", description="", is_custom=True
            )
            await session.commit()
            return created

    results = await asyncio.gather(*(add() for _ in range(5)))

    assert results.count(True) == 1
    assert (await db.execute(select(func.count()).select_from(models.AIModel))).scalar_one() == 1


async def test_a_model_with_a_chat_has_chats(db: AsyncSession):
    model = await add_model(db)
    db.add(models.ChatHistory(ai_model_id=model.id))
    await db.commit()

    assert await library.lock_and_check_usage(db, model.id) is library.ModelUsage.HAS_CHATS


async def test_a_model_that_embedded_a_collection_is_used_by_it(db: AsyncSession):
    model = await add_model(db, is_embedding=True)
    db.add(models.DocumentCollection(name="Docs", embedding_model_id=model.id, embedding_parameters="latest"))
    await db.commit()

    assert await library.lock_and_check_usage(db, model.id) is library.ModelUsage.USED_BY_COLLECTION


async def test_a_chat_cannot_sneak_in_between_the_check_and_the_delete(db: AsyncSession):
    """Deleting a model takes its chats with it, so once a model is checked
    and locked, creating a chat for it has to wait for the delete."""
    model = await add_model(db)
    assert await library.lock_and_check_usage(db, model.id) is library.ModelUsage.UNUSED

    async def create_chat() -> None:
        async with SessionLocal() as session:
            session.add(models.ChatHistory(ai_model_id=model.id))
            await session.commit()

    chat = asyncio.create_task(create_chat())
    await asyncio.sleep(0.2)
    assert not chat.done(), "the chat was created although the model was locked for deletion"

    await db.execute(models.AIModel.__table__.delete().where(models.AIModel.id == model.id))
    await db.commit()

    with pytest.raises(Exception, match="foreign key|violates"):
        await asyncio.wait_for(chat, timeout=5)
    assert (await db.execute(select(func.count()).select_from(models.ChatHistory))).scalar_one() == 0
