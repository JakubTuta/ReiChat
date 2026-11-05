"""Small catalog operations shared by every place that adds or removes models:
creating one only if it's missing, and checking that nothing depends on a model
before it is deleted."""

import enum

from sqlalchemy import exists, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AIModel, AIModelVersion, ChatHistory, DocumentCollection


async def get_or_create_model(
    session: AsyncSession, *, model: str, name: str, description: str, is_custom: bool
) -> tuple[int, bool]:
    """(id, created). An existing model is left exactly as it is."""
    created_id = (
        await session.execute(
            insert(AIModel)
            .values(model=model, name=name, description=description, is_custom=is_custom)
            .on_conflict_do_nothing(index_elements=[AIModel.model])
            .returning(AIModel.id)
        )
    ).scalar_one_or_none()
    if created_id is not None:
        return created_id, True

    existing_id = (await session.execute(select(AIModel.id).where(AIModel.model == model))).scalar_one()
    return existing_id, False


async def get_or_create_version(
    session: AsyncSession, *, ai_model_id: int, parameters: str, size: str, size_bytes: int | None = None
) -> bool:
    """True if the version was added, False if it was already there."""
    created_id = (
        await session.execute(
            insert(AIModelVersion)
            .values(ai_model_id=ai_model_id, parameters=parameters, size=size, size_bytes=size_bytes)
            .on_conflict_do_nothing(constraint="unique_ai_model_version")
            .returning(AIModelVersion.id)
        )
    ).scalar_one_or_none()
    return created_id is not None


class ModelUsage(enum.Enum):
    UNUSED = "unused"
    HAS_CHATS = "has_chats"
    USED_BY_COLLECTION = "used_by_collection"


async def lock_and_check_usage(session: AsyncSession, model_id: int) -> ModelUsage | None:
    """Locks the model's row, then says whether anything still depends on it.
    None if the model is already gone.

    Deleting an AIModel cascades into its chats, so "nothing uses it" has to
    stay true until the delete happens. Creating a chat takes a share lock on
    the model's row, so holding the row lock here means a chat can't appear
    between this check and the delete (it would wait, then fail).
    """
    locked = (
        await session.execute(select(AIModel.id).where(AIModel.id == model_id).with_for_update())
    ).scalar_one_or_none()
    if locked is None:
        return None

    if (await session.execute(select(exists().where(ChatHistory.ai_model_id == model_id)))).scalar_one():
        return ModelUsage.HAS_CHATS
    # A collection's vectors only make sense with the model that embedded them,
    # so the database refuses to delete that model; saying so here is kinder.
    if (
        await session.execute(select(exists().where(DocumentCollection.embedding_model_id == model_id)))
    ).scalar_one():
        return ModelUsage.USED_BY_COLLECTION
    return ModelUsage.UNUSED
