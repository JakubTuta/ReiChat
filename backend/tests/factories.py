"""Rows for tests to start from. Each one is committed, so the app's own
sessions (routes, sockets, background tasks) see it."""

import datetime
import itertools

from sqlalchemy.ext.asyncio import AsyncSession

from app.db import models

_counter = itertools.count(1)


async def make_model(session: AsyncSession, model: str | None = None, **fields) -> models.AIModel:
    model = model or f"model-{next(_counter)}"
    values = {"name": model, "model": model, "description": "", "index": 1, **fields}
    row = models.AIModel(**values)
    session.add(row)
    await session.commit()
    return row


async def make_chat(
    session: AsyncSession, model: models.AIModel | None = None, **fields
) -> models.ChatHistory:
    if model is None:
        model = await make_model(session)
    chat = models.ChatHistory(ai_model_id=model.id, **fields)
    session.add(chat)
    await session.commit()
    return chat


async def make_message(
    session: AsyncSession,
    chat: models.ChatHistory,
    role: str,
    content: str,
    parent: models.ChatMessage | None = None,
    *,
    activate: bool = False,
    **fields,
) -> models.ChatMessage:
    message = models.ChatMessage(
        chat_id=chat.id, parent_id=parent.id if parent else None, role=role, content=content, **fields
    )
    session.add(message)
    await session.flush()
    if activate:
        chat.active_leaf_id = message.id
    await session.commit()
    return message


async def make_exchange(
    session: AsyncSession,
    chat: models.ChatHistory,
    user: str,
    assistant: str,
    parent: models.ChatMessage | None = None,
) -> tuple[models.ChatMessage, models.ChatMessage]:
    """A user message and its reply, left as the chat's active branch."""
    user_message = await make_message(session, chat, "user", user, parent)
    assistant_message = await make_message(session, chat, "assistant", assistant, user_message, activate=True)
    return user_message, assistant_message


def minutes_ago(minutes: int) -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=minutes)
