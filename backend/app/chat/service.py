"""What the chat endpoints do beyond reading and writing messages (chat/history.py):
listing chats, creating and continuing them, changing their settings, and
reporting how full a chat's memory is."""

import typing

from fastapi import HTTPException
from sqlalchemy import delete, insert, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, load_only, selectinload

from app.chat import context_window, history
from app.chat.schemas import ChatCreate, ChatListItem, ChatRef, ChatUpdate, CollectionRef, PersonaRef
from app.db import models
from app.llm import runtime

_SIMPLE_FIELDS = (
    "title",
    "tools_enabled",
    "memory_mode",
    "think_level",
    "temperature",
    "num_ctx",
    "top_p",
    "seed",
)


async def list_chats(session: AsyncSession, ai_model: models.AIModel) -> list[ChatListItem]:
    # Only the settings the list shows: a chat's summaries can be long, and the
    # messages aren't wanted here at all.
    chats = (
        await session.execute(
            history.chats_of_model(ai_model.id, newest_first=True).options(
                load_only(
                    models.ChatHistory.id,
                    models.ChatHistory.title,
                    models.ChatHistory.persona_id,
                    models.ChatHistory.temperature,
                    models.ChatHistory.num_ctx,
                    models.ChatHistory.top_p,
                    models.ChatHistory.seed,
                    models.ChatHistory.tools_enabled,
                    models.ChatHistory.memory_mode,
                    models.ChatHistory.think_level,
                    models.ChatHistory.seed_source_id,
                ),
                joinedload(models.ChatHistory.persona).load_only(models.Persona.id, models.Persona.name),
                joinedload(models.ChatHistory.seed_source).load_only(
                    models.ChatHistory.id, models.ChatHistory.title
                ),
                selectinload(models.ChatHistory.active_collections).load_only(
                    models.DocumentCollection.id, models.DocumentCollection.name
                ),
            )
        )
    ).scalars()

    return [
        ChatListItem(
            id=chat.id,
            title=chat.title,
            persona=PersonaRef(id=chat.persona.id, name=chat.persona.name) if chat.persona else None,
            temperature=chat.temperature,
            num_ctx=chat.num_ctx,
            top_p=chat.top_p,
            seed=chat.seed,
            active_collections=[CollectionRef(id=c.id, name=c.name) for c in chat.active_collections],
            tools_enabled=chat.tools_enabled,
            memory_mode=chat.memory_mode,
            think_level=chat.think_level,
            continued_from=(
                ChatRef(id=chat.seed_source.id, title=chat.seed_source.title) if chat.seed_source else None
            ),
        )
        for chat in chats
    ]


async def create_chat(
    session: AsyncSession, ai_model: models.AIModel, body: ChatCreate
) -> models.ChatHistory:
    chat = models.ChatHistory(ai_model_id=ai_model.id, title=body.title)
    session.add(chat)
    await session.commit()
    return (
        await session.execute(
            select(models.ChatHistory)
            .where(models.ChatHistory.id == chat.id)
            .options(joinedload(models.ChatHistory.ai_model).selectinload(models.AIModel.versions))
        )
    ).scalar_one()


async def continue_chat(session: AsyncSession, source: models.ChatHistory) -> models.ChatHistory:
    """ "Start fresh with a summary": a new chat with the same settings whose
    first reply carries a summary of this one (generated then, over the chat
    socket — summarizing can outlast an HTTP request on CPU)."""
    continued = models.ChatHistory(
        ai_model_id=source.ai_model_id,
        title=f"{source.title} (continued)",
        persona_id=source.persona_id,
        temperature=source.temperature,
        num_ctx=source.num_ctx,
        top_p=source.top_p,
        seed=source.seed,
        tools_enabled=source.tools_enabled,
        memory_mode=source.memory_mode,
        think_level=source.think_level,
        seed_source_id=source.id,
    )
    session.add(continued)
    await session.flush()

    collection_ids = (
        await session.execute(
            select(models.chat_history_active_collection.c.document_collection_id).where(
                models.chat_history_active_collection.c.chat_history_id == source.id
            )
        )
    ).scalars()
    rows = [{"chat_history_id": continued.id, "document_collection_id": cid} for cid in collection_ids]
    if rows:
        await session.execute(insert(models.chat_history_active_collection), rows)

    await session.commit()
    return continued


async def chat_from_compare(
    session: AsyncSession, ai_model: models.AIModel, parameters: str, prompt: str, response: str
) -> models.ChatHistory:
    """Turns one answer from the Compare page into a regular chat with that
    model, so the conversation can carry on from the best reply."""
    chat = models.ChatHistory(ai_model_id=ai_model.id)
    session.add(chat)
    await session.flush()
    # One transaction: the chat and its two messages exist together or not at all.
    await history.add_messages_to_history(
        session, chat, prompt, response, model_label=f"{ai_model.model}:{parameters}"
    )
    return chat


async def update_chat_settings(session: AsyncSession, chat: models.ChatHistory, body: ChatUpdate) -> None:
    """Applies every sent field or none: everything that can be refused is
    checked before anything is written."""
    sent = body.model_fields_set

    if "persona_id" in sent and body.persona_id is not None:
        exists = (
            await session.execute(select(models.Persona.id).where(models.Persona.id == body.persona_id))
        ).scalar_one_or_none()
        if exists is None:
            raise HTTPException(status_code=404, detail="Persona not found")

    collection_ids: set[int] = set()
    if "collection_ids" in sent:
        collection_ids = set(body.collection_ids or [])
        found = (
            await session.execute(
                select(models.DocumentCollection.id).where(models.DocumentCollection.id.in_(collection_ids))
            )
        ).scalars()
        if len(set(found)) != len(collection_ids):
            raise HTTPException(status_code=404, detail="One or more collections not found")

    target: models.AIModel | None = None
    if "move_to_model" in sent:
        target = await history.get_ai_model(session, str(body.move_to_model))
        if target is None:
            raise HTTPException(status_code=404, detail="AI model not found")
        if target.is_embedding:
            raise HTTPException(status_code=400, detail="Embedding models can't chat — pick a chat model.")

    for field in _SIMPLE_FIELDS:
        if field in sent:
            setattr(chat, field, getattr(body, field))
    if "persona_id" in sent:
        chat.persona_id = body.persona_id
    if target is not None:
        chat.ai_model_id = target.id

    if "collection_ids" in sent:
        link = models.chat_history_active_collection
        await session.execute(delete(link).where(link.c.chat_history_id == chat.id))
        if collection_ids:
            await session.execute(
                insert(link),
                [{"chat_history_id": chat.id, "document_collection_id": cid} for cid in collection_ids],
            )

    await session.commit()


async def context_report(
    session: AsyncSession, ai_model: models.AIModel, chat: models.ChatHistory, parameters: str
) -> dict[str, typing.Any]:
    """How full the chat's memory is right now, before anything is sent — the
    same estimate the chat socket makes, minus document excerpts (those depend
    on the next question)."""
    model_info = await runtime.get_model_info(ai_model.model, parameters) if parameters else None
    model_max = model_info["context_length"] if model_info else None
    num_ctx = context_window.resolve_num_ctx(chat.num_ctx, model_max)

    messages = history.deserialize_messages(await history.get_messages_for_chat(session, chat))
    summary_blocks = [chat.seed_summary]
    message_ids = [m["id"] for m in messages]
    summarized = (
        chat.memory_mode == models.ChatHistory.MEMORY_MODE_SUMMARIZE
        and bool(chat.memory_summary)
        and chat.memory_summary_until_id in message_ids
    )
    if summarized:
        messages = messages[message_ids.index(chat.memory_summary_until_id) + 1 :]
        summary_blocks.append(chat.memory_summary)

    system_prompt = "\n\n".join(
        p for p in (await history.get_persona_system_prompt(session, chat), *summary_blocks) if p
    )
    fit = context_window.fit_history(
        system_prompt or None, messages, {"role": "user", "content": ""}, num_ctx
    )

    return {
        "limit": num_ctx,
        "used": fit.estimated_prompt_tokens,
        "model_max": model_max,
        "preset": context_window.preset_for(num_ctx),
        "dropped": fit.dropped,
        "summarized": summarized,
        "presets": context_window.MEMORY_PRESETS,
        "tokens_per_page": context_window.TOKENS_PER_PAGE,
        "thinking": model_info["thinking"] if model_info else None,
        "capabilities": model_info["capabilities"] if model_info else None,
    }
