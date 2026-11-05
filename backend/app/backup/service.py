"""Export and import everything the user created: chats (whole branch
trees), personas, prompt templates, MCP servers and, optionally, document
collections with their embeddings. Model weights aren't included — they're
re-downloadable and live in Docker volumes.

Import only ever adds. Nothing existing is overwritten or deleted, so
restoring the same backup twice just yields duplicates the user can delete,
never lost data.
"""

import dataclasses
import datetime
import typing

from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.db import models

FORMAT_NAME = "reichat-backup"
FORMAT_VERSION = 1


class BackupFormatError(ValueError):
    pass


async def _messages_by_chat(session: AsyncSession) -> dict[int, list[dict[str, typing.Any]]]:
    rows = await session.execute(
        select(models.ChatMessage)
        # created_at, id: the order siblings (branches) are shown in.
        .order_by(models.ChatMessage.chat_id, models.ChatMessage.created_at, models.ChatMessage.id)
    )
    by_chat: dict[int, list[dict[str, typing.Any]]] = {}
    for m in rows.scalars():
        by_chat.setdefault(m.chat_id, []).append(
            {
                "id": m.id,
                "parent": m.parent_id,
                "role": m.role,
                "content": m.content,
                "image": m.image,
                "thinking": m.thinking,
                "model_label": m.model_label,
            }
        )
    return by_chat


async def _active_collection_ids(session: AsyncSession) -> dict[int, list[int]]:
    link = models.chat_history_active_collection
    rows = await session.execute(
        select(link.c.chat_history_id, link.c.document_collection_id).order_by(link.c.document_collection_id)
    )
    by_chat: dict[int, list[int]] = {}
    for chat_id, collection_id in rows:
        by_chat.setdefault(chat_id, []).append(collection_id)
    return by_chat


async def export_data(session: AsyncSession, include_documents: bool) -> dict[str, typing.Any]:
    messages = await _messages_by_chat(session)
    collection_ids = await _active_collection_ids(session) if include_documents else {}
    chats = (
        await session.execute(
            select(models.ChatHistory)
            .options(joinedload(models.ChatHistory.ai_model), joinedload(models.ChatHistory.persona))
            .order_by(models.ChatHistory.id)
        )
    ).scalars()

    data: dict[str, typing.Any] = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "exported_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "personas": [
            {"name": p.name, "system_prompt": p.system_prompt}
            for p in (await session.execute(select(models.Persona).order_by(models.Persona.id))).scalars()
        ],
        "prompt_templates": [
            {"name": t.name, "description": t.description, "content": t.content}
            for t in (
                await session.execute(select(models.PromptTemplate).order_by(models.PromptTemplate.id))
            ).scalars()
        ],
        "mcp_servers": [
            {"name": s.name, "transport": s.transport, "command": s.command, "url": s.url}
            for s in (await session.execute(select(models.MCPServer).order_by(models.MCPServer.id))).scalars()
        ],
        "chats": [
            {
                "model": chat.ai_model.model,
                "title": chat.title,
                "persona": chat.persona.name if chat.persona else None,
                "settings": {
                    "temperature": chat.temperature,
                    "num_ctx": chat.num_ctx,
                    "top_p": chat.top_p,
                    "seed": chat.seed,
                    "tools_enabled": chat.tools_enabled,
                    "memory_mode": chat.memory_mode,
                    "think_level": chat.think_level,
                },
                "messages": messages.get(chat.id, []),
                "active_leaf": chat.active_leaf_id,
                "collections": collection_ids.get(chat.id, []),
            }
            for chat in chats
        ],
        "collections": [],
    }

    if include_documents:
        data["collections"] = await _export_collections(session)

    return data


async def _export_collections(session: AsyncSession) -> list[dict[str, typing.Any]]:
    collections = (
        await session.execute(
            select(models.DocumentCollection)
            .options(joinedload(models.DocumentCollection.embedding_model))
            .order_by(models.DocumentCollection.id)
        )
    ).scalars()

    exported = []
    for collection in collections:
        documents = (
            await session.execute(
                select(models.Document)
                .where(
                    models.Document.collection_id == collection.id,
                    models.Document.status == models.Document.STATUS_READY,
                )
                .order_by(models.Document.id)
            )
        ).scalars()
        exported_documents = []
        for document in documents:
            chunks = (
                await session.execute(
                    select(models.DocumentChunk)
                    .where(models.DocumentChunk.document_id == document.id)
                    .order_by(models.DocumentChunk.chunk_index)
                )
            ).scalars()
            exported_documents.append(
                {
                    "filename": document.filename,
                    "chunks": [
                        {
                            "chunk_index": c.chunk_index,
                            "content": c.content,
                            "char_start": c.char_start,
                            "page": c.page,
                            "embedding": [float(x) for x in c.embedding],
                        }
                        for c in chunks
                    ],
                }
            )
        exported.append(
            {
                "id": collection.id,
                "name": collection.name,
                "embedding_model": collection.embedding_model.model,
                "embedding_parameters": collection.embedding_parameters,
                "documents": exported_documents,
            }
        )
    return exported


@dataclasses.dataclass
class ImportReport:
    chats: int = 0
    messages: int = 0
    personas: int = 0
    prompt_templates: int = 0
    mcp_servers: int = 0
    collections: int = 0
    documents: int = 0
    # Plain-language notes the user should see (placeholders created, things
    # imported switched off).
    notes: list[str] = dataclasses.field(default_factory=list)


async def _model_for(
    session: AsyncSession, name: str, report: ImportReport, embedding: bool = False
) -> models.AIModel:
    """The catalog row for `name`, or a custom placeholder when this install
    doesn't list it — so the restored chats show up and can be moved to an
    installed model, instead of being dropped."""
    existing = (
        await session.execute(select(models.AIModel).where(models.AIModel.model == name))
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    report.notes.append(
        f'"{name}" isn\'t in your model list, so it was added as a custom model. Install it, or move its chats '
        "to a model you have."
    )
    placeholder = models.AIModel(
        name=name.rsplit("/", 1)[-1],
        model=name,
        description="Restored from a backup.",
        is_custom=True,
        is_embedding=embedding,
    )
    session.add(placeholder)
    await session.flush()
    return placeholder


def _as_list(data: dict[str, typing.Any], key: str) -> list[dict[str, typing.Any]]:
    value = data.get(key) or []
    if not isinstance(value, list) or not all(isinstance(v, dict) for v in value):
        raise BackupFormatError(f'"{key}" in the backup file is damaged.')
    return value


async def import_data(session: AsyncSession, data: typing.Any) -> ImportReport:
    """Adds everything in the backup in one transaction: a file that turns out
    to be damaged part-way through imports nothing (the caller's session is
    simply closed without a commit)."""
    if not isinstance(data, dict) or data.get("format") != FORMAT_NAME:
        raise BackupFormatError("This isn't a ReiChat backup file.")
    if not isinstance(data.get("version"), int) or data["version"] > FORMAT_VERSION:
        raise BackupFormatError(
            "This backup was made by a newer version of ReiChat. Update the app and try again."
        )

    report = ImportReport()

    personas_by_name: dict[str, models.Persona] = {}
    for row in _as_list(data, "personas"):
        persona = (
            (
                await session.execute(
                    select(models.Persona).where(
                        models.Persona.name == row["name"],
                        models.Persona.system_prompt == row["system_prompt"],
                    )
                )
            )
            .scalars()
            .first()
        )
        if persona is None:
            persona = models.Persona(name=row["name"], system_prompt=row["system_prompt"])
            session.add(persona)
            await session.flush()
            report.personas += 1
        personas_by_name[persona.name] = persona

    for row in _as_list(data, "prompt_templates"):
        known = (
            await session.execute(
                select(models.PromptTemplate.id).where(
                    models.PromptTemplate.name == row["name"], models.PromptTemplate.content == row["content"]
                )
            )
        ).first()
        if known is None:
            session.add(
                models.PromptTemplate(
                    name=row["name"], description=row.get("description", ""), content=row["content"]
                )
            )
            report.prompt_templates += 1

    imported_servers = 0
    for row in _as_list(data, "mcp_servers"):
        if (
            await session.execute(select(models.MCPServer.id).where(models.MCPServer.name == row["name"]))
        ).first():
            continue
        # A backup file can come from anywhere, and a stdio server is a
        # command this machine would run — off until the user looks.
        session.add(
            models.MCPServer(
                name=row["name"],
                transport=row["transport"],
                command=row.get("command", ""),
                url=row.get("url", ""),
                enabled=False,
                auto_approve=False,
            )
        )
        await session.flush()
        imported_servers += 1
    report.mcp_servers = imported_servers
    if imported_servers:
        report.notes.append(
            f"{imported_servers} tool server(s) were imported switched off. Check them on the Integrations page "
            "before turning them on."
        )

    collections_by_old_id: dict[int, models.DocumentCollection] = {}
    for row in _as_list(data, "collections"):
        embedder = await _model_for(session, row["embedding_model"], report, embedding=True)
        collection = models.DocumentCollection(
            name=row["name"], embedding_model_id=embedder.id, embedding_parameters=row["embedding_parameters"]
        )
        session.add(collection)
        await session.flush()
        collections_by_old_id[row["id"]] = collection
        report.collections += 1
        for document_row in row.get("documents") or []:
            chunks = document_row.get("chunks") or []
            document = models.Document(
                collection_id=collection.id,
                filename=document_row["filename"],
                status=models.Document.STATUS_READY,
                chunk_count=len(chunks),
                progress=100,
            )
            session.add(document)
            await session.flush()
            if chunks:
                await session.execute(
                    insert(models.DocumentChunk),
                    [
                        {
                            "document_id": document.id,
                            "chunk_index": c["chunk_index"],
                            "content": c["content"],
                            "char_start": c.get("char_start", 0),
                            "page": c.get("page"),
                            "embedding": c["embedding"],
                        }
                        for c in chunks
                    ],
                )
            report.documents += 1

    for row in _as_list(data, "chats"):
        settings = row.get("settings") or {}
        ai_model = await _model_for(session, row["model"], report)
        persona = personas_by_name.get(row.get("persona") or "")
        chat = models.ChatHistory(
            ai_model_id=ai_model.id,
            title=row.get("title") or models.DEFAULT_CHAT_TITLE,
            persona_id=persona.id if persona else None,
            temperature=settings.get("temperature"),
            num_ctx=settings.get("num_ctx"),
            top_p=settings.get("top_p"),
            seed=settings.get("seed"),
            tools_enabled=bool(settings.get("tools_enabled")),
            memory_mode=settings.get("memory_mode") or models.ChatHistory.MEMORY_MODE_TRIM,
            think_level=settings.get("think_level") or "",
        )
        session.add(chat)
        await session.flush()

        # Messages arrive parent-before-child (export order), so each parent
        # already has its new id when its children are created.
        new_ids: dict[int, models.ChatMessage] = {}
        for message_row in row.get("messages") or []:
            parent = new_ids.get(message_row["parent"]) if message_row.get("parent") else None
            message = models.ChatMessage(
                chat_id=chat.id,
                parent_id=parent.id if parent else None,
                role=message_row["role"],
                content=message_row["content"],
                image=message_row.get("image") or "",
                thinking=message_row.get("thinking") or "",
                model_label=message_row.get("model_label") or "",
            )
            session.add(message)
            await session.flush()
            new_ids[message_row["id"]] = message
            report.messages += 1

        leaf = new_ids.get(row.get("active_leaf") or -1)
        chat.active_leaf_id = leaf.id if leaf else None
        wanted = [
            collections_by_old_id[c].id for c in row.get("collections") or [] if c in collections_by_old_id
        ]
        if wanted:
            await session.execute(
                insert(models.chat_history_active_collection),
                [{"chat_history_id": chat.id, "document_collection_id": cid} for cid in set(wanted)],
            )
        report.chats += 1

    await session.commit()
    return report


async def import_single_chat(
    session: AsyncSession, data: typing.Any, fallback_model: models.AIModel
) -> tuple[models.ChatHistory, models.AIModel]:
    """A chat saved with the chat's own "Export conversation → json"
    ({title, model, messages: [{role, content, image}]}) — one linear
    conversation, added to `fallback_model` when its own isn't known."""
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        raise BackupFormatError("This isn't an exported ReiChat conversation.")

    messages = [
        m
        for m in data["messages"]
        if isinstance(m, dict)
        and m.get("role") in ("user", "assistant")
        and isinstance(m.get("content"), str)
    ]
    if not messages:
        raise BackupFormatError("That conversation has no messages to import.")

    own_model = None
    if isinstance(data.get("model"), str):
        own_model = (
            await session.execute(select(models.AIModel).where(models.AIModel.model == data["model"]))
        ).scalar_one_or_none()
    ai_model = own_model or fallback_model

    chat = models.ChatHistory(ai_model_id=ai_model.id, title=str(data.get("title") or "Imported chat"))
    session.add(chat)
    await session.flush()

    parent_id: int | None = None
    for m in messages:
        message = models.ChatMessage(
            chat_id=chat.id,
            parent_id=parent_id,
            role=m["role"],
            content=m["content"],
            image=m.get("image") or "",
        )
        session.add(message)
        await session.flush()
        parent_id = message.id
    chat.active_leaf_id = parent_id

    await session.commit()
    return chat, ai_model
