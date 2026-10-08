"""Export and import everything the user created: chats (whole branch
trees), personas, prompt templates, MCP servers and, optionally, document
collections with their embeddings. Model weights aren't included — they're
re-downloadable and live in Docker volumes.

Import only ever adds. Nothing existing is overwritten or deleted, so
restoring the same backup twice just yields duplicates the user can delete,
never lost data.
"""

from __future__ import annotations

import dataclasses
import typing

from django.db import transaction
from django.utils import timezone

from . import models

FORMAT_NAME = "reichat-backup"
FORMAT_VERSION = 1


class BackupFormatError(ValueError):
    pass


def _message_rows(chat: models.ChatHistory) -> list[dict[str, typing.Any]]:
    return [
        {
            "id": m.id,
            "parent": m.parent_id,
            "role": m.role,
            "content": m.content,
            "image": m.image,
            "thinking": m.thinking,
            "model_label": m.model_label,
        }
        # created_at, id: the order siblings (branches) are shown in.
        for m in models.ChatMessage.objects.filter(chat=chat).order_by("created_at", "id")
    ]


def export_data(include_documents: bool) -> dict[str, typing.Any]:
    chats = models.ChatHistory.objects.select_related("ai_model", "persona").prefetch_related("active_collections")
    data: dict[str, typing.Any] = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "exported_at": timezone.now().isoformat(),
        "personas": [
            {"name": p.name, "system_prompt": p.system_prompt} for p in models.Persona.objects.order_by("id")
        ],
        "prompt_templates": [
            {"name": t.name, "description": t.description, "content": t.content}
            for t in models.PromptTemplate.objects.order_by("id")
        ],
        "mcp_servers": [
            {"name": s.name, "transport": s.transport, "command": s.command, "url": s.url}
            for s in models.MCPServer.objects.order_by("id")
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
                "messages": _message_rows(chat),
                "active_leaf": chat.active_leaf_id,
                "collections": [c.id for c in chat.active_collections.all()] if include_documents else [],
            }
            for chat in chats.order_by("id")
        ],
        "collections": [],
    }

    if include_documents:
        for collection in models.DocumentCollection.objects.select_related("embedding_model").order_by("id"):
            data["collections"].append({
                "id": collection.id,
                "name": collection.name,
                "embedding_model": collection.embedding_model.model,
                "embedding_parameters": collection.embedding_parameters,
                "documents": [
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
                            for c in document.chunks.order_by("chunk_index")
                        ],
                    }
                    for document in collection.documents.filter(status=models.Document.STATUS_READY)
                ],
            })

    return data


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


def _model_for(name: str, report: ImportReport, embedding: bool = False) -> models.AIModel:
    """The catalog row for `name`, or a custom placeholder when this install
    doesn't list it — so the restored chats show up and can be moved to an
    installed model, instead of being dropped."""
    existing = models.AIModel.objects.filter(model=name).first()
    if existing is not None:
        return existing

    report.notes.append(
        f'"{name}" isn\'t in your model list, so it was added as a custom model. Install it, or move its chats '
        "to a model you have."
    )
    return models.AIModel.objects.create(
        name=name.rsplit("/", 1)[-1],
        model=name,
        description="Restored from a backup.",
        is_custom=True,
        is_embedding=embedding,
    )


def _as_list(data: dict[str, typing.Any], key: str) -> list[dict[str, typing.Any]]:
    value = data.get(key) or []
    if not isinstance(value, list) or not all(isinstance(v, dict) for v in value):
        raise BackupFormatError(f'"{key}" in the backup file is damaged.')
    return value


def import_data(data: typing.Any) -> ImportReport:
    if not isinstance(data, dict) or data.get("format") != FORMAT_NAME:
        raise BackupFormatError("This isn't a ReiChat backup file.")
    if not isinstance(data.get("version"), int) or data["version"] > FORMAT_VERSION:
        raise BackupFormatError("This backup was made by a newer version of ReiChat. Update the app and try again.")

    report = ImportReport()

    with transaction.atomic():
        personas_by_name: dict[str, models.Persona] = {}
        for row in _as_list(data, "personas"):
            persona = models.Persona.objects.filter(name=row["name"], system_prompt=row["system_prompt"]).first()
            if persona is None:
                persona = models.Persona.objects.create(name=row["name"], system_prompt=row["system_prompt"])
                report.personas += 1
            personas_by_name[persona.name] = persona

        for row in _as_list(data, "prompt_templates"):
            if not models.PromptTemplate.objects.filter(name=row["name"], content=row["content"]).exists():
                models.PromptTemplate.objects.create(
                    name=row["name"], description=row.get("description", ""), content=row["content"]
                )
                report.prompt_templates += 1

        imported_servers = 0
        for row in _as_list(data, "mcp_servers"):
            if models.MCPServer.objects.filter(name=row["name"]).exists():
                continue
            # A backup file can come from anywhere, and a stdio server is a
            # command this machine would run — off until the user looks.
            models.MCPServer.objects.create(
                name=row["name"],
                transport=row["transport"],
                command=row.get("command", ""),
                url=row.get("url", ""),
                enabled=False,
                auto_approve=False,
            )
            imported_servers += 1
        report.mcp_servers = imported_servers
        if imported_servers:
            report.notes.append(
                f"{imported_servers} tool server(s) were imported switched off. Check them on the Integrations page "
                "before turning them on."
            )

        collections_by_old_id: dict[int, models.DocumentCollection] = {}
        for row in _as_list(data, "collections"):
            collection = models.DocumentCollection.objects.create(
                name=row["name"],
                embedding_model=_model_for(row["embedding_model"], report, embedding=True),
                embedding_parameters=row["embedding_parameters"],
            )
            collections_by_old_id[row["id"]] = collection
            report.collections += 1
            for document_row in row.get("documents") or []:
                chunks = document_row.get("chunks") or []
                document = models.Document.objects.create(
                    collection=collection,
                    filename=document_row["filename"],
                    status=models.Document.STATUS_READY,
                    chunk_count=len(chunks),
                    progress=100,
                )
                models.DocumentChunk.objects.bulk_create([
                    models.DocumentChunk(
                        document=document,
                        chunk_index=c["chunk_index"],
                        content=c["content"],
                        char_start=c.get("char_start", 0),
                        page=c.get("page"),
                        embedding=c["embedding"],
                    )
                    for c in chunks
                ])
                report.documents += 1

        for row in _as_list(data, "chats"):
            settings = row.get("settings") or {}
            chat = models.ChatHistory.objects.create(
                ai_model=_model_for(row["model"], report),
                title=row.get("title") or models.DEFAULT_CHAT_TITLE,
                persona=personas_by_name.get(row.get("persona") or ""),
                temperature=settings.get("temperature"),
                num_ctx=settings.get("num_ctx"),
                top_p=settings.get("top_p"),
                seed=settings.get("seed"),
                tools_enabled=bool(settings.get("tools_enabled")),
                memory_mode=settings.get("memory_mode") or models.ChatHistory.MEMORY_MODE_TRIM,
                think_level=settings.get("think_level") or "",
            )

            # Messages arrive parent-before-child (export order), so each
            # parent already has its new id when its children are created.
            new_ids: dict[int, models.ChatMessage] = {}
            for message_row in row.get("messages") or []:
                new_ids[message_row["id"]] = models.ChatMessage.objects.create(
                    chat=chat,
                    parent=new_ids.get(message_row["parent"]) if message_row.get("parent") else None,
                    role=message_row["role"],
                    content=message_row["content"],
                    image=message_row.get("image") or "",
                    thinking=message_row.get("thinking") or "",
                    model_label=message_row.get("model_label") or "",
                )
                report.messages += 1

            chat.active_leaf = new_ids.get(row.get("active_leaf") or -1)
            chat.save(update_fields=["active_leaf"])
            chat.active_collections.set(
                [collections_by_old_id[c] for c in row.get("collections") or [] if c in collections_by_old_id]
            )
            report.chats += 1

    return report


def import_single_chat(data: typing.Any, fallback_model: str) -> models.ChatHistory:
    """A chat saved with the chat's own "Export conversation → json"
    ({title, model, messages: [{role, content, image}]}) — one linear
    conversation, added to `fallback_model` when its own isn't known."""
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        raise BackupFormatError("This isn't an exported ReiChat conversation.")

    messages = [
        m for m in data["messages"]
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
    ]
    if not messages:
        raise BackupFormatError("That conversation has no messages to import.")

    with transaction.atomic():
        model_name = data.get("model") if models.AIModel.objects.filter(model=data.get("model")).exists() else None
        ai_model = models.AIModel.objects.get(model=model_name or fallback_model)
        chat = models.ChatHistory.objects.create(ai_model=ai_model, title=str(data.get("title") or "Imported chat"))

        parent = None
        for m in messages:
            parent = models.ChatMessage.objects.create(
                chat=chat, parent=parent, role=m["role"], content=m["content"], image=m.get("image") or ""
            )
        chat.active_leaf = parent
        chat.save(update_fields=["active_leaf"])

    return chat
