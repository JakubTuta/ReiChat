import io
import json

import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.backup import service
from app.db import models
from tests.factories import make_chat, make_model

pytestmark = pytest.mark.usefixtures("db")


@pytest.fixture
async def populated(db: AsyncSession):
    ai_model = await make_model(db, "llama3.1-backup-test")
    embedder = await make_model(db, "nomic-backup-test", is_embedding=True, index=2)
    persona = models.Persona(name="Pirate", system_prompt="Arr.")
    db.add(persona)
    db.add(models.PromptTemplate(name="Summarize", content="Summarize {{text}}"))
    db.add(models.MCPServer(name="Shell", transport="stdio", command="python server.py", auto_approve=True))
    await db.flush()

    chat = await make_chat(db, ai_model, title="Trip", persona_id=persona.id, num_ctx=16384)
    user = models.ChatMessage(chat_id=chat.id, role="user", content="Plan a trip")
    db.add(user)
    await db.flush()
    old_reply = models.ChatMessage(chat_id=chat.id, parent_id=user.id, role="assistant", content="Rome?")
    db.add(old_reply)
    await db.flush()
    new_reply = models.ChatMessage(
        chat_id=chat.id,
        parent_id=user.id,
        role="assistant",
        content="Paris!",
        model_label="llama3.1:8b",
        thinking="hmm",
    )
    db.add(new_reply)
    await db.flush()
    chat.active_leaf_id = new_reply.id

    collection = models.DocumentCollection(
        name="Guides", embedding_model_id=embedder.id, embedding_parameters="latest"
    )
    db.add(collection)
    await db.flush()
    document = models.Document(collection_id=collection.id, filename="paris.pdf", status="ready")
    db.add(document)
    await db.flush()
    db.add(
        models.DocumentChunk(
            document_id=document.id,
            chunk_index=0,
            content="Eiffel tower",
            char_start=0,
            page=2,
            embedding=[1.0, 0.0],
        )
    )
    await db.execute(
        models.chat_history_active_collection.insert().values(
            chat_history_id=chat.id, document_collection_id=collection.id
        )
    )
    await db.commit()
    return chat, old_reply


async def count(db: AsyncSession, table) -> int:
    return (await db.execute(select(func.count()).select_from(table))).scalar_one()


async def restored_chat(db: AsyncSession) -> models.ChatHistory:
    return (
        await db.execute(
            select(models.ChatHistory)
            .where(models.ChatHistory.title == "Trip")
            .order_by(models.ChatHistory.id.desc())
            .limit(1)
            .options(
                selectinload(models.ChatHistory.active_collections), selectinload(models.ChatHistory.messages)
            )
        )
    ).scalar_one()


async def test_round_trip_restores_branches_settings_and_documents(db: AsyncSession, populated):
    data = json.loads(json.dumps(await service.export_data(db, include_documents=True)))

    report = await service.import_data(db, data)

    assert report.chats == 1
    assert (await count(db, models.ChatHistory)) == 2
    restored = await restored_chat(db)
    assert restored.num_ctx == 16384
    persona = (
        await db.execute(select(models.Persona).where(models.Persona.id == restored.persona_id))
    ).scalar_one()
    assert persona.name == "Pirate"
    leaf = (
        await db.execute(select(models.ChatMessage).where(models.ChatMessage.id == restored.active_leaf_id))
    ).scalar_one()
    assert (leaf.content, leaf.model_label, leaf.thinking) == ("Paris!", "llama3.1:8b", "hmm")
    siblings = [m for m in restored.messages if m.role == "assistant"]
    assert [m.content for m in siblings] == ["Rome?", "Paris!"]
    assert siblings[0].parent_id == siblings[1].parent_id

    [restored_collection] = restored.active_collections
    chunk = (
        await db.execute(
            select(models.DocumentChunk)
            .join(models.Document, models.Document.id == models.DocumentChunk.document_id)
            .where(models.Document.collection_id == restored_collection.id)
        )
    ).scalar_one()
    assert chunk.page == 2
    assert list(chunk.embedding) == [1.0, 0.0]

    # Same-named persona with the same prompt is reused, not duplicated.
    assert await count(db, models.Persona) == 1
    assert report.personas == 0


async def test_the_report_counts_what_was_added(db: AsyncSession, populated):
    data = await service.export_data(db, include_documents=True)

    report = await service.import_data(db, data)

    assert (report.chats, report.messages, report.collections, report.documents) == (1, 3, 1, 1)
    assert report.mcp_servers == 0  # a server of that name already exists
    assert report.prompt_templates == 0  # same name and text already exists


async def test_importing_adds_and_never_overwrites(db: AsyncSession, populated):
    chat, _ = populated
    before = await count(db, models.ChatMessage)
    data = await service.export_data(db, include_documents=False)

    await service.import_data(db, data)
    await service.import_data(db, data)

    assert await count(db, models.ChatHistory) == 3
    assert await count(db, models.ChatMessage) == before * 3
    original = (
        await db.execute(select(models.ChatHistory).where(models.ChatHistory.id == chat.id))
    ).scalar_one()
    assert original.title == "Trip"


async def test_a_backup_without_documents_has_none(db: AsyncSession, populated):
    data = await service.export_data(db, include_documents=False)

    assert data["collections"] == []
    assert data["chats"][0]["collections"] == []


async def test_an_export_carries_the_format_marker(db: AsyncSession, populated):
    data = await service.export_data(db, include_documents=False)

    assert (data["format"], data["version"]) == ("reichat-backup", 1)
    assert data["exported_at"].endswith("+00:00")
    assert [p["name"] for p in data["personas"]] == ["Pirate"]
    assert data["chats"][0]["settings"]["num_ctx"] == 16384
    assert data["chats"][0]["persona"] == "Pirate"
    assert [m["content"] for m in data["chats"][0]["messages"]] == ["Plan a trip", "Rome?", "Paris!"]


async def test_documents_that_are_not_ready_are_not_exported(db: AsyncSession, populated):
    collection = (await db.execute(select(models.DocumentCollection))).scalars().first()
    db.add(models.Document(collection_id=collection.id, filename="half.pdf", status="processing"))
    await db.commit()

    data = await service.export_data(db, include_documents=True)

    assert [d["filename"] for d in data["collections"][0]["documents"]] == ["paris.pdf"]


async def test_imported_tool_servers_arrive_switched_off(db: AsyncSession, populated):
    data = await service.export_data(db, include_documents=False)
    await db.execute(models.MCPServer.__table__.delete())
    await db.commit()

    report = await service.import_data(db, data)

    server = (await db.execute(select(models.MCPServer).where(models.MCPServer.name == "Shell"))).scalar_one()
    assert server.enabled is False
    assert server.auto_approve is False
    assert any("switched off" in note for note in report.notes)


async def test_chats_for_unknown_models_get_a_placeholder_model(db: AsyncSession, populated):
    data = await service.export_data(db, include_documents=False)
    data["chats"][0]["model"] = "never-installed-model"

    report = await service.import_data(db, data)

    placeholder = (
        await db.execute(select(models.AIModel).where(models.AIModel.model == "never-installed-model"))
    ).scalar_one()
    assert placeholder.is_custom is True
    assert (
        await db.execute(
            select(func.count())
            .select_from(models.ChatHistory)
            .where(models.ChatHistory.ai_model_id == placeholder.id)
        )
    ).scalar_one() == 1
    assert any("never-installed-model" in note for note in report.notes)


async def test_the_placeholder_note_is_given_once_per_model(db: AsyncSession, populated):
    data = await service.export_data(db, include_documents=False)
    data["chats"][0]["model"] = "missing-model"
    data["chats"].append({**data["chats"][0], "title": "Second"})

    report = await service.import_data(db, data)

    assert len([n for n in report.notes if "missing-model" in n]) == 1


async def test_rejects_files_that_are_not_backups(db: AsyncSession):
    with pytest.raises(service.BackupFormatError):
        await service.import_data(db, {"hello": "world"})
    with pytest.raises(service.BackupFormatError):
        await service.import_data(db, {"format": "reichat-backup", "version": 99})
    with pytest.raises(service.BackupFormatError):
        await service.import_data(db, ["not", "a", "dict"])


async def test_a_damaged_section_is_named(db: AsyncSession):
    with pytest.raises(service.BackupFormatError, match='"chats"'):
        await service.import_data(db, {"format": "reichat-backup", "version": 1, "chats": "oops"})


async def test_damaged_backup_imports_nothing(client: httpx.AsyncClient, db: AsyncSession, populated):
    data = await service.export_data(db, include_documents=False)
    chats_before = await count(db, models.ChatHistory)
    data["personas"].append({"name": "Would be new", "system_prompt": "x"})
    data["chats"].append({"title": "broken"})

    response = await client.post(
        "/backup/", files={"file": ("backup.json", json.dumps(data).encode(), "application/json")}
    )

    assert response.status_code == 400
    assert response.json() == {"error": "Part of that backup file is damaged, so nothing was imported."}
    assert await count(db, models.ChatHistory) == chats_before
    assert (
        await db.execute(select(models.Persona.name).where(models.Persona.name == "Would be new"))
    ).first() is None


async def test_a_value_the_database_refuses_is_a_damaged_file_not_a_server_error(
    client: httpx.AsyncClient, db: AsyncSession, populated
):
    data = await service.export_data(db, include_documents=True)
    data["collections"][0]["documents"][0]["chunks"][0]["embedding"] = "not a vector"

    response = await client.post(
        "/backup/", files={"file": ("backup.json", json.dumps(data).encode(), "application/json")}
    )

    assert response.status_code == 400
    assert "damaged" in response.json()["error"]


async def test_backup_download_and_upload_over_http(client: httpx.AsyncClient, populated):
    download = await client.get("/backup/?documents=true")
    assert download.status_code == 200
    assert download.headers["Content-Type"] == "application/json"
    assert download.headers["Content-Disposition"].startswith('attachment; filename="reichat-backup-')
    assert download.headers["Content-Disposition"].endswith('.json"')

    response = await client.post(
        "/backup/", files={"file": ("backup.json", download.content, "application/json")}
    )

    assert response.status_code == 200
    assert response.json()["chats"] == 1
    assert response.json()["documents"] == 1
    assert set(response.json()) == {
        "chats",
        "messages",
        "personas",
        "prompt_templates",
        "mcp_servers",
        "collections",
        "documents",
        "notes",
    }


async def test_a_download_without_documents_has_none(client: httpx.AsyncClient, populated):
    body = (await client.get("/backup/")).json()

    assert body["collections"] == []


async def test_uploading_something_that_is_not_json_is_explained(client: httpx.AsyncClient):
    response = await client.post(
        "/backup/", files={"file": ("backup.json", b"\xff\xfe not json", "application/json")}
    )

    assert response.status_code == 400
    assert response.json() == {"error": "That file couldn't be read — it isn't valid JSON."}


async def test_uploading_a_json_file_that_is_not_a_backup_is_explained(client: httpx.AsyncClient):
    response = await client.post(
        "/backup/", files={"file": ("x.json", b'{"hello": "world"}', "application/json")}
    )

    assert response.status_code == 400
    assert response.json() == {"error": "This isn't a ReiChat backup file."}


async def test_a_file_that_is_too_large_is_refused_before_it_is_read(client: httpx.AsyncClient, monkeypatch):
    from app.backup import router

    monkeypatch.setattr(router, "MAX_BACKUP_BYTES", 10)

    response = await client.post("/backup/", files={"file": ("big.json", b"x" * 100, "application/json")})

    assert response.status_code == 400
    assert response.json() == {"error": "That file is too large to import."}


async def test_restoring_needs_a_file(client: httpx.AsyncClient):
    response = await client.post("/backup/", data={})

    assert response.status_code == 400
    assert response.json()["missing_params"] == ["file"]


async def test_a_backup_cannot_be_restored_from_another_site(client: httpx.AsyncClient, db: AsyncSession):
    response = await client.post(
        "/backup/",
        files={"file": ("backup.json", b"{}", "application/json")},
        headers={"Origin": "https://evil.example"},
    )

    assert response.status_code == 403


# One conversation


def exported_conversation(**fields):
    return {
        "title": "Exported",
        "model": "not-installed-here",
        "messages": [
            {"role": "user", "content": "Hi", "image": ""},
            {"role": "assistant", "content": "Hello", "image": ""},
        ],
        **fields,
    }


def upload(payload: dict):
    return {"file": ("chat.json", json.dumps(payload).encode(), "application/json")}


async def test_single_exported_chat_imports_as_a_linear_conversation(
    client: httpx.AsyncClient, db: AsyncSession, populated
):
    chat, _ = populated

    response = await client.post("/chat-import/llama3.1-backup-test", files=upload(exported_conversation()))

    assert response.status_code == 201
    body = response.json()
    assert (body["title"], body["model"]) == ("Exported", "llama3.1-backup-test")
    imported = (
        await db.execute(
            select(models.ChatHistory)
            .where(models.ChatHistory.id == body["id"])
            .execution_options(populate_existing=True)
        )
    ).scalar_one()
    assert imported.ai_model_id == chat.ai_model_id
    path = [
        m["content"] for m in (await client.get(f"/chat-history/llama3.1-backup-test/{imported.id}")).json()
    ]
    assert path == ["Hi", "Hello"]


async def test_an_imported_conversation_keeps_its_own_model_when_it_is_installed(
    client: httpx.AsyncClient, db: AsyncSession, populated
):
    other = await make_model(db, "mistral")

    response = await client.post(
        "/chat-import/llama3.1-backup-test", files=upload(exported_conversation(model="mistral"))
    )

    assert response.json()["model"] == "mistral"
    chat = (
        await db.execute(select(models.ChatHistory).where(models.ChatHistory.id == response.json()["id"]))
    ).scalar_one()
    assert chat.ai_model_id == other.id


async def test_an_imported_conversation_keeps_images_and_drops_other_roles(
    client: httpx.AsyncClient, populated
):
    payload = exported_conversation(
        messages=[
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "Look", "image": "data:image/png;base64,AAA"},
            {"role": "assistant", "content": 5},
            {"role": "assistant", "content": "A cat."},
            "garbage",
        ]
    )

    response = await client.post("/chat-import/llama3.1-backup-test", files=upload(payload))

    path = (await client.get(f"/chat-history/llama3.1-backup-test/{response.json()['id']}")).json()
    assert [(m["role"], m["content"], m["image"]) for m in path] == [
        ("user", "Look", "data:image/png;base64,AAA"),
        ("assistant", "A cat.", ""),
    ]


async def test_an_imported_conversation_without_a_title_gets_one(client: httpx.AsyncClient, populated):
    response = await client.post(
        "/chat-import/llama3.1-backup-test", files=upload(exported_conversation(title=""))
    )

    assert response.json()["title"] == "Imported chat"


async def test_a_conversation_with_nothing_to_import_is_refused(
    client: httpx.AsyncClient, db: AsyncSession, populated
):
    chats_before = await count(db, models.ChatHistory)

    response = await client.post(
        "/chat-import/llama3.1-backup-test",
        files=upload(exported_conversation(messages=[{"role": "tool", "content": "x"}])),
    )

    assert response.status_code == 400
    assert response.json() == {"error": "That conversation has no messages to import."}
    assert await count(db, models.ChatHistory) == chats_before


async def test_a_file_that_is_not_a_conversation_is_refused(client: httpx.AsyncClient, populated):
    response = await client.post("/chat-import/llama3.1-backup-test", files=upload({"hello": "world"}))

    assert response.status_code == 400
    assert response.json() == {"error": "This isn't an exported ReiChat conversation."}


async def test_importing_into_an_unknown_model_is_a_404(client: httpx.AsyncClient):
    response = await client.post("/chat-import/nothing", files=upload(exported_conversation()))

    assert response.status_code == 404
    assert response.json() == {"error": "AI model not found"}


async def test_importing_a_conversation_that_is_not_json_is_explained(client: httpx.AsyncClient, populated):
    response = await client.post(
        "/chat-import/llama3.1-backup-test",
        files={"file": ("c.json", io.BytesIO(b"nope"), "application/json")},
    )

    assert response.status_code == 400
    assert "isn't valid JSON" in response.json()["error"]


async def test_model_names_with_slashes_work_for_imports(client: httpx.AsyncClient, db: AsyncSession):
    await make_model(db, "hf.co/u/repo", is_custom=True)

    response = await client.post("/chat-import/hf.co/u/repo", files=upload(exported_conversation()))

    assert response.status_code == 201
    assert response.json()["model"] == "hf.co/u/repo"
