from __future__ import annotations

import json

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile

from django_app import backup, models

pytestmark = pytest.mark.django_db


@pytest.fixture
def populated():
    ai_model = models.AIModel.objects.create(name="llama3.1", model="llama3.1-backup-test", index=1)
    embedder = models.AIModel.objects.create(name="nomic", model="nomic-backup-test", is_embedding=True, index=2)
    persona = models.Persona.objects.create(name="Pirate", system_prompt="Arr.")
    models.PromptTemplate.objects.create(name="Summarize", content="Summarize {{text}}")
    models.MCPServer.objects.create(name="Shell", transport="stdio", command="python server.py", auto_approve=True)

    chat = models.ChatHistory.objects.create(ai_model=ai_model, title="Trip", persona=persona, num_ctx=16384)
    user = models.ChatMessage.objects.create(chat=chat, role="user", content="Plan a trip")
    old_reply = models.ChatMessage.objects.create(chat=chat, parent=user, role="assistant", content="Rome?")
    new_reply = models.ChatMessage.objects.create(
        chat=chat, parent=user, role="assistant", content="Paris!", model_label="llama3.1:8b", thinking="hmm"
    )
    chat.active_leaf = new_reply
    chat.save(update_fields=["active_leaf"])

    collection = models.DocumentCollection.objects.create(
        name="Guides", embedding_model=embedder, embedding_parameters="latest"
    )
    document = models.Document.objects.create(collection=collection, filename="paris.pdf", status="ready")
    models.DocumentChunk.objects.create(
        document=document, chunk_index=0, content="Eiffel tower", char_start=0, page=2, embedding=[1.0, 0.0]
    )
    chat.active_collections.add(collection)
    return chat, old_reply


def test_round_trip_restores_branches_settings_and_documents(populated):
    data = json.loads(json.dumps(backup.export_data(include_documents=True)))

    report = backup.import_data(data)

    assert report.chats == models.ChatHistory.objects.count() // 2
    restored = models.ChatHistory.objects.filter(title="Trip").order_by("-id").first()
    assert restored.num_ctx == 16384
    assert restored.persona.name == "Pirate"
    assert restored.active_leaf.content == "Paris!"
    assert restored.active_leaf.model_label == "llama3.1:8b"
    assert restored.active_leaf.thinking == "hmm"
    siblings = models.ChatMessage.objects.filter(chat=restored, role="assistant").order_by("id")
    assert [m.content for m in siblings] == ["Rome?", "Paris!"]
    assert siblings[0].parent_id == siblings[1].parent_id

    restored_collection = restored.active_collections.get()
    chunk = models.DocumentChunk.objects.get(document__collection=restored_collection)
    assert chunk.page == 2
    assert list(chunk.embedding) == [1.0, 0.0]

    # Same-named persona with the same prompt is reused, not duplicated.
    assert models.Persona.objects.filter(name="Pirate").count() == 1


def test_imported_tool_servers_arrive_switched_off(populated):
    data = backup.export_data(include_documents=False)
    models.MCPServer.objects.all().delete()

    report = backup.import_data(data)

    server = models.MCPServer.objects.get(name="Shell")
    assert server.enabled is False
    assert server.auto_approve is False
    assert any("switched off" in note for note in report.notes)


def test_chats_for_unknown_models_get_a_placeholder_model(populated):
    data = backup.export_data(include_documents=False)
    data["chats"][0]["model"] = "never-installed-model"

    report = backup.import_data(data)

    placeholder = models.AIModel.objects.get(model="never-installed-model")
    assert placeholder.is_custom is True
    assert models.ChatHistory.objects.filter(ai_model=placeholder).count() == 1
    assert any("never-installed-model" in note for note in report.notes)


def test_rejects_files_that_are_not_backups():
    with pytest.raises(backup.BackupFormatError):
        backup.import_data({"hello": "world"})
    with pytest.raises(backup.BackupFormatError):
        backup.import_data({"format": "reichat-backup", "version": 99})


def test_damaged_backup_imports_nothing(client, populated):
    data = backup.export_data(include_documents=False)
    chats_before = models.ChatHistory.objects.count()
    data["chats"].append({"title": "broken"})
    upload = SimpleUploadedFile("backup.json", json.dumps(data).encode(), content_type="application/json")

    response = client.post("/backup/", data={"file": upload})

    assert response.status_code == 400
    assert models.ChatHistory.objects.count() == chats_before


def test_backup_download_and_upload_over_http(client, populated):
    download = client.get("/backup/?documents=true")
    assert download.status_code == 200
    assert "attachment" in download["Content-Disposition"]

    upload = SimpleUploadedFile("backup.json", download.content, content_type="application/json")
    response = client.post("/backup/", data={"file": upload})

    assert response.status_code == 200
    assert response.json()["chats"] == 1
    assert response.json()["documents"] == 1


def test_single_exported_chat_imports_as_a_linear_conversation(client, populated):
    chat, _ = populated
    exported = {
        "title": "Exported",
        "model": "not-installed-here",
        "messages": [
            {"role": "user", "content": "Hi", "image": ""},
            {"role": "assistant", "content": "Hello", "image": ""},
        ],
    }
    upload = SimpleUploadedFile("chat.json", json.dumps(exported).encode(), content_type="application/json")

    response = client.post(f"/chat-import/{chat.ai_model.model}", data={"file": upload})

    assert response.status_code == 201
    imported = models.ChatHistory.objects.get(id=response.json()["id"])
    assert imported.ai_model_id == chat.ai_model_id
    assert imported.active_leaf.content == "Hello"
    assert imported.active_leaf.parent.content == "Hi"
