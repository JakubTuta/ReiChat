"""DRF views over Django's test client — synchronous, no channel layer or
Ollama/Docker involved. Covers the branch-aware chat-history GET and the
branch-switch PATCH endpoint added alongside ChatMessage.parent.
"""

from __future__ import annotations

import json

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile

from django_app import models

pytestmark = pytest.mark.django_db


@pytest.fixture
def chat():
    models.AIModel.objects.filter(model="llama3.1").delete()
    ai_model = models.AIModel.objects.create(name="llama3.1", model="llama3.1", index=1)
    return models.ChatHistory.objects.create(ai_model=ai_model, title="Test chat")


@pytest.fixture
def branching_chat(chat):
    """user "Hi" -> assistant "Hello" (inactive) / assistant "Hi again" (active)."""
    user_msg = models.ChatMessage.objects.create(chat=chat, parent=None, role="user", content="Hi")
    models.ChatMessage.objects.create(chat=chat, parent=user_msg, role="assistant", content="Hello")
    active_reply = models.ChatMessage.objects.create(
        chat=chat, parent=user_msg, role="assistant", content="Hi again"
    )
    chat.active_leaf = active_reply
    chat.save(update_fields=["active_leaf"])
    return chat


def _history_url(chat_history: models.ChatHistory) -> str:
    return f"/chat-history/{chat_history.ai_model.model}/{chat_history.id}"


def test_get_chat_history_returns_active_path_with_branch_metadata(client, branching_chat):
    response = client.get(_history_url(branching_chat))

    assert response.status_code == 200
    assert response.json() == [
        {"role": "user", "content": "Hi", "image": "", "thinking": "", "model": "", "sibling_count": 1, "sibling_index": 0},
        {"role": "assistant", "content": "Hi again", "image": "", "thinking": "", "model": "", "sibling_count": 2, "sibling_index": 1},
    ]


def test_patch_switch_branch_moves_active_leaf_to_chosen_sibling(client, branching_chat):
    response = client.patch(
        _history_url(branching_chat),
        data=json.dumps({"index": 1, "sibling_index": 0}),
        content_type="application/json",
    )

    assert response.status_code == 200
    assert response.json() == [
        {"role": "user", "content": "Hi", "image": "", "thinking": "", "model": "", "sibling_count": 1, "sibling_index": 0},
        {"role": "assistant", "content": "Hello", "image": "", "thinking": "", "model": "", "sibling_count": 2, "sibling_index": 0},
    ]

    branching_chat.refresh_from_db()
    assert branching_chat.active_leaf.content == "Hello"


def test_patch_switch_branch_out_of_range_sibling_returns_400(client, branching_chat):
    response = client.patch(
        _history_url(branching_chat),
        data=json.dumps({"index": 1, "sibling_index": 5}),
        content_type="application/json",
    )

    assert response.status_code == 400
    assert "error" in response.json()


def test_patch_switch_branch_missing_body_params_returns_400(client, branching_chat):
    response = client.patch(
        _history_url(branching_chat),
        data=json.dumps({"index": 1}),
        content_type="application/json",
    )

    assert response.status_code == 400


def test_patch_switch_branch_non_integer_params_returns_400(client, branching_chat):
    response = client.patch(
        _history_url(branching_chat),
        data=json.dumps({"index": "1", "sibling_index": 0}),
        content_type="application/json",
    )

    assert response.status_code == 400


def test_patch_switch_branch_chat_not_found_returns_404(client, chat):
    response = client.patch(
        f"/chat-history/{chat.ai_model.model}/999999",
        data=json.dumps({"index": 0, "sibling_index": 0}),
        content_type="application/json",
    )

    assert response.status_code == 404


def test_patch_switch_branch_unknown_model_returns_404(client, branching_chat):
    response = client.patch(
        f"/chat-history/does-not-exist/{branching_chat.id}",
        data=json.dumps({"index": 0, "sibling_index": 0}),
        content_type="application/json",
    )

    assert response.status_code == 404


# --- Personas -----------------------------------------------------------


def test_get_personas_lists_alphabetically(client):
    models.Persona.objects.create(name="Zeta", system_prompt="z")
    models.Persona.objects.create(name="Alpha", system_prompt="a")

    response = client.get("/personas/")

    assert response.status_code == 200
    assert [p["name"] for p in response.json()] == ["Alpha", "Zeta"]


def test_post_persona_creates_it(client):
    response = client.post(
        "/personas/",
        data=json.dumps({"name": "Pirate", "system_prompt": "Talk like a pirate."}),
        content_type="application/json",
    )

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Pirate"
    assert body["system_prompt"] == "Talk like a pirate."
    assert models.Persona.objects.filter(name="Pirate").exists()


def test_post_persona_missing_fields_returns_400(client):
    response = client.post(
        "/personas/",
        data=json.dumps({"name": "Pirate"}),
        content_type="application/json",
    )

    assert response.status_code == 400


def test_put_persona_updates_it(client):
    persona = models.Persona.objects.create(name="Pirate", system_prompt="Arr.")

    response = client.put(
        "/personas/",
        data=json.dumps({"id": persona.id, "system_prompt": "Talk like a pirate, matey."}),
        content_type="application/json",
    )

    assert response.status_code == 200
    persona.refresh_from_db()
    assert persona.name == "Pirate"
    assert persona.system_prompt == "Talk like a pirate, matey."


def test_put_persona_not_found_returns_404(client):
    response = client.put(
        "/personas/",
        data=json.dumps({"id": 999999, "name": "x"}),
        content_type="application/json",
    )

    assert response.status_code == 404


def test_delete_persona_removes_it_and_unsets_it_on_chats(client, chat):
    persona = models.Persona.objects.create(name="Pirate", system_prompt="Arr.")
    chat.persona = persona
    chat.save(update_fields=["persona"])

    response = client.delete(
        "/personas/",
        data=json.dumps({"id": persona.id}),
        content_type="application/json",
    )

    assert response.status_code == 200
    assert not models.Persona.objects.filter(id=persona.id).exists()

    chat.refresh_from_db()
    assert chat.persona_id is None


def test_delete_persona_not_found_returns_404(client):
    response = client.delete(
        "/personas/",
        data=json.dumps({"id": 999999}),
        content_type="application/json",
    )

    assert response.status_code == 404


# --- Prompt templates -----------------------------------------------------


def test_get_prompt_templates_lists_alphabetically(client):
    models.PromptTemplate.objects.create(name="Zeta", content="z")
    models.PromptTemplate.objects.create(name="Alpha", content="a")

    response = client.get("/prompt-templates/")

    assert response.status_code == 200
    assert [t["name"] for t in response.json()] == ["Alpha", "Zeta"]


def test_post_prompt_template_creates_it(client):
    response = client.post(
        "/prompt-templates/",
        data=json.dumps({
            "name": "Bug report",
            "description": "Structured bug report starter",
            "content": "Bug: {{summary}}\nSteps to reproduce: {{steps}}",
        }),
        content_type="application/json",
    )

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Bug report"
    assert body["content"] == "Bug: {{summary}}\nSteps to reproduce: {{steps}}"
    assert models.PromptTemplate.objects.filter(name="Bug report").exists()


def test_post_prompt_template_defaults_description_to_empty(client):
    response = client.post(
        "/prompt-templates/",
        data=json.dumps({"name": "Plain", "content": "Just some text."}),
        content_type="application/json",
    )

    assert response.status_code == 201
    assert response.json()["description"] == ""


def test_post_prompt_template_accepts_explicit_empty_description(client):
    # The frontend always sends `description` (possibly ""), never omits it
    # — the model field needs blank=True or DRF rejects an explicit "" with
    # "This field may not be blank."
    response = client.post(
        "/prompt-templates/",
        data=json.dumps({"name": "Plain", "content": "Just some text.", "description": ""}),
        content_type="application/json",
    )

    assert response.status_code == 201
    assert response.json()["description"] == ""


def test_post_prompt_template_missing_fields_returns_400(client):
    response = client.post(
        "/prompt-templates/",
        data=json.dumps({"name": "Incomplete"}),
        content_type="application/json",
    )

    assert response.status_code == 400


def test_put_prompt_template_updates_it(client):
    template = models.PromptTemplate.objects.create(name="Draft", content="v1")

    response = client.put(
        "/prompt-templates/",
        data=json.dumps({"id": template.id, "content": "v2"}),
        content_type="application/json",
    )

    assert response.status_code == 200
    template.refresh_from_db()
    assert template.name == "Draft"
    assert template.content == "v2"


def test_put_prompt_template_not_found_returns_404(client):
    response = client.put(
        "/prompt-templates/",
        data=json.dumps({"id": 999999, "content": "x"}),
        content_type="application/json",
    )

    assert response.status_code == 404


def test_delete_prompt_template_removes_it(client):
    template = models.PromptTemplate.objects.create(name="Draft", content="v1")

    response = client.delete(
        "/prompt-templates/",
        data=json.dumps({"id": template.id}),
        content_type="application/json",
    )

    assert response.status_code == 200
    assert not models.PromptTemplate.objects.filter(id=template.id).exists()


def test_delete_prompt_template_not_found_returns_404(client):
    response = client.delete(
        "/prompt-templates/",
        data=json.dumps({"id": 999999}),
        content_type="application/json",
    )

    assert response.status_code == 404


# --- Chat <-> persona wiring ----------------------------------------------


def _all_chats_url(chat_history: models.ChatHistory) -> str:
    return f"/all-chats/{chat_history.ai_model.model}"


def test_get_all_chats_includes_persona(client, chat):
    persona = models.Persona.objects.create(name="Pirate", system_prompt="Arr.")
    chat.persona = persona
    chat.save(update_fields=["persona"])

    response = client.get(_all_chats_url(chat))

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": chat.id,
            "title": chat.title,
            "persona": {"id": persona.id, "name": "Pirate"},
            "temperature": None,
            "num_ctx": None,
            "top_p": None,
            "seed": None,
            "active_collections": [],
            "tools_enabled": False,
            "memory_mode": "trim",
            "think_level": "",
            "continued_from": None,
        },
    ]


def test_put_all_chats_sets_memory_mode(client, chat):
    response = client.put(
        _all_chats_url(chat), data=json.dumps({"id": chat.id, "memory_mode": "summarize"}),
        content_type="application/json",
    )

    assert response.status_code == 200
    chat.refresh_from_db()
    assert chat.memory_mode == "summarize"


def test_put_all_chats_rejects_unknown_memory_mode(client, chat):
    response = client.put(
        _all_chats_url(chat), data=json.dumps({"id": chat.id, "memory_mode": "forever"}),
        content_type="application/json",
    )

    assert response.status_code == 400


def test_continue_chat_copies_settings_and_links_the_source(client, chat):
    persona = models.Persona.objects.create(name="Pirate", system_prompt="Arr.")
    chat.persona = persona
    chat.num_ctx = 16384
    chat.memory_mode = "summarize"
    chat.save(update_fields=["persona", "num_ctx", "memory_mode"])

    response = client.post(
        f"/chat-continue/{chat.ai_model.model}", data=json.dumps({"chat_id": chat.id}),
        content_type="application/json",
    )

    assert response.status_code == 201
    continued = models.ChatHistory.objects.get(id=response.json()["id"])
    assert continued.seed_source_id == chat.id
    assert continued.persona_id == persona.id
    assert continued.num_ctx == 16384
    assert continued.memory_mode == "summarize"
    assert continued.title == "Test chat (continued)"


def test_chat_context_estimates_memory_use_before_sending(client, chat, monkeypatch):
    from django_app import functions

    monkeypatch.setattr(functions, "get_model_info", lambda model, params: {"context_length": 4096, "thinking": None})
    user = models.ChatMessage.objects.create(chat=chat, role="user", content="word " * 700)
    reply = models.ChatMessage.objects.create(chat=chat, parent=user, role="assistant", content="ok")
    chat.active_leaf = reply
    chat.save(update_fields=["active_leaf"])

    response = client.get(f"/chat-context/{chat.ai_model.model}/{chat.id}?parameters=8b")

    assert response.status_code == 200
    body = response.json()
    assert body["limit"] == 4096
    assert body["model_max"] == 4096
    assert body["preset"] == "short"
    assert 1000 < body["used"] < 1100
    assert body["presets"]["standard"] == 8192


def test_get_all_chats_persona_null_when_unset(client, chat):
    response = client.get(_all_chats_url(chat))

    assert response.status_code == 200
    assert response.json()[0]["persona"] is None


def test_put_all_chats_sets_persona(client, chat):
    persona = models.Persona.objects.create(name="Pirate", system_prompt="Arr.")

    response = client.put(
        _all_chats_url(chat),
        data=json.dumps({"id": chat.id, "persona_id": persona.id}),
        content_type="application/json",
    )

    assert response.status_code == 200
    chat.refresh_from_db()
    assert chat.persona_id == persona.id
    # Only persona_id was sent — title must be untouched.
    assert chat.title == "Test chat"


def test_put_all_chats_clears_persona(client, chat):
    persona = models.Persona.objects.create(name="Pirate", system_prompt="Arr.")
    chat.persona = persona
    chat.save(update_fields=["persona"])

    response = client.put(
        _all_chats_url(chat),
        data=json.dumps({"id": chat.id, "persona_id": None}),
        content_type="application/json",
    )

    assert response.status_code == 200
    chat.refresh_from_db()
    assert chat.persona_id is None


def test_put_all_chats_unknown_persona_returns_404(client, chat):
    response = client.put(
        _all_chats_url(chat),
        data=json.dumps({"id": chat.id, "persona_id": 999999}),
        content_type="application/json",
    )

    assert response.status_code == 404


def test_put_all_chats_no_fields_returns_400(client, chat):
    response = client.put(
        _all_chats_url(chat),
        data=json.dumps({"id": chat.id}),
        content_type="application/json",
    )

    assert response.status_code == 400


# --- Generation parameters -------------------------------------------------


def test_put_all_chats_sets_generation_params(client, chat):
    response = client.put(
        _all_chats_url(chat),
        data=json.dumps({"id": chat.id, "temperature": 0.8, "num_ctx": 8192, "top_p": 0.9, "seed": 42}),
        content_type="application/json",
    )

    assert response.status_code == 200
    chat.refresh_from_db()
    assert chat.temperature == 0.8
    assert chat.num_ctx == 8192
    assert chat.top_p == 0.9
    assert chat.seed == 42


def test_put_all_chats_clears_generation_param_with_null(client, chat):
    chat.temperature = 0.8
    chat.save(update_fields=["temperature"])

    response = client.put(
        _all_chats_url(chat),
        data=json.dumps({"id": chat.id, "temperature": None}),
        content_type="application/json",
    )

    assert response.status_code == 200
    chat.refresh_from_db()
    assert chat.temperature is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", -0.1),
        ("temperature", 2.1),
        ("top_p", -0.1),
        ("top_p", 1.1),
        ("num_ctx", 0),
        ("temperature", "hot"),
        ("temperature", True),
    ],
)
def test_put_all_chats_rejects_invalid_generation_param(client, chat, field, value):
    response = client.put(
        _all_chats_url(chat),
        data=json.dumps({"id": chat.id, field: value}),
        content_type="application/json",
    )

    assert response.status_code == 400
    chat.refresh_from_db()
    assert getattr(chat, field) is None


def test_put_all_chats_seed_has_no_upper_or_lower_bound(client, chat):
    response = client.put(
        _all_chats_url(chat),
        data=json.dumps({"id": chat.id, "seed": -12345}),
        content_type="application/json",
    )

    assert response.status_code == 200
    chat.refresh_from_db()
    assert chat.seed == -12345


# --- Search ----------------------------------------------------------------


def test_get_search_finds_matches_for_the_model(client, chat):
    models.ChatMessage.objects.create(chat=chat, role="user", content="tell me about quantum computing")

    response = client.get(f"/search/{chat.ai_model.model}?q=quantum")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["chat_title"] == chat.title
    assert "quantum" in body[0]["snippet"].lower()


def test_get_search_unknown_model_returns_404(client):
    response = client.get("/search/does-not-exist?q=hi")

    assert response.status_code == 404


def test_get_search_missing_query_returns_empty_list(client, chat):
    response = client.get(f"/search/{chat.ai_model.model}")

    assert response.status_code == 200
    assert response.json() == []


def test_get_search_no_matches_returns_empty_list(client, chat):
    models.ChatMessage.objects.create(chat=chat, role="user", content="hello there")

    response = client.get(f"/search/{chat.ai_model.model}?q=nonexistent")

    assert response.status_code == 200
    assert response.json() == []


# --- Collections & Documents ------------------------------------------------


class _SyncThread:
    """Stands in for threading.Thread in rag/ingest.py so Documents.post's
    background ingest runs inline instead of racing the test's assertions.
    """

    def __init__(self, target=None, args=(), kwargs=None, daemon=None, name=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        self._target(*self._args, **self._kwargs)


@pytest.fixture
def run_ingest_synchronously(monkeypatch):
    from django_app.rag import ingest

    monkeypatch.setattr(ingest.threading, "Thread", _SyncThread)


@pytest.fixture
def stub_ingest_embeddings(monkeypatch):
    """Documents.post always kicks off real ingest — stub the embedding
    call so upload tests don't need a running Ollama container.
    """
    from django_app.rag import ingest

    class _FakeEmbeddingsClient:
        def embed_documents(self, texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr(ingest.embeddings, "get_embeddings_client", lambda model, params: _FakeEmbeddingsClient())


@pytest.fixture
def embedding_model():
    models.AIModel.objects.filter(model="nomic-embed-text-views-test").delete()
    return models.AIModel.objects.create(
        name="Nomic Embed", model="nomic-embed-text-views-test", is_embedding=True, index=1
    )


@pytest.fixture
def chat_model():
    return models.AIModel.objects.create(name="llama3.1", model="llama3.1-rag-test", index=1)


def test_post_collection_creates_it(client, embedding_model):
    response = client.post(
        "/collections/",
        data=json.dumps(
            {"name": "My Docs", "embedding_model": embedding_model.id, "embedding_parameters": "latest"}
        ),
        content_type="application/json",
    )

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "My Docs"
    assert body["embedding_model_name"] == "nomic-embed-text-views-test"
    assert body["document_count"] == 0
    assert models.DocumentCollection.objects.filter(name="My Docs").exists()


def test_post_collection_rejects_non_embedding_model(client, chat_model):
    response = client.post(
        "/collections/",
        data=json.dumps(
            {"name": "My Docs", "embedding_model": chat_model.id, "embedding_parameters": "8b"}
        ),
        content_type="application/json",
    )

    assert response.status_code == 400


def test_post_collection_unknown_embedding_model_returns_404(client):
    response = client.post(
        "/collections/",
        data=json.dumps({"name": "My Docs", "embedding_model": 999999, "embedding_parameters": "latest"}),
        content_type="application/json",
    )

    assert response.status_code == 404


def test_get_collections_includes_document_count(client, embedding_model):
    collection = models.DocumentCollection.objects.create(
        name="Docs", embedding_model=embedding_model, embedding_parameters="latest"
    )
    models.Document.objects.create(collection=collection, filename="a.txt")
    models.Document.objects.create(collection=collection, filename="b.txt")

    response = client.get("/collections/")

    assert response.status_code == 200
    body = response.json()
    assert body[0]["document_count"] == 2


def test_get_collections_filtered_by_chat_id_includes_global_ones(client, embedding_model, chat_model):
    chat = models.ChatHistory.objects.create(ai_model=chat_model, title="t")
    other_chat = models.ChatHistory.objects.create(ai_model=chat_model, title="other")

    global_collection = models.DocumentCollection.objects.create(
        name="Global", embedding_model=embedding_model, embedding_parameters="latest"
    )
    own_collection = models.DocumentCollection.objects.create(
        name="Own", chat=chat, embedding_model=embedding_model, embedding_parameters="latest"
    )
    models.DocumentCollection.objects.create(
        name="Someone else's", chat=other_chat, embedding_model=embedding_model, embedding_parameters="latest"
    )

    response = client.get(f"/collections/?chat_id={chat.id}")

    names = {c["name"] for c in response.json()}
    assert names == {global_collection.name, own_collection.name}


def test_delete_collection_removes_it_and_cascades_documents(client, embedding_model):
    collection = models.DocumentCollection.objects.create(
        name="Docs", embedding_model=embedding_model, embedding_parameters="latest"
    )
    document = models.Document.objects.create(collection=collection, filename="a.txt")

    response = client.delete(
        "/collections/", data=json.dumps({"id": collection.id}), content_type="application/json"
    )

    assert response.status_code == 200
    assert not models.DocumentCollection.objects.filter(id=collection.id).exists()
    assert not models.Document.objects.filter(id=document.id).exists()


def test_delete_collection_not_found_returns_404(client):
    response = client.delete("/collections/", data=json.dumps({"id": 999999}), content_type="application/json")

    assert response.status_code == 404


def test_post_document_uploads_extracts_and_ingests(
    client, embedding_model, run_ingest_synchronously, stub_ingest_embeddings
):
    collection = models.DocumentCollection.objects.create(
        name="Docs", embedding_model=embedding_model, embedding_parameters="latest"
    )
    upload = SimpleUploadedFile("notes.txt", b"Some real content to chunk and embed. " * 20)

    response = client.post("/documents/", data={"collection_id": collection.id, "file": upload})

    assert response.status_code == 202
    body = response.json()
    assert body["filename"] == "notes.txt"

    document = models.Document.objects.get(id=body["id"])
    assert document.status == models.Document.STATUS_READY
    assert document.chunk_count > 0


def test_post_document_unsupported_file_type_returns_400(client, embedding_model):
    collection = models.DocumentCollection.objects.create(
        name="Docs", embedding_model=embedding_model, embedding_parameters="latest"
    )
    upload = SimpleUploadedFile("archive.zip", b"whatever")

    response = client.post("/documents/", data={"collection_id": collection.id, "file": upload})

    assert response.status_code == 400
    assert not models.Document.objects.filter(collection=collection).exists()


def test_post_document_unknown_collection_returns_404(client):
    upload = SimpleUploadedFile("notes.txt", b"content")

    response = client.post("/documents/", data={"collection_id": 999999, "file": upload})

    assert response.status_code == 404


def test_get_documents_requires_collection_id(client):
    response = client.get("/documents/")

    assert response.status_code == 400


def test_get_documents_lists_for_collection(client, embedding_model):
    collection = models.DocumentCollection.objects.create(
        name="Docs", embedding_model=embedding_model, embedding_parameters="latest"
    )
    models.Document.objects.create(collection=collection, filename="a.txt")

    response = client.get(f"/documents/?collection_id={collection.id}")

    assert response.status_code == 200
    assert response.json()[0]["filename"] == "a.txt"


def test_delete_document_removes_it(client, embedding_model):
    collection = models.DocumentCollection.objects.create(
        name="Docs", embedding_model=embedding_model, embedding_parameters="latest"
    )
    document = models.Document.objects.create(collection=collection, filename="a.txt")

    response = client.delete("/documents/", data=json.dumps({"id": document.id}), content_type="application/json")

    assert response.status_code == 200
    assert not models.Document.objects.filter(id=document.id).exists()


def test_delete_document_not_found_returns_404(client):
    response = client.delete("/documents/", data=json.dumps({"id": 999999}), content_type="application/json")

    assert response.status_code == 404


# --- Chat <-> collections wiring --------------------------------------------


def test_put_all_chats_sets_active_collections(client, chat, embedding_model):
    collection = models.DocumentCollection.objects.create(
        name="Docs", embedding_model=embedding_model, embedding_parameters="latest"
    )

    response = client.put(
        _all_chats_url(chat),
        data=json.dumps({"id": chat.id, "collection_ids": [collection.id]}),
        content_type="application/json",
    )

    assert response.status_code == 200
    assert list(chat.active_collections.values_list("id", flat=True)) == [collection.id]


def test_put_all_chats_clears_active_collections_with_empty_list(client, chat, embedding_model):
    collection = models.DocumentCollection.objects.create(
        name="Docs", embedding_model=embedding_model, embedding_parameters="latest"
    )
    chat.active_collections.add(collection)

    response = client.put(
        _all_chats_url(chat),
        data=json.dumps({"id": chat.id, "collection_ids": []}),
        content_type="application/json",
    )

    assert response.status_code == 200
    assert chat.active_collections.count() == 0


def test_put_all_chats_unknown_collection_id_returns_404(client, chat):
    response = client.put(
        _all_chats_url(chat),
        data=json.dumps({"id": chat.id, "collection_ids": [999999]}),
        content_type="application/json",
    )

    assert response.status_code == 404


def test_get_all_chats_includes_active_collections(client, chat, embedding_model):
    collection = models.DocumentCollection.objects.create(
        name="Docs", embedding_model=embedding_model, embedding_parameters="latest"
    )
    chat.active_collections.add(collection)

    response = client.get(_all_chats_url(chat))

    assert response.json()[0]["active_collections"] == [{"id": collection.id, "name": "Docs"}]


# --- Chat <-> tools wiring ---------------------------------------------------


def test_put_all_chats_enables_tools(client, chat):
    response = client.put(
        _all_chats_url(chat), data=json.dumps({"id": chat.id, "tools_enabled": True}), content_type="application/json"
    )

    assert response.status_code == 200
    chat.refresh_from_db()
    assert chat.tools_enabled is True


def test_put_all_chats_disables_tools(client, chat):
    chat.tools_enabled = True
    chat.save(update_fields=["tools_enabled"])

    response = client.put(
        _all_chats_url(chat), data=json.dumps({"id": chat.id, "tools_enabled": False}), content_type="application/json"
    )

    assert response.status_code == 200
    chat.refresh_from_db()
    assert chat.tools_enabled is False


def test_put_all_chats_tools_enabled_rejects_non_boolean(client, chat):
    response = client.put(
        _all_chats_url(chat), data=json.dumps({"id": chat.id, "tools_enabled": "yes"}), content_type="application/json"
    )

    assert response.status_code == 400
    chat.refresh_from_db()
    assert chat.tools_enabled is False


def test_get_all_chats_includes_tools_enabled(client, chat):
    chat.tools_enabled = True
    chat.save(update_fields=["tools_enabled"])

    response = client.get(_all_chats_url(chat))

    assert response.json()[0]["tools_enabled"] is True


# --- MCP servers -------------------------------------------------------------


def test_post_mcp_server_stdio_creates_it(client):
    response = client.post(
        "/mcp-servers/",
        data=json.dumps({"name": "Local Tools", "transport": "stdio", "command": "python server.py"}),
        content_type="application/json",
    )

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Local Tools"
    assert body["enabled"] is True
    assert models.MCPServer.objects.filter(name="Local Tools").exists()


def test_post_mcp_server_http_creates_it(client):
    response = client.post(
        "/mcp-servers/",
        data=json.dumps({"name": "Remote Tools", "transport": "http", "url": "http://localhost:9000/mcp"}),
        content_type="application/json",
    )

    assert response.status_code == 201
    assert models.MCPServer.objects.get(name="Remote Tools").url == "http://localhost:9000/mcp"


def test_post_mcp_server_stdio_without_command_returns_400(client):
    response = client.post(
        "/mcp-servers/",
        data=json.dumps({"name": "Local Tools", "transport": "stdio"}),
        content_type="application/json",
    )

    assert response.status_code == 400
    assert not models.MCPServer.objects.filter(name="Local Tools").exists()


def test_post_mcp_server_http_without_url_returns_400(client):
    response = client.post(
        "/mcp-servers/",
        data=json.dumps({"name": "Remote Tools", "transport": "http"}),
        content_type="application/json",
    )

    assert response.status_code == 400


def test_post_mcp_server_invalid_transport_returns_400(client):
    response = client.post(
        "/mcp-servers/",
        data=json.dumps({"name": "Bad", "transport": "carrier-pigeon"}),
        content_type="application/json",
    )

    assert response.status_code == 400


def test_get_mcp_servers_lists_alphabetically(client):
    models.MCPServer.objects.create(name="Zeta", transport="stdio", command="z")
    models.MCPServer.objects.create(name="Alpha", transport="stdio", command="a")

    response = client.get("/mcp-servers/")

    assert response.status_code == 200
    assert [s["name"] for s in response.json()] == ["Alpha", "Zeta"]


def test_put_mcp_server_toggles_enabled(client):
    server = models.MCPServer.objects.create(name="Local", transport="stdio", command="x", enabled=True)

    response = client.put(
        "/mcp-servers/", data=json.dumps({"id": server.id, "enabled": False}), content_type="application/json"
    )

    assert response.status_code == 200
    server.refresh_from_db()
    assert server.enabled is False


def test_put_mcp_server_not_found_returns_404(client):
    response = client.put(
        "/mcp-servers/", data=json.dumps({"id": 999999, "enabled": False}), content_type="application/json"
    )

    assert response.status_code == 404


def test_put_mcp_server_switching_to_http_without_url_returns_400(client):
    server = models.MCPServer.objects.create(name="Local", transport="stdio", command="x")

    response = client.put(
        "/mcp-servers/", data=json.dumps({"id": server.id, "transport": "http"}), content_type="application/json"
    )

    assert response.status_code == 400


def test_delete_mcp_server_removes_it(client):
    server = models.MCPServer.objects.create(name="Local", transport="stdio", command="x")

    response = client.delete(
        "/mcp-servers/", data=json.dumps({"id": server.id}), content_type="application/json"
    )

    assert response.status_code == 200
    assert not models.MCPServer.objects.filter(id=server.id).exists()


def test_delete_mcp_server_not_found_returns_404(client):
    response = client.delete(
        "/mcp-servers/", data=json.dumps({"id": 999999}), content_type="application/json"
    )

    assert response.status_code == 404


def test_mcp_server_test_lists_tools(client, monkeypatch):
    from django_app import views

    server = models.MCPServer.objects.create(name="Probe OK", transport=models.MCPServer.TRANSPORT_HTTP, url="http://x/mcp")

    async def _probe(srv):
        return [{"name": "search", "description": "Search", "input_schema": {}}]

    monkeypatch.setattr(views.mcp_client, "probe_server_tools", _probe)

    response = client.post("/mcp-servers/test/", data=json.dumps({"id": server.id}), content_type="application/json")

    assert response.status_code == 200
    assert response.json() == {"tools": [{"name": "search", "description": "Search"}]}


def test_mcp_server_test_reports_the_reason_it_failed(client, monkeypatch):
    from django_app import views

    server = models.MCPServer.objects.create(name="Probe bad", transport=models.MCPServer.TRANSPORT_STDIO, command="npx x")

    async def _probe(srv):
        raise views.mcp_client.MCPToolError("The command 'npx' isn't available.")

    monkeypatch.setattr(views.mcp_client, "probe_server_tools", _probe)

    response = client.post("/mcp-servers/test/", data=json.dumps({"id": server.id}), content_type="application/json")

    assert response.status_code == 502
    assert response.json() == {"error": "The command 'npx' isn't available."}


def test_get_chat_history_query_count_does_not_grow_with_chat_length(client, chat, django_assert_max_num_queries):
    parent = None
    for i in range(30):
        role = "user" if i % 2 == 0 else "assistant"
        parent = models.ChatMessage.objects.create(chat=chat, parent=parent, role=role, content=f"m{i}")
    chat.active_leaf = parent
    chat.save(update_fields=["active_leaf"])

    with django_assert_max_num_queries(6):
        response = client.get(_history_url(chat))

    assert response.status_code == 200
    assert [m["content"] for m in response.json()] == [f"m{i}" for i in range(30)]


def test_document_chunk_returns_the_cited_passage(client, embedding_model):
    collection = models.DocumentCollection.objects.create(
        name="Docs", embedding_model=embedding_model, embedding_parameters="latest"
    )
    document = models.Document.objects.create(collection=collection, filename="manual.pdf", status="ready")
    models.DocumentChunk.objects.create(
        document=document, chunk_index=3, content="Hold reset for 10 seconds.", char_start=0, page=4,
        embedding=[1.0, 0.0],
    )

    response = client.get(f"/documents/chunk/?document_id={document.id}&chunk_index=3")

    assert response.status_code == 200
    assert response.json() == {"filename": "manual.pdf", "page": 4, "content": "Hold reset for 10 seconds."}


def test_document_chunk_missing_says_so(client):
    response = client.get("/documents/chunk/?document_id=999999&chunk_index=0")

    assert response.status_code == 404


def test_moving_a_chat_to_another_model_keeps_who_wrote_what(client, chat):
    other = models.AIModel.objects.create(name="mistral", model="mistral-move-test", index=2)
    user = models.ChatMessage.objects.create(chat=chat, role="user", content="Hi")
    reply = models.ChatMessage.objects.create(
        chat=chat, parent=user, role="assistant", content="Hello", model_label="llama3.1:8b"
    )
    chat.active_leaf = reply
    chat.save(update_fields=["active_leaf"])

    response = client.put(
        _all_chats_url(chat), data=json.dumps({"id": chat.id, "move_to_model": "mistral-move-test"}),
        content_type="application/json",
    )

    assert response.status_code == 200
    chat.refresh_from_db()
    assert chat.ai_model_id == other.id
    history = client.get(f"/chat-history/mistral-move-test/{chat.id}").json()
    assert history[1]["model"] == "llama3.1:8b"


def test_chats_cannot_move_to_an_embedding_model(client, chat, embedding_model):
    response = client.put(
        _all_chats_url(chat), data=json.dumps({"id": chat.id, "move_to_model": embedding_model.model}),
        content_type="application/json",
    )

    assert response.status_code == 400


def test_compare_answer_becomes_a_chat(client, chat):
    response = client.post(
        f"/chat-from-compare/{chat.ai_model.model}",
        data=json.dumps({"parameters": "8b", "prompt": "Capital of France?", "response": "Paris."}),
        content_type="application/json",
    )

    assert response.status_code == 201
    new_id = response.json()["id"]
    history = client.get(f"/chat-history/{chat.ai_model.model}/{new_id}").json()
    assert [(m["role"], m["content"], m["model"]) for m in history] == [
        ("user", "Capital of France?", ""),
        ("assistant", "Paris.", "llama3.1:8b"),
    ]


@pytest.mark.parametrize(
    ("reference", "model", "tag"),
    [
        ("hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M", "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF", "Q4_K_M"),
        ("https://hf.co/user/repo", "hf.co/user/repo", "latest"),
        ("someuser/coolmodel:7b", "someuser/coolmodel", "7b"),
    ],
)
def test_custom_model_is_added_by_reference(client, reference, model, tag):
    response = client.post("/custom-models/", data=json.dumps({"reference": reference}), content_type="application/json")

    assert response.status_code == 201
    body = response.json()
    assert body["model"] == model
    assert body["is_custom"] is True
    assert [v["parameters"] for v in body["versions"]] == [tag]


def test_custom_model_rejects_nonsense(client):
    response = client.post(
        "/custom-models/", data=json.dumps({"reference": "rm -rf /; echo"}), content_type="application/json"
    )

    assert response.status_code == 400


def test_custom_model_with_chats_is_not_deleted(client):
    ai_model = models.AIModel.objects.create(name="repo", model="hf.co/u/repo", is_custom=True)
    models.ChatHistory.objects.create(ai_model=ai_model)

    response = client.delete("/custom-models/", data=json.dumps({"id": ai_model.id}), content_type="application/json")

    assert response.status_code == 409
    assert models.AIModel.objects.filter(id=ai_model.id).exists()


def test_routes_accept_model_names_with_slashes(client):
    ai_model = models.AIModel.objects.create(name="repo", model="hf.co/u/repo", is_custom=True)
    chat = models.ChatHistory.objects.create(ai_model=ai_model)

    assert client.get("/all-chats/hf.co/u/repo").status_code == 200
    assert client.get(f"/chat-history/hf.co/u/repo/{chat.id}").status_code == 200
