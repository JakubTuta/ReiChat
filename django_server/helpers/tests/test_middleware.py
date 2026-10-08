from __future__ import annotations

import json

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile

from django_app import models

pytestmark = pytest.mark.django_db


def test_cross_origin_form_post_is_rejected_before_it_reaches_the_view(client):
    response = client.post(
        "/mcp-servers/",
        data="name=evil&transport=stdio&command=sh+-c+id",
        content_type="application/x-www-form-urlencoded",
        HTTP_ORIGIN="https://evil.example",
    )

    assert response.status_code == 403
    assert not models.MCPServer.objects.filter(name="evil").exists()


def test_null_origin_is_rejected(client):
    response = client.post(
        "/personas/",
        data=json.dumps({"name": "x", "system_prompt": "y"}),
        content_type="application/json",
        HTTP_ORIGIN="null",
    )

    assert response.status_code == 403


def test_frontend_origin_is_allowed(client):
    response = client.post(
        "/personas/",
        data=json.dumps({"name": "Pirate", "system_prompt": "Talk like a pirate."}),
        content_type="application/json",
        HTTP_ORIGIN="http://localhost:3000",
    )

    assert response.status_code == 201


def test_request_without_origin_is_allowed_for_non_browser_clients(client):
    response = client.post(
        "/personas/",
        data=json.dumps({"name": "Pirate", "system_prompt": "Talk like a pirate."}),
        content_type="application/json",
    )

    assert response.status_code == 201


def test_safe_methods_ignore_origin(client):
    response = client.get("/personas/", HTTP_ORIGIN="https://evil.example")

    assert response.status_code == 200


def test_same_origin_form_encoded_body_is_unsupported(client):
    response = client.post(
        "/mcp-servers/",
        data="name=x&transport=stdio&command=echo",
        content_type="application/x-www-form-urlencoded",
        HTTP_ORIGIN="http://localhost:3000",
    )

    assert response.status_code == 415
    assert not models.MCPServer.objects.filter(name="x").exists()


def test_document_upload_still_accepts_multipart(client, monkeypatch):
    from django_app import views

    monkeypatch.setattr(views.rag_ingest, "start_ingest", lambda *args: None)
    models.AIModel.objects.filter(model="nomic-embed-text-mw-test").delete()
    embedder = models.AIModel.objects.create(
        name="nomic-embed-text-mw-test", model="nomic-embed-text-mw-test", is_embedding=True
    )
    collection = models.DocumentCollection.objects.create(
        name="Docs", embedding_model=embedder, embedding_parameters="latest"
    )
    upload = SimpleUploadedFile("notes.txt", b"hello world", content_type="text/plain")

    response = client.post(
        "/documents/",
        data={"collection_id": collection.id, "file": upload},
        HTTP_ORIGIN="http://localhost:3000",
    )

    assert response.status_code == 202
