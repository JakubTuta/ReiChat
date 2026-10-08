from __future__ import annotations

import pytest

from django_app import models
from django_app.functions import ModelUnavailableError
from django_app.rag import ingest

pytestmark = pytest.mark.django_db


class _FakeEmbeddingsClient:
    def __init__(self, dim=3):
        self._dim = dim

    def embed_documents(self, texts):
        return [[float(i)] * self._dim for i in range(len(texts))]


def _stub_embeddings(monkeypatch, dim=3):
    monkeypatch.setattr(ingest.embeddings, "get_embeddings_client", lambda model, params: _FakeEmbeddingsClient(dim))


@pytest.fixture
def collection():
    ai_model = models.AIModel.objects.create(
        name="nomic", model="nomic-embed-text-ingest-test", is_embedding=True, index=1
    )
    return models.DocumentCollection.objects.create(
        name="Docs", embedding_model=ai_model, embedding_parameters="latest"
    )


@pytest.fixture
def document(collection):
    return models.Document.objects.create(collection=collection, filename="notes.txt")


def test_ingest_document_creates_chunks_and_marks_ready(monkeypatch, document):
    _stub_embeddings(monkeypatch)

    ingest.ingest_document(document.id, "word " * 500)

    document.refresh_from_db()
    assert document.status == models.Document.STATUS_READY
    assert document.chunk_count > 0
    assert document.chunks.count() == document.chunk_count


def test_ingest_document_empty_text_marks_failed(monkeypatch, document):
    _stub_embeddings(monkeypatch)

    ingest.ingest_document(document.id, "   ")

    document.refresh_from_db()
    assert document.status == models.Document.STATUS_FAILED
    assert "No extractable text" in document.error_message
    assert document.chunks.count() == 0


def test_ingest_document_embedding_unavailable_marks_failed(monkeypatch, document):
    def _raise(model, params):
        raise ModelUnavailableError("container not running")

    monkeypatch.setattr(ingest.embeddings, "get_embeddings_client", _raise)

    ingest.ingest_document(document.id, "some real text content")

    document.refresh_from_db()
    assert document.status == models.Document.STATUS_FAILED
    assert "container not running" in document.error_message
    assert document.chunks.count() == 0


def test_ingest_document_missing_document_id_is_a_noop():
    # Shouldn't raise — the document may have been deleted before the
    # background thread got a chance to run.
    ingest.ingest_document(999999, "text")


def test_mark_interrupted_fails_orphaned_documents_but_not_live_ones(collection, monkeypatch):
    orphan = models.Document.objects.create(
        collection=collection, filename="orphan.txt", status=models.Document.STATUS_PROCESSING
    )
    live = models.Document.objects.create(
        collection=collection, filename="live.txt", status=models.Document.STATUS_PROCESSING
    )
    done = models.Document.objects.create(
        collection=collection, filename="done.txt", status=models.Document.STATUS_READY
    )
    monkeypatch.setattr(ingest, "_active_ids", {live.id})

    assert ingest.mark_interrupted_documents(collection.id) == 1

    orphan.refresh_from_db()
    live.refresh_from_db()
    done.refresh_from_db()
    assert orphan.status == models.Document.STATUS_FAILED
    assert orphan.error_message == ingest.INTERRUPTED_MESSAGE
    assert live.status == models.Document.STATUS_PROCESSING
    assert done.status == models.Document.STATUS_READY


def test_start_ingest_tracks_the_document_until_the_thread_finishes(monkeypatch):
    seen_during: list[bool] = []
    monkeypatch.setattr(
        ingest, "ingest_document", lambda doc_id, text, pages: seen_during.append(doc_id in ingest._active_ids)
    )

    class _InlineThread:
        def __init__(self, target, args, **kwargs):
            self._run = lambda: target(*args)

        def start(self):
            self._run()

    monkeypatch.setattr(ingest.threading, "Thread", _InlineThread)

    ingest.start_ingest(4242, "text")

    assert seen_during == [True]
    assert 4242 not in ingest._active_ids


def test_ingest_maps_chunks_to_pdf_pages_and_reports_progress(monkeypatch, document):
    _stub_embeddings(monkeypatch)
    page_one = "alpha " * 300
    text = page_one + "\n\n" + "beta " * 300

    ingest.ingest_document(document.id, text, page_starts=[0, len(page_one) + 2])

    document.refresh_from_db()
    pages = list(document.chunks.order_by("chunk_index").values_list("page", flat=True))
    assert pages[0] == 1
    assert pages[-1] == 2
    assert document.progress == 100
