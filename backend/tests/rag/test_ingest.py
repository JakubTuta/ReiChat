import asyncio

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import models
from app.llm.runtime import ModelUnavailableError
from app.rag import ingest
from tests.factories import make_model

pytestmark = pytest.mark.usefixtures("db")


class FakeEmbeddingsClient:
    def __init__(self, dim=3, on_batch=None):
        self.dim = dim
        self.on_batch = on_batch
        self.batches: list[list[str]] = []

    async def aembed_documents(self, texts):
        self.batches.append(list(texts))
        if self.on_batch:
            await self.on_batch(len(self.batches))
        return [[float(i)] * self.dim for i in range(len(texts))]


@pytest.fixture
def embeddings(monkeypatch):
    client = FakeEmbeddingsClient()

    async def get_client(model, params):
        client.asked_for = (model, params)
        return client

    monkeypatch.setattr(ingest.embeddings, "get_embeddings_client", get_client)
    return client


@pytest.fixture
async def collection(db: AsyncSession) -> models.DocumentCollection:
    embedder = await make_model(db, "nomic-embed-text-ingest-test", is_embedding=True)
    collection = models.DocumentCollection(
        name="Docs", embedding_model_id=embedder.id, embedding_parameters="latest"
    )
    db.add(collection)
    await db.commit()
    return collection


@pytest.fixture
async def document(db: AsyncSession, collection) -> models.Document:
    document = models.Document(collection_id=collection.id, filename="notes.txt")
    db.add(document)
    await db.commit()
    return document


async def fresh(db: AsyncSession, document: models.Document) -> models.Document:
    return (
        await db.execute(
            select(models.Document)
            .where(models.Document.id == document.id)
            .execution_options(populate_existing=True)
        )
    ).scalar_one()


async def chunk_count(db: AsyncSession, document: models.Document) -> int:
    return (
        await db.execute(
            select(func.count())
            .select_from(models.DocumentChunk)
            .where(models.DocumentChunk.document_id == document.id)
        )
    ).scalar_one()


async def test_ingest_document_creates_chunks_and_marks_ready(db: AsyncSession, document, embeddings):
    await ingest.ingest_document(document.id, "word " * 500)

    stored = await fresh(db, document)
    assert stored.status == models.Document.STATUS_READY
    assert stored.chunk_count > 0
    assert await chunk_count(db, document) == stored.chunk_count


async def test_the_embedding_model_of_the_collection_is_the_one_asked_for(
    db: AsyncSession, document, embeddings
):
    await ingest.ingest_document(document.id, "some text")

    assert embeddings.asked_for == ("nomic-embed-text-ingest-test", "latest")


async def test_chunks_keep_their_text_position_and_vector(db: AsyncSession, document, embeddings):
    await ingest.ingest_document(document.id, "word " * 500)

    first = (
        await db.execute(
            select(models.DocumentChunk).where(
                models.DocumentChunk.chunk_index == 0, models.DocumentChunk.document_id == document.id
            )
        )
    ).scalar_one()
    assert first.char_start == 0
    assert first.content.startswith("word word")
    assert first.embedding == pytest.approx([0.0, 0.0, 0.0])
    assert first.page is None


async def test_ingest_document_empty_text_marks_failed(db: AsyncSession, document, embeddings):
    await ingest.ingest_document(document.id, "   ")

    stored = await fresh(db, document)
    assert stored.status == models.Document.STATUS_FAILED
    assert "No extractable text" in stored.error_message
    assert await chunk_count(db, document) == 0


async def test_ingest_document_embedding_unavailable_marks_failed(db: AsyncSession, document, monkeypatch):
    async def unavailable(model, params):
        raise ModelUnavailableError("container not running")

    monkeypatch.setattr(ingest.embeddings, "get_embeddings_client", unavailable)

    await ingest.ingest_document(document.id, "some real text content")

    stored = await fresh(db, document)
    assert stored.status == models.Document.STATUS_FAILED
    assert "container not running" in stored.error_message
    assert await chunk_count(db, document) == 0


async def test_an_unexpected_crash_marks_failed_with_the_reason_and_is_logged(
    db: AsyncSession, document, embeddings, monkeypatch, caplog
):
    async def explode(texts):
        raise RuntimeError("embedder exploded")

    monkeypatch.setattr(embeddings, "aembed_documents", explode)

    await ingest.ingest_document(document.id, "some real text content")

    stored = await fresh(db, document)
    assert (stored.status, stored.error_message) == (models.Document.STATUS_FAILED, "embedder exploded")
    assert "Ingest crashed" in caplog.text


async def test_a_failure_halfway_leaves_no_chunks_behind(db: AsyncSession, document, embeddings, monkeypatch):
    monkeypatch.setattr(ingest, "EMBED_BATCH_SIZE", 2)

    async def die_on_the_second_batch(batch_number):
        if batch_number == 2:
            raise RuntimeError("lost the embedder")

    embeddings.on_batch = die_on_the_second_batch

    await ingest.ingest_document(document.id, "word " * 800)

    assert (await fresh(db, document)).status == models.Document.STATUS_FAILED
    assert await chunk_count(db, document) == 0


async def test_progress_is_visible_while_the_document_is_still_processing(
    db: AsyncSession, document, embeddings, monkeypatch
):
    monkeypatch.setattr(ingest, "EMBED_BATCH_SIZE", 2)
    seen: list[tuple[str, int]] = []

    async def look_at_the_document(batch_number):
        row = await fresh(db, document)
        seen.append((row.status, row.progress))

    embeddings.on_batch = look_at_the_document

    await ingest.ingest_document(document.id, "word " * 800)

    assert len(seen) > 2
    assert {status for status, _ in seen} == {models.Document.STATUS_PROCESSING}
    progress = [p for _, p in seen]
    assert progress == sorted(progress)
    assert progress[0] < 100
    assert (await fresh(db, document)).progress == 100


async def test_ingest_document_missing_document_id_is_a_noop(embeddings, caplog):
    # The document may have been deleted before the background task got to run.
    await ingest.ingest_document(999999, "text")

    assert embeddings.batches == []
    assert "vanished before ingest started" in caplog.text


async def test_ingest_maps_chunks_to_pdf_pages_and_reports_progress(db: AsyncSession, document, embeddings):
    page_one = "alpha " * 300
    text = page_one + "\n\n" + "beta " * 300

    await ingest.ingest_document(document.id, text, page_starts=[0, len(page_one) + 2])

    pages = list(
        (
            await db.execute(
                select(models.DocumentChunk.page)
                .where(models.DocumentChunk.document_id == document.id)
                .order_by(models.DocumentChunk.chunk_index)
            )
        ).scalars()
    )
    assert pages[0] == 1
    assert pages[-1] == 2
    assert (await fresh(db, document)).progress == 100


@pytest.mark.parametrize(
    ("char_start", "page_starts", "expected"),
    [
        (0, None, None),
        (0, [], None),
        (0, [0, 100], 1),
        (99, [0, 100], 1),
        (100, [0, 100], 2),
        (5000, [0, 100], 2),
    ],
)
def test_a_chunk_belongs_to_the_page_it_starts_on(char_start, page_starts, expected):
    assert ingest._page_for(char_start, page_starts) == expected


async def test_mark_interrupted_fails_orphaned_documents_but_not_live_ones(
    db: AsyncSession, collection, monkeypatch
):
    docs = {}
    for name, status in (
        ("orphan", models.Document.STATUS_PROCESSING),
        ("pending", models.Document.STATUS_PENDING),
        ("live", models.Document.STATUS_PROCESSING),
        ("done", models.Document.STATUS_READY),
    ):
        docs[name] = models.Document(collection_id=collection.id, filename=f"{name}.txt", status=status)
        db.add(docs[name])
    await db.commit()
    monkeypatch.setattr(ingest, "_active_ids", {docs["live"].id})

    assert await ingest.mark_interrupted_documents(collection.id) == 2

    assert (await fresh(db, docs["orphan"])).status == models.Document.STATUS_FAILED
    assert (await fresh(db, docs["orphan"])).error_message == ingest.INTERRUPTED_MESSAGE
    assert (await fresh(db, docs["pending"])).status == models.Document.STATUS_FAILED
    assert (await fresh(db, docs["live"])).status == models.Document.STATUS_PROCESSING
    assert (await fresh(db, docs["done"])).status == models.Document.STATUS_READY


async def test_mark_interrupted_only_touches_the_given_collection(db: AsyncSession, collection):
    other = models.DocumentCollection(
        name="Other", embedding_model_id=collection.embedding_model_id, embedding_parameters="latest"
    )
    db.add(other)
    await db.commit()
    stuck = models.Document(
        collection_id=other.id, filename="x.txt", status=models.Document.STATUS_PROCESSING
    )
    db.add(stuck)
    await db.commit()

    assert await ingest.mark_interrupted_documents(collection.id) == 0
    assert (await fresh(db, stuck)).status == models.Document.STATUS_PROCESSING


async def test_start_ingest_tracks_the_document_until_the_task_finishes(monkeypatch):
    seen_during: list[bool] = []

    async def fake_ingest(doc_id, text, pages):
        seen_during.append(doc_id in ingest._active_ids)

    monkeypatch.setattr(ingest, "ingest_document", fake_ingest)

    ingest.start_ingest(4242, "text")
    assert 4242 in ingest._active_ids
    await asyncio.gather(*(t for t in asyncio.all_tasks() if t.get_name() == "ingest-document-4242"))

    assert seen_during == [True]
    assert 4242 not in ingest._active_ids


async def test_a_crashing_ingest_still_stops_being_tracked(monkeypatch):
    async def explode(doc_id, text, pages):
        raise RuntimeError("boom")

    monkeypatch.setattr(ingest, "ingest_document", explode)

    ingest.start_ingest(4243, "text")
    await asyncio.gather(
        *(t for t in asyncio.all_tasks() if t.get_name() == "ingest-document-4243"), return_exceptions=True
    )

    assert 4243 not in ingest._active_ids
