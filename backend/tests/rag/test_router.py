import asyncio
import io

import docx
import httpx
import pytest
from pypdf import PdfWriter
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import models
from app.rag import ingest
from tests.factories import make_chat, make_model

pytestmark = pytest.mark.usefixtures("db")

FRONTEND = {"Origin": "http://localhost:3000"}


@pytest.fixture
async def embedding_model(db: AsyncSession) -> models.AIModel:
    return await make_model(db, "nomic-embed-text-views-test", name="Nomic Embed", is_embedding=True)


@pytest.fixture
async def collection(db: AsyncSession, embedding_model) -> models.DocumentCollection:
    return await add_collection(db, embedding_model)


async def add_collection(
    db: AsyncSession, embedding_model, name: str = "Docs", **fields
) -> models.DocumentCollection:
    collection = models.DocumentCollection(
        name=name, embedding_model_id=embedding_model.id, embedding_parameters="latest", **fields
    )
    db.add(collection)
    await db.commit()
    return collection


async def count(db: AsyncSession, table) -> int:
    return (await db.execute(select(func.count()).select_from(table))).scalar_one()


@pytest.fixture
def background_ingest(monkeypatch):
    """Replaces the real ingest with one that records what it was asked to do,
    so an upload test doesn't need a running embedding model."""
    calls = []

    def start_ingest(document_id, text, page_starts=None):
        calls.append((document_id, text, page_starts))

    monkeypatch.setattr(ingest, "start_ingest", start_ingest)
    return calls


# Collections


async def test_post_collection_creates_it(client: httpx.AsyncClient, embedding_model, db: AsyncSession):
    response = await client.post(
        "/collections/",
        json={"name": "My Docs", "embedding_model": embedding_model.id, "embedding_parameters": "latest"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "My Docs"
    assert body["embedding_model"] == embedding_model.id
    assert body["embedding_model_name"] == "nomic-embed-text-views-test"
    assert body["embedding_parameters"] == "latest"
    assert body["document_count"] == 0
    assert body["chat"] is None
    assert set(body) == {
        "id",
        "name",
        "chat",
        "embedding_model",
        "embedding_model_name",
        "embedding_parameters",
        "document_count",
        "created_at",
    }
    assert await count(db, models.DocumentCollection) == 1


async def test_post_collection_can_belong_to_a_chat(
    client: httpx.AsyncClient, embedding_model, db: AsyncSession
):
    chat = await make_chat(db)

    response = await client.post(
        "/collections/",
        json={
            "name": "For one chat",
            "embedding_model": embedding_model.id,
            "embedding_parameters": "latest",
            "chat": chat.id,
        },
    )

    assert response.status_code == 201
    assert response.json()["chat"] == chat.id


async def test_post_collection_for_an_unknown_chat_is_a_404(
    client: httpx.AsyncClient, embedding_model, db: AsyncSession
):
    response = await client.post(
        "/collections/",
        json={
            "name": "x",
            "embedding_model": embedding_model.id,
            "embedding_parameters": "latest",
            "chat": 999999,
        },
    )

    assert response.status_code == 404
    assert await count(db, models.DocumentCollection) == 0


async def test_post_collection_rejects_non_embedding_model(client: httpx.AsyncClient, db: AsyncSession):
    chat_model = await make_model(db, "llama3.1-rag-test")

    response = await client.post(
        "/collections/",
        json={"name": "My Docs", "embedding_model": chat_model.id, "embedding_parameters": "8b"},
    )

    assert response.status_code == 400
    assert response.json() == {"error": "That model isn't an embedding model"}


async def test_post_collection_unknown_embedding_model_returns_404(client: httpx.AsyncClient):
    response = await client.post(
        "/collections/", json={"name": "My Docs", "embedding_model": 999999, "embedding_parameters": "latest"}
    )

    assert response.status_code == 404
    assert response.json() == {"error": "Embedding model not found"}


@pytest.mark.parametrize(
    "body",
    [
        {"embedding_model": 1, "embedding_parameters": "latest"},
        {"name": "x", "embedding_parameters": "latest"},
        {"name": "x", "embedding_model": 1},
    ],
)
async def test_post_collection_needs_name_model_and_parameters(client: httpx.AsyncClient, body):
    response = await client.post("/collections/", json=body)

    assert response.status_code == 400
    assert response.json()["error"] == "Missing required body parameters"


@pytest.mark.parametrize("field", ["name", "embedding_parameters"])
async def test_post_collection_refuses_blank_text(client: httpx.AsyncClient, embedding_model, field):
    body = {
        "name": "Docs",
        "embedding_model": embedding_model.id,
        "embedding_parameters": "latest",
        field: "  ",
    }

    assert (await client.post("/collections/", json=body)).status_code == 400


async def test_get_collections_includes_document_count(
    client: httpx.AsyncClient, collection, db: AsyncSession
):
    db.add_all(
        [
            models.Document(collection_id=collection.id, filename="a.txt"),
            models.Document(collection_id=collection.id, filename="b.txt"),
        ]
    )
    await db.commit()

    response = await client.get("/collections/")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["document_count"] == 2
    assert body[0]["embedding_model_name"] == "nomic-embed-text-views-test"


async def test_a_collection_without_documents_has_a_count_of_zero(client: httpx.AsyncClient, collection):
    assert (await client.get("/collections/")).json()[0]["document_count"] == 0


async def test_newest_collections_are_listed_first(
    client: httpx.AsyncClient, embedding_model, db: AsyncSession
):
    await add_collection(db, embedding_model, "Older")
    await add_collection(db, embedding_model, "Newer")

    assert [c["name"] for c in (await client.get("/collections/")).json()] == ["Newer", "Older"]


async def test_get_collections_filtered_by_chat_id_includes_global_ones(
    client: httpx.AsyncClient, embedding_model, db: AsyncSession
):
    chat = await make_chat(db, title="t")
    other_chat = await make_chat(db, title="other")
    await add_collection(db, embedding_model, "Global")
    await add_collection(db, embedding_model, "Own", chat_id=chat.id)
    await add_collection(db, embedding_model, "Someone else's", chat_id=other_chat.id)

    response = await client.get(f"/collections/?chat_id={chat.id}")

    assert {c["name"] for c in response.json()} == {"Global", "Own"}


async def test_without_a_chat_filter_every_collection_is_listed(
    client: httpx.AsyncClient, embedding_model, db: AsyncSession
):
    chat = await make_chat(db)
    await add_collection(db, embedding_model, "Global")
    await add_collection(db, embedding_model, "Own", chat_id=chat.id)

    assert {c["name"] for c in (await client.get("/collections/")).json()} == {"Global", "Own"}


async def test_delete_collection_removes_it_and_cascades_documents(
    client: httpx.AsyncClient, collection, db: AsyncSession
):
    document = models.Document(collection_id=collection.id, filename="a.txt")
    db.add(document)
    await db.flush()
    db.add(
        models.DocumentChunk(
            document_id=document.id, chunk_index=0, content="x", char_start=0, embedding=[1.0]
        )
    )
    await db.commit()

    response = await client.request("DELETE", "/collections/", json={"id": collection.id})

    assert response.status_code == 200
    assert response.json() == {"status": "Collection deleted"}
    assert await count(db, models.DocumentCollection) == 0
    assert await count(db, models.Document) == 0
    assert await count(db, models.DocumentChunk) == 0


async def test_deleting_a_collection_keeps_the_chats_that_used_it(
    client: httpx.AsyncClient, collection, db: AsyncSession
):
    chat = await make_chat(db)
    await db.execute(
        models.chat_history_active_collection.insert().values(
            chat_history_id=chat.id, document_collection_id=collection.id
        )
    )
    await db.commit()

    await client.request("DELETE", "/collections/", json={"id": collection.id})

    assert await count(db, models.ChatHistory) == 1
    assert await count(db, models.chat_history_active_collection) == 0


async def test_delete_collection_not_found_returns_404(client: httpx.AsyncClient):
    response = await client.request("DELETE", "/collections/", json={"id": 999999})

    assert response.status_code == 404
    assert response.json() == {"error": "Collection not found"}


# Documents


def upload(collection_id, name="notes.txt", content=b"Some real content to chunk and embed. " * 20):
    return {"data": {"collection_id": str(collection_id)}, "files": {"file": (name, content, "text/plain")}}


async def test_post_document_uploads_extracts_and_ingests(
    client: httpx.AsyncClient, collection, db: AsyncSession, background_ingest
):
    response = await client.post("/documents/", **upload(collection.id), headers=FRONTEND)

    assert response.status_code == 202
    body = response.json()
    assert body["filename"] == "notes.txt"
    assert body["collection"] == collection.id
    assert (body["status"], body["chunk_count"], body["progress"], body["error_message"]) == (
        "pending",
        0,
        0,
        "",
    )
    assert set(body) == {
        "id",
        "collection",
        "filename",
        "status",
        "error_message",
        "chunk_count",
        "progress",
        "uploaded_at",
    }
    [(document_id, text, page_starts)] = background_ingest
    assert document_id == body["id"]
    assert text.startswith("Some real content")
    assert page_starts is None
    assert await count(db, models.Document) == 1


async def test_a_real_upload_ends_up_chunked_and_ready(
    client: httpx.AsyncClient, collection, db: AsyncSession, monkeypatch
):
    class Embedder:
        async def aembed_documents(self, texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

    async def get_client(model, params):
        return Embedder()

    monkeypatch.setattr(ingest.embeddings, "get_embeddings_client", get_client)

    body = (await client.post("/documents/", **upload(collection.id))).json()
    await asyncio.gather(*(t for t in asyncio.all_tasks() if t.get_name().startswith("ingest-document")))

    listed = (await client.get(f"/documents/?collection_id={collection.id}")).json()
    assert [(d["id"], d["status"]) for d in listed] == [(body["id"], "ready")]
    assert listed[0]["chunk_count"] > 0


async def test_a_path_in_the_filename_is_reduced_to_the_name(
    client: httpx.AsyncClient, collection, background_ingest
):
    response = await client.post("/documents/", **upload(collection.id, name="C:\\Users\\me\\notes.txt"))

    assert response.json()["filename"] == "notes.txt"
    traversal = await client.post("/documents/", **upload(collection.id, name="../../etc/notes.md"))
    assert traversal.json()["filename"] == "notes.md"


async def test_a_pdf_is_read_with_its_pages(
    client: httpx.AsyncClient, collection, background_ingest, monkeypatch
):
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    monkeypatch.setattr(
        "app.rag.router.extract.extract_document",
        lambda name, raw: ("page one text\n\npage two text", [0, 15]),
    )

    response = await client.post("/documents/", **upload(collection.id, "manual.pdf", buffer.getvalue()))

    assert response.status_code == 202
    assert background_ingest[0][2] == [0, 15]


async def test_a_word_document_is_read(client: httpx.AsyncClient, collection, background_ingest):
    document = docx.Document()
    document.add_paragraph("A paragraph in a Word file.")
    buffer = io.BytesIO()
    document.save(buffer)

    response = await client.post("/documents/", **upload(collection.id, "letter.docx", buffer.getvalue()))

    assert response.status_code == 202
    assert "A paragraph in a Word file." in background_ingest[0][1]


async def test_post_document_unsupported_file_type_returns_400(
    client: httpx.AsyncClient, collection, db: AsyncSession
):
    response = await client.post("/documents/", **upload(collection.id, "archive.zip", b"whatever"))

    assert response.status_code == 400
    assert "Unsupported file type: .zip" in response.json()["error"]
    assert await count(db, models.Document) == 0


async def test_a_damaged_file_is_explained_not_a_server_error(
    client: httpx.AsyncClient, collection, db: AsyncSession
):
    response = await client.post("/documents/", **upload(collection.id, "broken.pdf", b"this is not a pdf"))

    assert response.status_code == 400
    assert "damaged" in response.json()["error"]
    assert await count(db, models.Document) == 0


async def test_a_file_without_text_is_refused(client: httpx.AsyncClient, collection, db: AsyncSession):
    response = await client.post("/documents/", **upload(collection.id, "empty.txt", b"   \n  "))

    assert response.status_code == 400
    assert response.json() == {"error": "No extractable text found in this file."}
    assert await count(db, models.Document) == 0


async def test_post_document_unknown_collection_returns_404(client: httpx.AsyncClient):
    response = await client.post("/documents/", **upload(999999, content=b"content"))

    assert response.status_code == 404
    assert response.json() == {"error": "Collection not found"}


async def test_post_document_needs_a_collection_and_a_file(client: httpx.AsyncClient):
    response = await client.post("/documents/", data={})

    assert response.status_code == 400
    assert response.json() == {
        "error": "Missing required body parameters",
        "missing_params": ["collection_id", "file"],
    }


async def test_post_document_with_only_a_collection_says_the_file_is_missing(
    client: httpx.AsyncClient, collection
):
    response = await client.post("/documents/", data={"collection_id": str(collection.id)})

    assert response.status_code == 400
    assert response.json()["missing_params"] == ["file"]


async def test_post_document_from_another_site_is_refused(
    client: httpx.AsyncClient, collection, db: AsyncSession
):
    response = await client.post(
        "/documents/", **upload(collection.id), headers={"Origin": "https://evil.example"}
    )

    assert response.status_code == 403
    assert await count(db, models.Document) == 0


async def test_get_documents_requires_collection_id(client: httpx.AsyncClient):
    response = await client.get("/documents/")

    assert response.status_code == 400
    assert response.json() == {
        "error": "Missing required query parameters",
        "missing_params": ["collection_id"],
    }


async def test_get_documents_lists_for_collection_newest_first(
    client: httpx.AsyncClient, collection, db: AsyncSession
):
    db.add(models.Document(collection_id=collection.id, filename="old.txt"))
    await db.commit()
    db.add(models.Document(collection_id=collection.id, filename="new.txt"))
    await db.commit()

    response = await client.get(f"/documents/?collection_id={collection.id}")

    assert response.status_code == 200
    assert [d["filename"] for d in response.json()] == ["new.txt", "old.txt"]


async def test_get_documents_only_lists_that_collections_documents(
    client: httpx.AsyncClient, collection, embedding_model, db: AsyncSession
):
    other = await add_collection(db, embedding_model, "Other")
    db.add_all(
        [
            models.Document(collection_id=collection.id, filename="mine.txt"),
            models.Document(collection_id=other.id, filename="theirs.txt"),
        ]
    )
    await db.commit()

    body = (await client.get(f"/documents/?collection_id={collection.id}")).json()

    assert [d["filename"] for d in body] == ["mine.txt"]


async def test_listing_documents_fails_the_ones_a_restart_orphaned(
    client: httpx.AsyncClient, collection, db: AsyncSession
):
    db.add(
        models.Document(
            collection_id=collection.id, filename="a.txt", status=models.Document.STATUS_PROCESSING
        )
    )
    await db.commit()

    [listed] = (await client.get(f"/documents/?collection_id={collection.id}")).json()

    assert listed["status"] == "failed"
    assert listed["error_message"] == ingest.INTERRUPTED_MESSAGE


async def test_listing_documents_leaves_a_live_ingest_alone(
    client: httpx.AsyncClient, collection, db: AsyncSession, monkeypatch
):
    document = models.Document(
        collection_id=collection.id, filename="a.txt", status=models.Document.STATUS_PROCESSING
    )
    db.add(document)
    await db.commit()
    monkeypatch.setattr(ingest, "_active_ids", {document.id})

    [listed] = (await client.get(f"/documents/?collection_id={collection.id}")).json()

    assert listed["status"] == "processing"


async def test_get_documents_for_a_non_numeric_collection_is_a_400(client: httpx.AsyncClient):
    assert (await client.get("/documents/?collection_id=abc")).status_code == 400


async def test_delete_document_removes_it_and_its_chunks(
    client: httpx.AsyncClient, collection, db: AsyncSession
):
    document = models.Document(collection_id=collection.id, filename="a.txt")
    db.add(document)
    await db.flush()
    db.add(
        models.DocumentChunk(
            document_id=document.id, chunk_index=0, content="x", char_start=0, embedding=[1.0]
        )
    )
    await db.commit()

    response = await client.request("DELETE", "/documents/", json={"id": document.id})

    assert response.status_code == 200
    assert response.json() == {"status": "Document deleted"}
    assert await count(db, models.Document) == 0
    assert await count(db, models.DocumentChunk) == 0
    assert await count(db, models.DocumentCollection) == 1


async def test_delete_document_not_found_returns_404(client: httpx.AsyncClient):
    response = await client.request("DELETE", "/documents/", json={"id": 999999})

    assert response.status_code == 404
    assert response.json() == {"error": "Document not found"}


# The passage behind a citation


async def test_document_chunk_returns_the_cited_passage(
    client: httpx.AsyncClient, collection, db: AsyncSession
):
    document = models.Document(collection_id=collection.id, filename="manual.pdf", status="ready")
    db.add(document)
    await db.flush()
    db.add(
        models.DocumentChunk(
            document_id=document.id,
            chunk_index=3,
            content="Hold reset for 10 seconds.",
            char_start=0,
            page=4,
            embedding=[1.0, 0.0],
        )
    )
    await db.commit()

    response = await client.get(f"/documents/chunk/?document_id={document.id}&chunk_index=3")

    assert response.status_code == 200
    assert response.json() == {"filename": "manual.pdf", "page": 4, "content": "Hold reset for 10 seconds."}


async def test_document_chunk_missing_says_so(client: httpx.AsyncClient):
    response = await client.get("/documents/chunk/?document_id=999999&chunk_index=0")

    assert response.status_code == 404
    assert "no longer exists" in response.json()["error"]


async def test_document_chunk_needs_both_ids(client: httpx.AsyncClient):
    response = await client.get("/documents/chunk/")

    assert response.status_code == 400
    assert response.json()["missing_params"] == ["document_id", "chunk_index"]
