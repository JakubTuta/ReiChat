import asyncio
import bisect
import logging
from typing import Any

from sqlalchemy import insert, select, update

from app.core import tasks
from app.db import models
from app.db.session import SessionLocal
from app.llm.runtime import ModelUnavailableError
from app.rag import chunking, embeddings

logger = logging.getLogger(__name__)

INTERRUPTED_MESSAGE = "Processing was interrupted (the app restarted). Upload the file again."

# Ids with a live ingest task in this process. Ingest runs as a background
# task that dies with the server, and only the extracted text (never stored)
# could resume it — so a pending/processing row not in here is stuck for good
# and has to be told so, instead of spinning forever.
_active_ids: set[int] = set()


# Embedding a whole long document in one request gives no progress and can
# run into request timeouts; batches give both a heartbeat and a percentage.
EMBED_BATCH_SIZE = 32


def start_ingest(document_id: int, text: str, page_starts: list[int] | None = None) -> None:
    _active_ids.add(document_id)
    tasks.spawn(_run_ingest(document_id, text, page_starts), name=f"ingest-document-{document_id}")


async def _run_ingest(document_id: int, text: str, page_starts: list[int] | None) -> None:
    try:
        await ingest_document(document_id, text, page_starts)
    finally:
        _active_ids.discard(document_id)


async def mark_interrupted_documents(collection_id: int) -> int:
    """Fails the pending/processing documents nothing is working on any more.
    Committed here: the listing that asked for it shows the result at once."""
    async with SessionLocal() as session:
        result = await session.execute(
            update(models.Document)
            .where(
                models.Document.collection_id == collection_id,
                models.Document.status.in_(
                    [models.Document.STATUS_PENDING, models.Document.STATUS_PROCESSING]
                ),
                models.Document.id.not_in(list(_active_ids)),
            )
            .values(status=models.Document.STATUS_FAILED, error_message=INTERRUPTED_MESSAGE)
            .returning(models.Document.id)
        )
        interrupted = len(result.all())
        await session.commit()
    return interrupted


def _page_for(char_start: int, page_starts: list[int] | None) -> int | None:
    if not page_starts:
        return None
    return bisect.bisect_right(page_starts, char_start)


async def _update_document(document_id: int, **values: Any) -> None:
    async with SessionLocal() as session:
        await session.execute(
            update(models.Document).where(models.Document.id == document_id).values(**values)
        )
        await session.commit()


async def ingest_document(document_id: int, text: str, page_starts: list[int] | None = None) -> None:
    """Chunks, embeds, and saves a document's content. Runs as a background
    task (see rag/router.py) — real embedding calls are too slow to do inline
    in the upload request. Only the id and already-extracted text are passed
    in; the file itself is gone once the upload request ends.

    Every step commits on its own (status, then each batch's progress), so the
    document list shows a live percentage instead of one result at the end.
    """
    async with SessionLocal() as session:
        row = (
            await session.execute(
                select(models.AIModel.model, models.DocumentCollection.embedding_parameters)
                .select_from(models.Document)
                .join(
                    models.DocumentCollection, models.DocumentCollection.id == models.Document.collection_id
                )
                .join(models.AIModel, models.AIModel.id == models.DocumentCollection.embedding_model_id)
                .where(models.Document.id == document_id)
            )
        ).one_or_none()
    if row is None:
        logger.warning("Document %s vanished before ingest started", document_id)
        return
    model_name, parameters = row

    await _update_document(document_id, status=models.Document.STATUS_PROCESSING)

    try:
        # Splitting a long text is CPU work.
        chunks = await asyncio.to_thread(chunking.chunk_text, text)
        if not chunks:
            raise ValueError("No extractable text found in this file.")

        client = await embeddings.get_embeddings_client(model_name, parameters)
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[start : start + EMBED_BATCH_SIZE]
            vectors.extend(await client.aembed_documents([c.content for c in batch]))
            await _update_document(document_id, progress=int(100 * len(vectors) / len(chunks)))

        # One transaction: a document is never "ready" with only some of its chunks.
        async with SessionLocal() as session, session.begin():
            await session.execute(
                insert(models.DocumentChunk),
                [
                    {
                        "document_id": document_id,
                        "chunk_index": i,
                        "content": chunk.content,
                        "char_start": chunk.char_start,
                        "page": _page_for(chunk.char_start, page_starts),
                        "embedding": vector,
                    }
                    for i, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
                ],
            )
            await session.execute(
                update(models.Document)
                .where(models.Document.id == document_id)
                .values(status=models.Document.STATUS_READY, chunk_count=len(chunks))
            )

    except ModelUnavailableError as e:
        logger.warning("Ingest failed for document %s: %s", document_id, e)
        await _update_document(document_id, status=models.Document.STATUS_FAILED, error_message=str(e))
    except Exception as e:
        logger.exception("Ingest crashed for document %s", document_id)
        await _update_document(document_id, status=models.Document.STATUS_FAILED, error_message=str(e))
