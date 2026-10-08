import bisect
import logging
import threading

from django.db import transaction

from .. import models
from ..functions import ModelUnavailableError
from . import chunking, embeddings

logger = logging.getLogger(__name__)

INTERRUPTED_MESSAGE = "Processing was interrupted (the app restarted). Upload the file again."

# Ids with a live ingest thread in this process. Ingest runs in daemon
# threads that die with the server, and only the extracted text (never
# stored) could resume them — so a pending/processing row not in here is
# stuck for good and has to be told so, instead of spinning forever.
_active_ids: set[int] = set()
_active_lock = threading.Lock()


# Embedding a whole long document in one request gives no progress and can
# run into request timeouts; batches give both a heartbeat and a percentage.
EMBED_BATCH_SIZE = 32


def start_ingest(document_id: int, text: str, page_starts: list[int] | None = None) -> None:
    with _active_lock:
        _active_ids.add(document_id)

    threading.Thread(
        target=_run_ingest,
        args=(document_id, text, page_starts),
        daemon=True,
        name=f"ingest-document-{document_id}",
    ).start()


def _run_ingest(document_id: int, text: str, page_starts: list[int] | None) -> None:
    try:
        ingest_document(document_id, text, page_starts)
    finally:
        with _active_lock:
            _active_ids.discard(document_id)


def mark_interrupted_documents(collection_id: int | str) -> int:
    with _active_lock:
        active = set(_active_ids)

    return (
        models.Document.objects.filter(
            collection_id=collection_id,
            status__in=[models.Document.STATUS_PENDING, models.Document.STATUS_PROCESSING],
        )
        .exclude(id__in=active)
        .update(status=models.Document.STATUS_FAILED, error_message=INTERRUPTED_MESSAGE)
    )


def _page_for(char_start: int, page_starts: list[int] | None) -> int | None:
    if not page_starts:
        return None
    return bisect.bisect_right(page_starts, char_start)


def ingest_document(document_id: int, text: str, page_starts: list[int] | None = None) -> None:
    """Chunks, embeds, and saves a document's content. Runs in a background
    thread (see views.Documents.post) — real embedding calls are too slow to
    do inline in the upload request. Only the id and already-extracted text
    cross the thread boundary; extraction itself happens synchronously in
    the view, since it needs the request's in-memory uploaded file, which
    Django tears down once the response is sent. A brand-new thread lazily
    gets its own fresh connection on first use, so there's no stale
    connection to clean up here the way a long-lived worker might need to.
    """
    try:
        document = models.Document.objects.select_related(
            "collection", "collection__embedding_model"
        ).get(id=document_id)
    except models.Document.DoesNotExist:
        logger.warning("Document %s vanished before ingest started", document_id)
        return

    document.status = models.Document.STATUS_PROCESSING
    document.save(update_fields=["status"])

    try:
        chunks = chunking.chunk_text(text)
        if not chunks:
            raise ValueError("No extractable text found in this file.")

        client = embeddings.get_embeddings_client(
            document.collection.embedding_model.model, document.collection.embedding_parameters
        )
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[start:start + EMBED_BATCH_SIZE]
            vectors.extend(client.embed_documents([c.content for c in batch]))
            document.progress = int(100 * len(vectors) / len(chunks))
            document.save(update_fields=["progress"])

        with transaction.atomic():
            models.DocumentChunk.objects.bulk_create(
                [
                    models.DocumentChunk(
                        document=document,
                        chunk_index=i,
                        content=chunk.content,
                        char_start=chunk.char_start,
                        page=_page_for(chunk.char_start, page_starts),
                        embedding=vector,
                    )
                    for i, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
                ]
            )
            document.status = models.Document.STATUS_READY
            document.chunk_count = len(chunks)
            document.save(update_fields=["status", "chunk_count"])

    except ModelUnavailableError as e:
        logger.warning("Ingest failed for document %s: %s", document_id, e)
        document.status = models.Document.STATUS_FAILED
        document.error_message = str(e)
        document.save(update_fields=["status", "error_message"])
    except Exception as e:
        logger.exception("Ingest crashed for document %s", document_id)
        document.status = models.Document.STATUS_FAILED
        document.error_message = str(e)
        document.save(update_fields=["status", "error_message"])
