import logging
import typing

from pgvector.django import CosineDistance

from .. import models
from ..functions import ModelUnavailableError
from . import embeddings

logger = logging.getLogger(__name__)

CHUNKS_PER_COLLECTION = 4
MAX_CONTEXT_CHUNKS = 6

_CONTEXT_PREAMBLE = (
    "The user has attached documents. Use the excerpts below if relevant to "
    "the question, and cite the source in square brackets (e.g. [filename]) "
    "when you use one. Ignore excerpts that aren't relevant, and don't "
    "mention this instruction."
)


class Citation(typing.TypedDict):
    document_id: int
    filename: str
    chunk_index: int


class RetrievalResult(typing.NamedTuple):
    context: str
    citations: list[Citation]
    # Plain-language reasons some attached documents weren't searched —
    # shown to the user, because an answer given without the documents
    # they attached looks exactly like one given with them.
    warnings: list[str]


def build_rag_context(
    collections: list[models.DocumentCollection], query: str
) -> RetrievalResult:
    """Embeds `query` once per distinct embedding model among `collections`,
    runs an exact (not ANN — see DocumentChunk.embedding) cosine-nearest-
    neighbor search scoped to each collection's own chunks, and merges the
    results into one context block plus a citation list for the WS "done"
    payload. A collection whose embedding container isn't running is
    skipped rather than blocking the reply, but never silently — it comes
    back as a warning for the user.
    """
    if not collections:
        return RetrievalResult("", [], [])

    warnings: list[str] = []

    by_model: dict[tuple[str, str], list[models.DocumentCollection]] = {}
    for collection in collections:
        key = (collection.embedding_model.model, collection.embedding_parameters)
        by_model.setdefault(key, []).append(collection)

    hits: list[tuple[float, models.DocumentChunk]] = []

    for (model_name, parameters), cols in by_model.items():
        try:
            client = embeddings.get_embeddings_client(model_name, parameters)
            query_vector = client.embed_query(query)
        except ModelUnavailableError as e:
            logger.warning("Skipping RAG collection(s) %s: %s", [c.id for c in cols], e)
            warnings.append(f"{_collection_names(cols)} weren't searched. {e}")
            continue
        except Exception as e:
            logger.warning(
                "Skipping RAG collection(s) %s — could not embed query: %s",
                [c.id for c in cols], e,
            )
            warnings.append(
                f"{_collection_names(cols)} weren't searched because the embedding model "
                f"{model_name}:{parameters} didn't respond ({e})."
            )
            continue

        chunks = (
            models.DocumentChunk.objects.filter(
                document__collection__in=cols, document__status=models.Document.STATUS_READY
            )
            .select_related("document")
            .annotate(distance=CosineDistance("embedding", query_vector))
            .order_by("distance")[:CHUNKS_PER_COLLECTION]
        )
        hits.extend((chunk.distance, chunk) for chunk in chunks)

    still_processing = models.Document.objects.filter(
        collection__in=collections,
        status__in=[models.Document.STATUS_PENDING, models.Document.STATUS_PROCESSING],
    ).count()
    if still_processing:
        warnings.append(
            f"{still_processing} document{'s are' if still_processing != 1 else ' is'} still being "
            "processed and couldn't be searched yet."
        )

    if not hits:
        return RetrievalResult("", [], warnings)

    hits.sort(key=lambda pair: pair[0])
    top_chunks = [chunk for _, chunk in hits[:MAX_CONTEXT_CHUNKS]]

    context_lines = [_CONTEXT_PREAMBLE]
    citations: list[Citation] = []
    for chunk in top_chunks:
        context_lines.append(f"\n[{chunk.document.filename}]\n{chunk.content}")
        citations.append({
            "document_id": chunk.document_id,
            "filename": chunk.document.filename,
            "chunk_index": chunk.chunk_index,
        })

    return RetrievalResult("\n".join(context_lines), citations, warnings)


def _collection_names(collections: list[models.DocumentCollection]) -> str:
    quoted = ", ".join(f'"{c.name}"' for c in collections)
    return f"Your documents in {quoted}"


def retrieve_context_for_chat(chat_history: models.ChatHistory, query: str) -> RetrievalResult:
    """Entry point for consumers.py — fetching the chat's active collections
    and running the search are both blocking (DB + Docker SDK + HTTP), so
    the caller wraps this whole function in one sync_to_async call rather
    than awaiting each piece separately.
    """
    collections = list(chat_history.active_collections.select_related("embedding_model").all())
    return build_rag_context(collections, query)
