import logging
import typing

from django.contrib.postgres.search import SearchQuery, SearchRank, SearchVector
from pgvector.django import CosineDistance

from .. import models
from ..functions import ModelUnavailableError
from . import embeddings

logger = logging.getLogger(__name__)

# Candidates taken from each search (vector and keyword) per embedding model,
# before fusion picks the final MAX_CONTEXT_CHUNKS.
CANDIDATES_PER_SEARCH = 8
MAX_CONTEXT_CHUNKS = 6
# Reciprocal rank fusion's damping constant — the usual 60 from the RRF
# paper; it keeps a single first place from drowning out agreement between
# the two searches further down.
RRF_K = 60
# A vector-only hit this much further away than the best one is noise
# riding along, not a second relevant passage.
RELATIVE_DISTANCE_CUTOFF = 0.25
# "simple" = no stemming or stop words for any particular language, so
# keyword matching behaves the same for every document language.
TEXT_SEARCH_CONFIG = "simple"
# Shorter than this, a message is probably a follow-up ("and the second
# one?") that only makes sense together with the previous question.
FOLLOW_UP_MAX_CHARS = 80

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
    page: int | None


class RetrievalResult(typing.NamedTuple):
    context: str
    citations: list[Citation]
    # Plain-language reasons some attached documents weren't searched —
    # shown to the user, because an answer given without the documents
    # they attached looks exactly like one given with them.
    warnings: list[str]


def build_search_query(question: str, previous_question: str | None) -> str:
    if previous_question and len(question) < FOLLOW_UP_MAX_CHARS:
        return f"{previous_question}\n{question}"
    return question


def _fuse(
    vector_hits: list[tuple[models.DocumentChunk, float]],
    keyword_hits: list[models.DocumentChunk],
) -> list[tuple[float, models.DocumentChunk]]:
    """Reciprocal rank fusion of the two result lists. Vector search finds
    passages that mean the same thing in other words; keyword search finds
    exact names, numbers and codes that embeddings blur — a passage both
    agree on ranks highest."""
    scores: dict[int, float] = {}
    chunks: dict[int, models.DocumentChunk] = {}

    if vector_hits:
        best_distance = vector_hits[0][1]
        keyword_ids = {c.id for c in keyword_hits}
        for rank, (chunk, distance) in enumerate(vector_hits):
            if distance - best_distance > RELATIVE_DISTANCE_CUTOFF and chunk.id not in keyword_ids:
                continue
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1 / (RRF_K + rank + 1)
            chunks[chunk.id] = chunk

    for rank, chunk in enumerate(keyword_hits):
        scores[chunk.id] = scores.get(chunk.id, 0.0) + 1 / (RRF_K + rank + 1)
        chunks[chunk.id] = chunk

    return sorted(((score, chunks[cid]) for cid, score in scores.items()), key=lambda pair: -pair[0])


def build_rag_context(
    collections: list[models.DocumentCollection], query: str
) -> RetrievalResult:
    """Per distinct embedding model among `collections`: an exact (not ANN —
    see DocumentChunk.embedding) cosine-nearest-neighbor search plus a
    Postgres full-text search over the same chunks, fused by rank. Results
    become one context block plus a citation list for the WS "done"
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

    fused: list[tuple[float, models.DocumentChunk]] = []

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

        ready_chunks = models.DocumentChunk.objects.filter(
            document__collection__in=cols, document__status=models.Document.STATUS_READY
        ).select_related("document")

        vector_hits = [
            (chunk, chunk.distance)
            for chunk in ready_chunks.annotate(distance=CosineDistance("embedding", query_vector))
            .order_by("distance")[:CANDIDATES_PER_SEARCH]
        ]
        search_query = SearchQuery(query, config=TEXT_SEARCH_CONFIG, search_type="websearch")
        keyword_hits = list(
            ready_chunks.annotate(
                rank=SearchRank(SearchVector("content", config=TEXT_SEARCH_CONFIG), search_query)
            )
            .filter(rank__gt=0)
            .order_by("-rank")[:CANDIDATES_PER_SEARCH]
        )
        fused.extend(_fuse(vector_hits, keyword_hits))

    still_processing = models.Document.objects.filter(
        collection__in=collections,
        status__in=[models.Document.STATUS_PENDING, models.Document.STATUS_PROCESSING],
    ).count()
    if still_processing:
        warnings.append(
            f"{still_processing} document{'s are' if still_processing != 1 else ' is'} still being "
            "processed and couldn't be searched yet."
        )

    if not fused:
        return RetrievalResult("", [], warnings)

    fused.sort(key=lambda pair: -pair[0])
    top_chunks = [chunk for _, chunk in fused[:MAX_CONTEXT_CHUNKS]]

    context_lines = [_CONTEXT_PREAMBLE]
    citations: list[Citation] = []
    for chunk in top_chunks:
        label = f"{chunk.document.filename}, page {chunk.page}" if chunk.page else chunk.document.filename
        context_lines.append(f"\n[{label}]\n{chunk.content}")
        citations.append({
            "document_id": chunk.document_id,
            "filename": chunk.document.filename,
            "chunk_index": chunk.chunk_index,
            "page": chunk.page,
        })

    return RetrievalResult("\n".join(context_lines), citations, warnings)


def _collection_names(collections: list[models.DocumentCollection]) -> str:
    quoted = ", ".join(f'"{c.name}"' for c in collections)
    return f"Your documents in {quoted}"


def retrieve_context_for_chat(
    chat_history: models.ChatHistory, query: str, previous_question: str | None = None
) -> RetrievalResult:
    """Entry point for consumers.py — fetching the chat's active collections
    and running the search are both blocking (DB + Docker SDK + HTTP), so
    the caller wraps this whole function in one sync_to_async call rather
    than awaiting each piece separately.
    """
    collections = list(chat_history.active_collections.select_related("embedding_model").all())
    return build_rag_context(collections, build_search_query(query, previous_question))
