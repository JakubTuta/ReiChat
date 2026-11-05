import logging
import typing

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql.ext import to_tsvector, websearch_to_tsquery
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import contains_eager, joinedload

from app.db import models
from app.llm.runtime import ModelUnavailableError
from app.rag import embeddings

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


async def _search_ready_chunks(
    session: AsyncSession, collection_ids: list[int], query: str, query_vector: list[float]
) -> tuple[list[tuple[models.DocumentChunk, float]], list[models.DocumentChunk]]:
    """Exact (not ANN — see DocumentChunk.embedding) cosine-nearest-neighbor
    search plus a Postgres full-text search over the same chunks."""
    in_scope = (
        models.Document.collection_id.in_(collection_ids),
        models.Document.status == models.Document.STATUS_READY,
    )

    distance = models.DocumentChunk.embedding.cosine_distance(query_vector).label("distance")
    vector_rows = await session.execute(
        select(models.DocumentChunk, distance)
        .join(models.Document, models.Document.id == models.DocumentChunk.document_id)
        .options(contains_eager(models.DocumentChunk.document))
        .where(*in_scope)
        .order_by(distance)
        .limit(CANDIDATES_PER_SEARCH)
    )
    vector_hits = [(chunk, float(chunk_distance)) for chunk, chunk_distance in vector_rows]

    # SQLAlchemy leaves these two constructors unannotated; they add the
    # ::regconfig cast asyncpg needs for the config name.
    document_vector = to_tsvector(TEXT_SEARCH_CONFIG, models.DocumentChunk.content)  # type: ignore[no-untyped-call]
    search_query = websearch_to_tsquery(TEXT_SEARCH_CONFIG, query)  # type: ignore[no-untyped-call]
    rank = func.ts_rank(document_vector, search_query).label("rank")
    keyword_rows = await session.execute(
        select(models.DocumentChunk)
        .join(models.Document, models.Document.id == models.DocumentChunk.document_id)
        .options(contains_eager(models.DocumentChunk.document))
        .where(*in_scope, rank > 0)
        .order_by(rank.desc())
        .limit(CANDIDATES_PER_SEARCH)
    )
    return vector_hits, list(keyword_rows.scalars())


async def build_rag_context(
    session: AsyncSession, collections: list[models.DocumentCollection], query: str
) -> RetrievalResult:
    """Per distinct embedding model among `collections`: a vector search plus a
    keyword search over the same chunks, fused by rank. Results become one
    context block plus a citation list for the socket's "done" payload. A
    collection whose embedding container isn't running is skipped rather than
    blocking the reply, but never silently — it comes back as a warning for
    the user. Each collection needs its `embedding_model` loaded.
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
            client = await embeddings.get_embeddings_client(model_name, parameters)
            query_vector = await client.aembed_query(query)
        except ModelUnavailableError as e:
            logger.warning("Skipping RAG collection(s) %s: %s", [c.id for c in cols], e)
            warnings.append(f"{_collection_names(cols)} weren't searched. {e}")
            continue
        except Exception as e:
            logger.warning(
                "Skipping RAG collection(s) %s — could not embed query: %s",
                [c.id for c in cols],
                e,
            )
            warnings.append(
                f"{_collection_names(cols)} weren't searched because the embedding model "
                f"{model_name}:{parameters} didn't respond ({e})."
            )
            continue

        vector_hits, keyword_hits = await _search_ready_chunks(
            session, [c.id for c in cols], query, query_vector
        )
        fused.extend(_fuse(vector_hits, keyword_hits))

    still_processing = (
        await session.execute(
            select(func.count())
            .select_from(models.Document)
            .where(
                models.Document.collection_id.in_([c.id for c in collections]),
                models.Document.status.in_(
                    [models.Document.STATUS_PENDING, models.Document.STATUS_PROCESSING]
                ),
            )
        )
    ).scalar_one()
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
        citations.append(
            {
                "document_id": chunk.document_id,
                "filename": chunk.document.filename,
                "chunk_index": chunk.chunk_index,
                "page": chunk.page,
            }
        )

    return RetrievalResult("\n".join(context_lines), citations, warnings)


def _collection_names(collections: list[models.DocumentCollection]) -> str:
    quoted = ", ".join(f'"{c.name}"' for c in collections)
    return f"Your documents in {quoted}"


async def retrieve_context_for_chat(
    session: AsyncSession, chat: models.ChatHistory, query: str, previous_question: str | None = None
) -> RetrievalResult:
    """Entry point for the chat socket."""
    collections = list(
        (
            await session.execute(
                select(models.DocumentCollection)
                .join(
                    models.chat_history_active_collection,
                    models.chat_history_active_collection.c.document_collection_id
                    == models.DocumentCollection.id,
                )
                .where(models.chat_history_active_collection.c.chat_history_id == chat.id)
                .options(joinedload(models.DocumentCollection.embedding_model))
                .order_by(models.DocumentCollection.id)
            )
        )
        .scalars()
        .all()
    )
    return await build_rag_context(session, collections, build_search_query(query, previous_question))
