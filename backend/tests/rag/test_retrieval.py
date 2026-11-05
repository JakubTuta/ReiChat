"""build_rag_context/retrieve_context_for_chat against a real Postgres +
pgvector — cosine distance and full-text search are database-side operations,
not something worth faking. The embedding HTTP call is the only thing mocked out.
"""

import pytest
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import models
from app.llm.runtime import ModelUnavailableError
from app.rag import retrieval
from tests.factories import make_chat, make_model

pytestmark = pytest.mark.usefixtures("db")


class FakeEmbeddingsClient:
    def __init__(self, query_vector):
        self.query_vector = query_vector
        self.queries: list[str] = []

    async def aembed_query(self, text):
        self.queries.append(text)
        return self.query_vector


@pytest.fixture
def embed_query_as(monkeypatch):
    def configure(vector):
        client = FakeEmbeddingsClient(vector)

        async def get_client(model, params):
            return client

        monkeypatch.setattr(retrieval.embeddings, "get_embeddings_client", get_client)
        return client

    return configure


@pytest.fixture
async def embedding_model(db: AsyncSession) -> models.AIModel:
    return await make_model(db, "nomic-embed-text-retrieval-test", is_embedding=True)


async def make_collection(db, embedding_model, name="Docs", **fields) -> models.DocumentCollection:
    collection = models.DocumentCollection(
        name=name, embedding_model=embedding_model, embedding_parameters="latest", **fields
    )
    db.add(collection)
    await db.commit()
    return collection


@pytest.fixture
async def collection(db: AsyncSession, embedding_model) -> models.DocumentCollection:
    return await make_collection(db, embedding_model)


async def make_chunk(
    db: AsyncSession,
    collection,
    content,
    embedding,
    chunk_index=0,
    filename="doc.txt",
    status=models.Document.STATUS_READY,
    page=None,
):
    document = models.Document(collection_id=collection.id, filename=filename, status=status)
    db.add(document)
    await db.flush()
    chunk = models.DocumentChunk(
        document_id=document.id,
        chunk_index=chunk_index,
        content=content,
        char_start=0,
        page=page,
        embedding=embedding,
    )
    db.add(chunk)
    await db.commit()
    return chunk


async def test_build_rag_context_empty_collections_returns_nothing(db: AsyncSession):
    context, citations, warnings = await retrieval.build_rag_context(db, [], "anything")

    assert (context, citations, warnings) == ("", [], [])


async def test_build_rag_context_returns_closest_chunk_first(db: AsyncSession, collection, embed_query_as):
    await make_chunk(db, collection, "about cats", [1.0, 0.0], filename="cats.txt")
    await make_chunk(db, collection, "about dogs", [0.0, 1.0], filename="dogs.txt")
    embed_query_as([1.0, 0.0])

    context, citations, warnings = await retrieval.build_rag_context(db, [collection], "cats")

    assert citations[0]["filename"] == "cats.txt"
    assert "about cats" in context
    assert "[cats.txt]" in context
    assert context.startswith("The user has attached documents.")


async def test_the_question_is_what_gets_embedded(db: AsyncSession, collection, embed_query_as):
    client = embed_query_as([1.0, 0.0])

    await retrieval.build_rag_context(db, [collection], "what is the warranty?")

    assert client.queries == ["what is the warranty?"]


async def test_build_rag_context_ignores_non_ready_documents(db: AsyncSession, collection, embed_query_as):
    await make_chunk(db, collection, "not ready yet", [1.0, 0.0], status=models.Document.STATUS_PROCESSING)
    embed_query_as([1.0, 0.0])

    context, citations, warnings = await retrieval.build_rag_context(db, [collection], "anything")

    assert (context, citations) == ("", [])


async def test_build_rag_context_skips_collection_when_embedding_fails(
    db: AsyncSession, collection, monkeypatch
):
    await make_chunk(db, collection, "content", [1.0, 0.0])

    async def broken(model, params):
        raise Exception("container not running")  # noqa: TRY002

    monkeypatch.setattr(retrieval.embeddings, "get_embeddings_client", broken)

    context, citations, warnings = await retrieval.build_rag_context(db, [collection], "anything")

    assert (context, citations) == ("", [])
    assert len(warnings) == 1
    assert '"Docs"' in warnings[0]
    assert "didn't respond (container not running)" in warnings[0]


async def test_a_failing_embed_call_is_skipped_with_a_warning_too(db: AsyncSession, collection, monkeypatch):
    class Failing:
        async def aembed_query(self, text):
            raise ConnectionError("embedder went away")

    async def get_client(model, params):
        return Failing()

    monkeypatch.setattr(retrieval.embeddings, "get_embeddings_client", get_client)

    _, _, warnings = await retrieval.build_rag_context(db, [collection], "anything")

    assert "embedder went away" in warnings[0]


async def test_build_rag_context_warns_with_the_actionable_message_when_embedder_is_stopped(
    db: AsyncSession, collection, monkeypatch
):
    await make_chunk(db, collection, "content", [1.0, 0.0])

    async def stopped(model, params):
        raise ModelUnavailableError(
            "Embedding model nomic:latest isn't running. Start it on the Models page."
        )

    monkeypatch.setattr(retrieval.embeddings, "get_embeddings_client", stopped)

    _, _, warnings = await retrieval.build_rag_context(db, [collection], "anything")

    assert warnings == [
        'Your documents in "Docs" weren\'t searched. '
        "Embedding model nomic:latest isn't running. Start it on the Models page."
    ]


async def test_build_rag_context_warns_about_documents_still_processing(
    db: AsyncSession, collection, embed_query_as
):
    await make_chunk(db, collection, "not ready yet", [1.0, 0.0], status=models.Document.STATUS_PROCESSING)
    embed_query_as([1.0, 0.0])

    _, _, warnings = await retrieval.build_rag_context(db, [collection], "anything")

    assert warnings == ["1 document is still being processed and couldn't be searched yet."]


async def test_several_documents_still_processing_are_counted(db: AsyncSession, collection, embed_query_as):
    for i in range(3):
        await make_chunk(
            db, collection, "x", [1.0, 0.0], filename=f"{i}.txt", status=models.Document.STATUS_PENDING
        )
    embed_query_as([1.0, 0.0])

    _, _, warnings = await retrieval.build_rag_context(db, [collection], "anything")

    assert warnings == ["3 documents are still being processed and couldn't be searched yet."]


async def test_build_rag_context_scopes_to_given_collections_only(
    db: AsyncSession, embedding_model, embed_query_as
):
    collection_a = await make_collection(db, embedding_model, "A")
    collection_b = await make_collection(db, embedding_model, "B")
    await make_chunk(db, collection_a, "in collection A", [1.0, 0.0], filename="a.txt")
    await make_chunk(db, collection_b, "in collection B", [1.0, 0.0], filename="b.txt")
    embed_query_as([1.0, 0.0])

    _, citations, _ = await retrieval.build_rag_context(db, [collection_a], "anything")

    assert {c["filename"] for c in citations} == {"a.txt"}


async def test_collections_with_the_same_embedding_model_are_searched_together(
    db: AsyncSession, embedding_model, embed_query_as
):
    collection_a = await make_collection(db, embedding_model, "A")
    collection_b = await make_collection(db, embedding_model, "B")
    await make_chunk(db, collection_a, "in collection A", [1.0, 0.0], filename="a.txt")
    await make_chunk(db, collection_b, "in collection B", [1.0, 0.0], filename="b.txt")
    client = embed_query_as([1.0, 0.0])

    _, citations, _ = await retrieval.build_rag_context(db, [collection_a, collection_b], "anything")

    assert {c["filename"] for c in citations} == {"a.txt", "b.txt"}
    assert len(client.queries) == 1


async def test_each_embedding_model_gets_its_own_query(db: AsyncSession, embedding_model, monkeypatch):
    other = await make_model(db, "mxbai-embed-large", is_embedding=True)
    collection_a = await make_collection(db, embedding_model, "A")
    collection_b = await make_collection(db, other, "B")
    await make_chunk(db, collection_a, "in A", [1.0, 0.0], filename="a.txt")
    await make_chunk(db, collection_b, "in B", [1.0, 0.0, 0.0], filename="b.txt")
    asked = []

    class Client:
        def __init__(self, vector):
            self.vector = vector

        async def aembed_query(self, text):
            return self.vector

    async def get_client(model, params):
        asked.append(model)
        return Client([1.0, 0.0] if model == embedding_model.model else [1.0, 0.0, 0.0])

    monkeypatch.setattr(retrieval.embeddings, "get_embeddings_client", get_client)

    _, citations, _ = await retrieval.build_rag_context(db, [collection_a, collection_b], "anything")

    assert sorted(asked) == sorted([embedding_model.model, other.model])
    assert {c["filename"] for c in citations} == {"a.txt", "b.txt"}


async def test_retrieve_context_for_chat_uses_chats_active_collections(
    db: AsyncSession, collection, embed_query_as
):
    chat = await make_chat(db, await make_model(db, "llama-retrieval-test"), title="t")
    await db.execute(
        insert(models.chat_history_active_collection).values(
            chat_history_id=chat.id, document_collection_id=collection.id
        )
    )
    await db.commit()
    await make_chunk(db, collection, "relevant content", [1.0, 0.0], filename="doc.txt")
    embed_query_as([1.0, 0.0])

    _, citations, _ = await retrieval.retrieve_context_for_chat(db, chat, "anything")

    assert citations[0]["filename"] == "doc.txt"


async def test_retrieve_context_for_chat_ignores_collections_that_are_not_active(
    db: AsyncSession, collection, embed_query_as
):
    chat = await make_chat(db, await make_model(db, "llama-retrieval-test"), title="t")
    await make_chunk(db, collection, "relevant content", [1.0, 0.0], filename="doc.txt")
    embed_query_as([1.0, 0.0])

    assert await retrieval.retrieve_context_for_chat(db, chat, "anything") == ("", [], [])


async def test_retrieve_context_for_chat_no_active_collections_returns_nothing(db: AsyncSession):
    chat = await make_chat(db, await make_model(db, "llama-retrieval-test-2"), title="t")

    context, citations, warnings = await retrieval.retrieve_context_for_chat(db, chat, "anything")

    assert (context, citations, warnings) == ("", [], [])


async def test_retrieve_context_for_chat_searches_follow_ups_together_with_the_previous_question(
    db: AsyncSession, collection, embed_query_as
):
    chat = await make_chat(db, await make_model(db, "llama-retrieval-test-3"), title="t")
    await db.execute(
        insert(models.chat_history_active_collection).values(
            chat_history_id=chat.id, document_collection_id=collection.id
        )
    )
    await db.commit()
    client = embed_query_as([1.0, 0.0])

    await retrieval.retrieve_context_for_chat(
        db, chat, "and the second?", previous_question="What are the warranty options?"
    )

    assert client.queries == ["What are the warranty options?\nand the second?"]


async def test_keyword_search_finds_exact_codes_that_embeddings_miss(
    db: AsyncSession, collection, embed_query_as
):
    await make_chunk(db, collection, "general product overview", [1.0, 0.0], filename="overview.txt")
    await make_chunk(
        db, collection, "error ZX4471 means the fan is blocked", [0.0, 1.0], filename="codes.txt"
    )
    embed_query_as([1.0, 0.0])

    _, citations, _ = await retrieval.build_rag_context(db, [collection], "ZX4471")

    assert "codes.txt" in {c["filename"] for c in citations}


async def test_keyword_search_ignores_case_and_punctuation_in_the_question(
    db: AsyncSession, collection, embed_query_as
):
    await make_chunk(db, collection, "The WARRANTY lasts two years.", [0.0, 1.0], filename="terms.txt")
    await make_chunk(db, collection, "unrelated", [1.0, 0.0], filename="other.txt")
    embed_query_as([1.0, 0.0])

    _, citations, _ = await retrieval.build_rag_context(db, [collection], "warranty?!")

    assert "terms.txt" in {c["filename"] for c in citations}


async def test_a_question_with_search_operator_syntax_does_not_break_the_search(
    db: AsyncSession, collection, embed_query_as
):
    await make_chunk(db, collection, "plain text", [1.0, 0.0], filename="doc.txt")
    embed_query_as([1.0, 0.0])

    for question in ['"unterminated', "a OR", "-", "(((", "'; DROP TABLE document_chunk; --"]:
        await retrieval.build_rag_context(db, [collection], question)


async def test_weak_vector_only_matches_are_dropped_when_a_strong_one_exists(
    db: AsyncSession, collection, embed_query_as
):
    await make_chunk(db, collection, "about cats", [1.0, 0.0], filename="cats.txt")
    await make_chunk(db, collection, "about dogs", [0.0, 1.0], filename="dogs.txt")
    embed_query_as([1.0, 0.0])

    _, citations, _ = await retrieval.build_rag_context(db, [collection], "felines")

    assert [c["filename"] for c in citations] == ["cats.txt"]


async def test_citations_and_context_carry_the_page(db: AsyncSession, collection, embed_query_as):
    await make_chunk(db, collection, "reset instructions", [1.0, 0.0], filename="manual.pdf", page=7)
    embed_query_as([1.0, 0.0])

    context, citations, _ = await retrieval.build_rag_context(db, [collection], "reset")

    assert citations[0]["page"] == 7
    assert "[manual.pdf, page 7]" in context


async def test_a_citation_names_the_document_and_the_chunk(db: AsyncSession, collection, embed_query_as):
    chunk = await make_chunk(db, collection, "text", [1.0, 0.0], chunk_index=4, filename="a.txt")
    embed_query_as([1.0, 0.0])

    _, citations, _ = await retrieval.build_rag_context(db, [collection], "text")

    assert citations == [
        {"document_id": chunk.document_id, "filename": "a.txt", "chunk_index": 4, "page": None}
    ]


async def test_no_more_than_six_passages_reach_the_prompt(db: AsyncSession, collection, embed_query_as):
    for i in range(12):
        await make_chunk(db, collection, f"passage {i}", [1.0, 0.0], filename=f"{i}.txt")
    embed_query_as([1.0, 0.0])

    _, citations, _ = await retrieval.build_rag_context(db, [collection], "passage")

    assert len(citations) == retrieval.MAX_CONTEXT_CHUNKS


def test_short_follow_up_questions_are_searched_together_with_the_previous_one():
    assert retrieval.build_search_query("and the second?", "What are the two warranty options?") == (
        "What are the two warranty options?\nand the second?"
    )
    long_question = "Explain in detail how the warranty registration process works for business customers"
    assert retrieval.build_search_query(long_question, "previous") == long_question
    assert retrieval.build_search_query("short", None) == "short"


def chunk_of(chunk_id: int) -> models.DocumentChunk:
    return models.DocumentChunk(
        id=chunk_id, document_id=1, chunk_index=chunk_id, content=str(chunk_id), char_start=0
    )


def test_a_passage_both_searches_agree_on_outranks_one_only_a_single_search_found():
    both, vector_only, keyword_only = chunk_of(1), chunk_of(2), chunk_of(3)

    fused = retrieval._fuse([(vector_only, 0.1), (both, 0.12)], [keyword_only, both])

    assert [chunk.id for _, chunk in fused][0] == 1


def test_fusion_of_nothing_is_nothing():
    assert retrieval._fuse([], []) == []


def test_a_far_off_vector_hit_is_dropped_unless_the_keyword_search_found_it_too():
    near, far_agreed, far_alone = chunk_of(1), chunk_of(2), chunk_of(3)

    fused = retrieval._fuse([(near, 0.1), (far_agreed, 0.9), (far_alone, 0.95)], [far_agreed])

    assert {chunk.id for _, chunk in fused} == {1, 2}
