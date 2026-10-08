"""build_rag_context/retrieve_context_for_chat against a real Postgres +
pgvector — CosineDistance is a database-side operation, not something worth
faking. The embedding HTTP call is the only thing mocked out.
"""

from __future__ import annotations

import pytest

from django_app import models
from django_app.rag import retrieval

pytestmark = pytest.mark.django_db


class _FakeEmbeddingsClient:
    def __init__(self, query_vector):
        self._query_vector = query_vector

    def embed_query(self, text):
        return self._query_vector


def _stub_embeddings(monkeypatch, vector):
    monkeypatch.setattr(
        retrieval.embeddings, "get_embeddings_client",
        lambda model, params: _FakeEmbeddingsClient(vector),
    )


@pytest.fixture
def embedding_model():
    return models.AIModel.objects.create(
        name="nomic", model="nomic-embed-text-retrieval-test", is_embedding=True, index=1
    )


@pytest.fixture
def collection(embedding_model):
    return models.DocumentCollection.objects.create(
        name="Docs", embedding_model=embedding_model, embedding_parameters="latest"
    )


def _make_chunk(
    collection, content, embedding, chunk_index=0, filename="doc.txt",
    status=models.Document.STATUS_READY,
):
    document = models.Document.objects.create(collection=collection, filename=filename, status=status)
    return models.DocumentChunk.objects.create(
        document=document, chunk_index=chunk_index, content=content, char_start=0, embedding=embedding
    )


def test_build_rag_context_empty_collections_returns_nothing():
    context, citations, warnings = retrieval.build_rag_context([], "anything")

    assert context == ""
    assert citations == []


def test_build_rag_context_returns_closest_chunk_first(monkeypatch, collection):
    _make_chunk(collection, "about cats", [1.0, 0.0], filename="cats.txt")
    _make_chunk(collection, "about dogs", [0.0, 1.0], filename="dogs.txt")
    _stub_embeddings(monkeypatch, [1.0, 0.0])

    context, citations, warnings = retrieval.build_rag_context([collection], "cats")

    assert citations[0]["filename"] == "cats.txt"
    assert "about cats" in context
    assert "[cats.txt]" in context


def test_build_rag_context_ignores_non_ready_documents(monkeypatch, collection):
    _make_chunk(collection, "not ready yet", [1.0, 0.0], status=models.Document.STATUS_PROCESSING)
    _stub_embeddings(monkeypatch, [1.0, 0.0])

    context, citations, warnings = retrieval.build_rag_context([collection], "anything")

    assert context == ""
    assert citations == []


def test_build_rag_context_skips_collection_when_embedding_fails(monkeypatch, collection):
    _make_chunk(collection, "content", [1.0, 0.0])

    def _raise(model, params):
        raise Exception("container not running")  # noqa: TRY002

    monkeypatch.setattr(retrieval.embeddings, "get_embeddings_client", _raise)

    context, citations, warnings = retrieval.build_rag_context([collection], "anything")

    assert context == ""
    assert citations == []
    assert len(warnings) == 1
    assert '"Docs"' in warnings[0]


def test_build_rag_context_warns_with_the_actionable_message_when_embedder_is_stopped(monkeypatch, collection):
    from django_app.functions import ModelUnavailableError

    _make_chunk(collection, "content", [1.0, 0.0])

    def _raise(model, params):
        raise ModelUnavailableError("Embedding model nomic:latest isn't running. Start it on the Models page.")

    monkeypatch.setattr(retrieval.embeddings, "get_embeddings_client", _raise)

    _, _, warnings = retrieval.build_rag_context([collection], "anything")

    assert warnings == [
        "Your documents in \"Docs\" weren't searched. "
        "Embedding model nomic:latest isn't running. Start it on the Models page."
    ]


def test_build_rag_context_warns_about_documents_still_processing(monkeypatch, collection):
    _make_chunk(collection, "not ready yet", [1.0, 0.0], status=models.Document.STATUS_PROCESSING)
    _stub_embeddings(monkeypatch, [1.0, 0.0])

    _, _, warnings = retrieval.build_rag_context([collection], "anything")

    assert warnings == ["1 document is still being processed and couldn't be searched yet."]


def test_build_rag_context_scopes_to_given_collections_only(monkeypatch, embedding_model):
    collection_a = models.DocumentCollection.objects.create(
        name="A", embedding_model=embedding_model, embedding_parameters="latest"
    )
    collection_b = models.DocumentCollection.objects.create(
        name="B", embedding_model=embedding_model, embedding_parameters="latest"
    )
    _make_chunk(collection_a, "in collection A", [1.0, 0.0], filename="a.txt")
    _make_chunk(collection_b, "in collection B", [1.0, 0.0], filename="b.txt")
    _stub_embeddings(monkeypatch, [1.0, 0.0])

    context, citations, warnings = retrieval.build_rag_context([collection_a], "anything")

    assert {c["filename"] for c in citations} == {"a.txt"}


def test_retrieve_context_for_chat_uses_chats_active_collections(monkeypatch, collection):
    ai_model = models.AIModel.objects.create(name="llama", model="llama-retrieval-test", index=1)
    chat = models.ChatHistory.objects.create(ai_model=ai_model, title="t")
    chat.active_collections.add(collection)
    _make_chunk(collection, "relevant content", [1.0, 0.0], filename="doc.txt")
    _stub_embeddings(monkeypatch, [1.0, 0.0])

    context, citations, warnings = retrieval.retrieve_context_for_chat(chat, "anything")

    assert citations[0]["filename"] == "doc.txt"


def test_retrieve_context_for_chat_no_active_collections_returns_nothing():
    ai_model = models.AIModel.objects.create(name="llama", model="llama-retrieval-test-2", index=1)
    chat = models.ChatHistory.objects.create(ai_model=ai_model, title="t")

    context, citations, warnings = retrieval.retrieve_context_for_chat(chat, "anything")

    assert context == ""
    assert citations == []


def test_keyword_search_finds_exact_codes_that_embeddings_miss(monkeypatch, collection):
    _make_chunk(collection, "general product overview", [1.0, 0.0], filename="overview.txt")
    _make_chunk(collection, "error ZX4471 means the fan is blocked", [0.0, 1.0], filename="codes.txt")
    _stub_embeddings(monkeypatch, [1.0, 0.0])

    _, citations, _ = retrieval.build_rag_context([collection], "ZX4471")

    assert "codes.txt" in {c["filename"] for c in citations}


def test_weak_vector_only_matches_are_dropped_when_a_strong_one_exists(monkeypatch, collection):
    _make_chunk(collection, "about cats", [1.0, 0.0], filename="cats.txt")
    _make_chunk(collection, "about dogs", [0.0, 1.0], filename="dogs.txt")
    _stub_embeddings(monkeypatch, [1.0, 0.0])

    _, citations, _ = retrieval.build_rag_context([collection], "felines")

    assert [c["filename"] for c in citations] == ["cats.txt"]


def test_citations_and_context_carry_the_page(monkeypatch, collection):
    document = models.Document.objects.create(
        collection=collection, filename="manual.pdf", status=models.Document.STATUS_READY
    )
    models.DocumentChunk.objects.create(
        document=document, chunk_index=0, content="reset instructions", char_start=0, page=7, embedding=[1.0, 0.0]
    )
    _stub_embeddings(monkeypatch, [1.0, 0.0])

    context, citations, _ = retrieval.build_rag_context([collection], "reset")

    assert citations[0]["page"] == 7
    assert "[manual.pdf, page 7]" in context


def test_short_follow_up_questions_are_searched_together_with_the_previous_one():
    assert retrieval.build_search_query("and the second?", "What are the two warranty options?") == (
        "What are the two warranty options?\nand the second?"
    )
    long_question = "Explain in detail how the warranty registration process works for business customers"
    assert retrieval.build_search_query(long_question, "previous") == long_question
    assert retrieval.build_search_query("short", None) == "short"
