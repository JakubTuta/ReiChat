from unittest.mock import AsyncMock

import pytest
from langchain_ollama import OllamaEmbeddings

from app.llm import runtime
from app.llm.runtime import ModelUnavailableError
from app.rag import embeddings


async def test_a_running_embedding_model_gets_a_client_pointed_at_it(monkeypatch):
    monkeypatch.setattr(
        runtime, "get_ollama_url", AsyncMock(return_value=("http://o:11434", "nomic-embed-text:latest"))
    )

    client = await embeddings.get_embeddings_client("nomic-embed-text", "latest")

    assert isinstance(client, OllamaEmbeddings)
    assert (client.model, client.base_url) == ("nomic-embed-text:latest", "http://o:11434")


async def test_an_embedding_model_that_is_not_running_says_how_to_start_it(monkeypatch):
    monkeypatch.setattr(runtime, "get_ollama_url", AsyncMock(return_value=None))

    with pytest.raises(ModelUnavailableError, match="nomic-embed-text:latest isn't running.*Models page"):
        await embeddings.get_embeddings_client("nomic-embed-text", "latest")


async def test_an_embedding_model_that_is_still_downloading_passes_its_own_message_on(monkeypatch):
    monkeypatch.setattr(
        runtime, "get_ollama_url", AsyncMock(side_effect=ModelUnavailableError("still downloading"))
    )

    with pytest.raises(ModelUnavailableError, match="still downloading"):
        await embeddings.get_embeddings_client("nomic-embed-text", "latest")
