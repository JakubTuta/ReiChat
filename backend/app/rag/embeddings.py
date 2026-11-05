from langchain_ollama import OllamaEmbeddings

from app.llm import runtime
from app.llm.runtime import ModelUnavailableError


async def get_embeddings_client(model_name: str, parameters: str) -> OllamaEmbeddings:
    """Raises ModelUnavailableError (same as the chat path) if the model's
    container isn't up — an embedding model has to be running exactly like a
    chat model does, just never shown in the chat picker (AIModel.is_embedding).
    """
    url_info = await runtime.get_ollama_url(model_name, parameters)
    if url_info is None:
        raise ModelUnavailableError(
            f"Embedding model {model_name}:{parameters} isn't running. "
            "Start its container on the Models page and try again."
        )

    base_url, full_model_string = url_info

    return OllamaEmbeddings(model=full_model_string, base_url=base_url)
