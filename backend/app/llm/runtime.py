"""Where a model is running right now and what it says it can do — the
questions to answer before generating anything.

A model is "available" when its container is running and holds its weights (or,
with the user's own Ollama, when that Ollama has it). Everything here answers
from live state: Docker via containers/, Ollama via llm/ollama_api.py.
"""

import typing

from app import runtime_settings
from app.containers import external, service
from app.containers.manager import LABEL_GPU, ContainerManager
from app.llm import ollama_api
from app.realtime import progress


class ModelUnavailableError(Exception):
    """The model can't be used right now; the message says what to do about it."""


class ModelInfo(typing.TypedDict):
    context_length: int | None
    thinking: ollama_api.ThinkingControls | None
    # What the installed model itself says it can do ("tools", "vision",
    # ...) — more reliable than the catalog scraped from ollama.com.
    capabilities: list[str]


def model_can(capability: str, info: ModelInfo | None, catalog_flag: bool) -> bool:
    """The model's own answer when it can be asked, the catalog's guess
    otherwise (container down, older Ollama without capabilities)."""
    if info and info["capabilities"]:
        return capability in info["capabilities"]
    return catalog_flag


# ChatHistory.think_level values besides "" (use the model's default) and a
# model-specific level name.
THINK_OFF = "off"
THINK_ON = "on"


def resolve_reasoning(think_level: str, info: ModelInfo | None) -> bool | str | None:
    """The `reasoning` value for ChatOllama. None only for models that
    can't think: for a thinking model, None would let Ollama think anyway
    (most default to it) while langchain-ollama throws the separate
    thinking text away — the user would just stare at a long silence."""
    controls = info["thinking"] if info else None
    if controls is None:
        return None

    values = controls["values"]
    if think_level == THINK_OFF:
        return False if False in values else controls["default"]
    if think_level == THINK_ON:
        return True if True in values else controls["default"]
    if think_level and think_level in values:
        return think_level

    # "Model default = don't think" stays off rather than becoming None.
    return controls["default"]


async def get_ollama_url(model_name: str, parameters: str) -> tuple[str, str] | None:
    """(base_url, "model:tag") for a running container that holds its model,
    None if the container isn't running. Raises ModelUnavailableError for the
    "running but not usable yet" cases, each with its own next step."""
    manager = ContainerManager()
    container_name = ContainerManager.container_name(model_name, parameters)

    if progress.is_in_progress(container_name):
        raise ModelUnavailableError(
            "This model is still downloading. Wait for it to finish on the Models page, then try again."
        )

    if (external_url := await runtime_settings.external_ollama_url()) is not None:
        installed = await external.has_model(external_url, model_name, parameters)
        if installed is None:
            raise ModelUnavailableError(
                f"Can't reach your Ollama at {external_url}. Check that it's running, or switch back to "
                "app-managed models on the Models page."
            )
        if not installed:
            raise ModelUnavailableError(
                "This model isn't installed on your Ollama yet. Install it from the Models page first."
            )
        return external_url, f"{model_name}:{parameters}"

    container = await manager.get_container(container_name)
    if container is None or container.status != "running":
        return None

    base_url = ContainerManager.base_url_for(container)
    if base_url is None:
        return None

    # None (Ollama not answering yet) falls through: the request itself then
    # fails with the "may still be starting up" message.
    if await service.check_model_installed(container) is False:
        raise ModelUnavailableError(
            "This model's download didn't finish. Open the Models page and press 'Resume download'."
        )

    return base_url, f"{model_name}:{parameters}"


async def external_url_in_use() -> bool:
    return await runtime_settings.external_ollama_url() is not None


async def get_model_info(model_name: str, parameters: str) -> ModelInfo | None:
    """What the model's own Ollama container says about it: its trained
    maximum context and whether/how it can think. None when that can't be
    determined (container down, older Ollama) — callers then neither cap
    the context nor offer thinking controls."""
    try:
        url_info = await get_ollama_url(model_name, parameters)
    except ModelUnavailableError:
        return None
    if url_info is None:
        return None

    show_info = await ollama_api.show_model(*url_info)
    if show_info is None:
        return None

    return {
        "context_length": ollama_api.context_length(show_info),
        "thinking": ollama_api.thinking_controls(show_info),
        "capabilities": ollama_api.capabilities(show_info),
    }


async def get_gpu_share(model_name: str, parameters: str) -> float | None:
    """Fraction (0–1) of the loaded model that sits in GPU memory, per
    /api/ps. Below 1 means part of it spilled to the CPU and every token is
    slower — usually because a bigger memory (num_ctx) grew the KV cache
    past what the GPU holds. None when there's no GPU in play at all, so
    CPU-only machines are never nagged about it."""
    external_url = await runtime_settings.external_ollama_url()
    if external_url is not None:
        # The user's own Ollama: no container label to say whether a GPU is
        # involved, so trust /api/ps — a CPU-only machine reports no VRAM.
        base_url: str | None = external_url
    else:
        container = await ContainerManager().get_container(
            ContainerManager.container_name(model_name, parameters)
        )
        if container is None or container.status != "running":
            return None
        if container.attrs.get("Config", {}).get("Labels", {}).get(LABEL_GPU) != "true":
            return None
        base_url = ContainerManager.base_url_for(container)

    running = await ollama_api.running_models(base_url) if base_url else None
    entry = next((m for m in running or [] if m["name"] == f"{model_name}:{parameters}"), None)
    if entry is None or not entry["size"] or (external_url is not None and not entry["size_vram"]):
        return None

    return round(entry["size_vram"] / entry["size"], 2)
