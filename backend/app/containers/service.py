"""What the app knows about its models' containers: which exist, and whether
each one actually holds its model. That takes both Docker (manager.py) and the
Ollama inside the container (llm/ollama_api.py), so it lives above both.
"""

import asyncio
import typing

from app import runtime_settings
from app.containers import external
from app.containers.manager import Container, ContainerManager
from app.llm import ollama_api
from app.realtime import progress

# Container ids whose model Ollama has confirmed it holds. A model only
# disappears when its container is removed, so once confirmed this never needs
# asking again — it keeps /api/tags off the 2s snapshot path.
_installed_ids: set[str] = set()


def mark_installed(container: Container) -> None:
    if container.id:
        _installed_ids.add(container.id)


def is_marked_installed(container: Container) -> bool:
    return container.id in _installed_ids


def forget_installed(container_id: str | None) -> None:
    _installed_ids.discard(container_id or "")


def reset() -> None:
    """For tests, which share this module's state."""
    _installed_ids.clear()


async def check_model_installed(container: Container) -> bool | None:
    """Whether the container's Ollama holds its model, or None when Ollama
    can't be asked yet (still booting, or the address is unknown)."""
    if is_marked_installed(container):
        return True

    base_url = ContainerManager.base_url_for(container)
    model = ContainerManager.get_container_metadata(container, "model")
    parameters = ContainerManager.get_container_metadata(container, "parameters")
    if base_url is None or model is None or parameters is None:
        return None

    installed = await ollama_api.is_model_installed(base_url, model, parameters)
    if installed:
        mark_installed(container)
    return installed


async def resolve_status(container: Container) -> str:
    if container.status != ContainerManager.CONTAINER_STATUS["RUNNING"]:
        return str(container.status)

    if progress.is_in_progress(container.name or ""):
        return ContainerManager.CONTAINER_STATUS["PULLING_MODEL"]

    installed = await check_model_installed(container)
    if installed is None:
        return ContainerManager.CONTAINER_STATUS["STARTING"]

    return ContainerManager.CONTAINER_STATUS["RUNNING" if installed else "INCOMPLETE"]


async def get_available_containers() -> list[dict[str, typing.Any]]:
    """The models the user can see on the Models page: the user's own Ollama's
    models when that is in use, otherwise this app's containers."""
    if (external_url := await runtime_settings.external_ollama_url()) is not None:
        return await external.list_models(external_url)

    manager = ContainerManager()
    if not await manager.ensure_connected():
        return []

    containers = await manager.list_managed()
    statuses = await asyncio.gather(*(resolve_status(container) for container in containers))
    return [
        ContainerManager.map_container(container, status=status)
        for container, status in zip(containers, statuses, strict=True)
    ]


async def remove_model_container(container_name: str) -> None:
    removed_id = await ContainerManager().remove_container(container_name)
    forget_installed(removed_id)
