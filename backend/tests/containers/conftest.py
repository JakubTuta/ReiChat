from unittest.mock import MagicMock

import docker.errors
import pytest

from app.containers import manager as manager_module
from app.containers.manager import ContainerManager


def fake_container(
    name: str = "llama3.1_8b",
    status: str = "running",
    labels: dict | None = None,
    env: list[str] | None = None,
    image: str = "ollama/ollama:latest",
    container_id: str = "abc123",
) -> MagicMock:
    container = MagicMock()
    container.name = name
    container.status = status
    container.id = container_id
    container.attrs = {"Config": {"Labels": labels or {}, "Env": env or [], "Image": image}}
    return container


def managed_labels(model: str = "llama3.1", parameters: str = "8b", port: str = "11434") -> dict:
    return {
        "reichat.managed": "true",
        "reichat.model": model,
        "reichat.parameters": parameters,
        "reichat.port": port,
    }


@pytest.fixture
def connected_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """A Docker SDK client that answers pings and knows no containers."""
    client = MagicMock()
    client.ping.return_value = True
    client.containers.get.side_effect = docker.errors.NotFound("no such container")
    monkeypatch.setattr(ContainerManager, "_client", client)

    async def allocate_port(container_name: str) -> int:
        return 11500

    monkeypatch.setattr(manager_module.ports, "allocate_port", allocate_port)
    return client
