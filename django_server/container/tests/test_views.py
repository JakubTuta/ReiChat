"""Hardware/DiskUsage views over a mocked ContainerManager — never a real
Docker daemon, same style as test_container_manager.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from container.ContainerManager import ContainerManager
from django_app import models

pytestmark = pytest.mark.django_db


@pytest.fixture(autouse=True)
def _reset_singleton():
    ContainerManager._client = None
    yield
    ContainerManager._client = None


@pytest.fixture
def connected_docker():
    fake_client = MagicMock()
    fake_client.ping.return_value = True
    ContainerManager._client = fake_client
    return fake_client


def test_get_hardware_returns_ram_and_vram(client, connected_docker, monkeypatch):
    monkeypatch.setattr("container.views.hardware.get_host_ram_bytes", lambda cm: 16_000_000_000)
    monkeypatch.setattr("container.views.hardware.detect_vram_bytes", lambda cm: 8_000_000_000)
    monkeypatch.setattr(ContainerManager, "gpu_status", lambda self: False)

    response = client.get("/docker/hardware/")

    assert response.status_code == 200
    assert response.json() == {"ram_bytes": 16_000_000_000, "vram_bytes": 8_000_000_000, "gpu_available": False}


def test_get_hardware_missing_vram_is_null(client, connected_docker, monkeypatch):
    monkeypatch.setattr("container.views.hardware.get_host_ram_bytes", lambda cm: 16_000_000_000)
    monkeypatch.setattr("container.views.hardware.detect_vram_bytes", lambda cm: None)

    response = client.get("/docker/hardware/")

    assert response.status_code == 200
    assert response.json()["vram_bytes"] is None


def test_get_hardware_docker_not_connected_returns_500(client, monkeypatch):
    monkeypatch.setattr(ContainerManager, "connect_to_docker", lambda self: False)

    response = client.get("/docker/hardware/")

    assert response.status_code == 500


def test_get_disk_usage_matches_containers_to_catalog_sizes(client, connected_docker, monkeypatch):
    ai_model = models.AIModel.objects.create(name="llama3.1", model="llama3.1-disk-test", index=1)
    models.AIModelVersion.objects.create(
        ai_model=ai_model, parameters="8b", size="4.6 GB", size_bytes=4_920_000_000
    )

    monkeypatch.setattr(
        ContainerManager,
        "get_available_containers",
        lambda self: [
            {
                "name": "llama3.1-disk-test_8b", "status": "running", "port": "11434",
                "environment": {"model": "llama3.1-disk-test", "parameters": "8b"},
            },
        ],
    )

    response = client.get("/docker/disk-usage/")

    assert response.status_code == 200
    body = response.json()
    assert body["total_bytes"] == 4_920_000_000
    assert body["models"] == [
        {"model": "llama3.1-disk-test", "parameters": "8b", "status": "running", "size_bytes": 4_920_000_000},
    ]


def test_get_disk_usage_unknown_model_has_null_size_and_excluded_from_total(client, connected_docker, monkeypatch):
    monkeypatch.setattr(
        ContainerManager,
        "get_available_containers",
        lambda self: [
            {
                "name": "ghost_1b", "status": "running", "port": "11434",
                "environment": {"model": "ghost-disk-test", "parameters": "1b"},
            },
        ],
    )

    response = client.get("/docker/disk-usage/")

    assert response.status_code == 200
    body = response.json()
    assert body["total_bytes"] == 0
    assert body["models"][0]["size_bytes"] is None


def test_get_disk_usage_no_containers_returns_empty(client, connected_docker, monkeypatch):
    monkeypatch.setattr(ContainerManager, "get_available_containers", lambda self: [])

    response = client.get("/docker/disk-usage/")

    assert response.status_code == 200
    assert response.json() == {"total_bytes": 0, "models": []}


@pytest.fixture
def catalog_model():
    ai_model = models.AIModel.objects.create(name="llama3.1", model="llama3.1-post-test", index=1)
    models.AIModelVersion.objects.create(ai_model=ai_model, parameters="8b", size="4.6 GB")
    return ai_model


@pytest.fixture
def install_calls(monkeypatch):
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "container.views.install.start_install", lambda model, params: calls.append((model, params)) or True
    )
    return calls


def _existing_container(container_id: str = "c1") -> MagicMock:
    container = MagicMock()
    container.id = container_id
    container.name = "llama3.1-post-test_8b"
    container.status = "running"
    container.attrs = {"Config": {"Labels": {}, "Env": []}}
    return container


def test_post_existing_confirmed_container_returns_200_without_installing(
    client, connected_docker, catalog_model, install_calls, monkeypatch
):
    container = _existing_container()
    monkeypatch.setattr(ContainerManager, "get_container", lambda self, name: container)
    monkeypatch.setattr(ContainerManager, "_installed_ids", {container.id})

    response = client.post("/docker/container/llama3.1-post-test?parameters=8b")

    assert response.status_code == 200
    assert install_calls == []


def test_post_existing_unconfirmed_container_resumes_install(
    client, connected_docker, catalog_model, install_calls, monkeypatch
):
    monkeypatch.setattr(ContainerManager, "get_container", lambda self, name: _existing_container())
    monkeypatch.setattr(ContainerManager, "_installed_ids", set())

    response = client.post("/docker/container/llama3.1-post-test?parameters=8b")

    assert response.status_code == 202
    assert install_calls == [("llama3.1-post-test", "8b")]


def test_post_new_container_starts_install(client, connected_docker, catalog_model, install_calls, monkeypatch):
    monkeypatch.setattr(ContainerManager, "get_container", lambda self, name: None)

    response = client.post("/docker/container/llama3.1-post-test?parameters=8b")

    assert response.status_code == 202
    assert install_calls == [("llama3.1-post-test", "8b")]


def test_post_existing_container_that_fails_to_start_returns_readable_error(
    client, connected_docker, catalog_model, install_calls, monkeypatch
):
    from container.ContainerManager import ContainerError

    stopped = _existing_container()
    stopped.status = "exited"
    monkeypatch.setattr(ContainerManager, "get_container", lambda self, name: stopped)

    def _fail(self, container):
        raise ContainerError("Couldn't start the model container: boom")

    monkeypatch.setattr(ContainerManager, "start_container", _fail)

    response = client.post("/docker/container/llama3.1-post-test?parameters=8b")

    assert response.status_code == 500
    assert response.json() == {"error": "Couldn't start the model container: boom"}
    assert install_calls == []


def test_get_disk_usage_docker_not_connected_returns_500(client, monkeypatch):
    monkeypatch.setattr(ContainerManager, "connect_to_docker", lambda self: False)

    response = client.get("/docker/disk-usage/")

    assert response.status_code == 500
