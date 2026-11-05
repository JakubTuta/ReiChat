"""The /docker/* endpoints over a mocked ContainerManager — never a real Docker
daemon — and a mocked Ollama."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import runtime_settings
from app.containers import hardware, install, router, searxng, service
from app.containers.manager import ContainerError, ContainerManager
from app.db import models
from tests.containers.conftest import fake_container, managed_labels

EXTERNAL = "http://host.docker.internal:11434"
DOCKER_DOWN = {"error": "Error connecting to Docker"}


@pytest.fixture
def docker_up(monkeypatch):
    monkeypatch.setattr(ContainerManager, "ensure_connected", AsyncMock(return_value=True))


@pytest.fixture
def docker_down(monkeypatch):
    monkeypatch.setattr(ContainerManager, "ensure_connected", AsyncMock(return_value=False))


@pytest.fixture
def external_mode(monkeypatch):
    monkeypatch.setattr(runtime_settings, "external_ollama_url", AsyncMock(return_value=EXTERNAL))


@pytest.fixture
def ollama(monkeypatch):
    mocks = MagicMock()
    mocks.version = AsyncMock(return_value="0.12.0")
    mocks.installed_model_sizes = AsyncMock(return_value={})
    mocks.running_models = AsyncMock(return_value=[])
    mocks.unload_model = AsyncMock(return_value=True)
    mocks.delete_model = AsyncMock(return_value=True)
    for name in ("version", "installed_model_sizes", "running_models", "unload_model", "delete_model"):
        monkeypatch.setattr(router.ollama_api, name, getattr(mocks, name))
    return mocks


@pytest.fixture
def install_calls(monkeypatch):
    calls = MagicMock()
    calls.start_install = MagicMock(return_value=True)
    calls.start_model_update = MagicMock(return_value=True)
    calls.start_ollama_update = MagicMock(return_value=True)
    for name in ("start_install", "start_model_update", "start_ollama_update"):
        monkeypatch.setattr(install, name, getattr(calls, name))
    return calls


async def add_catalog_model(
    db: AsyncSession, model: str = "llama3.1", parameters: str = "8b", size_bytes=None
):
    ai_model = models.AIModel(name=model, model=model, description="", index=1)
    db.add(ai_model)
    await db.flush()
    db.add(
        models.AIModelVersion(
            ai_model_id=ai_model.id, parameters=parameters, size="4.6 GB", size_bytes=size_bytes
        )
    )
    await db.commit()


# Docker health


async def test_docker_status_when_docker_is_running(client: httpx.AsyncClient, docker_up):
    response = await client.get("/docker/")

    assert response.status_code == 200
    assert response.json() == {"status": "Docker is running"}


async def test_docker_status_when_docker_is_down(client: httpx.AsyncClient, docker_down):
    response = await client.get("/docker/")

    assert response.status_code == 500
    assert response.json() == DOCKER_DOWN


# Listing


async def test_containers_are_listed(client: httpx.AsyncClient, docker_up, monkeypatch):
    listed = [{"name": "llama3.1_8b", "status": "running", "port": "11500", "environment": {}}]
    monkeypatch.setattr(service, "get_available_containers", AsyncMock(return_value=listed))

    response = await client.get("/docker/containers/")

    assert response.status_code == 200
    assert response.json() == listed


async def test_containers_list_needs_docker(client: httpx.AsyncClient, docker_down):
    response = await client.get("/docker/containers/")

    assert response.status_code == 500
    assert response.json() == DOCKER_DOWN


async def test_listing_with_the_users_own_ollama_first_registers_its_models(
    client: httpx.AsyncClient, external_mode, monkeypatch, db: AsyncSession
):
    register = AsyncMock(return_value=1)
    monkeypatch.setattr(router.external, "register_models", register)
    monkeypatch.setattr(service, "get_available_containers", AsyncMock(return_value=[]))

    response = await client.get("/docker/containers/")

    assert response.status_code == 200
    register.assert_awaited_once()
    assert register.await_args.args[1] == EXTERNAL


# Hardware


async def test_get_hardware_returns_ram_and_vram(client: httpx.AsyncClient, docker_up, monkeypatch):
    monkeypatch.setattr(ContainerManager, "host_memory_bytes", AsyncMock(return_value=16_000_000_000))
    monkeypatch.setattr(hardware, "detect_vram_bytes", AsyncMock(return_value=8_000_000_000))
    monkeypatch.setattr(ContainerManager, "gpu_status", AsyncMock(return_value=False))

    response = await client.get("/docker/hardware/")

    assert response.status_code == 200
    assert response.json() == {
        "ram_bytes": 16_000_000_000,
        "vram_bytes": 8_000_000_000,
        "gpu_available": False,
    }


# Disk usage


def listed_container(model: str, parameters: str = "8b", status: str = "running") -> dict:
    return {
        "name": f"{model}_{parameters}",
        "status": status,
        "port": "11434",
        "environment": {"model": model, "parameters": parameters},
    }


async def test_get_disk_usage_matches_containers_to_catalog_sizes(
    client: httpx.AsyncClient, docker_up, monkeypatch, db: AsyncSession, ollama
):
    await add_catalog_model(db, "llama3.1-disk-test", size_bytes=4_920_000_000)
    monkeypatch.setattr(
        service, "get_available_containers", AsyncMock(return_value=[listed_container("llama3.1-disk-test")])
    )
    monkeypatch.setattr(ContainerManager, "get_ollama_base_url", AsyncMock(return_value=None))

    response = await client.get("/docker/disk-usage/")

    assert response.status_code == 200
    assert response.json() == {
        "total_bytes": 4_920_000_000,
        "models": [
            {
                "model": "llama3.1-disk-test",
                "parameters": "8b",
                "status": "running",
                "size_bytes": 4_920_000_000,
            }
        ],
    }


# Starting, inspecting and removing one model's container


@pytest.fixture
def existing_container(monkeypatch):
    container = fake_container(name="llama3.1_8b", labels=managed_labels(), container_id="c1")
    monkeypatch.setattr(ContainerManager, "get_container", AsyncMock(return_value=container))
    monkeypatch.setattr(ContainerManager, "start_container", AsyncMock())
    return container


async def test_post_existing_confirmed_container_returns_200_without_installing(
    client: httpx.AsyncClient, docker_up, db, install_calls, existing_container
):
    await add_catalog_model(db)
    service.mark_installed(existing_container)

    response = await client.post("/docker/container/llama3.1?parameters=8b")

    assert response.status_code == 200
    assert response.json()["status"] == "Container is running"
    assert response.json()["container"]["name"] == "llama3.1_8b"
    install_calls.start_install.assert_not_called()


async def test_post_existing_unconfirmed_container_resumes_install(
    client: httpx.AsyncClient, docker_up, db, install_calls, existing_container
):
    await add_catalog_model(db)

    response = await client.post("/docker/container/llama3.1?parameters=8b")

    assert response.status_code == 202
    assert response.json() == {"status": "Model setup started"}
    install_calls.start_install.assert_called_once_with("llama3.1", "8b")
    ContainerManager.start_container.assert_awaited_once_with(existing_container)


async def test_post_new_container_starts_install(
    client: httpx.AsyncClient, docker_up, db, install_calls, monkeypatch
):
    await add_catalog_model(db)
    monkeypatch.setattr(ContainerManager, "get_container", AsyncMock(return_value=None))

    response = await client.post("/docker/container/llama3.1?parameters=8b")

    assert response.status_code == 202
    install_calls.start_install.assert_called_once_with("llama3.1", "8b")


async def test_post_existing_container_that_fails_to_start_returns_readable_error(
    client: httpx.AsyncClient, docker_up, db, install_calls, existing_container, monkeypatch
):
    await add_catalog_model(db)
    monkeypatch.setattr(
        ContainerManager,
        "start_container",
        AsyncMock(side_effect=ContainerError("Couldn't start the model container: boom")),
    )

    response = await client.post("/docker/container/llama3.1?parameters=8b")

    assert response.status_code == 500
    assert response.json() == {"error": "Couldn't start the model container: boom"}
    install_calls.start_install.assert_not_called()


async def test_post_for_a_model_not_in_the_catalog_is_a_404(client: httpx.AsyncClient, docker_up, db):
    response = await client.post("/docker/container/nothing?parameters=8b")

    assert response.status_code == 404
    assert response.json() == {"error": "Model not found"}


async def test_post_needs_docker(client: httpx.AsyncClient, docker_down, db):
    response = await client.post("/docker/container/llama3.1?parameters=8b")

    assert response.status_code == 500
    assert response.json() == DOCKER_DOWN


async def test_post_for_a_namespaced_model_keeps_its_slashes(
    client: httpx.AsyncClient, docker_up, db, install_calls, monkeypatch
):
    await add_catalog_model(db, "hf.co/user/repo", "Q4_K_M")
    monkeypatch.setattr(ContainerManager, "get_container", AsyncMock(return_value=None))

    response = await client.post("/docker/container/hf.co/user/repo?parameters=Q4_K_M")

    assert response.status_code == 202
    install_calls.start_install.assert_called_once_with("hf.co/user/repo", "Q4_K_M")


async def test_installing_on_the_users_ollama_when_the_model_is_missing_there(
    client: httpx.AsyncClient, external_mode, db, install_calls, monkeypatch
):
    await add_catalog_model(db, "llama3.2-ext-test", "3b")
    monkeypatch.setattr(router.external, "has_model", AsyncMock(return_value=False))

    response = await client.post("/docker/container/llama3.2-ext-test?parameters=3b")

    assert response.status_code == 202
    install_calls.start_install.assert_called_once_with("llama3.2-ext-test", "3b")


async def test_delete_stop_stops_the_container(client: httpx.AsyncClient, docker_up, monkeypatch):
    stop = AsyncMock()
    monkeypatch.setattr(ContainerManager, "stop_container", stop)

    response = await client.delete("/docker/container/llama3.1?parameters=8b&method=stop")

    assert response.status_code == 200
    assert response.json() == {"status": "Container stopped"}
    stop.assert_awaited_once_with("llama3.1_8b")


async def test_delete_remove_removes_the_container(client: httpx.AsyncClient, docker_up, monkeypatch):
    remove = AsyncMock()
    monkeypatch.setattr(service, "remove_model_container", remove)

    response = await client.delete("/docker/container/llama3.1?parameters=8b&method=remove")

    assert response.status_code == 200
    assert response.json() == {"status": "Container removed"}
    remove.assert_awaited_once_with("llama3.1_8b")


async def test_stopping_a_model_on_the_users_ollama_unloads_it(
    client: httpx.AsyncClient, external_mode, ollama
):
    response = await client.delete("/docker/container/llama3.2?parameters=3b&method=stop")

    assert response.status_code == 200
    assert response.json() == {"status": "Model unloaded"}
    ollama.unload_model.assert_awaited_once_with(EXTERNAL, "llama3.2:3b")


async def test_removing_a_model_from_the_users_ollama_deletes_it_there(
    client: httpx.AsyncClient, external_mode, ollama
):
    response = await client.delete("/docker/container/llama3.2?parameters=3b&method=remove")

    assert response.status_code == 200
    assert response.json() == {"status": "Model removed"}
    ollama.delete_model.assert_awaited_once_with(EXTERNAL, "llama3.2", "3b")


# Unload, update, runtime


async def test_unload_frees_a_running_model(client: httpx.AsyncClient, monkeypatch, ollama):
    monkeypatch.setattr(ContainerManager, "get_ollama_base_url", AsyncMock(return_value="http://o"))

    response = await client.post("/docker/container/llama3.1/unload?parameters=8b")

    assert response.status_code == 200
    assert response.json() == {"status": "Model unloaded"}
    ollama.unload_model.assert_awaited_once_with("http://o", "llama3.1:8b")


async def test_update_starts_a_re_pull_for_an_installed_model(
    client: httpx.AsyncClient, docker_up, existing_container, install_calls
):
    response = await client.post("/docker/container/llama3.1/update?parameters=8b")

    assert response.status_code == 202
    assert response.json() == {"status": "Checking for updates"}
    install_calls.start_model_update.assert_called_once_with("llama3.1", "8b")


async def test_update_of_a_model_that_is_not_installed_is_a_404(
    client: httpx.AsyncClient, docker_up, install_calls, monkeypatch
):
    monkeypatch.setattr(ContainerManager, "get_container", AsyncMock(return_value=None))

    response = await client.post("/docker/container/llama3.1/update?parameters=8b")

    assert response.status_code == 404
    install_calls.start_model_update.assert_not_called()


async def test_runtime_lists_loaded_models_per_container(client: httpx.AsyncClient, connected_client, ollama):
    container = fake_container(labels=managed_labels(port="11500"))
    connected_client.containers.list.return_value = [container]
    ollama.running_models.return_value = [
        {"name": "llama3.1:8b", "size": 100, "size_vram": 80, "context_length": 8192, "expires_at": "t"}
    ]

    response = await client.get("/docker/runtime/")

    assert response.status_code == 200
    assert response.json()["loaded"] == [
        {
            "container": container.name,
            "model": "llama3.1:8b",
            "size_bytes": 100,
            "vram_bytes": 80,
            "context_length": 8192,
            "expires_at": "t",
        }
    ]
    ollama.running_models.assert_awaited_once_with("http://127.0.0.1:11500")


# The Ollama image and updating it


async def test_ollama_image_present(client: httpx.AsyncClient, docker_up, monkeypatch):
    monkeypatch.setattr(ContainerManager, "is_ollama_image_pulled", AsyncMock(return_value=True))

    assert (await client.get("/docker/ollama-image/")).status_code == 200


async def test_pulling_the_ollama_image_runs_in_the_background(
    client: httpx.AsyncClient, docker_up, monkeypatch
):
    pull = AsyncMock()
    monkeypatch.setattr(ContainerManager, "pull_ollama_image", pull)

    response = await client.post("/docker/ollama-image/")
    await asyncio.sleep(0)

    assert response.status_code == 202
    assert response.json() == {"status": "Pulling ollama/ollama image in background"}
    pull.assert_awaited_once()


async def test_ollama_update_starts(client: httpx.AsyncClient, docker_up, install_calls):
    response = await client.post("/docker/ollama-update/")

    assert response.status_code == 202
    assert response.json() == {"status": "Ollama update started"}
    install_calls.start_ollama_update.assert_called_once()


# Web search


async def test_web_search_can_be_turned_on(client: httpx.AsyncClient, docker_up, monkeypatch):
    enable = MagicMock(return_value=True)
    monkeypatch.setattr(searxng, "start_enable", enable)

    response = await client.post("/docker/web-search/")

    assert response.status_code == 202
    assert response.json() == {"status": "installing"}
    enable.assert_called_once()


async def test_web_search_can_be_turned_off(client: httpx.AsyncClient, monkeypatch):
    disable = AsyncMock()
    monkeypatch.setattr(searxng, "disable", disable)

    response = await client.delete("/docker/web-search/")

    assert response.status_code == 200
    assert response.json() == {"status": "off"}


# Where models run


async def test_connection_reports_the_users_own_ollama(client: httpx.AsyncClient, external_mode, ollama):
    response = await client.get("/docker/ollama-connection/")

    assert response.json() == {"mode": "external", "url": EXTERNAL, "version": "0.12.0"}


async def test_connection_is_only_saved_when_ollama_answers(
    client: httpx.AsyncClient, db, ollama, monkeypatch
):
    async def stored_url():
        return await runtime_settings.get(runtime_settings.EXTERNAL_OLLAMA_URL) or None

    ollama.version.return_value = None
    refused = await client.put(
        "/docker/ollama-connection/", json={"mode": "external", "url": "http://nowhere:11434"}
    )
    assert refused.status_code == 400
    assert "Nothing answered like Ollama at http://nowhere:11434" in refused.json()["error"]
    assert await stored_url() is None

    ollama.version.return_value = "0.12.0"
    monkeypatch.setattr(router.external, "register_models", AsyncMock(return_value=0))
    saved = await client.put(
        "/docker/ollama-connection/", json={"mode": "external", "url": "http://host:11434/"}
    )
    assert saved.status_code == 200
    assert saved.json() == {"mode": "external", "url": "http://host:11434", "version": "0.12.0"}
    assert await stored_url() == "http://host:11434"

    back = await client.put("/docker/ollama-connection/", json={"mode": "managed"})
    assert back.json() == {"mode": "managed", "url": None, "version": None}
    assert await stored_url() is None


async def test_switching_to_the_users_ollama_makes_its_models_selectable(
    client: httpx.AsyncClient, db: AsyncSession, ollama
):
    ollama.installed_model_sizes.return_value = {"brand-new:latest": 1_000}

    response = await client.put("/docker/ollama-connection/", json={"mode": "external", "url": EXTERNAL})

    assert response.status_code == 200
    created = (await db.execute(select(models.AIModel.model, models.AIModel.is_custom))).all()
    assert [(row.model, row.is_custom) for row in created] == [("brand-new", True)]


async def test_connection_with_an_unknown_mode_is_a_400(client: httpx.AsyncClient):
    body = {"mode": "nonsense"}

    response = await client.put("/docker/ollama-connection/", json=body)

    assert response.status_code == 400
    assert response.json() == {"error": "mode must be 'managed' or 'external'"}


async def test_connection_url_has_to_start_with_http(client: httpx.AsyncClient):
    url = "host:11434"

    response = await client.put("/docker/ollama-connection/", json={"mode": "external", "url": url})

    assert response.status_code == 400
    assert "http://" in response.json()["error"]
