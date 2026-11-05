"""Container creation, discovery and removal against a mocked Docker SDK —
never a real daemon.

The client used to be constructed per instance (a `self.__client` name-mangling
bug defeated the intended class-level singleton), leaking a connection on every
chat message, every 2s snapshot tick, and every view.
"""

from unittest.mock import MagicMock

import docker.errors
import pytest

from app.containers import manager as manager_module
from app.containers.manager import ContainerError, ContainerManager
from tests.containers.conftest import fake_container, managed_labels


async def test_not_connected_when_the_daemon_is_unreachable(monkeypatch):
    def raise_no_daemon():
        raise docker.errors.DockerException("no daemon")

    monkeypatch.setattr(manager_module.docker, "DockerClient", raise_no_daemon)

    assert await ContainerManager().ensure_connected() is False


async def test_a_client_that_fails_its_ping_is_dropped_and_replaced(monkeypatch):
    stale = MagicMock()
    stale.ping.side_effect = docker.errors.APIError("daemon gone")
    monkeypatch.setattr(ContainerManager, "_client", stale)
    fresh = MagicMock()
    monkeypatch.setattr(manager_module.docker, "DockerClient", lambda: fresh)

    assert await ContainerManager().ensure_connected() is True
    assert ContainerManager._client is fresh


async def test_create_labels_container_and_mounts_a_named_volume_for_weights(connected_client):
    connected_client.containers.create.return_value = fake_container(status="created")

    await ContainerManager().create_and_start_container("llama3.1", "8b")

    kwargs = connected_client.containers.create.call_args.kwargs
    assert kwargs["labels"]["reichat.managed"] == "true"
    assert kwargs["labels"]["reichat.model"] == "llama3.1"
    assert kwargs["labels"]["reichat.parameters"] == "8b"
    assert kwargs["labels"]["reichat.port"] == "11500"
    assert kwargs["volumes"] == {"reichat-ollama-llama3.1_8b": {"bind": "/root/.ollama", "mode": "rw"}}
    assert kwargs["device_requests"]
    assert kwargs["ports"] == {"11434/tcp": ("127.0.0.1", 11500)}
    assert kwargs["network"] == "chatbot-network"
    assert ContainerManager._gpu_supported is True


async def test_create_falls_back_to_cpu_when_host_has_no_gpu_runtime(connected_client):
    gpu_container = fake_container(status="created", container_id="gpu")
    gpu_container.start.side_effect = docker.errors.APIError(
        "500 Server Error", explanation='could not select device driver "" with capabilities: [[gpu]]'
    )
    cpu_container = fake_container(status="created", container_id="cpu")
    connected_client.containers.create.side_effect = [gpu_container, cpu_container]

    result = await ContainerManager().create_and_start_container("llama3.1", "8b")

    assert result is cpu_container
    gpu_container.remove.assert_called_once_with(force=True)
    cpu_container.start.assert_called_once()
    second_kwargs = connected_client.containers.create.call_args_list[1].kwargs
    assert second_kwargs["device_requests"] == []
    assert second_kwargs["labels"]["reichat.gpu"] == "false"
    assert ContainerManager._gpu_supported is False


async def test_non_gpu_start_failure_raises_readable_error_without_recreating(connected_client):
    broken = fake_container(status="created")
    broken.start.side_effect = docker.errors.APIError("500", explanation="port is already allocated")
    connected_client.containers.create.return_value = broken

    with pytest.raises(ContainerError, match="port is already allocated"):
        await ContainerManager().create_and_start_container("llama3.1", "8b")

    assert connected_client.containers.create.call_count == 1


async def test_an_existing_container_is_started_not_recreated(connected_client):
    stopped = fake_container(status="exited")
    connected_client.containers.get.side_effect = None
    connected_client.containers.get.return_value = stopped

    result = await ContainerManager().create_and_start_container("llama3.1", "8b")

    assert result is stopped
    stopped.start.assert_called_once()
    connected_client.containers.create.assert_not_called()


async def test_creating_without_docker_explains_what_to_do(monkeypatch):
    def raise_no_daemon():
        raise docker.errors.DockerException("no daemon")

    monkeypatch.setattr(manager_module.docker, "DockerClient", raise_no_daemon)

    with pytest.raises(ContainerError, match="Docker isn't running"):
        await ContainerManager().create_and_start_container("llama3.1", "8b")


async def test_starting_existing_gpu_container_on_gpu_less_host_explains_how_to_recover(connected_client):
    stopped = fake_container(status="exited")
    stopped.start.side_effect = docker.errors.APIError("500", explanation="could not select device driver")

    with pytest.raises(ContainerError, match="create it again"):
        await ContainerManager().start_container(stopped)


async def test_managed_containers_include_labelled_and_legacy_env_containers_only(connected_client):
    labelled = fake_container(name="llama3.1_8b", status="exited", labels=managed_labels())
    legacy = fake_container(
        name="qwen2.5_0.5b", status="exited", env=["model=qwen2.5", "parameters=0.5b", "port=11435"]
    )
    unrelated = fake_container(name="postgres", status="running", image="pgvector/pgvector:pg16")
    connected_client.containers.list.return_value = [labelled, legacy, unrelated]

    result = await ContainerManager().list_managed()

    assert result == [labelled, legacy]
    connected_client.containers.list.assert_called_once_with(all=True)


async def test_remove_container_also_deletes_its_weights_volume(connected_client):
    container = fake_container(labels=managed_labels())
    connected_client.containers.get.side_effect = None
    connected_client.containers.get.return_value = container

    removed_id = await ContainerManager().remove_container("llama3.1_8b")

    assert removed_id == container.id
    container.remove.assert_called_once()
    connected_client.volumes.get.assert_called_once_with("reichat-ollama-llama3.1_8b")
    connected_client.volumes.get.return_value.remove.assert_called_once()


async def test_remove_container_that_will_not_go_leaves_the_volume_alone(connected_client):
    container = fake_container(labels=managed_labels())
    container.remove.side_effect = docker.errors.APIError("500", explanation="busy")
    connected_client.containers.get.side_effect = None
    connected_client.containers.get.return_value = container

    assert await ContainerManager().remove_container("llama3.1_8b") is None
    connected_client.volumes.get.assert_not_called()


def test_metadata_falls_back_to_env_for_containers_older_than_labels():
    container = fake_container(env=["model=qwen2.5", "parameters=0.5b"])

    assert ContainerManager.get_container_metadata(container, "model") == "qwen2.5"
    assert ContainerManager.get_container_metadata(container, "port") is None


async def test_recreate_keeps_name_port_labels_and_weights_volume(connected_client):
    old = fake_container(labels={**managed_labels(port="11500"), "reichat.gpu": "true"})
    old.attrs["Mounts"] = [{"Name": "reichat-ollama-llama3.1_8b"}]
    connected_client.containers.create.return_value = fake_container(status="created", container_id="new")

    await ContainerManager().recreate_on_current_image(old)

    old.remove.assert_called_once_with(force=True)
    kwargs = connected_client.containers.create.call_args.kwargs
    assert kwargs["name"] == "llama3.1_8b"
    assert kwargs["ports"] == {"11434/tcp": ("127.0.0.1", 11500)}
    assert kwargs["volumes"] == {"reichat-ollama-llama3.1_8b": {"bind": "/root/.ollama", "mode": "rw"}}
    assert kwargs["labels"]["reichat.model"] == "llama3.1"


async def test_recreate_refuses_containers_whose_weights_would_be_lost(connected_client):
    legacy = fake_container(labels=managed_labels())
    legacy.attrs["Mounts"] = []

    with pytest.raises(ContainerError, match="downloading the model again"):
        await ContainerManager().recreate_on_current_image(legacy)

    legacy.remove.assert_not_called()


def test_container_names_are_docker_safe_for_namespaced_models():
    assert ContainerManager.container_name("hf.co/user/Repo-GGUF", "Q4_K_M") == "hf.co-user-Repo-GGUF_Q4_K_M"
    assert ContainerManager.container_name("llama3.2", "3b") == "llama3.2_3b"


def test_a_container_is_addressed_by_name_inside_docker_and_by_host_port_outside(monkeypatch):
    from app.core.config import get_settings

    container = fake_container(name="llama3.1_8b", labels=managed_labels(port="11500"))

    assert ContainerManager.base_url_for(container) == "http://127.0.0.1:11500"

    monkeypatch.setenv("DOCKER", "true")
    get_settings.cache_clear()
    try:
        assert ContainerManager.base_url_for(container) == "http://llama3.1_8b:11434"
    finally:
        get_settings.cache_clear()


def test_map_container_describes_it_for_the_ui():
    container = fake_container(name="llama3.1_8b", status="exited", labels=managed_labels(port="11500"))

    assert ContainerManager.map_container(container) == {
        "name": "llama3.1_8b",
        "status": "exited",
        "port": "11500",
        "environment": {"model": "llama3.1", "parameters": "8b"},
    }
    assert ContainerManager.map_container(container, status="incomplete")["status"] == "incomplete"


async def test_gpu_status_is_what_the_first_container_start_showed(connected_client):
    ContainerManager._gpu_supported = True

    assert await ContainerManager().gpu_status() is True
    connected_client.containers.list.assert_not_called()


async def test_ollama_image_pull_reports_overall_progress_across_layers(connected_client):
    connected_client.images.get.side_effect = docker.errors.ImageNotFound("absent")
    connected_client.api.pull.return_value = iter(
        [
            {"id": "aaaaaaaaaaaaaaaa", "progressDetail": {"total": 100, "current": 50}},
            {"id": "bbbbbbbbbbbbbbbb", "progressDetail": {"total": 100, "current": 0}},
            {"id": "aaaaaaaaaaaaaaaa", "progressDetail": {"total": 100, "current": 100}},
        ]
    )
    calls = []

    assert (
        await ContainerManager().ensure_ollama_image(lambda pct, detail: calls.append((pct, detail))) is True
    )

    assert calls == [
        (50, "Layer aaaaaaaaaaaa — 50%"),
        (25, "Layer bbbbbbbbbbbb — 0%"),
        (50, "Layer aaaaaaaaaaaa — 100%"),
        (100, "Base image ready"),
    ]


async def test_exec_in_gpu_container_runs_the_command_in_a_gpu_container(connected_client):
    gpu_container = MagicMock()
    gpu_container.exec_run.return_value = (0, b"12288\n")
    connected_client.containers.list.return_value = [gpu_container]

    assert await ContainerManager().exec_in_gpu_container(["nvidia-smi"]) == "12288\n"
    filters = connected_client.containers.list.call_args.kwargs["filters"]
    assert "reichat.gpu=true" in filters["label"]
    assert filters["status"] == "running"
