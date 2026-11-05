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


async def test_client_is_a_singleton_across_instances(monkeypatch):
    fake_clients = []

    def docker_client():
        client = MagicMock()
        fake_clients.append(client)
        return client

    monkeypatch.setattr(manager_module.docker, "DockerClient", docker_client)

    for _ in range(3):
        await ContainerManager().ensure_connected()

    assert len(fake_clients) == 1


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


async def test_a_stale_client_is_dropped_even_when_nothing_can_replace_it(monkeypatch):
    stale = MagicMock()
    stale.ping.side_effect = docker.errors.APIError("daemon gone")
    monkeypatch.setattr(ContainerManager, "_client", stale)

    def raise_no_daemon():
        raise docker.errors.DockerException("no daemon")

    monkeypatch.setattr(manager_module.docker, "DockerClient", raise_no_daemon)

    assert await ContainerManager().ensure_connected() is False
    assert ContainerManager._client is None


async def test_a_live_client_is_kept(connected_client):
    assert await ContainerManager().ensure_connected() is True
    assert ContainerManager._client is connected_client


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


async def test_once_gpu_is_known_missing_new_containers_skip_the_gpu_request(connected_client):
    ContainerManager._gpu_supported = False
    connected_client.containers.create.return_value = fake_container(status="created")

    await ContainerManager().create_and_start_container("llama3.1", "8b")

    assert connected_client.containers.create.call_args.kwargs["device_requests"] == []


async def test_non_gpu_start_failure_raises_readable_error_without_recreating(connected_client):
    broken = fake_container(status="created")
    broken.start.side_effect = docker.errors.APIError("500", explanation="port is already allocated")
    connected_client.containers.create.return_value = broken

    with pytest.raises(ContainerError, match="port is already allocated"):
        await ContainerManager().create_and_start_container("llama3.1", "8b")

    assert connected_client.containers.create.call_count == 1


async def test_a_container_that_cannot_even_be_created_is_reported_readably(connected_client):
    connected_client.containers.create.side_effect = docker.errors.APIError(
        "500", explanation="no such network"
    )

    with pytest.raises(ContainerError, match="Couldn't create the model container: no such network"):
        await ContainerManager().create_and_start_container("llama3.1", "8b")


async def test_an_existing_container_is_started_not_recreated(connected_client):
    stopped = fake_container(status="exited")
    connected_client.containers.get.side_effect = None
    connected_client.containers.get.return_value = stopped

    result = await ContainerManager().create_and_start_container("llama3.1", "8b")

    assert result is stopped
    stopped.start.assert_called_once()
    connected_client.containers.create.assert_not_called()


async def test_starting_an_already_running_container_does_nothing(connected_client):
    running = fake_container(status="running")

    await ContainerManager().start_container(running)

    running.start.assert_not_called()


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


async def test_get_container_is_none_when_it_does_not_exist(connected_client):
    assert await ContainerManager().get_container("llama3.1_8b") is None


async def test_get_container_is_none_when_docker_fails(connected_client):
    connected_client.containers.get.side_effect = docker.errors.DockerException("boom")

    assert await ContainerManager().get_container("llama3.1_8b") is None


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


async def test_listing_only_running_containers_asks_docker_for_just_those(connected_client):
    running = fake_container(labels=managed_labels())
    connected_client.containers.list.return_value = [running]

    assert await ContainerManager().list_managed(running_only=True) == [running]
    connected_client.containers.list.assert_called_once_with(filters={"status": "running"})


async def test_listing_without_docker_is_empty(monkeypatch):
    def raise_no_daemon():
        raise docker.errors.DockerException("no daemon")

    monkeypatch.setattr(manager_module.docker, "DockerClient", raise_no_daemon)

    assert await ContainerManager().list_managed() == []


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


async def test_remove_container_still_clears_a_leftover_volume_when_the_container_is_gone(connected_client):
    assert await ContainerManager().remove_container("llama3.1_8b") is None

    connected_client.volumes.get.return_value.remove.assert_called_once()


async def test_stop_container_stops_it(connected_client):
    container = fake_container()
    connected_client.containers.get.side_effect = None
    connected_client.containers.get.return_value = container

    await ContainerManager().stop_container("llama3.1_8b")

    container.stop.assert_called_once()


async def test_stop_container_survives_docker_refusing(connected_client):
    container = fake_container()
    container.stop.side_effect = docker.errors.APIError("500", explanation="busy")
    connected_client.containers.get.side_effect = None
    connected_client.containers.get.return_value = container

    await ContainerManager().stop_container("llama3.1_8b")


def test_metadata_prefers_labels_over_env():
    container = fake_container(labels=managed_labels(port="11500"), env=["port=99"])

    assert ContainerManager.get_container_metadata(container, "port") == "11500"


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


async def test_recreate_restarts_a_container_that_was_running(connected_client):
    old = fake_container(labels=managed_labels(port="11500"), status="running")
    old.attrs["Mounts"] = [{"Name": "reichat-ollama-llama3.1_8b"}]
    replacement = fake_container(status="created", container_id="new")
    connected_client.containers.create.return_value = replacement

    result = await ContainerManager().recreate_on_current_image(old)

    assert result is replacement
    replacement.start.assert_called_once()


async def test_recreate_leaves_a_stopped_container_stopped(connected_client):
    old = fake_container(labels=managed_labels(port="11500"), status="exited")
    old.attrs["Mounts"] = [{"Name": "reichat-ollama-llama3.1_8b"}]
    replacement = fake_container(status="created", container_id="new")
    connected_client.containers.create.return_value = replacement

    await ContainerManager().recreate_on_current_image(old)

    replacement.start.assert_not_called()


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


def test_a_container_with_no_known_port_has_no_address_outside_docker():
    assert ContainerManager.base_url_for(fake_container(labels={})) is None


async def test_ollama_base_url_only_for_running_containers(connected_client):
    running = fake_container(labels=managed_labels(port="11500"), status="running")
    connected_client.containers.get.side_effect = None
    connected_client.containers.get.return_value = running

    assert await ContainerManager().get_ollama_base_url("llama3.1", "8b") == "http://127.0.0.1:11500"

    running.status = "exited"
    assert await ContainerManager().get_ollama_base_url("llama3.1", "8b") is None


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


async def test_gpu_status_falls_back_to_how_existing_containers_were_created(connected_client):
    connected_client.containers.list.return_value = [
        fake_container(labels={**managed_labels(), "reichat.gpu": "false"})
    ]

    assert await ContainerManager().gpu_status() is False


async def test_gpu_status_is_unknown_when_nothing_has_been_started(connected_client):
    connected_client.containers.list.return_value = []

    assert await ContainerManager().gpu_status() is None


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


async def test_ollama_image_already_present_is_not_pulled_again(connected_client):
    calls = []

    assert (
        await ContainerManager().ensure_ollama_image(lambda pct, detail: calls.append((pct, detail))) is True
    )

    assert calls == [(100, "Base image already present")]
    connected_client.api.pull.assert_not_called()


async def test_ollama_image_pull_failure_is_reported_not_raised(connected_client):
    connected_client.images.get.side_effect = docker.errors.ImageNotFound("absent")
    connected_client.api.pull.side_effect = docker.errors.APIError("offline")

    assert await ContainerManager().ensure_ollama_image() is False


async def test_pulling_the_ollama_image_failing_says_why(connected_client):
    connected_client.images.pull.side_effect = docker.errors.APIError("offline")

    with pytest.raises(ContainerError, match="Couldn't download the Ollama image"):
        await ContainerManager().pull_ollama_image()


async def test_image_id_is_what_docker_reports(connected_client):
    connected_client.images.get.return_value = MagicMock(id="sha256:abc")

    assert await ContainerManager().image_id() == "sha256:abc"


async def test_image_id_is_none_when_the_image_is_absent(connected_client):
    connected_client.images.get.side_effect = docker.errors.ImageNotFound("absent")

    assert await ContainerManager().image_id() is None


async def test_host_memory_is_what_docker_reports(connected_client):
    connected_client.info.return_value = {"MemTotal": 16_000_000_000}

    assert await ContainerManager().host_memory_bytes() == 16_000_000_000


async def test_host_memory_is_none_when_docker_cannot_say(connected_client):
    connected_client.info.side_effect = Exception("daemon unreachable")

    assert await ContainerManager().host_memory_bytes() is None


async def test_exec_in_gpu_container_runs_the_command_in_a_gpu_container(connected_client):
    gpu_container = MagicMock()
    gpu_container.exec_run.return_value = (0, b"12288\n")
    connected_client.containers.list.return_value = [gpu_container]

    assert await ContainerManager().exec_in_gpu_container(["nvidia-smi"]) == "12288\n"
    filters = connected_client.containers.list.call_args.kwargs["filters"]
    assert "reichat.gpu=true" in filters["label"]
    assert filters["status"] == "running"


async def test_exec_in_gpu_container_is_none_without_one(connected_client):
    connected_client.containers.list.return_value = []

    assert await ContainerManager().exec_in_gpu_container(["nvidia-smi"]) is None


async def test_exec_in_gpu_container_tries_the_next_container_when_one_fails(connected_client):
    broken, working = MagicMock(), MagicMock()
    broken.exec_run.return_value = (1, b"")
    working.exec_run.return_value = (0, b"4096")
    connected_client.containers.list.return_value = [broken, working]

    assert await ContainerManager().exec_in_gpu_container(["nvidia-smi"]) == "4096"
