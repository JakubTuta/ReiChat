"""Container creation, discovery and install-state detection against a
mocked Docker SDK and mocked Ollama HTTP — never a real daemon."""

from __future__ import annotations

from unittest.mock import MagicMock

import docker.errors
import pytest
import requests

from container import ContainerManager as container_manager_module
from container.ContainerManager import ContainerError, ContainerManager


@pytest.fixture(autouse=True)
def _reset_class_state(monkeypatch):
    monkeypatch.setattr(ContainerManager, "_client", None)
    monkeypatch.setattr(ContainerManager, "_gpu_supported", None)
    monkeypatch.setattr(ContainerManager, "_installed_ids", set())


def _fake_container(
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


def _managed_labels(model: str = "llama3.1", parameters: str = "8b", port: str = "11434") -> dict:
    return {
        "reichat.managed": "true",
        "reichat.model": model,
        "reichat.parameters": parameters,
        "reichat.port": port,
    }


@pytest.fixture
def connected_client(monkeypatch):
    client = MagicMock()
    client.ping.return_value = True
    client.containers.get.side_effect = docker.errors.NotFound("no such container")
    monkeypatch.setattr(ContainerManager, "_client", client)
    monkeypatch.setattr(ContainerManager, "allocate_port", staticmethod(lambda name: 11500))
    return client


def test_create_labels_container_and_mounts_a_named_volume_for_weights(connected_client):
    connected_client.containers.create.return_value = _fake_container(status="created")

    ContainerManager().create_and_start_container("llama3.1", "8b")

    kwargs = connected_client.containers.create.call_args.kwargs
    assert kwargs["labels"]["reichat.managed"] == "true"
    assert kwargs["labels"]["reichat.model"] == "llama3.1"
    assert kwargs["labels"]["reichat.parameters"] == "8b"
    assert kwargs["labels"]["reichat.port"] == "11500"
    assert kwargs["volumes"] == {"reichat-ollama-llama3.1_8b": {"bind": "/root/.ollama", "mode": "rw"}}
    assert kwargs["device_requests"]
    assert kwargs["ports"] == {"11434/tcp": ("127.0.0.1", 11500)}
    assert ContainerManager._gpu_supported is True


def test_create_falls_back_to_cpu_when_host_has_no_gpu_runtime(connected_client):
    gpu_container = _fake_container(status="created", container_id="gpu")
    gpu_container.start.side_effect = docker.errors.APIError(
        "500 Server Error",
        explanation='could not select device driver "" with capabilities: [[gpu]]',
    )
    cpu_container = _fake_container(status="created", container_id="cpu")
    connected_client.containers.create.side_effect = [gpu_container, cpu_container]

    result = ContainerManager().create_and_start_container("llama3.1", "8b")

    assert result is cpu_container
    gpu_container.remove.assert_called_once_with(force=True)
    cpu_container.start.assert_called_once()
    second_kwargs = connected_client.containers.create.call_args_list[1].kwargs
    assert second_kwargs["device_requests"] == []
    assert second_kwargs["labels"]["reichat.gpu"] == "false"
    assert ContainerManager._gpu_supported is False


def test_once_gpu_is_known_missing_new_containers_skip_the_gpu_request(connected_client):
    ContainerManager._gpu_supported = False
    connected_client.containers.create.return_value = _fake_container(status="created")

    ContainerManager().create_and_start_container("llama3.1", "8b")

    assert connected_client.containers.create.call_args.kwargs["device_requests"] == []


def test_non_gpu_start_failure_raises_readable_error_without_recreating(connected_client):
    broken = _fake_container(status="created")
    broken.start.side_effect = docker.errors.APIError("500", explanation="port is already allocated")
    connected_client.containers.create.return_value = broken

    with pytest.raises(ContainerError, match="port is already allocated"):
        ContainerManager().create_and_start_container("llama3.1", "8b")

    assert connected_client.containers.create.call_count == 1


def test_starting_existing_gpu_container_on_gpu_less_host_explains_how_to_recover(connected_client):
    stopped = _fake_container(status="exited")
    stopped.start.side_effect = docker.errors.APIError("500", explanation="could not select device driver")

    with pytest.raises(ContainerError, match="create it again"):
        ContainerManager().start_container(stopped)


def test_available_containers_include_labelled_and_legacy_env_containers_only(connected_client):
    labelled = _fake_container(name="llama3.1_8b", status="exited", labels=_managed_labels())
    legacy = _fake_container(
        name="qwen2.5_0.5b", status="exited", env=["model=qwen2.5", "parameters=0.5b", "port=11435"],
    )
    unrelated = _fake_container(name="postgres", status="running", image="pgvector/pgvector:pg16")
    connected_client.containers.list.return_value = [labelled, legacy, unrelated]

    result = ContainerManager().get_available_containers()

    assert [c["name"] for c in result] == ["llama3.1_8b", "qwen2.5_0.5b"]
    assert result[1]["environment"] == {"model": "qwen2.5", "parameters": "0.5b"}
    assert result[1]["port"] == "11435"


@pytest.mark.parametrize(
    ("in_progress", "installed", "expected"),
    [
        (True, True, "pulling_model"),
        (False, True, "running"),
        (False, False, "incomplete"),
        (False, None, "starting"),
    ],
)
def test_resolve_status_reflects_install_state(connected_client, monkeypatch, in_progress, installed, expected):
    monkeypatch.setattr(container_manager_module.progress_state, "is_in_progress", lambda name: in_progress)
    monkeypatch.setattr(ContainerManager, "is_model_installed", staticmethod(lambda *a: installed))

    status = ContainerManager().resolve_status(_fake_container(labels=_managed_labels()))

    assert status == expected


def test_confirmed_install_is_cached_so_tags_are_not_asked_again(connected_client, monkeypatch):
    calls = []

    def _installed(*args):
        calls.append(args)
        return True

    monkeypatch.setattr(ContainerManager, "is_model_installed", staticmethod(_installed))
    container = _fake_container(labels=_managed_labels())
    manager = ContainerManager()

    assert manager.check_model_installed(container) is True
    assert manager.check_model_installed(container) is True
    assert len(calls) == 1


class _FakeTagsResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_is_model_installed_reads_ollama_tags(monkeypatch):
    payload = {"models": [{"name": "llama3.1:8b", "model": "llama3.1:8b"}]}
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeTagsResponse(payload))

    assert ContainerManager.is_model_installed("http://x", "llama3.1", "8b") is True
    assert ContainerManager.is_model_installed("http://x", "llama3.1", "70b") is False


def test_is_model_installed_is_none_when_ollama_unreachable(monkeypatch):
    def _refuse(*a, **k):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(requests, "get", _refuse)

    assert ContainerManager.is_model_installed("http://x", "llama3.1", "8b") is None


def test_remove_container_also_deletes_its_weights_volume(connected_client):
    container = _fake_container(labels=_managed_labels())
    connected_client.containers.get.side_effect = None
    connected_client.containers.get.return_value = container
    ContainerManager._installed_ids.add(container.id)

    ContainerManager().remove_container("llama3.1_8b")

    container.remove.assert_called_once()
    connected_client.volumes.get.assert_called_once_with("reichat-ollama-llama3.1_8b")
    connected_client.volumes.get.return_value.remove.assert_called_once()
    assert container.id not in ContainerManager._installed_ids


def test_metadata_prefers_labels_over_env():
    container = _fake_container(labels=_managed_labels(port="11500"), env=["port=99"])

    assert ContainerManager.get_container_metadata(container, "port") == "11500"


def test_recreate_keeps_name_port_labels_and_weights_volume(connected_client):
    old = _fake_container(labels={**_managed_labels(port="11500"), "reichat.gpu": "true"})
    old.attrs["Mounts"] = [{"Name": "reichat-ollama-llama3.1_8b"}]
    connected_client.containers.create.return_value = _fake_container(status="created", container_id="new")

    ContainerManager().recreate_on_current_image(old)

    old.remove.assert_called_once_with(force=True)
    kwargs = connected_client.containers.create.call_args.kwargs
    assert kwargs["name"] == "llama3.1_8b"
    assert kwargs["ports"] == {"11434/tcp": ("127.0.0.1", 11500)}
    assert kwargs["volumes"] == {"reichat-ollama-llama3.1_8b": {"bind": "/root/.ollama", "mode": "rw"}}
    assert kwargs["labels"]["reichat.model"] == "llama3.1"


def test_recreate_refuses_containers_whose_weights_would_be_lost(connected_client):
    legacy = _fake_container(labels=_managed_labels())
    legacy.attrs["Mounts"] = []

    with pytest.raises(ContainerError, match="downloading the model again"):
        ContainerManager().recreate_on_current_image(legacy)

    legacy.remove.assert_not_called()
