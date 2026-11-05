from unittest.mock import AsyncMock, MagicMock

import pytest

from app import runtime_settings
from app.containers import external, service
from app.containers.manager import ContainerManager
from app.llm import ollama_api, runtime
from app.llm.runtime import ModelUnavailableError
from app.realtime import progress
from tests.containers.conftest import fake_container, managed_labels

EXTERNAL = "http://host.docker.internal:11434"


@pytest.fixture
def external_mode(monkeypatch):
    monkeypatch.setattr(runtime_settings, "external_ollama_url", AsyncMock(return_value=EXTERNAL))


@pytest.fixture
def running_container(monkeypatch):
    container = fake_container(labels=managed_labels(port="11500"))
    monkeypatch.setattr(ContainerManager, "get_container", AsyncMock(return_value=container))
    monkeypatch.setattr(service, "check_model_installed", AsyncMock(return_value=True))
    return container


# Where a model is running


async def test_a_running_model_is_reached_at_its_containers_address(running_container):
    assert await runtime.get_ollama_url("llama3.1", "8b") == ("http://127.0.0.1:11500", "llama3.1:8b")


async def test_a_model_without_a_container_is_not_running(monkeypatch):
    monkeypatch.setattr(ContainerManager, "get_container", AsyncMock(return_value=None))

    assert await runtime.get_ollama_url("llama3.1", "8b") is None


async def test_a_stopped_container_is_not_running(running_container):
    running_container.status = "exited"

    assert await runtime.get_ollama_url("llama3.1", "8b") is None


async def test_a_container_with_no_known_address_is_not_running(running_container):
    running_container.attrs["Config"]["Labels"] = {}

    assert await runtime.get_ollama_url("llama3.1", "8b") is None


async def test_get_ollama_url_reports_unfinished_download(running_container, monkeypatch):
    monkeypatch.setattr(service, "check_model_installed", AsyncMock(return_value=False))

    with pytest.raises(ModelUnavailableError, match="Resume download"):
        await runtime.get_ollama_url("llama3.1", "8b")


async def test_an_ollama_that_is_not_answering_yet_is_left_to_the_request_to_report(
    running_container, monkeypatch
):
    monkeypatch.setattr(service, "check_model_installed", AsyncMock(return_value=None))

    assert await runtime.get_ollama_url("llama3.1", "8b") == ("http://127.0.0.1:11500", "llama3.1:8b")


async def test_get_ollama_url_reports_download_in_progress():
    progress.try_claim("busy-model_1b", "Starting…")

    with pytest.raises(ModelUnavailableError, match="still downloading"):
        await runtime.get_ollama_url("busy-model", "1b")


async def test_get_ollama_url_points_every_model_at_the_external_server(external_mode, monkeypatch):
    monkeypatch.setattr(external, "has_model", AsyncMock(return_value=True))

    assert await runtime.get_ollama_url("llama3.2", "3b") == (EXTERNAL, "llama3.2:3b")


async def test_get_ollama_url_explains_a_missing_model(external_mode, monkeypatch):
    monkeypatch.setattr(external, "has_model", AsyncMock(return_value=False))

    with pytest.raises(ModelUnavailableError, match="isn't installed on your Ollama"):
        await runtime.get_ollama_url("llama3.2", "3b")


async def test_get_ollama_url_explains_an_unreachable_server(external_mode, monkeypatch):
    monkeypatch.setattr(external, "has_model", AsyncMock(return_value=None))

    with pytest.raises(ModelUnavailableError, match="Can't reach your Ollama"):
        await runtime.get_ollama_url("llama3.2", "3b")


async def test_a_download_in_progress_wins_over_the_external_server(external_mode, monkeypatch):
    monkeypatch.setattr(external, "has_model", AsyncMock(return_value=True))
    progress.try_claim("llama3.2_3b", "Starting…")

    with pytest.raises(ModelUnavailableError, match="still downloading"):
        await runtime.get_ollama_url("llama3.2", "3b")


async def test_external_url_in_use(external_mode):
    assert await runtime.external_url_in_use() is True


async def test_external_url_not_in_use_by_default():
    assert await runtime.external_url_in_use() is False


# What the model says about itself


async def test_model_info_is_what_the_models_own_ollama_says(monkeypatch):
    monkeypatch.setattr(runtime, "get_ollama_url", AsyncMock(return_value=("http://o", "llama3.1:8b")))
    show = AsyncMock(
        return_value={
            "model_info": {"llama.context_length": 131072},
            "capabilities": ["completion", "tools"],
        }
    )
    monkeypatch.setattr(ollama_api, "show_model", show)

    assert await runtime.get_model_info("llama3.1", "8b") == {
        "context_length": 131072,
        "thinking": None,
        "capabilities": ["completion", "tools"],
    }
    show.assert_awaited_once_with("http://o", "llama3.1:8b")


async def test_model_info_is_none_when_the_model_is_not_available(monkeypatch):
    monkeypatch.setattr(
        runtime, "get_ollama_url", AsyncMock(side_effect=ModelUnavailableError("downloading"))
    )

    assert await runtime.get_model_info("llama3.1", "8b") is None


async def test_model_info_is_none_when_the_container_is_down(monkeypatch):
    monkeypatch.setattr(runtime, "get_ollama_url", AsyncMock(return_value=None))

    assert await runtime.get_model_info("llama3.1", "8b") is None


async def test_model_info_is_none_when_ollama_cannot_describe_the_model(monkeypatch):
    monkeypatch.setattr(runtime, "get_ollama_url", AsyncMock(return_value=("http://o", "llama3.1:8b")))
    monkeypatch.setattr(ollama_api, "show_model", AsyncMock(return_value=None))

    assert await runtime.get_model_info("llama3.1", "8b") is None


def test_model_can_prefers_the_models_own_answer():
    info = {"context_length": 8192, "thinking": None, "capabilities": ["completion", "tools"]}

    assert runtime.model_can("tools", info, catalog_flag=False) is True
    assert runtime.model_can("vision", info, catalog_flag=True) is False
    assert runtime.model_can("tools", None, catalog_flag=True) is True
    assert runtime.model_can("tools", {**info, "capabilities": []}, catalog_flag=False) is False


def _info(values, default):
    return {"context_length": 8192, "thinking": {"values": values, "default": default}}


def test_resolve_reasoning_is_none_only_for_models_that_cannot_think():
    assert runtime.resolve_reasoning("", None) is None
    assert runtime.resolve_reasoning("on", {"context_length": 8192, "thinking": None}) is None


def test_resolve_reasoning_never_leaves_a_thinking_model_on_none():
    # None would let Ollama think while langchain-ollama drops the text.
    assert runtime.resolve_reasoning("", _info([False, True], True)) is True
    assert runtime.resolve_reasoning("", _info(["low", "medium", "high"], "medium")) == "medium"


def test_resolve_reasoning_maps_user_choices():
    on_off = _info([False, True], True)
    levels = _info(["low", "medium", "high"], "medium")

    assert runtime.resolve_reasoning("off", on_off) is False
    assert runtime.resolve_reasoning("on", on_off) is True
    assert runtime.resolve_reasoning("high", levels) == "high"
    # Can't switch gpt-oss-style models off — falls back to their default.
    assert runtime.resolve_reasoning("off", levels) == "medium"
    assert runtime.resolve_reasoning("bogus", levels) == "medium"


def test_a_model_that_defaults_to_not_thinking_stays_off_rather_than_becoming_none():
    assert runtime.resolve_reasoning("", _info([False, True], False)) is False


# GPU share


async def test_gpu_share_is_none_for_cpu_only_containers(monkeypatch):
    container = MagicMock(status="running", attrs={"Config": {"Labels": {"reichat.gpu": "false"}}})
    monkeypatch.setattr(ContainerManager, "get_container", AsyncMock(return_value=container))

    assert await runtime.get_gpu_share("llama3.1", "8b") is None


async def test_gpu_share_is_none_when_the_container_is_not_running(monkeypatch):
    monkeypatch.setattr(ContainerManager, "get_container", AsyncMock(return_value=None))

    assert await runtime.get_gpu_share("llama3.1", "8b") is None


async def test_gpu_share_reads_the_loaded_model_from_ps(monkeypatch):
    container = fake_container(labels={**managed_labels(port="11500"), "reichat.gpu": "true"})
    monkeypatch.setattr(ContainerManager, "get_container", AsyncMock(return_value=container))
    running = AsyncMock(
        return_value=[
            {
                "name": "llama3.1:8b",
                "size": 1000,
                "size_vram": 750,
                "context_length": 8192,
                "expires_at": None,
            }
        ]
    )
    monkeypatch.setattr(ollama_api, "running_models", running)

    assert await runtime.get_gpu_share("llama3.1", "8b") == 0.75
    running.assert_awaited_once_with("http://127.0.0.1:11500")


async def test_gpu_share_is_none_when_the_model_is_not_loaded(monkeypatch):
    container = fake_container(labels={**managed_labels(), "reichat.gpu": "true"})
    monkeypatch.setattr(ContainerManager, "get_container", AsyncMock(return_value=container))
    monkeypatch.setattr(ollama_api, "running_models", AsyncMock(return_value=[]))

    assert await runtime.get_gpu_share("llama3.1", "8b") is None


async def test_gpu_share_on_the_users_own_ollama_trusts_ps(external_mode, monkeypatch):
    running = AsyncMock(
        return_value=[
            {
                "name": "llama3.2:3b",
                "size": 1000,
                "size_vram": 500,
                "context_length": None,
                "expires_at": None,
            }
        ]
    )
    monkeypatch.setattr(ollama_api, "running_models", running)

    assert await runtime.get_gpu_share("llama3.2", "3b") == 0.5
    running.assert_awaited_once_with(EXTERNAL)


async def test_a_cpu_only_users_own_ollama_reports_no_gpu_share(external_mode, monkeypatch):
    running = AsyncMock(
        return_value=[
            {"name": "llama3.2:3b", "size": 1000, "size_vram": 0, "context_length": None, "expires_at": None}
        ]
    )
    monkeypatch.setattr(ollama_api, "running_models", running)

    assert await runtime.get_gpu_share("llama3.2", "3b") is None
