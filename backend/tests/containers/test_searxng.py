import asyncio
import io
import tarfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.containers import searxng
from app.containers.manager import ContainerError, ContainerManager
from app.core.config import get_settings
from app.realtime import progress
from app.realtime.hub import hub


@pytest.fixture
def searxng_container(monkeypatch):
    """What Docker reports for the SearXNG container; mutate .status or set get_container to None."""
    container = MagicMock(status="running")
    get_container = AsyncMock(return_value=container)
    monkeypatch.setattr(ContainerManager, "get_container", get_container)
    container.get_container = get_container
    return container


async def test_status_is_installing_while_setup_is_in_progress(searxng_container):
    progress.try_claim(searxng.PROGRESS_KEY, "Setting up web search…")

    assert await searxng.status() == "installing"


async def test_status_is_off_when_there_is_no_container(searxng_container):
    searxng_container.get_container.return_value = None

    assert await searxng.status() == "off"


@pytest.mark.parametrize(("docker_status", "expected"), [("running", "running"), ("exited", "stopped")])
async def test_status_follows_the_container(searxng_container, docker_status, expected):
    searxng_container.status = docker_status

    assert await searxng.status() == expected


async def test_base_url_is_none_unless_it_is_running(searxng_container):
    searxng_container.status = "exited"
    assert await searxng.base_url() is None

    searxng_container.get_container.return_value = None
    assert await searxng.base_url() is None


async def test_base_url_is_the_published_loopback_port_from_source(searxng_container):
    assert await searxng.base_url() == "http://127.0.0.1:8888"


async def test_base_url_is_the_container_name_inside_docker(searxng_container, monkeypatch):
    monkeypatch.setenv("DOCKER", "true")
    get_settings.cache_clear()
    try:
        assert await searxng.base_url() == "http://reichat-searxng:8080"
    finally:
        get_settings.cache_clear()


def test_the_settings_it_installs_allow_json_results():
    with tarfile.open(fileobj=io.BytesIO(searxng._settings_archive())) as archive:
        member = archive.getmember("settings.yml")
        content = archive.extractfile(member).read().decode()

    assert member.mode == 0o644
    assert "- json" in content
    assert "limiter: false" in content


def received_after(action):
    """Runs `action` and returns every event the hub carried meanwhile."""

    async def run():
        hub.bind(asyncio.get_running_loop())
        queue = hub.subscribe()
        await action()
        await asyncio.sleep(0)
        out = []
        while not queue.empty():
            out.append(queue.get_nowait())
        return out

    return run()


async def test_enabling_installs_the_service_container_and_announces_it(monkeypatch):
    ensure = AsyncMock()
    monkeypatch.setattr(ContainerManager, "ensure_service_running", ensure)
    progress.try_claim(searxng.PROGRESS_KEY, "x")

    sent = await received_after(searxng._enable_worker)

    kwargs = ensure.await_args.kwargs
    assert kwargs["image"] == "searxng/searxng:latest"
    assert kwargs["name"] == "reichat-searxng"
    assert (kwargs["internal_port"], kwargs["host_port"]) == (8080, 8888)
    assert kwargs["labels"] == {"reichat.service": "searxng"}
    assert kwargs["config_dir"] == "/etc/searxng"
    assert len(kwargs["environment"]["SEARXNG_SECRET"]) == 64
    assert kwargs["failure"] == "Couldn't set up web search"
    assert sent == [
        {"type": "container_update", "container": "web-search", "info": "Web search is ready."},
    ]
    assert not progress.is_in_progress(searxng.PROGRESS_KEY)


async def test_a_container_error_while_enabling_reaches_the_user(monkeypatch):
    monkeypatch.setattr(
        ContainerManager,
        "ensure_service_running",
        AsyncMock(side_effect=ContainerError("Docker isn't running. Start Docker Desktop and try again.")),
    )
    progress.try_claim(searxng.PROGRESS_KEY, "x")

    sent = await received_after(searxng._enable_worker)

    assert sent == [
        {
            "type": "container_update",
            "container": "web-search",
            "error": "Docker isn't running. Start Docker Desktop and try again.",
        }
    ]
    assert not progress.is_in_progress(searxng.PROGRESS_KEY)


async def test_an_unexpected_crash_while_enabling_is_logged_and_reaches_the_user(monkeypatch, caplog):
    monkeypatch.setattr(
        ContainerManager, "ensure_service_running", AsyncMock(side_effect=RuntimeError("kaboom"))
    )
    progress.try_claim(searxng.PROGRESS_KEY, "x")

    sent = await received_after(searxng._enable_worker)

    assert sent[0]["error"] == "Setting up web search failed: kaboom"
    assert "Enabling web search crashed" in caplog.text
    assert not progress.is_in_progress(searxng.PROGRESS_KEY)


async def test_enabling_twice_at_once_only_starts_one_setup(monkeypatch):
    spawned = []

    def spawn(coroutine, name):
        coroutine.close()
        spawned.append(name)

    monkeypatch.setattr(searxng.tasks, "spawn", spawn)

    assert searxng.start_enable() is True
    assert searxng.start_enable() is False
    assert spawned == ["searxng-enable"]


async def test_disabling_removes_the_container(monkeypatch):
    remove = AsyncMock()
    monkeypatch.setattr(ContainerManager, "force_remove_container", remove)

    await searxng.disable()

    remove.assert_awaited_once_with("reichat-searxng", failure="Couldn't turn web search off")
