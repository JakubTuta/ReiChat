"""containers/install.py's worker steps, run inline against a mocked
ContainerManager and mocked Ollama — no Docker, no Ollama."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import runtime_settings
from app.containers import install, service
from app.containers.manager import ContainerError, ContainerManager
from app.db import models
from app.realtime import progress
from app.realtime.hub import hub

EXTERNAL = "http://host.docker.internal:11434"


@pytest.fixture
def quiet(monkeypatch):
    monkeypatch.setattr(install, "_sync_custom_model_capabilities", AsyncMock())


@pytest.fixture
def manager(monkeypatch, quiet):
    """The container manager's I/O methods, scripted for a healthy host that
    already has the model's container."""
    mocks = SimpleNamespace(
        ensure_connected=AsyncMock(return_value=True),
        get_container=AsyncMock(return_value=MagicMock(id="c1")),
        ensure_ollama_image=AsyncMock(return_value=True),
        create_and_start_container=AsyncMock(return_value=MagicMock(id="c2")),
        image_id=AsyncMock(),
        pull_ollama_image=AsyncMock(),
        list_managed=AsyncMock(return_value=[]),
        recreate_on_current_image=AsyncMock(),
    )
    for name, mock in vars(mocks).items():
        monkeypatch.setattr(ContainerManager, name, mock)
    monkeypatch.setattr(ContainerManager, "base_url_for", staticmethod(lambda container: "http://ollama"))
    return mocks


@pytest.fixture
def ollama(monkeypatch):
    mocks = SimpleNamespace(
        wait_until_ready=AsyncMock(return_value=True),
        is_model_installed=AsyncMock(return_value=False),
        pull_model=AsyncMock(return_value=(True, None)),
        version=AsyncMock(return_value="0.12.0"),
    )
    for name, mock in vars(mocks).items():
        monkeypatch.setattr(install.ollama_api, name, mock)
    return mocks


@pytest.fixture
async def events():
    hub.bind(asyncio.get_running_loop())
    queue = hub.subscribe()

    async def received():
        await asyncio.sleep(0)
        out = []
        while not queue.empty():
            out.append(queue.get_nowait())
        return out

    return received


async def test_resume_on_existing_container_pulls_missing_model(manager, ollama):
    ollama.is_model_installed.return_value = False

    await install._install("llama3.1", "8b", "llama3.1_8b")

    manager.ensure_ollama_image.assert_not_awaited()
    ollama.pull_model.assert_awaited_once()
    assert service.is_marked_installed(manager.create_and_start_container.return_value) is True


async def test_resume_when_model_already_present_skips_the_pull(manager, ollama):
    ollama.is_model_installed.return_value = True

    await install._install("llama3.1", "8b", "llama3.1_8b")

    ollama.pull_model.assert_not_awaited()
    assert service.is_marked_installed(manager.create_and_start_container.return_value) is True


async def test_failed_pull_raises_with_resume_hint(manager, ollama):
    ollama.pull_model.return_value = (False, "disk full")

    with pytest.raises(ContainerError, match="disk full.*Resume download"):
        await install._install("llama3.1", "8b", "llama3.1_8b")


async def test_new_install_pulls_base_image_first(manager, ollama):
    manager.get_container.return_value = None

    await install._install("llama3.1", "8b", "llama3.1_8b")

    manager.ensure_ollama_image.assert_awaited_once()
    manager.create_and_start_container.assert_awaited_once_with("llama3.1", "8b")
    ollama.pull_model.assert_awaited_once()


async def test_install_without_docker_says_to_start_it(manager, ollama):
    manager.ensure_connected.return_value = False

    with pytest.raises(ContainerError, match="Docker isn't running"):
        await install._install("llama3.1", "8b", "llama3.1_8b")


async def test_a_model_update_pulls_even_when_the_model_is_already_there(manager, ollama):
    ollama.is_model_installed.return_value = True

    await install._install("llama3.1", "8b", "llama3.1_8b", force_pull=True)

    ollama.pull_model.assert_awaited_once()


async def test_a_finished_install_records_what_the_model_can_do(manager, ollama):
    await install._install("llama3.1", "8b", "llama3.1_8b")

    install._sync_custom_model_capabilities.assert_awaited_once_with("llama3.1", "8b", "http://ollama")


async def test_installing_onto_the_users_own_ollama_pulls_straight_there(monkeypatch, manager, ollama):
    monkeypatch.setattr(runtime_settings, "external_ollama_url", AsyncMock(return_value=EXTERNAL))

    await install._install("llama3.2", "3b", "llama3.2_3b")

    ollama.pull_model.assert_awaited_once()
    assert ollama.pull_model.await_args.args[:3] == (EXTERNAL, "llama3.2", "3b")
    manager.create_and_start_container.assert_not_awaited()


def recording_spawn(monkeypatch):
    spawned = []

    def spawn(coroutine, name):
        coroutine.close()
        spawned.append(name)

    monkeypatch.setattr(install.tasks, "spawn", spawn)
    return spawned


def test_start_install_refuses_a_second_concurrent_install(monkeypatch):
    spawned = recording_spawn(monkeypatch)

    assert install.start_install("dup-model", "1b") is True
    assert install.start_install("dup-model", "1b") is False

    assert spawned == ["install-dup-model_1b"]


def test_start_model_update_refuses_while_the_model_is_installing(monkeypatch):
    spawned = recording_spawn(monkeypatch)
    install.start_install("dup-model", "1b")

    assert install.start_model_update("dup-model", "1b") is False
    assert spawned == ["install-dup-model_1b"]


async def test_a_started_install_runs_and_leaves_nothing_in_progress(manager, ollama):
    ollama.is_model_installed.return_value = True
    ollama.pull_model.return_value = (True, None)

    assert install.start_install("llama3.1", "8b") is True
    assert progress.is_in_progress("llama3.1_8b")
    await asyncio.gather(*[t for t in asyncio.all_tasks() if t.get_name() == "install-llama3.1_8b"])

    assert not progress.is_in_progress("llama3.1_8b")


async def test_a_container_error_reaches_the_user_and_clears_the_progress(manager, ollama, events):
    manager.ensure_connected.return_value = False
    progress.try_claim("llama3.1_8b", "Starting…")

    await install._install_worker("llama3.1", "8b", "llama3.1_8b")

    sent = await events()
    assert {
        "type": "container_update",
        "container": "llama3.1_8b",
        "error": "Docker isn't running. Start Docker Desktop and try again.",
    } in sent
    assert not progress.is_in_progress("llama3.1_8b")


async def test_an_unexpected_crash_reaches_the_user_and_is_logged(manager, ollama, events, caplog):
    manager.ensure_connected.side_effect = RuntimeError("kaboom")

    await install._install_worker("llama3.1", "8b", "llama3.1_8b")

    errors = [event["error"] for event in await events() if "error" in event]
    assert errors == ["Something went wrong while setting up the model: kaboom"]
    assert "Install worker crashed" in caplog.text


async def test_a_cancelled_install_clears_its_progress_and_stays_quiet(manager, ollama, events):
    started = asyncio.Event()

    async def hang(*args):
        started.set()
        await asyncio.sleep(60)

    ollama.wait_until_ready.side_effect = hang
    progress.try_claim("llama3.1_8b", "Starting…")
    task = asyncio.create_task(install._install_worker("llama3.1", "8b", "llama3.1_8b"))
    await started.wait()

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert not progress.is_in_progress("llama3.1_8b")
    assert [event for event in await events() if event["type"] in ("snapshot", "container_update")] == []


async def test_ollama_update_is_a_no_op_when_the_image_did_not_change(manager, monkeypatch):
    manager.image_id.return_value = "sha256:same"
    infos = []
    monkeypatch.setattr(install, "_report_info", infos.append)
    progress.try_claim(install.UPDATE_KEY, "x")

    await install._ollama_update_worker()

    assert infos == ["Ollama is already up to date."]
    manager.recreate_on_current_image.assert_not_awaited()
    assert not progress.is_in_progress(install.UPDATE_KEY)


async def test_ollama_update_moves_every_model_and_reports_the_ones_it_could_not(manager, monkeypatch):
    manager.image_id.side_effect = ["sha256:old", "sha256:new"]
    good, legacy = MagicMock(id="g"), MagicMock(id="l")
    good.name, legacy.name = "llama3.1_8b", "old_1b"
    manager.list_managed.return_value = [good, legacy]
    service.mark_installed(good)
    service.mark_installed(legacy)

    async def recreate(container):
        if container is legacy:
            raise ContainerError("no volume")

    manager.recreate_on_current_image.side_effect = recreate
    infos = []
    monkeypatch.setattr(install, "_report_info", infos.append)
    progress.try_claim(install.UPDATE_KEY, "x")

    await install._ollama_update_worker()

    assert manager.recreate_on_current_image.await_count == 2
    assert "old_1b" in infos[0]
    assert "llama3.1_8b" not in infos[0]
    # Replaced containers have to prove they hold their model again.
    assert service.is_marked_installed(good) is False
    assert service.is_marked_installed(legacy) is False


class TestCapabilitySync:
    """What the installed model says it can do overrides the ollama.com listing."""

    @pytest.fixture
    async def catalog(self, db: AsyncSession):
        model = models.AIModel(name="qwen3", model="qwen3", description="", index=1)
        custom = models.AIModel(name="mine", model="hf.co/me/mine", description="", index=2, is_custom=True)
        db.add_all([model, custom])
        await db.flush()
        db.add_all(
            [
                models.AIModelVersion(ai_model_id=model.id, parameters="8b", size="5 GB"),
                models.AIModelVersion(ai_model_id=custom.id, parameters="Q4", size="unknown"),
            ]
        )
        await db.commit()
        return db

    @pytest.fixture
    def model_says(self, monkeypatch):
        async def configure(info, sizes=None):
            monkeypatch.setattr(install.ollama_api, "show_model", AsyncMock(return_value=info))
            monkeypatch.setattr(install.ollama_api, "installed_model_sizes", AsyncMock(return_value=sizes))

        return configure

    async def row(self, db: AsyncSession, model: str) -> models.AIModel:
        db.expire_all()
        return (await db.execute(select(models.AIModel).where(models.AIModel.model == model))).scalar_one()

    async def test_the_models_own_capabilities_replace_the_catalogs(self, catalog, model_says):
        await model_says({"capabilities": ["completion", "tools", "vision"]})

        await install._sync_custom_model_capabilities("qwen3", "8b", "http://ollama")

        model = await self.row(catalog, "qwen3")
        assert (model.can_use_tools, model.can_process_image, model.is_embedding) == (True, True, False)

    async def test_an_embedding_only_model_is_flagged(self, catalog, model_says):
        await model_says({"capabilities": ["embedding"]})

        await install._sync_custom_model_capabilities("qwen3", "8b", "http://ollama")

        assert (await self.row(catalog, "qwen3")).is_embedding is True

    async def test_a_custom_models_size_is_recorded_from_the_install(self, catalog, model_says):
        await model_says({"capabilities": ["completion"]}, sizes={"hf.co/me/mine:Q4": 4_700_000_000})

        await install._sync_custom_model_capabilities("hf.co/me/mine", "Q4", "http://ollama")

        version = (
            await catalog.execute(
                select(models.AIModelVersion).where(models.AIModelVersion.parameters == "Q4")
            )
        ).scalar_one()
        await catalog.refresh(version)
        assert version.size_bytes == 4_700_000_000
        assert version.size == "4.4 GB"
