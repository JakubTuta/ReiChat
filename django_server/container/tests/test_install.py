"""container/install.py's worker steps, run inline against a mocked
ContainerManager — no Docker, no Ollama, no threads."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from container import install
from container.ContainerManager import ContainerError, ContainerManager
from django_app import functions, progress_state
from django_app.functions import ModelUnavailableError


@pytest.fixture
def manager(monkeypatch):
    fake = MagicMock(spec=ContainerManager)
    fake.is_connected.return_value = True
    fake.get_container.return_value = MagicMock(id="c1")
    fake.base_url_for.return_value = "http://ollama"
    fake.wait_for_ollama_ready.return_value = True
    fake.pull_model_with_progress.return_value = (True, None)
    monkeypatch.setattr(install, "ContainerManager", MagicMock(return_value=fake))
    monkeypatch.setattr(install.ContainerManager, "mark_installed", MagicMock())
    monkeypatch.setattr(install, "_progress", lambda *a: None)
    return fake


def test_resume_on_existing_container_pulls_missing_model(manager):
    manager.is_model_installed.return_value = False

    install._install("llama3.1", "8b", "llama3.1_8b")

    manager.ensure_ollama_image.assert_not_called()
    manager.pull_model_with_progress.assert_called_once()
    install.ContainerManager.mark_installed.assert_called_once()


def test_resume_when_model_already_present_skips_the_pull(manager):
    manager.is_model_installed.return_value = True

    install._install("llama3.1", "8b", "llama3.1_8b")

    manager.pull_model_with_progress.assert_not_called()
    install.ContainerManager.mark_installed.assert_called_once()


def test_failed_pull_raises_with_resume_hint(manager):
    manager.is_model_installed.return_value = False
    manager.pull_model_with_progress.return_value = (False, "disk full")

    with pytest.raises(ContainerError, match="disk full.*Resume download"):
        install._install("llama3.1", "8b", "llama3.1_8b")


def test_new_install_pulls_base_image_first(manager):
    manager.get_container.return_value = None
    manager.create_and_start_container.return_value = MagicMock(id="c2")
    manager.ensure_ollama_image.return_value = True
    manager.is_model_installed.return_value = False

    install._install("llama3.1", "8b", "llama3.1_8b")

    manager.ensure_ollama_image.assert_called_once()
    manager.pull_model_with_progress.assert_called_once()


def test_start_install_refuses_a_second_concurrent_install(monkeypatch):
    started = []
    monkeypatch.setattr(install.threading, "Thread", lambda **kw: MagicMock(start=lambda: started.append(kw)))

    try:
        assert install.start_install("dup-model", "1b") is True
        assert install.start_install("dup-model", "1b") is False
    finally:
        progress_state.clear_progress("dup-model_1b")

    assert len(started) == 1


def test_get_ollama_url_reports_unfinished_download(monkeypatch):
    container = MagicMock(status="running")
    monkeypatch.setattr(ContainerManager, "_client", MagicMock())
    monkeypatch.setattr(ContainerManager, "get_container", lambda self, name: container)
    monkeypatch.setattr(ContainerManager, "base_url_for", lambda self, c: "http://ollama")
    monkeypatch.setattr(ContainerManager, "check_model_installed", lambda self, c: False)

    with pytest.raises(ModelUnavailableError, match="Resume download"):
        functions.get_ollama_url("llama3.1", "8b")


def test_get_ollama_url_reports_download_in_progress(monkeypatch):
    monkeypatch.setattr(ContainerManager, "_client", MagicMock())
    progress_state.try_claim("busy-model_1b", "Starting…")

    try:
        with pytest.raises(ModelUnavailableError, match="still downloading"):
            functions.get_ollama_url("busy-model", "1b")
    finally:
        progress_state.clear_progress("busy-model_1b")


def test_ollama_update_is_a_no_op_when_the_image_did_not_change(monkeypatch):
    fake = MagicMock(spec=ContainerManager)
    fake.is_connected.return_value = True
    fake.image_id.return_value = "sha256:same"
    monkeypatch.setattr(install, "ContainerManager", MagicMock(return_value=fake))
    infos = []
    monkeypatch.setattr(install, "_report_info", infos.append)
    monkeypatch.setattr(install, "_progress", lambda *a: None)
    monkeypatch.setattr(install, "_broadcast_snapshot", lambda: None)
    monkeypatch.setattr(install, "close_old_connections", lambda: None)
    progress_state.try_claim(install.UPDATE_KEY, "x")

    install._ollama_update_worker()

    assert infos == ["Ollama is already up to date."]
    fake.recreate_on_current_image.assert_not_called()
    assert not progress_state.is_in_progress(install.UPDATE_KEY)


def test_ollama_update_moves_every_model_and_reports_the_ones_it_could_not(monkeypatch):
    fake = MagicMock(spec=ContainerManager)
    fake.is_connected.return_value = True
    fake.image_id.side_effect = ["sha256:old", "sha256:new"]
    good, legacy = MagicMock(), MagicMock()
    good.name, legacy.name = "llama3.1_8b", "old_1b"
    fake.client = MagicMock()
    fake.client.containers.list.return_value = [good, legacy]

    def _recreate(container):
        if container is legacy:
            raise ContainerError("no volume")

    fake.recreate_on_current_image.side_effect = _recreate
    manager_cls = MagicMock(return_value=fake)
    manager_cls.is_managed_container = lambda c: True
    monkeypatch.setattr(install, "ContainerManager", manager_cls)
    infos = []
    monkeypatch.setattr(install, "_report_info", infos.append)
    monkeypatch.setattr(install, "_progress", lambda *a: None)
    monkeypatch.setattr(install, "_broadcast_snapshot", lambda: None)
    monkeypatch.setattr(install, "close_old_connections", lambda: None)
    progress_state.try_claim(install.UPDATE_KEY, "x")

    install._ollama_update_worker()

    assert fake.recreate_on_current_image.call_count == 2
    assert "old_1b" in infos[0]
