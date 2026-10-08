"""The "my own Ollama" runtime, with the external server's HTTP API mocked."""

from __future__ import annotations

import json

import pytest

from container import external, install
from container.ContainerManager import ContainerManager
from django_app import app_settings, functions, models
from django_app.app_settings import external_ollama_url as real_external_ollama_url
from django_app.functions import ModelUnavailableError

EXTERNAL = "http://host.docker.internal:11434"


@pytest.fixture
def external_mode(monkeypatch):
    monkeypatch.setattr(app_settings, "external_ollama_url", lambda: EXTERNAL)


@pytest.mark.django_db
def test_setting_round_trip(monkeypatch):
    monkeypatch.setattr(app_settings, "external_ollama_url", real_external_ollama_url)

    app_settings.set_external_ollama_url("http://127.0.0.1:11434/")
    assert app_settings.external_ollama_url() == "http://127.0.0.1:11434"

    app_settings.set_external_ollama_url(None)
    assert app_settings.external_ollama_url() is None


def test_external_models_are_listed_like_containers(external_mode, monkeypatch):
    monkeypatch.setattr(
        external.ollama_api, "installed_model_sizes", lambda url: {"llama3.2:3b": 2_000, "hf.co/u/repo:Q4_K_M": 1}
    )

    listed = ContainerManager().get_available_containers()

    assert {(c["environment"]["model"], c["environment"]["parameters"], c["status"]) for c in listed} == {
        ("llama3.2", "3b", "running"),
        ("hf.co/u/repo", "Q4_K_M", "running"),
    }
    assert {c["name"] for c in listed} == {"llama3.2_3b", "hf.co-u-repo_Q4_K_M"}


def test_get_ollama_url_points_every_model_at_the_external_server(external_mode, monkeypatch):
    monkeypatch.setattr(external, "has_model", lambda url, model, params: True)

    assert functions.get_ollama_url("llama3.2", "3b") == (EXTERNAL, "llama3.2:3b")


def test_get_ollama_url_explains_a_missing_model(external_mode, monkeypatch):
    monkeypatch.setattr(external, "has_model", lambda url, model, params: False)

    with pytest.raises(ModelUnavailableError, match="isn't installed on your Ollama"):
        functions.get_ollama_url("llama3.2", "3b")


def test_get_ollama_url_explains_an_unreachable_server(external_mode, monkeypatch):
    monkeypatch.setattr(external, "has_model", lambda url, model, params: None)

    with pytest.raises(ModelUnavailableError, match="Can't reach your Ollama"):
        functions.get_ollama_url("llama3.2", "3b")


def test_install_pulls_straight_onto_the_external_server(external_mode, monkeypatch):
    pulls = []
    monkeypatch.setattr(external, "version", lambda url: "0.12.0")
    monkeypatch.setattr(
        ContainerManager, "pull_model_with_progress",
        lambda self, model, params, base_url, progress_cb=None: pulls.append((model, params, base_url)) or (True, None),
    )
    monkeypatch.setattr(install, "_progress", lambda *a: None)
    monkeypatch.setattr(install, "_sync_custom_model_capabilities", lambda *a: None)

    install._install("llama3.2", "3b", "llama3.2_3b")

    assert pulls == [("llama3.2", "3b", EXTERNAL)]


def test_ollama_update_is_refused_for_the_users_own_ollama(external_mode, monkeypatch):
    errors = []
    monkeypatch.setattr(install, "_report_error", lambda key, message: errors.append(message))
    monkeypatch.setattr(install, "_broadcast_snapshot", lambda: None)
    monkeypatch.setattr(install, "close_old_connections", lambda: None)

    install._ollama_update_worker()

    assert "update it the way you installed it" in errors[0]


@pytest.mark.django_db
def test_connection_is_only_saved_when_ollama_answers(client, monkeypatch):
    monkeypatch.setattr(app_settings, "external_ollama_url", real_external_ollama_url)
    monkeypatch.setattr(external, "version", lambda url: None)

    refused = client.put(
        "/docker/ollama-connection/", data=json.dumps({"mode": "external", "url": "http://nowhere:11434"}),
        content_type="application/json",
    )
    assert refused.status_code == 400
    assert app_settings.external_ollama_url() is None

    monkeypatch.setattr(external, "version", lambda url: "0.12.0")
    saved = client.put(
        "/docker/ollama-connection/", data=json.dumps({"mode": "external", "url": "http://host:11434/"}),
        content_type="application/json",
    )
    assert saved.json() == {"mode": "external", "url": "http://host:11434", "version": "0.12.0"}

    back = client.put(
        "/docker/ollama-connection/", data=json.dumps({"mode": "managed"}), content_type="application/json"
    )
    assert back.json()["mode"] == "managed"
    assert app_settings.external_ollama_url() is None


@pytest.mark.django_db
def test_installing_and_removing_go_to_the_external_server(client, external_mode, monkeypatch):
    ai_model = models.AIModel.objects.create(name="llama3.2", model="llama3.2-ext-test", index=1)
    models.AIModelVersion.objects.create(ai_model=ai_model, parameters="3b", size="2 GB")
    started, deleted = [], []
    monkeypatch.setattr(external, "has_model", lambda url, model, params: False)
    monkeypatch.setattr(install, "start_install", lambda m, p: started.append((m, p)) or True)
    monkeypatch.setattr(external, "delete_model", lambda url, m, p: deleted.append((m, p)) or True)

    assert client.post("/docker/container/llama3.2-ext-test?parameters=3b").status_code == 202
    assert client.delete("/docker/container/llama3.2-ext-test?parameters=3b&method=remove").status_code == 200

    assert started == [("llama3.2-ext-test", "3b")]
    assert deleted == [("llama3.2-ext-test", "3b")]


def test_health_check_reports_the_external_server(client, external_mode, monkeypatch):
    monkeypatch.setattr(external, "version", lambda url: None)

    response = client.get("/docker/")

    assert response.status_code == 503
    assert "Can't reach your Ollama" in response.json()["error"]


@pytest.mark.django_db
def test_models_already_on_the_users_ollama_become_selectable(monkeypatch):
    models.AIModel.objects.create(name="llama3.2", model="llama3.2-reg-test", index=1)
    monkeypatch.setattr(
        external.ollama_api, "installed_model_sizes",
        lambda url: {"llama3.2-reg-test:3b": 2_000_000_000, "brand-new-model:latest": 1_000},
    )

    assert external.register_models(EXTERNAL) == 2

    assert models.AIModelVersion.objects.filter(ai_model__model="llama3.2-reg-test", parameters="3b").exists()
    new = models.AIModel.objects.get(model="brand-new-model")
    assert new.is_custom is True
    assert new.versions.get().size_bytes == 1_000
    assert external.register_models(EXTERNAL) == 0
