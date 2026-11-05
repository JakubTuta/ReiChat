import pytest

from app import runtime_settings
from app.containers import service
from app.realtime import progress
from tests.containers.conftest import fake_container, managed_labels

EXTERNAL = "http://host.docker.internal:11434"


@pytest.fixture
def tags(monkeypatch: pytest.MonkeyPatch):
    """Scripts what Ollama's /api/tags says; records how often it was asked."""
    state = {"installed": True, "asked": 0}

    async def is_model_installed(base_url, model, parameters):
        state["asked"] += 1
        return state["installed"]

    monkeypatch.setattr(service.ollama_api, "is_model_installed", is_model_installed)
    return state


@pytest.mark.parametrize(
    ("in_progress", "installed", "expected"),
    [
        (True, True, "pulling_model"),
        (False, False, "incomplete"),
        (False, None, "starting"),
    ],
)
async def test_resolve_status_reflects_install_state(tags, in_progress, installed, expected):
    tags["installed"] = installed
    if in_progress:
        progress.try_claim("llama3.1_8b", "Starting…")

    assert await service.resolve_status(fake_container(labels=managed_labels())) == expected


async def test_confirmed_install_is_cached_so_tags_are_not_asked_again(tags):
    container = fake_container(labels=managed_labels())

    assert await service.check_model_installed(container) is True
    assert await service.check_model_installed(container) is True
    assert tags["asked"] == 1


async def test_a_missing_model_is_asked_about_every_time(tags):
    tags["installed"] = False
    container = fake_container(labels=managed_labels())

    assert await service.check_model_installed(container) is False
    assert await service.check_model_installed(container) is False
    assert tags["asked"] == 2


async def test_forgetting_a_container_makes_it_prove_its_model_again(tags):
    container = fake_container(labels=managed_labels())
    await service.check_model_installed(container)

    service.forget_installed(container.id)

    assert service.is_marked_installed(container) is False
    await service.check_model_installed(container)
    assert tags["asked"] == 2


async def test_available_containers_are_this_apps_containers_with_their_resolved_status(
    connected_client, tags
):
    labelled = fake_container(name="llama3.1_8b", labels=managed_labels(port="11500"))
    legacy = fake_container(
        name="qwen2.5_0.5b",
        status="exited",
        env=["model=qwen2.5", "parameters=0.5b", "port=11435"],
        container_id="old",
    )
    unrelated = fake_container(name="postgres", image="pgvector/pgvector:pg16", container_id="pg")
    connected_client.containers.list.return_value = [labelled, legacy, unrelated]

    result = await service.get_available_containers()

    assert result == [
        {
            "name": "llama3.1_8b",
            "status": "running",
            "port": "11500",
            "environment": {"model": "llama3.1", "parameters": "8b"},
        },
        {
            "name": "qwen2.5_0.5b",
            "status": "exited",
            "port": "11435",
            "environment": {"model": "qwen2.5", "parameters": "0.5b"},
        },
    ]


async def test_with_the_users_own_ollama_its_models_are_listed_like_containers(monkeypatch):
    async def external_url():
        return EXTERNAL

    async def installed_model_sizes(url):
        return {"llama3.2:3b": 2_000, "hf.co/u/repo:Q4_K_M": 1}

    monkeypatch.setattr(runtime_settings, "external_ollama_url", external_url)
    monkeypatch.setattr(service.external.ollama_api, "installed_model_sizes", installed_model_sizes)

    listed = await service.get_available_containers()

    assert {(c["environment"]["model"], c["environment"]["parameters"], c["status"]) for c in listed} == {
        ("llama3.2", "3b", "running"),
        ("hf.co/u/repo", "Q4_K_M", "running"),
    }
    assert {c["name"] for c in listed} == {"llama3.2_3b", "hf.co-u-repo_Q4_K_M"}


async def test_removing_a_model_container_forgets_that_it_held_its_model(connected_client, tags):
    container = fake_container(labels=managed_labels())
    await service.check_model_installed(container)
    connected_client.containers.get.side_effect = None
    connected_client.containers.get.return_value = container

    await service.remove_model_container("llama3.1_8b")

    assert service.is_marked_installed(container) is False
