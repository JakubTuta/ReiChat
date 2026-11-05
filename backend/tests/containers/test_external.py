"""The "my own Ollama" runtime, with the external server's HTTP API mocked."""

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.containers import external
from app.db import models
from app.realtime import progress

EXTERNAL = "http://host.docker.internal:11434"


@pytest.fixture
def installed_on_external(monkeypatch: pytest.MonkeyPatch):
    sizes: dict[str, int] | None = {}

    async def installed_model_sizes(url):
        return sizes

    monkeypatch.setattr(external.ollama_api, "installed_model_sizes", installed_model_sizes)

    def set_sizes(value):
        nonlocal sizes
        sizes = value

    return set_sizes


async def test_external_models_are_listed_like_containers(installed_on_external):
    installed_on_external({"llama3.2:3b": 2_000, "hf.co/u/repo:Q4_K_M": 1})

    listed = await external.list_models(EXTERNAL)

    assert listed == [
        {
            "name": "llama3.2_3b",
            "status": "running",
            "port": None,
            "environment": {"model": "llama3.2", "parameters": "3b"},
        },
        {
            "name": "hf.co-u-repo_Q4_K_M",
            "status": "running",
            "port": None,
            "environment": {"model": "hf.co/u/repo", "parameters": "Q4_K_M"},
        },
    ]


async def test_a_model_without_a_tag_is_listed_as_latest(installed_on_external):
    installed_on_external({"nomic-embed-text": 1})

    [entry] = await external.list_models(EXTERNAL)

    assert entry["environment"] == {"model": "nomic-embed-text", "parameters": "latest"}


async def test_nothing_is_listed_when_the_users_ollama_does_not_answer(installed_on_external):
    installed_on_external(None)

    assert await external.list_models(EXTERNAL) == []


async def test_a_download_that_has_not_reached_the_tag_list_yet_is_shown_as_pulling(installed_on_external):
    installed_on_external({"llama3.2:3b": 2_000})
    progress.try_claim("brand-new_1b", "Starting…")
    progress.try_claim("llama3.2_3b", "Starting…")

    listed = {entry["name"]: entry for entry in await external.list_models(EXTERNAL)}

    assert listed["brand-new_1b"]["status"] == "pulling_model"
    assert listed["brand-new_1b"]["environment"] == {"model": "brand-new", "parameters": "1b"}
    assert listed["llama3.2_3b"]["status"] == "pulling_model"


async def test_other_progress_keys_are_not_mistaken_for_models(installed_on_external):
    installed_on_external({})
    progress.try_claim("ollama-update", "Checking…")
    progress.try_claim("web-search", "Setting up…")

    assert await external.list_models(EXTERNAL) == []


async def test_has_model_checks_the_exact_tag(installed_on_external):
    installed_on_external({"llama3.2:3b": 1})

    assert await external.has_model(EXTERNAL, "llama3.2", "3b") is True
    assert await external.has_model(EXTERNAL, "llama3.2", "1b") is False


async def test_has_model_is_none_when_the_users_ollama_does_not_answer(installed_on_external):
    installed_on_external(None)

    assert await external.has_model(EXTERNAL, "llama3.2", "3b") is None


async def test_models_already_on_the_users_ollama_become_selectable(db: AsyncSession, installed_on_external):
    db.add(models.AIModel(name="llama3.2", model="llama3.2", description="", index=1))
    await db.commit()
    installed_on_external({"llama3.2:3b": 2_000_000_000, "brand-new-model:latest": 1_000})

    assert await external.register_models(db, EXTERNAL) == 2

    versions = (
        await db.execute(
            select(
                models.AIModel.model, models.AIModelVersion.parameters, models.AIModelVersion.size_bytes
            ).join(models.AIModelVersion, models.AIModelVersion.ai_model_id == models.AIModel.id)
        )
    ).all()
    assert {(v.model, v.parameters, v.size_bytes) for v in versions} == {
        ("llama3.2", "3b", 2_000_000_000),
        ("brand-new-model", "latest", 1_000),
    }
    new = (
        await db.execute(select(models.AIModel).where(models.AIModel.model == "brand-new-model"))
    ).scalar_one()
    assert new.is_custom is True
    assert new.name == "brand-new-model"
    assert new.description == "Found on your Ollama."
    existing = (
        await db.execute(select(models.AIModel).where(models.AIModel.model == "llama3.2"))
    ).scalar_one()
    assert existing.is_custom is False


async def test_registering_twice_adds_nothing_the_second_time(db: AsyncSession, installed_on_external):
    installed_on_external({"llama3.2:3b": 2_000, "brand-new-model:latest": 1_000})

    assert await external.register_models(db, EXTERNAL) == 2
    assert await external.register_models(db, EXTERNAL) == 0

    assert (await db.execute(select(func.count()).select_from(models.AIModel))).scalar_one() == 2


async def test_a_namespaced_model_gets_its_short_name_for_display(db: AsyncSession, installed_on_external):
    installed_on_external({"hf.co/user/Repo-GGUF:Q4_K_M": 5})

    await external.register_models(db, EXTERNAL)

    model = (await db.execute(select(models.AIModel))).scalar_one()
    assert (model.model, model.name) == ("hf.co/user/Repo-GGUF", "Repo-GGUF")


async def test_two_tags_of_one_new_model_make_one_model_with_two_versions(
    db: AsyncSession, installed_on_external
):
    installed_on_external({"brand-new:1b": 1, "brand-new:3b": 3})

    assert await external.register_models(db, EXTERNAL) == 2

    assert (await db.execute(select(func.count()).select_from(models.AIModel))).scalar_one() == 1
    assert (await db.execute(select(func.count()).select_from(models.AIModelVersion))).scalar_one() == 2


async def test_a_new_tag_of_a_known_model_counts_as_added(db: AsyncSession, installed_on_external):
    db.add(models.AIModel(name="x", model="known", description="", index=1))
    await db.commit()
    installed_on_external({"known:7b": 7})

    assert await external.register_models(db, EXTERNAL) == 1


async def test_nothing_is_registered_when_the_users_ollama_does_not_answer(
    db: AsyncSession, installed_on_external
):
    installed_on_external(None)

    assert await external.register_models(db, EXTERNAL) == 0
