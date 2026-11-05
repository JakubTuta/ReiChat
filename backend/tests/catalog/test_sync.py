"""refresh_catalog is the fix for the reported bug's root cause: the old
flow deleted every AIModel *before* scraping, so a parse that (through no fault
of its own) returned 0 models — which is exactly what happened once ollama.com
dropped the markup the old scraper keyed off — wiped the entire catalog and, via
CASCADE, every chat.

These tests assert the new rule holds: nothing is ever deleted unless a
*successful* parse proves it's gone, and even then only if it has no chat
history, no document collection and (when checkable) no live container.
"""

from unittest.mock import AsyncMock

import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.catalog import sync
from app.catalog.parser import CatalogParseError, ParsedModel
from app.catalog.registry import VerifyResult
from app.containers.manager import ContainerManager
from app.db import models

pytestmark = pytest.mark.usefixtures("db")


def _parsed(name: str, variants=("8b",), popularity=1_000_000, capabilities=frozenset()) -> ParsedModel:
    return ParsedModel(
        name=name,
        description=f"{name} description",
        popularity=popularity,
        capabilities=capabilities,
        variants=variants,
    )


@pytest.fixture(autouse=True)
def docker(monkeypatch):
    """Pruning asks Docker whether a model has a live container. By default
    Docker is unreachable; tests about that branch say otherwise."""
    state = AsyncMock(return_value=False)
    monkeypatch.setattr(ContainerManager, "ensure_connected", state)
    monkeypatch.setattr(ContainerManager, "get_container", AsyncMock(return_value=None))
    return state


@pytest.fixture
def library(monkeypatch):
    """Scripts ollama.com and the registry for one refresh."""

    def serve(parsed: list[ParsedModel], verified: dict | None = None) -> None:
        monkeypatch.setattr(sync, "fetch_library_html", AsyncMock(return_value="<html></html>"))
        monkeypatch.setattr(sync, "parse_library", lambda html: parsed)
        monkeypatch.setattr(sync.registry, "verify_tags", AsyncMock(return_value=verified or {}))

    return serve


async def add_model(db: AsyncSession, name: str = "llama3.1", **fields) -> models.AIModel:
    model = models.AIModel(
        name=name, model=name, description=fields.pop("description", ""), index=1, **fields
    )
    db.add(model)
    await db.commit()
    return model


async def get_model(db: AsyncSession, name: str) -> models.AIModel | None:
    return (
        await db.execute(
            select(models.AIModel)
            .where(models.AIModel.model == name)
            .options(selectinload(models.AIModel.versions))
            .execution_options(populate_existing=True)
        )
    ).scalar_one_or_none()


async def count(db: AsyncSession, table) -> int:
    return (await db.execute(select(func.count()).select_from(table))).scalar_one()


async def test_parse_failure_leaves_existing_catalog_untouched(db, monkeypatch):
    await add_model(db)
    monkeypatch.setattr(sync, "fetch_library_html", AsyncMock(return_value="<html></html>"))

    def fail(html):
        raise CatalogParseError("0 models parsed")

    monkeypatch.setattr(sync, "parse_library", fail)

    result = await sync.refresh_catalog(min_pull_count=0)

    assert result.error == "0 models parsed"
    assert await count(db, models.AIModel) == 1
    assert (await get_model(db, "llama3.1")).model == "llama3.1"


async def test_network_failure_leaves_existing_catalog_untouched(db, monkeypatch):
    await add_model(db)
    monkeypatch.setattr(
        sync, "fetch_library_html", AsyncMock(side_effect=httpx.ConnectError("connection refused"))
    )

    result = await sync.refresh_catalog(min_pull_count=0)

    assert result.error is not None
    assert "Could not reach ollama.com" in result.error
    assert await count(db, models.AIModel) == 1


async def test_successful_refresh_creates_model_and_verified_variant(db, library):
    library(
        [_parsed("llama3.1", variants=("8b",))],
        {("llama3.1", "8b"): VerifyResult(status="ok", size_bytes=4_700_000_000)},
    )

    result = await sync.refresh_catalog(min_pull_count=0)

    assert result.error is None
    assert result.created == 1
    assert result.variants_verified == 1

    ai_model = await get_model(db, "llama3.1")
    [version] = ai_model.versions
    assert (version.parameters, version.size_bytes, version.size) == ("8b", 4_700_000_000, "4.4 GB")
    assert version.verified_at is not None


async def test_a_new_model_takes_the_parsed_fields(db, library):
    library(
        [_parsed("llava", capabilities=frozenset({"vision", "tools"}), popularity=5_000)],
        {("llava", "8b"): VerifyResult(status="ok", size_bytes=1)},
    )

    await sync.refresh_catalog(min_pull_count=0)

    model = await get_model(db, "llava")
    assert (model.name, model.description, model.popularity) == ("llava", "llava description", 5_000)
    assert (model.can_process_image, model.can_use_tools, model.is_embedding, model.is_custom) == (
        True,
        True,
        False,
        False,
    )


async def test_refresh_persists_tools_capability(db, library):
    library(
        [_parsed("llama3.1", capabilities=frozenset({"tools"}))],
        {("llama3.1", "8b"): VerifyResult(status="ok", size_bytes=4_700_000_000)},
    )

    await sync.refresh_catalog(min_pull_count=0)

    assert (await get_model(db, "llama3.1")).can_use_tools is True


async def test_refresh_updates_tools_capability_on_existing_model(db, library):
    await add_model(db, can_use_tools=True)
    library(
        [_parsed("llama3.1", capabilities=frozenset())],
        {("llama3.1", "8b"): VerifyResult(status="ok", size_bytes=4_700_000_000)},
    )

    result = await sync.refresh_catalog(min_pull_count=0)

    assert result.updated == 1
    assert (await get_model(db, "llama3.1")).can_use_tools is False


async def test_an_existing_model_takes_the_new_description_and_popularity(db, library):
    await add_model(db, description="old", popularity=1)
    library([_parsed("llama3.1", popularity=9_000_000)])

    await sync.refresh_catalog(min_pull_count=0)

    model = await get_model(db, "llama3.1")
    assert (model.description, model.popularity) == ("llama3.1 description", 9_000_000)


async def test_new_models_are_numbered_after_the_existing_ones(db, library):
    await add_model(db, "old")
    library([_parsed("new-a"), _parsed("new-b")])

    await sync.refresh_catalog(min_pull_count=0)

    assert (await get_model(db, "new-a")).index == 2
    assert (await get_model(db, "new-b")).index == 3


async def test_below_min_pull_count_is_not_imported(db, library):
    library([_parsed("obscure-model", popularity=10)])

    result = await sync.refresh_catalog(min_pull_count=1_000_000)

    assert result.created == 0
    assert await get_model(db, "obscure-model") is None


async def test_variant_verified_not_found_is_dropped_but_model_stays(db, library):
    model = await add_model(db)
    db.add(models.AIModelVersion(ai_model_id=model.id, parameters="8b", size="4.4 GB"))
    await db.commit()
    library([_parsed("llama3.1")], {("llama3.1", "8b"): VerifyResult(status="not_found")})

    await sync.refresh_catalog(min_pull_count=0)

    refreshed = await get_model(db, "llama3.1")
    assert refreshed is not None
    assert refreshed.versions == []


async def test_a_variant_that_could_not_be_checked_is_left_as_it_was(db, library):
    model = await add_model(db)
    db.add(models.AIModelVersion(ai_model_id=model.id, parameters="8b", size="4.4 GB", size_bytes=123))
    await db.commit()
    library([_parsed("llama3.1")], {("llama3.1", "8b"): VerifyResult(status="error")})

    result = await sync.refresh_catalog(min_pull_count=0)

    [version] = (await get_model(db, "llama3.1")).versions
    assert (version.size, version.size_bytes) == ("4.4 GB", 123)
    assert result.variants_verified == 0


async def test_a_variant_nobody_checked_is_left_as_it_was(db, library):
    model = await add_model(db)
    db.add(models.AIModelVersion(ai_model_id=model.id, parameters="8b", size="4.4 GB"))
    await db.commit()
    library([_parsed("llama3.1")], {})

    await sync.refresh_catalog(min_pull_count=0)

    assert len((await get_model(db, "llama3.1")).versions) == 1


async def test_an_existing_variant_gets_its_size_and_check_time_refreshed(db, library):
    model = await add_model(db)
    db.add(models.AIModelVersion(ai_model_id=model.id, parameters="8b", size="old"))
    await db.commit()
    library([_parsed("llama3.1")], {("llama3.1", "8b"): VerifyResult(status="ok", size_bytes=2048)})

    await sync.refresh_catalog(min_pull_count=0)

    [version] = (await get_model(db, "llama3.1")).versions
    assert (version.size, version.size_bytes) == ("2.0 KB", 2048)
    assert version.verified_at is not None


async def test_a_new_variant_is_added_to_an_existing_model(db, library):
    model = await add_model(db)
    db.add(models.AIModelVersion(ai_model_id=model.id, parameters="8b", size="4.4 GB"))
    await db.commit()
    library(
        [_parsed("llama3.1", variants=("8b", "70b"))],
        {
            ("llama3.1", "8b"): VerifyResult(status="ok", size_bytes=1),
            ("llama3.1", "70b"): VerifyResult(status="ok", size_bytes=2),
        },
    )

    await sync.refresh_catalog(min_pull_count=0)

    assert sorted(v.parameters for v in (await get_model(db, "llama3.1")).versions) == ["70b", "8b"]


async def test_refreshing_twice_changes_nothing_the_second_time(db, library):
    library([_parsed("llama3.1")], {("llama3.1", "8b"): VerifyResult(status="ok", size_bytes=1)})

    await sync.refresh_catalog(min_pull_count=0)
    second = await sync.refresh_catalog(min_pull_count=0)

    assert second.created == 0
    assert second.updated == 1
    assert await count(db, models.AIModel) == 1
    assert await count(db, models.AIModelVersion) == 1


async def test_progress_is_reported_once_per_verified_tag(db, library, monkeypatch):
    parsed = [_parsed("a", variants=("1b", "2b")), _parsed("b", variants=("3b",))]
    monkeypatch.setattr(sync, "fetch_library_html", AsyncMock(return_value="<html></html>"))
    monkeypatch.setattr(sync, "parse_library", lambda html: parsed)

    async def verify_tags(names_and_tags, on_result=None):
        for name, tag in names_and_tags:
            on_result(name, tag, VerifyResult(status="ok", size_bytes=1))
        return {}

    monkeypatch.setattr(sync.registry, "verify_tags", verify_tags)
    calls = []

    await sync.refresh_catalog(
        min_pull_count=0, progress_cb=lambda done, total, name: calls.append((done, total, name))
    )

    # Three tags verified, but only two models: the bar never passes 100%.
    assert calls == [(1, 2, "a"), (2, 2, "a"), (2, 2, "b")]


async def test_a_refresh_that_fails_halfway_changes_nothing(db, library, monkeypatch):
    await add_model(db, "keeper", popularity=1)
    library([_parsed("keeper", popularity=999), _parsed("brand-new")])
    monkeypatch.setattr(sync, "_prune_missing", AsyncMock(side_effect=RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        await sync.refresh_catalog(min_pull_count=0)

    assert (await get_model(db, "keeper")).popularity == 1
    assert await get_model(db, "brand-new") is None


async def test_prune_never_touches_a_model_with_chat_history(db, library, docker):
    docker.return_value = True
    model = await add_model(db, "discontinued")
    db.add(models.ChatHistory(ai_model_id=model.id, title="An old chat"))
    await db.commit()
    # The model no longer appears on ollama.com at all.
    library([_parsed("some-other-model")])

    result = await sync.refresh_catalog(min_pull_count=0)

    assert result.pruned == 0
    assert await get_model(db, "discontinued") is not None
    assert await count(db, models.ChatHistory) == 1


async def test_prune_never_touches_a_model_a_document_collection_depends_on(db, library, docker):
    docker.return_value = True
    model = await add_model(db, "discontinued-embedder", is_embedding=True)
    db.add(models.DocumentCollection(name="Docs", embedding_model_id=model.id, embedding_parameters="latest"))
    await db.commit()
    library([_parsed("some-other-model")])

    result = await sync.refresh_catalog(min_pull_count=0)

    assert result.error is None
    assert result.pruned == 0
    assert await get_model(db, "discontinued-embedder") is not None


async def test_prune_removes_model_with_no_history_and_no_live_container(db, library, docker):
    docker.return_value = True
    await add_model(db, "discontinued")
    library([_parsed("some-other-model")])

    result = await sync.refresh_catalog(min_pull_count=0)

    assert result.pruned == 1
    assert await get_model(db, "discontinued") is None


async def test_prune_takes_the_models_versions_with_it(db, library, docker):
    docker.return_value = True
    model = await add_model(db, "discontinued")
    db.add(models.AIModelVersion(ai_model_id=model.id, parameters="8b", size="1 GB"))
    await db.commit()
    library([_parsed("some-other-model")])

    await sync.refresh_catalog(min_pull_count=0)

    assert await count(db, models.AIModelVersion) == 0


async def test_prune_leaves_a_model_with_a_live_container(db, library, docker, monkeypatch):
    docker.return_value = True
    model = await add_model(db, "discontinued")
    db.add(models.AIModelVersion(ai_model_id=model.id, parameters="8b", size="1 GB"))
    await db.commit()
    asked = AsyncMock(return_value=object())
    monkeypatch.setattr(ContainerManager, "get_container", asked)
    library([_parsed("some-other-model")])

    result = await sync.refresh_catalog(min_pull_count=0)

    assert result.pruned == 0
    asked.assert_awaited_once_with("discontinued_8b")


async def test_prune_leaves_model_alone_when_docker_unreachable(db, library):
    # Can't prove there's no live container right now — must not guess.
    await add_model(db, "discontinued")
    library([_parsed("some-other-model")])

    result = await sync.refresh_catalog(min_pull_count=0)

    assert result.pruned == 0
    assert await get_model(db, "discontinued") is not None


async def test_prune_leaves_custom_models_alone(db, library, docker):
    docker.return_value = True
    await add_model(db, "hf.co/me/mine", is_custom=True)
    library([_parsed("some-other-model")])

    result = await sync.refresh_catalog(min_pull_count=0)

    assert result.pruned == 0
    assert await get_model(db, "hf.co/me/mine") is not None


async def test_a_model_still_on_ollama_com_is_never_pruned_even_below_the_pull_floor(db, library, docker):
    docker.return_value = True
    await add_model(db, "obscure")
    library([_parsed("obscure", popularity=1)])

    result = await sync.refresh_catalog(min_pull_count=1_000_000)

    assert result.pruned == 0
    assert await get_model(db, "obscure") is not None
