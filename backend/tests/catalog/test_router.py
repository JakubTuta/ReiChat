import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.catalog import refresh
from app.db import models
from app.realtime import progress

pytestmark = pytest.mark.usefixtures("db")


async def add_model(db: AsyncSession, name: str, **fields) -> models.AIModel:
    model = models.AIModel(
        name=name, model=name, description=fields.pop("description", ""), index=1, **fields
    )
    db.add(model)
    await db.commit()
    return model


async def model_count(db: AsyncSession) -> int:
    return (await db.execute(select(func.count()).select_from(models.AIModel))).scalar_one()


# Listing


async def test_models_are_listed_most_popular_first_with_their_versions(
    client: httpx.AsyncClient, db: AsyncSession
):
    quiet = await add_model(db, "quiet", popularity=10)
    loud = await add_model(db, "loud", popularity=1_000, can_use_tools=True, description="Loud model")
    db.add_all(
        [
            models.AIModelVersion(
                ai_model_id=loud.id, parameters="8b", size="4.6 GB", size_bytes=4_920_000_000
            ),
            models.AIModelVersion(ai_model_id=loud.id, parameters="70b", size="40 GB"),
        ]
    )
    await db.commit()

    response = await client.get("/ai-models/")

    assert response.status_code == 200
    body = response.json()
    assert [m["model"] for m in body] == ["loud", "quiet"]
    assert body[0] == {
        "id": loud.id,
        "name": "loud",
        "model": "loud",
        "description": "Loud model",
        "popularity": 1_000,
        "can_process_image": False,
        "is_embedding": False,
        "can_use_tools": True,
        "is_custom": False,
        "versions": [
            {"parameters": "8b", "size": "4.6 GB", "size_bytes": 4_920_000_000, "verified_at": None},
            {"parameters": "70b", "size": "40 GB", "size_bytes": None, "verified_at": None},
        ],
        "index": 1,
    }
    assert body[1]["versions"] == []
    assert quiet.id == body[1]["id"]


async def test_an_empty_catalog_is_an_empty_list(client: httpx.AsyncClient):
    response = await client.get("/ai-models/")

    assert response.status_code == 200
    assert response.json() == []


async def test_verified_at_is_an_iso_timestamp(client: httpx.AsyncClient, db: AsyncSession):
    import datetime

    model = await add_model(db, "m")
    db.add(
        models.AIModelVersion(
            ai_model_id=model.id,
            parameters="8b",
            size="1 GB",
            verified_at=datetime.datetime(2026, 3, 4, 5, 6, 7, tzinfo=datetime.UTC),
        )
    )
    await db.commit()

    body = (await client.get("/ai-models/")).json()

    assert body[0]["versions"][0]["verified_at"] == "2026-03-04T05:06:07Z"


# Refreshing


@pytest.fixture
def worker(monkeypatch):
    """Stops the real refresh worker from running; records what it was asked."""
    calls = []

    def spawn(coroutine, name):
        coroutine.close()
        calls.append(name)

    monkeypatch.setattr(refresh.tasks, "spawn", spawn)
    return calls


async def test_a_refresh_is_started_in_the_background(client: httpx.AsyncClient, worker):
    response = await client.put("/ai-models/?minPullCount=1000")

    assert response.status_code == 202
    assert response.json() == {"status": "Catalog refresh started"}
    assert worker == ["catalog-refresh"]
    assert progress.get_snapshot()["scrape"]["running"] is True


async def test_a_second_refresh_while_one_runs_is_a_409(client: httpx.AsyncClient, worker):
    await client.put("/ai-models/?minPullCount=1000")

    response = await client.put("/ai-models/?minPullCount=1000")

    assert response.status_code == 409
    assert response.json() == {"error": "A catalog refresh is already running"}
    assert worker == ["catalog-refresh"]


async def test_the_pull_count_floor_is_passed_on(client: httpx.AsyncClient, monkeypatch):
    start = MagicMock(return_value=True)
    monkeypatch.setattr(refresh, "start_refresh", start)

    await client.put("/ai-models/?minPullCount=5000")
    await client.put("/ai-models/?minPullCount=")

    assert [call.args[0] for call in start.call_args_list] == [5000, 200_000]


async def test_a_non_numeric_pull_count_is_a_400(client: httpx.AsyncClient, worker):
    response = await client.put("/ai-models/?minPullCount=lots")

    assert response.status_code == 400
    assert response.json() == {"error": "minPullCount must be a whole number"}
    assert worker == []
    assert progress.get_snapshot()["scrape"]["running"] is False


async def test_the_pull_count_has_to_be_given(client: httpx.AsyncClient, worker):
    response = await client.put("/ai-models/")

    assert response.status_code == 400
    assert response.json() == {
        "error": "Missing required query parameters",
        "missing_params": ["minPullCount"],
    }


async def test_refreshing_for_real_updates_the_catalog_and_reports_progress(
    client: httpx.AsyncClient, db: AsyncSession, monkeypatch
):
    from app.catalog import sync
    from app.catalog.parser import ParsedModel
    from app.catalog.registry import VerifyResult

    parsed = ParsedModel("llama3.1", "desc", 5_000_000, frozenset(), ("8b",))
    monkeypatch.setattr(sync, "fetch_library_html", AsyncMock(return_value="<html></html>"))
    monkeypatch.setattr(sync, "parse_library", lambda html: [parsed])
    monkeypatch.setattr(
        sync.registry,
        "verify_tags",
        AsyncMock(return_value={("llama3.1", "8b"): VerifyResult(status="ok", size_bytes=1024)}),
    )

    response = await client.put("/ai-models/?minPullCount=0")
    await asyncio.wait_for(_the_refresh_task(), timeout=5)

    assert response.status_code == 202
    listed = (await client.get("/ai-models/")).json()
    assert [(m["model"], [v["parameters"] for v in m["versions"]]) for m in listed] == [("llama3.1", ["8b"])]
    assert progress.get_snapshot()["scrape"]["error"] is None


async def _the_refresh_task():
    await asyncio.gather(*(t for t in asyncio.all_tasks() if t.get_name() == "catalog-refresh"))


# Models added by name


@pytest.mark.parametrize(
    ("reference", "model", "tag"),
    [
        (
            "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M",
            "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF",
            "Q4_K_M",
        ),
        ("https://hf.co/user/repo", "hf.co/user/repo", "latest"),
        ("http://hf.co/user/repo:7b", "hf.co/user/repo", "7b"),
        ("someuser/coolmodel:7b", "someuser/coolmodel", "7b"),
        ("  llama3.2:3b  ", "llama3.2", "3b"),
    ],
)
async def test_custom_model_is_added_by_reference(client: httpx.AsyncClient, reference, model, tag):
    response = await client.post("/custom-models/", json={"reference": reference})

    assert response.status_code == 201
    body = response.json()
    assert body["model"] == model
    assert body["name"] == model.rsplit("/", 1)[-1]
    assert body["description"] == "Added by you by name."
    assert body["is_custom"] is True
    assert body["versions"] == [
        {"parameters": tag, "size": "unknown", "size_bytes": None, "verified_at": None}
    ]


async def test_adding_the_same_model_again_is_a_200_and_adds_nothing(
    client: httpx.AsyncClient, db: AsyncSession
):
    await client.post("/custom-models/", json={"reference": "user/model:7b"})

    response = await client.post("/custom-models/", json={"reference": "user/model:7b"})

    assert response.status_code == 200
    assert len(response.json()["versions"]) == 1
    assert await model_count(db) == 1


async def test_adding_another_tag_of_a_known_model_adds_a_version(client: httpx.AsyncClient):
    await client.post("/custom-models/", json={"reference": "user/model:7b"})

    response = await client.post("/custom-models/", json={"reference": "user/model:13b"})

    assert response.status_code == 200
    assert [v["parameters"] for v in response.json()["versions"]] == ["7b", "13b"]


async def test_a_catalog_model_added_by_name_is_not_turned_custom(
    client: httpx.AsyncClient, db: AsyncSession
):
    await add_model(db, "llama3.2")

    response = await client.post("/custom-models/", json={"reference": "llama3.2:1b"})

    assert response.status_code == 200
    assert response.json()["is_custom"] is False
    assert [v["parameters"] for v in response.json()["versions"]] == ["1b"]


@pytest.mark.parametrize(
    "reference", ["rm -rf /; echo", "", "UPPER", "a" * 201, "model:", "/leading/slash", "x:y:z"]
)
async def test_custom_model_rejects_nonsense(client: httpx.AsyncClient, reference):
    response = await client.post("/custom-models/", json={"reference": reference})

    assert response.status_code == 400
    assert "doesn't look like a model name" in response.json()["error"]


async def test_custom_model_needs_a_reference(client: httpx.AsyncClient):
    response = await client.post("/custom-models/", json={})

    assert response.status_code == 400
    assert response.json() == {"error": "Missing required body parameters", "missing_params": ["reference"]}


async def test_a_custom_model_without_chats_can_be_removed(client: httpx.AsyncClient, db: AsyncSession):
    model = await add_model(db, "hf.co/u/repo", is_custom=True)
    db.add(models.AIModelVersion(ai_model_id=model.id, parameters="Q4", size="1 GB"))
    await db.commit()

    response = await client.request("DELETE", "/custom-models/", json={"id": model.id})

    assert response.status_code == 200
    assert response.json() == {"status": "Custom model removed"}
    assert await model_count(db) == 0
    assert (await db.execute(select(func.count()).select_from(models.AIModelVersion))).scalar_one() == 0


async def test_custom_model_with_chats_is_not_deleted(client: httpx.AsyncClient, db: AsyncSession):
    model = await add_model(db, "hf.co/u/repo", is_custom=True)
    db.add(models.ChatHistory(ai_model_id=model.id))
    await db.commit()

    response = await client.request("DELETE", "/custom-models/", json={"id": model.id})

    assert response.status_code == 409
    assert "still has chats" in response.json()["error"]
    assert await model_count(db) == 1
    assert (await db.execute(select(func.count()).select_from(models.ChatHistory))).scalar_one() == 1


async def test_custom_model_used_by_a_document_collection_is_not_deleted(
    client: httpx.AsyncClient, db: AsyncSession
):
    model = await add_model(db, "hf.co/u/embedder", is_custom=True, is_embedding=True)
    db.add(models.DocumentCollection(name="Docs", embedding_model_id=model.id, embedding_parameters="latest"))
    await db.commit()

    response = await client.request("DELETE", "/custom-models/", json={"id": model.id})

    assert response.status_code == 409
    assert "document collection" in response.json()["error"]
    assert await model_count(db) == 1


async def test_a_catalog_model_cannot_be_removed_through_the_custom_endpoint(
    client: httpx.AsyncClient, db: AsyncSession
):
    model = await add_model(db, "llama3.1")

    response = await client.request("DELETE", "/custom-models/", json={"id": model.id})

    assert response.status_code == 404
    assert response.json() == {"error": "Custom model not found"}
    assert await model_count(db) == 1


async def test_removing_a_model_that_does_not_exist_is_a_404(client: httpx.AsyncClient):
    response = await client.request("DELETE", "/custom-models/", json={"id": 999})

    assert response.status_code == 404


async def test_removing_needs_an_id(client: httpx.AsyncClient):
    response = await client.request("DELETE", "/custom-models/", json={})

    assert response.status_code == 400
    assert response.json() == {"error": "Missing required body parameters", "missing_params": ["id"]}


async def test_a_non_numeric_id_is_a_400_not_a_server_error(client: httpx.AsyncClient):
    response = await client.request("DELETE", "/custom-models/", json={"id": "abc"})

    assert response.status_code == 400
    assert response.json()["details"][0]["field"] == "id"
