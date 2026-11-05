import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from asgi_lifespan import LifespanManager

from app.core import tasks
from app.main import create_app


async def test_health_check(client: httpx.AsyncClient):
    response = await client.get("/health/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_api_docs_are_off_by_default(client: httpx.AsyncClient):
    assert (await client.get("/docs")).status_code == 404
    assert (await client.get("/openapi.json")).status_code == 404


async def test_shutdown_closes_the_database_connections(monkeypatch: pytest.MonkeyPatch):
    dispose = AsyncMock()
    monkeypatch.setattr("app.main.engine", SimpleNamespace(dispose=dispose))

    async with LifespanManager(create_app()):
        dispose.assert_not_awaited()

    dispose.assert_awaited_once()


async def test_shutdown_stops_background_work():
    async with LifespanManager(create_app()):
        task = tasks.spawn(asyncio.sleep(60), name="long-running")

    assert task.cancelled()
