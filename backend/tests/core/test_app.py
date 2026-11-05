import httpx
import pytest
from asgi_lifespan import LifespanManager

from app.core.config import get_settings
from app.main import create_app


async def test_health_check(client: httpx.AsyncClient):
    response = await client.get("/health/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_api_docs_are_off_by_default(client: httpx.AsyncClient):
    assert (await client.get("/docs")).status_code == 404
    assert (await client.get("/openapi.json")).status_code == 404


async def test_api_docs_are_served_when_debug_is_on(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DEBUG", "true")
    get_settings.cache_clear()
    try:
        async with LifespanManager(create_app()) as manager:
            transport = httpx.ASGITransport(app=manager.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
                assert (await client.get("/openapi.json")).status_code == 200
    finally:
        get_settings.cache_clear()
