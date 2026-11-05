import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

import httpx
import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from httpx_ws import AsyncWebSocketSession, aconnect_ws
from httpx_ws.transport import ASGIWebSocketTransport
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.types import ASGIApp

from app import runtime_settings
from app.containers import hardware, service
from app.containers.manager import ContainerManager
from app.core import tasks
from app.db.base import Base
from app.db.session import SessionLocal, engine
from app.llm import ollama_api
from app.main import create_app
from app.realtime import progress
from app.realtime.hub import hub
from tests.database import migrate, recreate_database

FRONTEND_ORIGIN = "http://localhost:3000"

ConnectWs = Callable[..., AbstractAsyncContextManager[AsyncWebSocketSession]]


@pytest.fixture(autouse=True)
async def _reset_process_wide_state(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[None]:
    # Module-level registries live for the whole test run, so one test's
    # leftovers would otherwise leak into the next.
    progress.reset()
    service.reset()
    hardware.reset()
    ollama_api._show_cache.clear()
    monkeypatch.setattr(ContainerManager, "_client", None)
    monkeypatch.setattr(ContainerManager, "_gpu_supported", None)
    yield
    await tasks.cancel_all()
    hub.unbind()


@pytest.fixture(autouse=True)
def _app_managed_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    # Reading the runtime setting is a database query, which most tests here
    # can't make. Default every test to app-managed containers; tests about
    # the user's own Ollama override this.
    async def not_set() -> None:
        return None

    monkeypatch.setattr(runtime_settings, "external_ollama_url", not_set)


@pytest.fixture(scope="session")
def migrated_database() -> None:
    # Built by the real migrations rather than metadata.create_all, so a broken
    # migration fails here and not on someone's first `docker compose up`.
    asyncio.run(recreate_database("chatbot_test"))
    migrate("chatbot_test")


@pytest.fixture
async def db(migrated_database: None) -> AsyncIterator[AsyncSession]:
    # Truncating (not rolling back) so the app's own sessions — request handlers,
    # websocket handlers, background tasks — see exactly what the test wrote.
    # Tables start empty; the seeded catalog is covered by the migration tests.
    tables = ", ".join(table.name for table in Base.metadata.sorted_tables)
    async with engine.begin() as connection:
        await connection.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))
    async with SessionLocal() as session:
        yield session


@pytest.fixture
def app() -> FastAPI:
    return create_app()


@pytest.fixture
async def running_app(app: FastAPI) -> AsyncIterator[ASGIApp]:
    async with LifespanManager(app) as manager:
        yield manager.app


@pytest.fixture
async def client(running_app: ASGIApp) -> AsyncIterator[httpx.AsyncClient]:
    # base_url's host has to be one TrustedHostMiddleware accepts. App and test
    # share one event loop, so nothing here needs a thread or a live server.
    transport = httpx.ASGITransport(app=running_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as http_client:
        yield http_client


@pytest.fixture
def connect_ws(running_app: ASGIApp) -> ConnectWs:
    @asynccontextmanager
    async def connect(
        path: str, *, origin: str | None = FRONTEND_ORIGIN
    ) -> AsyncIterator[AsyncWebSocketSession]:
        # The WebSocket transport owns an anyio task group, which has to be
        # entered and exited in one task. A fixture's setup and teardown run in
        # different tasks, so this is built per connection, inside the test.
        headers = {} if origin is None else {"Origin": origin}
        transport = ASGIWebSocketTransport(app=running_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as ws_client:
            async with aconnect_ws(f"http://localhost{path}", ws_client, headers=headers) as ws:
                yield ws

    return connect
