from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

import httpx
import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from httpx_ws import AsyncWebSocketSession, aconnect_ws
from httpx_ws.transport import ASGIWebSocketTransport
from starlette.types import ASGIApp

from app.main import create_app

FRONTEND_ORIGIN = "http://localhost:3000"

ConnectWs = Callable[..., AbstractAsyncContextManager[AsyncWebSocketSession]]


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
