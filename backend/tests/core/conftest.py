from collections.abc import AsyncIterator

import pytest
from fastapi import APIRouter, Depends, FastAPI, HTTPException, WebSocket
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from app.core.security import require_json_body
from app.main import create_app

probe_router = APIRouter(prefix="/probe", dependencies=[Depends(require_json_body)])
# Separate: a router-level HTTP dependency can't be injected into a WebSocket route.
probe_ws_router = APIRouter(prefix="/probe")


class EchoBody(BaseModel):
    name: str
    count: int


@probe_router.get("/ok")
async def probe_ok() -> dict[str, bool]:
    return {"ok": True}


@probe_router.post("/echo", status_code=201)
async def probe_echo(body: EchoBody) -> EchoBody:
    return body


@probe_router.get("/query")
async def probe_query(limit: int) -> dict[str, int]:
    return {"limit": limit}


@probe_router.get("/items/{item_id}")
async def probe_item(item_id: int) -> dict[str, int]:
    return {"item_id": item_id}


@probe_router.get("/conflict")
async def probe_conflict() -> None:
    raise HTTPException(status_code=409, detail="Already exists.")


@probe_router.get("/boom")
async def probe_boom() -> None:
    raise RuntimeError("kaboom")


@probe_router.get("/boom-mid-stream")
async def probe_boom_mid_stream() -> StreamingResponse:
    async def chunks() -> AsyncIterator[bytes]:
        yield b"first"
        raise RuntimeError("died mid-stream")

    return StreamingResponse(chunks())


@probe_ws_router.websocket("/ws")
async def probe_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_text("hi")
    await websocket.close()


@pytest.fixture
def app() -> FastAPI:
    application = create_app()
    application.include_router(probe_router)
    application.include_router(probe_ws_router)
    return application
