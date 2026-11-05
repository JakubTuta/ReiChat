import asyncio
from unittest.mock import AsyncMock

import pytest
from httpx_ws import WebSocketDisconnect

from app.containers import socket as containers_socket
from app.realtime import progress
from app.realtime.hub import hub
from tests.conftest import ConnectWs

CONTAINERS = [
    {"name": "llama3.1_8b", "status": "running", "port": "11500", "environment": {"model": "llama3.1"}}
]


@pytest.fixture(autouse=True)
def listing(monkeypatch):
    mock = AsyncMock(return_value=CONTAINERS)
    monkeypatch.setattr(containers_socket.service, "get_available_containers", mock)
    return mock


async def settle() -> None:
    await asyncio.sleep(0.05)


async def test_a_new_connection_gets_a_snapshot_straight_away(connect_ws: ConnectWs):
    progress.try_claim("llama3.1_8b", "Starting…")

    async with connect_ws("/ws/containers/") as ws:
        frame = await ws.receive_json()

    assert frame["type"] == "snapshot"
    assert frame["containers"] == CONTAINERS
    assert frame["progress"]["llama3.1_8b"]["detail"] == "Starting…"
    assert frame["scrape"]["running"] is False


async def test_progress_events_reach_every_connected_client(connect_ws: ConnectWs):
    async with connect_ws("/ws/containers/") as first, connect_ws("/ws/containers/") as second:
        await first.receive_json()
        await second.receive_json()

        progress.broadcast(
            "model_pull_progress",
            {"type": "model_pull_progress", "percent": 5},
            key="llama3.1_8b",
            force=True,
        )

        assert await first.receive_json() == {"type": "model_pull_progress", "percent": 5}
        assert await second.receive_json() == {"type": "model_pull_progress", "percent": 5}


async def test_a_snapshot_keeps_arriving_while_someone_is_connected(connect_ws: ConnectWs, monkeypatch):
    monkeypatch.setattr(containers_socket, "SNAPSHOT_INTERVAL_SECONDS", 0.02)
    changed = [{"name": "qwen2.5_0.5b", "status": "running", "port": None, "environment": {}}]

    async with connect_ws("/ws/containers/") as ws:
        await ws.receive_json()
        containers_socket.service.get_available_containers.return_value = changed

        frame = await asyncio.wait_for(_next_snapshot_with(ws, changed), timeout=2)

    assert frame["containers"] == changed


async def _next_snapshot_with(ws, containers):
    while True:
        frame = await ws.receive_json()
        if frame["type"] == "snapshot" and frame["containers"] == containers:
            return frame


async def test_the_snapshot_loop_stops_when_the_last_client_leaves(connect_ws: ConnectWs, monkeypatch):
    monkeypatch.setattr(containers_socket, "SNAPSHOT_INTERVAL_SECONDS", 0.01)

    async with connect_ws("/ws/containers/") as ws:
        await ws.receive_json()
        loop_task = containers_socket._snapshot_task
        assert loop_task is not None
        assert not loop_task.done()

    await asyncio.wait_for(asyncio.shield(loop_task), timeout=2)

    assert loop_task.done()


async def test_the_loop_is_not_started_twice_for_two_clients(connect_ws: ConnectWs):
    async with connect_ws("/ws/containers/") as first:
        await first.receive_json()
        loop_task = containers_socket._snapshot_task

        async with connect_ws("/ws/containers/") as second:
            await second.receive_json()

            assert containers_socket._snapshot_task is loop_task


async def test_a_loop_that_ended_is_started_again_for_the_next_client(connect_ws: ConnectWs, monkeypatch):
    monkeypatch.setattr(containers_socket, "SNAPSHOT_INTERVAL_SECONDS", 0.01)
    async with connect_ws("/ws/containers/") as ws:
        await ws.receive_json()
        first_loop = containers_socket._snapshot_task
    await asyncio.wait_for(asyncio.shield(first_loop), timeout=2)

    async with connect_ws("/ws/containers/") as ws:
        await ws.receive_json()

        assert containers_socket._snapshot_task is not first_loop
        assert not containers_socket._snapshot_task.done()


async def test_leaving_stops_the_server_listening_for_that_client(connect_ws: ConnectWs):
    async with connect_ws("/ws/containers/") as ws:
        await ws.receive_json()
        assert hub.subscriber_count == 1

    await settle()

    assert hub.subscriber_count == 0


async def test_a_snapshot_that_cannot_be_built_does_not_close_the_connection(
    connect_ws: ConnectWs, listing: AsyncMock, caplog: pytest.LogCaptureFixture
):
    listing.side_effect = RuntimeError("docker exploded")

    async with connect_ws("/ws/containers/") as ws:
        progress.broadcast("container_update", {"type": "container_update", "info": "still here"}, force=True)

        assert await ws.receive_json() == {"type": "container_update", "info": "still here"}

    assert "Failed to send initial snapshot" in caplog.text


async def test_frames_the_client_sends_are_ignored(connect_ws: ConnectWs):
    async with connect_ws("/ws/containers/") as ws:
        await ws.receive_json()

        await ws.send_text("hello")
        await ws.send_bytes(b"\x00\x01")
        progress.broadcast("container_update", {"type": "container_update", "info": "after"}, force=True)

        assert await ws.receive_json() == {"type": "container_update", "info": "after"}


async def test_a_page_on_another_site_cannot_listen_in(connect_ws: ConnectWs):
    with pytest.RaisesGroup(WebSocketDisconnect):
        async with connect_ws("/ws/containers/", origin="https://evil.example"):
            pass

    assert hub.subscriber_count == 0
