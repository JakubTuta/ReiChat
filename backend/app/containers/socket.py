import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.containers import service
from app.core import tasks
from app.realtime import progress
from app.realtime.hub import hub

logger = logging.getLogger(__name__)

router = APIRouter()

SNAPSHOT_INTERVAL_SECONDS = 2.0

_snapshot_task: asyncio.Task[None] | None = None


async def build_snapshot() -> dict[str, object]:
    return {
        "type": "snapshot",
        "containers": await service.get_available_containers(),
        **progress.get_snapshot(),
    }


async def _snapshot_loop() -> None:
    # Only runs while someone is listening: it used to run forever, building a
    # fresh Docker client and hitting the Docker socket every 2s with nobody
    # watching.
    while hub.subscriber_count > 0:
        await asyncio.sleep(SNAPSHOT_INTERVAL_SECONDS)
        if hub.subscriber_count == 0:
            return
        try:
            hub.publish(await build_snapshot())
        except Exception as e:
            logger.debug("Snapshot error: %s", e)


def _ensure_snapshot_loop_running() -> None:
    global _snapshot_task
    if _snapshot_task is None or _snapshot_task.done():
        _snapshot_task = tasks.spawn(_snapshot_loop(), name="container-snapshot")


async def _watch_for_disconnect(websocket: WebSocket) -> None:
    """The client sends nothing; this only notices when it goes away, and
    ignores any frame it does send."""
    while (await websocket.receive())["type"] != "websocket.disconnect":
        pass


@router.websocket("/ws/containers/")
async def container_status(websocket: WebSocket) -> None:
    await websocket.accept()
    queue = hub.subscribe()
    _ensure_snapshot_loop_running()
    gone = asyncio.create_task(_watch_for_disconnect(websocket), name="container-socket-disconnect")

    try:
        try:
            await websocket.send_json(await build_snapshot())
        except WebSocketDisconnect:
            return
        except Exception as e:
            logger.warning("Failed to send initial snapshot: %s", e)

        while not gone.done():
            next_event = asyncio.create_task(queue.get())
            await asyncio.wait({next_event, gone}, return_when=asyncio.FIRST_COMPLETED)
            if not next_event.done():
                next_event.cancel()
                break
            try:
                await websocket.send_json(next_event.result())
            except Exception as e:
                logger.debug("Relay error: %s", e)
                break
    finally:
        hub.unsubscribe(queue)
        gone.cancel()
        await asyncio.gather(gone, return_exceptions=True)
