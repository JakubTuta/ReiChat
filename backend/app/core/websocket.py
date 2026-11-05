import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class Client:
    """The browser end of a socket, for handlers that keep working after it
    leaves: a reply that was cancelled by the disconnect is still saved, and
    its code still tries to say how it ended."""

    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket
        self.gone = False

    async def accept(self) -> None:
        await self._websocket.accept()

    async def send(self, payload: dict[str, Any]) -> None:
        # Starlette raises on a send to a closed socket, which would turn a
        # normal "user closed the tab" into a crash in whichever task was
        # mid-reply.
        if self.gone:
            return
        try:
            await self._websocket.send_text(json.dumps(payload))
        except (WebSocketDisconnect, RuntimeError):
            self.gone = True

    async def text_frames(self) -> AsyncIterator[str]:
        """Text frames until the client leaves. The protocol is JSON text only,
        so a binary frame ends the conversation."""
        try:
            while True:
                message = await self._websocket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                text = message.get("text")
                if text is None:
                    await self._close_after_binary_frame()
                    return
                yield text
        finally:
            self.gone = True

    async def _close_after_binary_frame(self) -> None:
        logger.warning("Closing a WebSocket that sent a binary frame")
        try:
            await self._websocket.close(1000)
        except (WebSocketDisconnect, RuntimeError):
            pass
