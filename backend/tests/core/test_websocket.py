from typing import Any

import pytest
from fastapi import WebSocketDisconnect

from app.core.websocket import Client


class StubSocket:
    """Just the parts of Starlette's WebSocket that Client touches."""

    def __init__(
        self, incoming: list[dict[str, Any]] | None = None, send_error: Exception | None = None
    ) -> None:
        self.incoming = list(incoming or [])
        self.send_error = send_error
        self.sent: list[str] = []
        self.closed_with: int | None = None

    async def accept(self) -> None:
        pass

    async def send_text(self, text: str) -> None:
        if self.send_error is not None:
            raise self.send_error
        self.sent.append(text)

    async def receive(self) -> dict[str, Any]:
        return self.incoming.pop(0)

    async def close(self, code: int = 1000) -> None:
        self.closed_with = code


def client_for(socket: StubSocket) -> Client:
    return Client(socket)  # type: ignore[arg-type]  # a stub with the same methods


def text(body: str) -> dict[str, Any]:
    return {"type": "websocket.receive", "text": body}


DISCONNECT = {"type": "websocket.disconnect", "code": 1001}


@pytest.mark.parametrize(
    "error",
    [WebSocketDisconnect(1006), RuntimeError('Cannot call "send" once a close message has been sent.')],
)
async def test_a_send_to_a_client_that_left_is_dropped_and_remembered(error: Exception):
    socket = StubSocket(send_error=error)
    client = client_for(socket)

    await client.send({"a": 1})

    assert client.gone is True
    socket.send_error = AssertionError("must not be called again")
    await client.send({"a": 2})


async def test_other_failures_while_sending_are_not_swallowed():
    client = client_for(StubSocket(send_error=ValueError("bad payload")))

    with pytest.raises(ValueError, match="bad payload"):
        await client.send({"a": 1})

    assert client.gone is False


async def test_text_frames_end_when_the_client_disconnects():
    client = client_for(StubSocket([text("one"), text("two"), DISCONNECT]))

    assert [frame async for frame in client.text_frames()] == ["one", "two"]
    assert client.gone is True


async def test_a_binary_frame_closes_the_socket_normally():
    socket = StubSocket([text("one"), {"type": "websocket.receive", "bytes": b"\x00"}, text("never read")])
    client = client_for(socket)

    assert [frame async for frame in client.text_frames()] == ["one"]
    assert socket.closed_with == 1000
    assert client.gone is True
