import asyncio
import logging
import typing

logger = logging.getLogger(__name__)

# A subscriber that stops reading (a stalled browser tab) must not make the
# server buffer events for it forever; past this its events are dropped.
QUEUE_SIZE = 100


class Hub:
    """Fans events out to every connected /ws/containers/ socket.

    It lives inside the one server process, which is all the Redis channel layer
    ever connected: there is a single backend worker, so no second process
    exists to share events with.
    """

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, typing.Any]]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        """Called once the server's event loop is running. Events can be
        published from worker threads (the Docker image-pull callback), and
        hand-over to the loop is the only thread-safe way to reach a queue."""
        self._loop = loop

    def unbind(self) -> None:
        self._loop = None
        self._subscribers.clear()

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def subscribe(self) -> asyncio.Queue[dict[str, typing.Any]]:
        queue: asyncio.Queue[dict[str, typing.Any]] = asyncio.Queue(maxsize=QUEUE_SIZE)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, typing.Any]]) -> None:
        self._subscribers.discard(queue)

    def publish(self, payload: dict[str, typing.Any]) -> None:
        """Safe to call from any thread."""
        loop = self._loop
        if loop is None:
            # Not serving yet (or shut down): nobody can be subscribed.
            logger.debug("Dropped %s event: the hub isn't bound to an event loop", payload.get("type"))
            return
        try:
            loop.call_soon_threadsafe(self._deliver, payload)
        except RuntimeError:
            logger.debug("Dropped %s event: the event loop is closed", payload.get("type"))

    def _deliver(self, payload: dict[str, typing.Any]) -> None:
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                logger.warning(
                    "A /ws/containers/ client isn't keeping up; dropped a %s event", payload.get("type")
                )


hub = Hub()
