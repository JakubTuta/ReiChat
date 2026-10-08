import json
import logging
import threading

from asgiref.sync import async_to_sync
from channels.generic.websocket import WebsocketConsumer
from channels.layers import get_channel_layer

logger = logging.getLogger(__name__)

_lifecycle_lock = threading.Lock()
_subscriber_count = 0
_stop_event: threading.Event | None = None


def _snapshot_loop(stop_event: threading.Event) -> None:
    while not stop_event.wait(2):
        try:
            from container.ContainerManager import ContainerManager
            from django_app import progress_state

            docker_client = ContainerManager()
            containers = docker_client.get_available_containers()
            snapshot = progress_state.get_snapshot()
            channel_layer = get_channel_layer()
            if channel_layer is None:
                continue
            async_to_sync(channel_layer.group_send)(
                "container_status",
                {
                    "type": "snapshot",
                    "payload": {
                        "type": "snapshot",
                        "containers": containers,
                        **snapshot,
                    },
                },
            )
        except Exception as e:
            logger.debug("Snapshot thread error: %s", e)


def _ensure_snapshot_thread_running() -> None:
    global _stop_event
    with _lifecycle_lock:
        if _stop_event is not None:
            return
        _stop_event = threading.Event()
        threading.Thread(
            target=_snapshot_loop, args=(_stop_event,), daemon=True, name="container-snapshot"
        ).start()


def _maybe_stop_snapshot_thread() -> None:
    global _stop_event
    with _lifecycle_lock:
        if _subscriber_count <= 0 and _stop_event is not None:
            _stop_event.set()
            _stop_event = None


class ContainerStatusConsumer(WebsocketConsumer):
    def connect(self) -> None:
        global _subscriber_count

        async_to_sync(self.channel_layer.group_add)(  # type: ignore
            "container_status", self.channel_name
        )
        self.accept()

        with _lifecycle_lock:
            _subscriber_count += 1
        # No client needs a live feed while nobody's subscribed — this loop
        # used to run forever regardless of subscriber count, building a
        # fresh ContainerManager (and hitting the Docker socket) every 2s.
        _ensure_snapshot_thread_running()
        self._send_snapshot()

    def disconnect(self, code: int) -> None:
        global _subscriber_count

        async_to_sync(self.channel_layer.group_discard)(  # type: ignore
            "container_status", self.channel_name
        )

        with _lifecycle_lock:
            _subscriber_count -= 1
        _maybe_stop_snapshot_thread()

    def _send_snapshot(self) -> None:
        try:
            from container.ContainerManager import ContainerManager
            from django_app import progress_state

            docker_client = ContainerManager()
            containers = docker_client.get_available_containers()
            snapshot = progress_state.get_snapshot()
            self.send(
                text_data=json.dumps(
                    {
                        "type": "snapshot",
                        "containers": containers,
                        **snapshot,
                    }
                )
            )
        except Exception as e:
            logger.warning("Failed to send initial snapshot: %s", e)

    def _relay(self, event: dict) -> None:
        try:
            self.send(text_data=json.dumps(event["payload"]))
        except Exception as e:
            logger.debug("Relay error: %s", e)

    def container_update(self, event: dict) -> None:
        self._relay(event)

    def image_pull_progress(self, event: dict) -> None:
        self._relay(event)

    def model_pull_progress(self, event: dict) -> None:
        self._relay(event)

    def scrape_progress(self, event: dict) -> None:
        self._relay(event)

    def ollama_update(self, event: dict) -> None:
        self._relay(event)

    def snapshot(self, event: dict) -> None:
        self._relay(event)
