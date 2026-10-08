import logging
import threading
import time

from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer

logger = logging.getLogger(__name__)

_lock = threading.Lock()

progress_registry: dict[str, dict] = {}
scrape_state: dict = {
    "running": False,
    "total": None,
    "completed": 0,
    "current": None,
    "error": None,
}

_last_broadcast: dict[str, float] = {}
_THROTTLE_S = 0.5


def set_progress(name: str, **kwargs) -> None:
    with _lock:
        if name not in progress_registry:
            progress_registry[name] = {}
        progress_registry[name].update(kwargs)


def try_claim(name: str, detail: str) -> bool:
    """Atomically mark `name` as having an install in flight. False if one
    already is, so two clicks (or two tabs) can't run two pulls at once."""
    with _lock:
        if name in progress_registry:
            return False
        progress_registry[name] = {"phase": "model_pull_progress", "percent": 0, "detail": detail}
        return True


def is_in_progress(name: str) -> bool:
    with _lock:
        return name in progress_registry


def clear_progress(name: str) -> None:
    with _lock:
        progress_registry.pop(name, None)
        _last_broadcast.pop(f"image_pull_progress:{name}", None)
        _last_broadcast.pop(f"model_pull_progress:{name}", None)


def get_snapshot() -> dict:
    with _lock:
        return {
            "progress": dict(progress_registry),
            "scrape": dict(scrape_state),
        }


def set_scrape(**kwargs) -> None:
    with _lock:
        scrape_state.update(kwargs)


def try_start_scrape() -> bool:
    """Atomically claim the scrape/refresh slot. Returns False if one is
    already running, so a double-click or a second browser tab can't start
    two concurrent catalog refreshes racing each other's writes.
    """
    with _lock:
        if scrape_state.get("running"):
            return False
        scrape_state.update(running=True, total=None, completed=0, current=None, error=None)
        return True


def broadcast(event_type: str, payload: dict, key: str = "", force: bool = False) -> None:
    if not force:
        now = time.monotonic()
        throttle_key = f"{event_type}:{key}"
        should_send = False
        with _lock:
            last = _last_broadcast.get(throttle_key, 0)
            if now - last >= _THROTTLE_S:
                _last_broadcast[throttle_key] = now
                should_send = True
        if not should_send:
            return

    try:
        channel_layer = get_channel_layer()
        if channel_layer is None:
            return
        async_to_sync(channel_layer.group_send)(
            "container_status",
            {"type": event_type.replace("-", "_"), "payload": payload},
        )
    except Exception:
        logger.warning("Failed to broadcast %s event", event_type, exc_info=True)
