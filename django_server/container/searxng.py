"""The app's own SearXNG metasearch container, behind the opt-in web search
tool. Self-hosted so web search needs no account or API key; queries still
go out to public search engines (that's what searching the web is), but
nothing else from the user's chats does.
"""

import io
import logging
import os
import secrets
import tarfile
import threading

import docker.errors

from django_app import progress_state

from .ContainerManager import ContainerError, ContainerManager, _docker_error_text

logger = logging.getLogger(__name__)

IMAGE = "searxng/searxng:latest"
CONTAINER_NAME = "reichat-searxng"
INTERNAL_PORT = 8080
HOST_PORT = 8888
PROGRESS_KEY = "web-search"
LABEL_SERVICE = "reichat.service"

# The stock settings only allow HTML results; the tool needs JSON. The
# limiter is for public instances and would throttle the app's own calls.
SETTINGS_YML = """use_default_settings: true
server:
  limiter: false
  image_proxy: false
search:
  safe_search: 1
  formats:
    - html
    - json
"""


def _settings_archive() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        data = SETTINGS_YML.encode()
        info = tarfile.TarInfo("settings.yml")
        info.size = len(data)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def _container():
    manager = ContainerManager()
    if not manager.is_connected() and not manager.connect_to_docker():
        return None
    return manager.get_container(CONTAINER_NAME)


def status() -> str:
    """"installing", "running", "stopped" or "off" (not set up)."""
    if progress_state.is_in_progress(PROGRESS_KEY):
        return "installing"
    container = _container()
    if container is None:
        return "off"
    return "running" if container.status == "running" else "stopped"


def base_url() -> str | None:
    container = _container()
    if container is None or container.status != "running":
        return None
    if os.getenv("DOCKER", "false") == "true":
        return f"http://{CONTAINER_NAME}:{INTERNAL_PORT}"
    return f"http://127.0.0.1:{HOST_PORT}"


def start_enable() -> bool:
    if not progress_state.try_claim(PROGRESS_KEY, "Setting up web search…"):
        return False
    threading.Thread(target=_enable_worker, daemon=True, name="searxng-enable").start()
    return True


def _enable_worker() -> None:
    try:
        _enable()
    except ContainerError as e:
        progress_state.broadcast(
            "container_update", {"type": "container_update", "container": PROGRESS_KEY, "error": str(e)},
            key=PROGRESS_KEY, force=True,
        )
    except Exception as e:
        logger.exception("Enabling web search crashed")
        progress_state.broadcast(
            "container_update",
            {"type": "container_update", "container": PROGRESS_KEY, "error": f"Setting up web search failed: {e}"},
            key=PROGRESS_KEY, force=True,
        )
    else:
        progress_state.broadcast(
            "container_update",
            {"type": "container_update", "container": PROGRESS_KEY, "info": "Web search is ready."},
            key=PROGRESS_KEY, force=True,
        )
    finally:
        progress_state.clear_progress(PROGRESS_KEY)


def _enable() -> None:
    manager = ContainerManager()
    if not manager.is_connected() and not manager.connect_to_docker():
        raise ContainerError("Docker isn't running. Start Docker Desktop and try again.")
    assert manager.client is not None

    existing = manager.get_container(CONTAINER_NAME)
    if existing is not None:
        manager.start_container(existing)
        return

    try:
        manager.client.images.pull(IMAGE)
        container = manager.client.containers.create(
            IMAGE,
            name=CONTAINER_NAME,
            detach=True,
            network="chatbot-network",
            ports={f"{INTERNAL_PORT}/tcp": ("127.0.0.1", HOST_PORT)},
            environment={"SEARXNG_SECRET": secrets.token_hex(32), "SEARXNG_PORT": str(INTERNAL_PORT)},
            labels={LABEL_SERVICE: "searxng"},
            restart_policy={"Name": "unless-stopped"},
        )
        container.put_archive("/etc/searxng", _settings_archive())
        container.start()
    except docker.errors.APIError as e:
        raise ContainerError(f"Couldn't set up web search: {_docker_error_text(e)}") from e


def disable() -> None:
    container = _container()
    if container is None:
        return
    try:
        container.remove(force=True)
    except docker.errors.APIError as e:
        raise ContainerError(f"Couldn't turn web search off: {_docker_error_text(e)}") from e
