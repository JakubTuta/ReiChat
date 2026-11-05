"""The app's own SearXNG metasearch container, behind the opt-in web search
tool. Self-hosted so web search needs no account or API key; queries still
go out to public search engines (that's what searching the web is), but
nothing else from the user's chats does.
"""

import io
import logging
import secrets
import tarfile

from app.containers.manager import ContainerError, ContainerManager
from app.core import tasks
from app.core.config import get_settings
from app.realtime import progress

logger = logging.getLogger(__name__)

IMAGE = "searxng/searxng:latest"
CONTAINER_NAME = "reichat-searxng"
INTERNAL_PORT = 8080
HOST_PORT = 8888
PROGRESS_KEY = "web-search"
LABEL_SERVICE = "reichat.service"
CONFIG_DIR = "/etc/searxng"

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


async def status() -> str:
    """ "installing", "running", "stopped" or "off" (not set up)."""
    if progress.is_in_progress(PROGRESS_KEY):
        return "installing"
    container = await ContainerManager().get_container(CONTAINER_NAME)
    if container is None:
        return "off"
    return "running" if container.status == "running" else "stopped"


async def base_url() -> str | None:
    container = await ContainerManager().get_container(CONTAINER_NAME)
    if container is None or container.status != "running":
        return None
    if get_settings().docker:
        return f"http://{CONTAINER_NAME}:{INTERNAL_PORT}"
    return f"http://127.0.0.1:{HOST_PORT}"


def start_enable() -> bool:
    if not progress.try_claim(PROGRESS_KEY, "Setting up web search…"):
        return False
    tasks.spawn(_enable_worker(), name="searxng-enable")
    return True


async def _enable_worker() -> None:
    try:
        await _enable()
    except ContainerError as e:
        progress.broadcast(
            "container_update",
            {"type": "container_update", "container": PROGRESS_KEY, "error": str(e)},
            key=PROGRESS_KEY,
            force=True,
        )
    except Exception as e:
        logger.exception("Enabling web search crashed")
        progress.broadcast(
            "container_update",
            {
                "type": "container_update",
                "container": PROGRESS_KEY,
                "error": f"Setting up web search failed: {e}",
            },
            key=PROGRESS_KEY,
            force=True,
        )
    else:
        progress.broadcast(
            "container_update",
            {"type": "container_update", "container": PROGRESS_KEY, "info": "Web search is ready."},
            key=PROGRESS_KEY,
            force=True,
        )
    finally:
        progress.clear_progress(PROGRESS_KEY)


async def _enable() -> None:
    await ContainerManager().ensure_service_running(
        image=IMAGE,
        name=CONTAINER_NAME,
        internal_port=INTERNAL_PORT,
        host_port=HOST_PORT,
        environment={"SEARXNG_SECRET": secrets.token_hex(32), "SEARXNG_PORT": str(INTERNAL_PORT)},
        labels={LABEL_SERVICE: "searxng"},
        config_dir=CONFIG_DIR,
        config_archive=_settings_archive(),
        failure="Couldn't set up web search",
    )


async def disable() -> None:
    await ContainerManager().force_remove_container(CONTAINER_NAME, failure="Couldn't turn web search off")
