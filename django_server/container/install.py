"""Background install of a model: base image → container → Ollama ready →
model pulled. Also the resume path: the same steps against an existing
container skip whatever is already done, and Ollama itself resumes a
partially downloaded model from the blobs it already has, so a pull cut off
by a failure or a backend restart continues instead of starting over.
"""

import logging
import threading

from django.db import close_old_connections

from django_app import progress_state

from .ContainerManager import ContainerError, ContainerManager

logger = logging.getLogger(__name__)


def start_install(model_name: str, parameters: str) -> bool:
    """Starts the install worker. False if one is already running for this
    model — the caller's progress is already on its way over the socket."""
    container_name = ContainerManager.container_name(model_name, parameters)

    if not progress_state.try_claim(container_name, "Starting…"):
        return False

    threading.Thread(
        target=_install_worker,
        args=(model_name, parameters, container_name),
        daemon=True,
        name=f"install-{container_name}",
    ).start()
    return True


def _install_worker(model_name: str, parameters: str, container_name: str) -> None:
    close_old_connections()

    try:
        _install(model_name, parameters, container_name)
    except ContainerError as e:
        _report_error(container_name, str(e))
    except Exception as e:
        logger.exception("Install worker crashed for %s", container_name)
        _report_error(container_name, f"Something went wrong while setting up the model: {e}")
    finally:
        progress_state.clear_progress(container_name)
        _broadcast_snapshot()
        close_old_connections()


def _install(model_name: str, parameters: str, container_name: str) -> None:
    manager = ContainerManager()
    if not manager.is_connected() and not manager.connect_to_docker():
        raise ContainerError("Docker isn't running. Start Docker Desktop and try again.")

    if manager.get_container(container_name) is None:
        _progress(container_name, "image_pull_progress", 0, "Checking base image…")
        if not manager.ensure_ollama_image(
            progress_cb=lambda pct, detail: _progress(container_name, "image_pull_progress", pct, detail)
        ):
            raise ContainerError(
                "Couldn't download the Ollama base image. Check your internet connection and try again."
            )

    _progress(container_name, "model_pull_progress", 0, "Starting container…")
    container = manager.create_and_start_container(model_name, parameters)

    base_url = manager.base_url_for(container)
    if base_url is None:
        raise ContainerError("The model container started but can't be reached. Try stopping and starting it again.")

    _progress(container_name, "model_pull_progress", 0, "Waiting for Ollama to start…")
    if not manager.wait_for_ollama_ready(base_url):
        raise ContainerError("Ollama inside the container didn't start in time. Try starting it again.")

    if manager.is_model_installed(base_url, model_name, parameters):
        ContainerManager.mark_installed(container)
        return

    pulled, pull_error = manager.pull_model_with_progress(
        model_name,
        parameters,
        base_url,
        progress_cb=lambda pct, detail: _progress(container_name, "model_pull_progress", pct, detail),
    )
    if not pulled:
        raise ContainerError(
            f"The download stopped: {pull_error or 'unknown error'}. "
            "Press 'Resume download' on the Models page to continue where it left off."
        )

    ContainerManager.mark_installed(container)


def _progress(container_name: str, phase: str, percent: int, detail: str) -> None:
    progress_state.set_progress(container_name, phase=phase, percent=percent, detail=detail)
    progress_state.broadcast(
        phase,
        {"type": phase, "container": container_name, "phase": phase, "percent": percent, "detail": detail},
        key=container_name,
    )


def _report_error(container_name: str, message: str) -> None:
    progress_state.broadcast(
        "container_update",
        {"type": "container_update", "container": container_name, "error": message},
        key=container_name,
        force=True,
    )


def _broadcast_snapshot() -> None:
    try:
        containers = ContainerManager().get_available_containers()
        progress_state.broadcast(
            "snapshot",
            {"type": "snapshot", "containers": containers, **progress_state.get_snapshot()},
            key="final",
            force=True,
        )
    except Exception as e:
        logger.warning("Couldn't broadcast the final container snapshot: %s", e)
