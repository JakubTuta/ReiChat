"""Background install of a model: base image → container → Ollama ready →
model pulled. Also the resume path: the same steps against an existing
container skip whatever is already done, and Ollama itself resumes a
partially downloaded model from the blobs it already has, so a pull cut off
by a failure or a backend restart continues instead of starting over.
"""

import logging
import threading

from django.db import close_old_connections

from django_app import ollama_api, progress_state
from django_app.catalog import registry
from django_app.models import AIModel, AIModelVersion

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


UPDATE_KEY = "ollama-update"


def start_ollama_update() -> bool:
    """Pulls the newest ollama/ollama image and moves every model onto it,
    keeping their downloaded files. False if an update is already running."""
    if not progress_state.try_claim(UPDATE_KEY, "Checking for a newer Ollama…"):
        return False

    threading.Thread(target=_ollama_update_worker, daemon=True, name="ollama-update").start()
    return True


def _ollama_update_worker() -> None:
    close_old_connections()
    try:
        manager = ContainerManager()
        if not manager.is_connected() and not manager.connect_to_docker():
            raise ContainerError("Docker isn't running. Start Docker Desktop and try again.")

        before = manager.image_id()
        _progress(UPDATE_KEY, "ollama_update", 0, "Downloading the latest Ollama…")
        manager.pull_ollama_image()
        after = manager.image_id()

        if before is not None and before == after:
            _report_info("Ollama is already up to date.")
            return

        assert manager.client is not None
        containers = [c for c in manager.client.containers.list(all=True) if ContainerManager.is_managed_container(c)]
        skipped: list[str] = []
        for index, container in enumerate(containers):
            _progress(UPDATE_KEY, "ollama_update", int(100 * index / max(len(containers), 1)),
                      f"Moving {container.name} to the new version…")
            try:
                manager.recreate_on_current_image(container)
            except ContainerError as e:
                logger.warning("Skipped updating %s: %s", container.name, e)
                skipped.append(container.name or "")

        if skipped:
            _report_info(
                "Ollama was updated. These models were set up before updates were possible and still run the old "
                f"version until you remove and re-create them: {', '.join(skipped)}."
            )
        else:
            _report_info("Ollama was updated. Your models keep their downloaded files.")
    except ContainerError as e:
        _report_error(UPDATE_KEY, str(e))
    except Exception as e:
        logger.exception("Ollama update crashed")
        _report_error(UPDATE_KEY, f"Updating Ollama failed: {e}")
    finally:
        progress_state.clear_progress(UPDATE_KEY)
        _broadcast_snapshot()
        close_old_connections()


def start_model_update(model_name: str, parameters: str) -> bool:
    """Re-pulls an installed model: Ollama compares digests and downloads
    only what changed, so an up-to-date model finishes almost instantly."""
    container_name = ContainerManager.container_name(model_name, parameters)
    if not progress_state.try_claim(container_name, "Checking for a newer version…"):
        return False

    threading.Thread(
        target=_install_worker,
        args=(model_name, parameters, container_name, True),
        daemon=True,
        name=f"update-{container_name}",
    ).start()
    return True


def _install_worker(model_name: str, parameters: str, container_name: str, force_pull: bool = False) -> None:
    close_old_connections()

    try:
        _install(model_name, parameters, container_name, force_pull)
    except ContainerError as e:
        _report_error(container_name, str(e))
    except Exception as e:
        logger.exception("Install worker crashed for %s", container_name)
        _report_error(container_name, f"Something went wrong while setting up the model: {e}")
    finally:
        progress_state.clear_progress(container_name)
        _broadcast_snapshot()
        close_old_connections()


def _install(model_name: str, parameters: str, container_name: str, force_pull: bool = False) -> None:
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

    if not force_pull and manager.is_model_installed(base_url, model_name, parameters):
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
    _sync_custom_model_capabilities(model_name, parameters, base_url)


def _sync_custom_model_capabilities(model_name: str, parameters: str, base_url: str) -> None:
    """A model added by name has no ollama.com listing to say whether it
    can see images or use tools — ask the model itself once it's in."""
    ai_model = AIModel.objects.filter(model=model_name, is_custom=True).first()
    if ai_model is None:
        return

    info = ollama_api.show_model(base_url, f"{model_name}:{parameters}")
    if info is None:
        return

    capabilities = set(ollama_api.capabilities(info))
    ai_model.can_process_image = "vision" in capabilities
    ai_model.can_use_tools = "tools" in capabilities
    ai_model.is_embedding = capabilities == {"embedding"}
    ai_model.save(update_fields=["can_process_image", "can_use_tools", "is_embedding"])

    sizes = ollama_api.installed_model_sizes(base_url) or {}
    if size := sizes.get(f"{model_name}:{parameters}"):
        AIModelVersion.objects.filter(ai_model=ai_model, parameters=parameters).update(
            size_bytes=size, size=registry.format_size(size)
        )


def _progress(container_name: str, phase: str, percent: int, detail: str) -> None:
    progress_state.set_progress(container_name, phase=phase, percent=percent, detail=detail)
    progress_state.broadcast(
        phase,
        {"type": phase, "container": container_name, "phase": phase, "percent": percent, "detail": detail},
        key=container_name,
    )


def _report_info(message: str) -> None:
    progress_state.broadcast(
        "container_update",
        {"type": "container_update", "container": UPDATE_KEY, "info": message},
        key=UPDATE_KEY,
        force=True,
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
