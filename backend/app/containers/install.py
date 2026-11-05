"""Background install of a model: base image → container → Ollama ready →
model pulled. Also the resume path: the same steps against an existing
container skip whatever is already done, and Ollama itself resumes a
partially downloaded model from the blobs it already has, so a pull cut off
by a failure or a backend restart continues instead of starting over.
"""

import logging

from sqlalchemy import select, update

from app import runtime_settings
from app.catalog import registry
from app.containers import service
from app.containers.manager import ContainerError, ContainerManager
from app.core import tasks
from app.db.models import AIModel, AIModelVersion
from app.db.session import SessionLocal
from app.llm import ollama_api
from app.realtime import progress

logger = logging.getLogger(__name__)

UPDATE_KEY = "ollama-update"


def start_install(model_name: str, parameters: str) -> bool:
    """Starts the install worker. False if one is already running for this
    model — the caller's progress is already on its way over the socket."""
    container_name = ContainerManager.container_name(model_name, parameters)

    if not progress.try_claim(container_name, "Starting…"):
        return False

    tasks.spawn(_install_worker(model_name, parameters, container_name), name=f"install-{container_name}")
    return True


def start_model_update(model_name: str, parameters: str) -> bool:
    """Re-pulls an installed model: Ollama compares digests and downloads
    only what changed, so an up-to-date model finishes almost instantly."""
    container_name = ContainerManager.container_name(model_name, parameters)
    if not progress.try_claim(container_name, "Checking for a newer version…"):
        return False

    tasks.spawn(
        _install_worker(model_name, parameters, container_name, force_pull=True),
        name=f"update-{container_name}",
    )
    return True


def start_ollama_update() -> bool:
    """Pulls the newest ollama/ollama image and moves every model onto it,
    keeping their downloaded files. False if an update is already running."""
    if not progress.try_claim(UPDATE_KEY, "Checking for a newer Ollama…"):
        return False

    tasks.spawn(_ollama_update_worker(), name="ollama-update")
    return True


async def _ollama_update_worker() -> None:
    try:
        try:
            await _update_ollama()
        except ContainerError as e:
            _report_error(UPDATE_KEY, str(e))
        except Exception as e:
            logger.exception("Ollama update crashed")
            _report_error(UPDATE_KEY, f"Updating Ollama failed: {e}")
    finally:
        progress.clear_progress(UPDATE_KEY)
    await _broadcast_snapshot()


async def _update_ollama() -> None:
    if await runtime_settings.external_ollama_url() is not None:
        raise ContainerError(
            "You're using your own Ollama, so update it the way you installed it (for example, the "
            "Ollama app updates itself)."
        )

    manager = ContainerManager()
    if not await manager.ensure_connected():
        raise ContainerError("Docker isn't running. Start Docker Desktop and try again.")

    before = await manager.image_id()
    _progress(UPDATE_KEY, "ollama_update", 0, "Downloading the latest Ollama…")
    await manager.pull_ollama_image()
    after = await manager.image_id()

    if before is not None and before == after:
        _report_info("Ollama is already up to date.")
        return

    containers = await manager.list_managed()
    skipped: list[str] = []
    for index, container in enumerate(containers):
        _progress(
            UPDATE_KEY,
            "ollama_update",
            int(100 * index / max(len(containers), 1)),
            f"Moving {container.name} to the new version…",
        )
        try:
            await manager.recreate_on_current_image(container)
        except ContainerError as e:
            logger.warning("Skipped updating %s: %s", container.name, e)
            skipped.append(container.name or "")
        finally:
            # The replacement is a new container, so it has to prove it holds
            # its model again.
            service.forget_installed(container.id)

    if skipped:
        _report_info(
            "Ollama was updated. These models were set up before updates were possible and still run the old "
            f"version until you remove and re-create them: {', '.join(skipped)}."
        )
    else:
        _report_info("Ollama was updated. Your models keep their downloaded files.")


async def _install_worker(
    model_name: str, parameters: str, container_name: str, force_pull: bool = False
) -> None:
    try:
        try:
            await _install(model_name, parameters, container_name, force_pull)
        except ContainerError as e:
            _report_error(container_name, str(e))
        except Exception as e:
            logger.exception("Install worker crashed for %s", container_name)
            _report_error(container_name, f"Something went wrong while setting up the model: {e}")
    finally:
        progress.clear_progress(container_name)
    await _broadcast_snapshot()


async def _install(model_name: str, parameters: str, container_name: str, force_pull: bool = False) -> None:
    if (external_url := await runtime_settings.external_ollama_url()) is not None:
        await _install_on_external(model_name, parameters, container_name, external_url)
        return

    manager = ContainerManager()
    if not await manager.ensure_connected():
        raise ContainerError("Docker isn't running. Start Docker Desktop and try again.")

    if await manager.get_container(container_name) is None:
        _progress(container_name, "image_pull_progress", 0, "Checking base image…")
        if not await manager.ensure_ollama_image(
            progress_cb=lambda pct, detail: _progress(container_name, "image_pull_progress", pct, detail)
        ):
            raise ContainerError(
                "Couldn't download the Ollama base image. Check your internet connection and try again."
            )

    _progress(container_name, "model_pull_progress", 0, "Starting container…")
    container = await manager.create_and_start_container(model_name, parameters)

    base_url = ContainerManager.base_url_for(container)
    if base_url is None:
        raise ContainerError(
            "The model container started but can't be reached. Try stopping and starting it again."
        )

    _progress(container_name, "model_pull_progress", 0, "Waiting for Ollama to start…")
    if not await ollama_api.wait_until_ready(base_url):
        raise ContainerError("Ollama inside the container didn't start in time. Try starting it again.")

    if not force_pull and await ollama_api.is_model_installed(base_url, model_name, parameters):
        service.mark_installed(container)
        return

    pulled, pull_error = await ollama_api.pull_model(
        base_url,
        model_name,
        parameters,
        progress_cb=lambda pct, detail: _progress(container_name, "model_pull_progress", pct, detail),
    )
    if not pulled:
        raise ContainerError(
            f"The download stopped: {pull_error or 'unknown error'}. "
            "Press 'Resume download' on the Models page to continue where it left off."
        )

    service.mark_installed(container)
    await _sync_custom_model_capabilities(model_name, parameters, base_url)


async def _install_on_external(model_name: str, parameters: str, container_name: str, base_url: str) -> None:
    if await ollama_api.version(base_url) is None:
        raise ContainerError(f"Can't reach your Ollama at {base_url}. Check that it's running and try again.")

    pulled, pull_error = await ollama_api.pull_model(
        base_url,
        model_name,
        parameters,
        progress_cb=lambda pct, detail: _progress(container_name, "model_pull_progress", pct, detail),
    )
    if not pulled:
        raise ContainerError(
            f"The download stopped: {pull_error or 'unknown error'}. Try again to continue where it left off."
        )
    await _sync_custom_model_capabilities(model_name, parameters, base_url)


async def _sync_custom_model_capabilities(model_name: str, parameters: str, base_url: str) -> None:
    """Records what the installed model says it can do. A model added by
    name has no ollama.com listing at all, and the scraped listing for
    catalog models has been wrong (qwen3 shown without tools) — the model's
    own answer is the truth for the version actually installed."""
    async with SessionLocal() as session:
        row = (
            await session.execute(select(AIModel.id, AIModel.is_custom).where(AIModel.model == model_name))
        ).one_or_none()
    if row is None:
        return

    info = await ollama_api.show_model(base_url, f"{model_name}:{parameters}")
    if info is None:
        return

    capabilities = set(ollama_api.capabilities(info))
    if not capabilities:
        return

    size = None
    if row.is_custom:
        sizes = await ollama_api.installed_model_sizes(base_url) or {}
        size = sizes.get(f"{model_name}:{parameters}")

    async with SessionLocal() as session:
        await session.execute(
            update(AIModel)
            .where(AIModel.id == row.id)
            .values(
                can_process_image="vision" in capabilities,
                can_use_tools="tools" in capabilities,
                is_embedding=capabilities == {"embedding"},
            )
        )
        if size:
            await session.execute(
                update(AIModelVersion)
                .where(AIModelVersion.ai_model_id == row.id, AIModelVersion.parameters == parameters)
                .values(size_bytes=size, size=registry.format_size(size))
            )
        await session.commit()


def _progress(container_name: str, phase: str, percent: int, detail: str) -> None:
    progress.set_progress(container_name, phase=phase, percent=percent, detail=detail)
    progress.broadcast(
        phase,
        {"type": phase, "container": container_name, "phase": phase, "percent": percent, "detail": detail},
        key=container_name,
    )


def _report_info(message: str) -> None:
    progress.broadcast(
        "container_update",
        {"type": "container_update", "container": UPDATE_KEY, "info": message},
        key=UPDATE_KEY,
        force=True,
    )


def _report_error(container_name: str, message: str) -> None:
    progress.broadcast(
        "container_update",
        {"type": "container_update", "container": container_name, "error": message},
        key=container_name,
        force=True,
    )


async def _broadcast_snapshot() -> None:
    try:
        containers = await service.get_available_containers()
        progress.broadcast(
            "snapshot",
            {"type": "snapshot", "containers": containers, **progress.get_snapshot()},
            key="final",
            force=True,
        )
    except Exception as e:
        logger.warning("Couldn't broadcast the final container snapshot: %s", e)
