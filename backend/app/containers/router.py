import typing

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app import runtime_settings
from app.containers import external, hardware, install, searxng, service
from app.containers.manager import ContainerError, ContainerManager
from app.core import tasks
from app.core.security import require_json_body
from app.db.models import AIModel, AIModelVersion
from app.db.session import SessionDep
from app.llm import ollama_api

# Registered most specific first: `{model:path}` swallows slashes, so the
# `/unload` and `/update` routes have to be tried before the bare container one.
router = APIRouter(prefix="/docker", dependencies=[Depends(require_json_body)])

MISSING_PARAMETERS = "Invalid request: missing 'parameters' query param"


async def _require_docker(manager: ContainerManager) -> None:
    if not await manager.ensure_connected():
        raise HTTPException(status_code=500, detail="Error connecting to Docker")


@router.get("/")
async def docker_status() -> dict[str, str]:
    # With the user's own Ollama, Docker isn't needed for models at all —
    # what matters is whether that Ollama answers.
    if (external_url := await runtime_settings.external_ollama_url()) is not None:
        if await ollama_api.version(external_url) is None:
            raise HTTPException(
                status_code=503,
                detail=f"Can't reach your Ollama at {external_url}. Start it, or switch back to "
                "app-managed models on the Models page.",
            )
        return {"status": "Ollama is reachable"}

    await _require_docker(ContainerManager())
    return {"status": "Docker is running"}


@router.get("/containers/")
async def list_containers(session: SessionDep) -> list[dict[str, typing.Any]]:
    if (external_url := await runtime_settings.external_ollama_url()) is not None:
        await external.register_models(session, external_url)
        return await service.get_available_containers()

    await _require_docker(ContainerManager())
    return await service.get_available_containers()


@router.get("/hardware/")
async def hardware_info() -> dict[str, int | bool | None]:
    # Docker's numbers describe Docker's VM, not where the user's own Ollama
    # runs — better to admit not knowing than to mislead.
    if await runtime_settings.external_ollama_url() is not None:
        return {"ram_bytes": None, "vram_bytes": None, "gpu_available": None}

    manager = ContainerManager()
    await _require_docker(manager)
    return {
        "ram_bytes": await manager.host_memory_bytes(),
        "vram_bytes": await hardware.detect_vram_bytes(manager),
        "gpu_available": await manager.gpu_status(),
    }


@router.get("/disk-usage/")
async def disk_usage(session: SessionDep) -> dict[str, typing.Any]:
    # Real installed-model disk usage, not the registry's advertised
    # size — matched against the catalog by (model, parameters) since
    # Docker itself doesn't know which Ollama model a container holds.
    manager = ContainerManager()
    external_url = await runtime_settings.external_ollama_url()

    if external_url is None:
        await _require_docker(manager)

    containers = await service.get_available_containers()
    external_sizes = (await ollama_api.installed_model_sizes(external_url) or {}) if external_url else {}

    rows = await session.execute(
        select(AIModel.model, AIModelVersion.parameters, AIModelVersion.size_bytes).join(
            AIModelVersion, AIModelVersion.ai_model_id == AIModel.id
        )
    )
    versions_by_key = {(row.model, row.parameters): row.size_bytes for row in rows}

    entries = []
    total_bytes = 0
    for container in containers:
        model = container["environment"].get("model")
        parameters = container["environment"].get("parameters")
        # The real size on disk when the container can be asked (a
        # running one), the registry's advertised size otherwise.
        size_bytes = external_sizes.get(f"{model}:{parameters}")
        if (
            size_bytes is None
            and not external_url
            and container["status"] in ("running", "incomplete")
            and model
            and parameters
        ):
            base_url = await manager.get_ollama_base_url(model, parameters)
            sizes = await ollama_api.installed_model_sizes(base_url) if base_url else None
            size_bytes = (sizes or {}).get(f"{model}:{parameters}")
        if size_bytes is None:
            size_bytes = versions_by_key.get((model, parameters))
        if size_bytes:
            total_bytes += size_bytes

        entries.append(
            {
                "model": model,
                "parameters": parameters,
                "status": container["status"],
                "size_bytes": size_bytes,
            }
        )

    return {"total_bytes": total_bytes, "models": entries}


@router.get("/ollama-image/")
async def ollama_image_status() -> Response:
    manager = ContainerManager()
    await _require_docker(manager)

    return Response(status_code=200 if await manager.is_ollama_image_pulled() else 404)


@router.post("/ollama-image/", status_code=202)
async def pull_ollama_image() -> dict[str, str]:
    manager = ContainerManager()
    await _require_docker(manager)

    tasks.spawn(manager.pull_ollama_image(), name="pull-ollama-image")
    return {"status": "Pulling ollama/ollama image in background"}


@router.post("/container/{model:path}/unload")
async def unload_container(model: str, parameters: str | None = None) -> dict[str, str]:
    if parameters is None:
        raise HTTPException(status_code=400, detail=MISSING_PARAMETERS)

    base_url = await ContainerManager().get_ollama_base_url(model, parameters)
    if base_url is None:
        raise HTTPException(status_code=404, detail="That model isn't running.")

    if not await ollama_api.unload_model(base_url, f"{model}:{parameters}"):
        raise HTTPException(status_code=502, detail="Ollama didn't respond — try again in a moment.")

    return {"status": "Model unloaded"}


@router.post("/container/{model:path}/update", status_code=202)
async def update_container(model: str, parameters: str | None = None) -> dict[str, str]:
    if parameters is None:
        raise HTTPException(status_code=400, detail=MISSING_PARAMETERS)

    external_url = await runtime_settings.external_ollama_url()
    installed = (
        await external.has_model(external_url, model, parameters)
        if external_url
        else await ContainerManager().get_container(ContainerManager.container_name(model, parameters))
        is not None
    )
    if not installed:
        raise HTTPException(status_code=404, detail="That model isn't installed.")

    install.start_model_update(model, parameters)
    return {"status": "Checking for updates"}


@router.get("/container/{model:path}")
async def get_container(model: str) -> dict[str, typing.Any]:
    manager = ContainerManager()
    await _require_docker(manager)

    if not (container := await manager.get_container(model)):
        raise HTTPException(status_code=404, detail="Container not found")

    return {"status": "Container found", "container": ContainerManager.map_container(container)}


@router.post("/container/{model:path}")
async def start_container(model: str, session: SessionDep, parameters: str | None = None) -> JSONResponse:
    # 200 when the model is already confirmed running; otherwise 202 and
    # progress arrives over the container socket (see containers/install.py).
    if parameters is None:
        raise HTTPException(status_code=400, detail=MISSING_PARAMETERS)

    manager = ContainerManager()
    external_url = await runtime_settings.external_ollama_url()
    if external_url is None:
        await _require_docker(manager)

    ai_model = (
        await session.execute(
            select(AIModel).where(AIModel.model == model).options(selectinload(AIModel.versions))
        )
    ).scalar_one_or_none()
    if ai_model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    ai_model_version = next(
        (version for version in ai_model.versions if version.parameters == parameters), None
    )
    if ai_model_version is None:
        raise HTTPException(status_code=404, detail="Invalid model version")

    container_name = ContainerManager.container_name(model, parameters)

    if external_url is not None:
        if await external.has_model(external_url, model, parameters):
            return JSONResponse({"status": "Model is installed"}, status_code=200)
        install.start_install(ai_model.model, ai_model_version.parameters)
        return JSONResponse({"status": "Model setup started"}, status_code=202)

    existing = await manager.get_container(container_name)

    if existing is not None:
        try:
            await manager.start_container(existing)
        except ContainerError as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

        # Already confirmed to hold its model — nothing left to do. Any
        # other case (first start since the backend restarted, or a pull
        # that never finished) goes through the install worker, which
        # just checks /api/tags when the model is there and resumes the
        # download when it isn't.
        if service.is_marked_installed(existing):
            return JSONResponse(
                {"status": "Container is running", "container": ContainerManager.map_container(existing)},
                status_code=200,
            )

    install.start_install(ai_model.model, ai_model_version.parameters)

    return JSONResponse({"status": "Model setup started"}, status_code=202)


@router.delete("/container/{model:path}")
async def stop_or_remove_container(
    model: str, parameters: str | None = None, method: str | None = None
) -> dict[str, str]:
    if parameters is None or method is None:
        raise HTTPException(
            status_code=400, detail="Invalid request: missing 'parameters' or 'method' query params"
        )

    if (external_url := await runtime_settings.external_ollama_url()) is not None:
        # No container to stop: "stop" frees the model's memory, "remove"
        # deletes it from the user's Ollama.
        if method == "stop":
            await ollama_api.unload_model(external_url, f"{model}:{parameters}")
            return {"status": "Model unloaded"}
        if method == "remove":
            if not await ollama_api.delete_model(external_url, model, parameters):
                raise HTTPException(
                    status_code=502, detail="Your Ollama didn't delete the model — try again."
                )
            return {"status": "Model removed"}

    manager = ContainerManager()
    await _require_docker(manager)

    container_name = ContainerManager.container_name(model, parameters)

    if method == "stop":
        await manager.stop_container(container_name)
        return {"status": "Container stopped"}

    if method == "remove":
        await service.remove_model_container(container_name)
        return {"status": "Container removed"}

    raise HTTPException(status_code=400, detail="Invalid method: use 'stop' or 'remove'")


@router.get("/runtime/")
async def runtime() -> dict[str, list[dict[str, typing.Any]]]:
    """What each running model container has loaded in memory right now."""
    if (external_url := await runtime_settings.external_ollama_url()) is not None:
        return {
            "loaded": [
                {
                    "container": None,
                    "model": entry["name"],
                    "size_bytes": entry["size"],
                    "vram_bytes": entry["size_vram"],
                    "context_length": entry["context_length"],
                    "expires_at": entry["expires_at"],
                }
                for entry in await ollama_api.running_models(external_url) or []
            ]
        }

    manager = ContainerManager()
    await _require_docker(manager)

    loaded = []
    for container in await manager.list_managed(running_only=True):
        base_url = ContainerManager.base_url_for(container)
        for entry in (await ollama_api.running_models(base_url) if base_url else None) or []:
            loaded.append(
                {
                    "container": container.name,
                    "model": entry["name"],
                    "size_bytes": entry["size"],
                    "vram_bytes": entry["size_vram"],
                    "context_length": entry["context_length"],
                    "expires_at": entry["expires_at"],
                }
            )

    return {"loaded": loaded}


@router.post("/ollama-update/", status_code=202)
async def update_ollama() -> dict[str, str]:
    await _require_docker(ContainerManager())

    install.start_ollama_update()
    return {"status": "Ollama update started"}


@router.get("/web-search/")
async def web_search_status() -> dict[str, str]:
    return {"status": await searxng.status()}


@router.post("/web-search/", status_code=202)
async def enable_web_search() -> dict[str, str]:
    await _require_docker(ContainerManager())

    searxng.start_enable()
    return {"status": "installing"}


@router.delete("/web-search/")
async def disable_web_search() -> dict[str, str]:
    try:
        await searxng.disable()
    except ContainerError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"status": "off"}


class OllamaConnectionBody(BaseModel):
    mode: object = None
    url: object = None


@router.get("/ollama-connection/")
async def ollama_connection() -> dict[str, str | None]:
    """Where models run: app-managed containers (default) or the user's own
    Ollama at a URL."""
    external_url = await runtime_settings.external_ollama_url()
    return {
        "mode": "external" if external_url else "managed",
        "url": external_url,
        "version": await ollama_api.version(external_url) if external_url else None,
    }


@router.put("/ollama-connection/")
async def set_ollama_connection(
    session: SessionDep, body: OllamaConnectionBody | None = None
) -> dict[str, str | None]:
    body = body or OllamaConnectionBody()

    if body.mode == "managed":
        await runtime_settings.set_external_ollama_url(None)
        return {"mode": "managed", "url": None, "version": None}

    if body.mode != "external":
        raise HTTPException(status_code=400, detail="mode must be 'managed' or 'external'")

    url = str(body.url or "").strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=400,
            detail="Enter the address with http:// in front, e.g. http://host.docker.internal:11434",
        )

    # Checked before saving: switching to an address where nothing
    # answers would make every model disappear at once.
    found = await ollama_api.version(url)
    if found is None:
        raise HTTPException(
            status_code=400,
            detail=f"Nothing answered like Ollama at {url}. Make sure Ollama is running — and if it runs "
            "on this computer while ReiChat runs in Docker, use http://host.docker.internal:11434.",
        )

    await runtime_settings.set_external_ollama_url(url)
    await external.register_models(session, url)
    return {"mode": "external", "url": url, "version": found}
