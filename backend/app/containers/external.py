"""The "my own Ollama" runtime: one Ollama the user already runs (the
native app — the only way to get GPU acceleration on a Mac — or a server
elsewhere), instead of one app-managed container per model. Ollama then
schedules memory across models itself.

Its models are reported in the same shape as the managed containers
(ContainerManager.map_container), so the rest of the app and the frontend
don't need a second code path for listing them.
"""

import typing

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.catalog import registry
from app.containers.manager import ContainerManager
from app.db.models import AIModel, AIModelVersion
from app.llm import ollama_api
from app.realtime import progress


async def list_models(base_url: str) -> list[dict[str, typing.Any]]:
    sizes = await ollama_api.installed_model_sizes(base_url)
    if sizes is None:
        return []

    listed = []
    for name in sizes:
        model, _, parameters = name.partition(":")
        parameters = parameters or "latest"
        container_name = ContainerManager.container_name(model, parameters)
        listed.append(
            {
                "name": container_name,
                "status": "pulling_model" if progress.is_in_progress(container_name) else "running",
                "port": None,
                "environment": {"model": model, "parameters": parameters},
            }
        )

    # A first-time pull hasn't shown up in /api/tags yet but is in progress.
    known = {entry["name"] for entry in listed}
    for name in progress.get_snapshot()["progress"]:
        if name not in known and progress.is_in_progress(name) and "_" in name:
            model, _, parameters = name.rpartition("_")
            listed.append(
                {
                    "name": name,
                    "status": "pulling_model",
                    "port": None,
                    "environment": {"model": model, "parameters": parameters},
                }
            )
    return listed


async def register_models(session: AsyncSession, base_url: str) -> int:
    """Makes every model already on the user's Ollama selectable: one that
    the ollama.com catalog doesn't list (or a tag it doesn't list) is added
    as a custom model. Returns how many entries were added."""
    added = 0
    for name, size in (await ollama_api.installed_model_sizes(base_url) or {}).items():
        model, _, parameters = name.partition(":")

        model_id = (
            await session.execute(
                insert(AIModel)
                .values(
                    model=model,
                    name=model.rsplit("/", 1)[-1],
                    description="Found on your Ollama.",
                    is_custom=True,
                )
                .on_conflict_do_nothing(index_elements=[AIModel.model])
                .returning(AIModel.id)
            )
        ).scalar_one_or_none()
        model_created = model_id is not None
        if model_id is None:
            model_id = (await session.execute(select(AIModel.id).where(AIModel.model == model))).scalar_one()

        version_created = (
            await session.execute(
                insert(AIModelVersion)
                .values(
                    ai_model_id=model_id,
                    parameters=parameters or "latest",
                    size=registry.format_size(size),
                    size_bytes=size,
                )
                .on_conflict_do_nothing(constraint="unique_ai_model_version")
                .returning(AIModelVersion.id)
            )
        ).scalar_one_or_none() is not None

        added += int(model_created or version_created)

    await session.commit()
    return added


async def has_model(base_url: str, model: str, parameters: str) -> bool | None:
    sizes = await ollama_api.installed_model_sizes(base_url)
    if sizes is None:
        return None
    return f"{model}:{parameters}" in sizes
