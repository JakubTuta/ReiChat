"""The "my own Ollama" runtime: one Ollama the user already runs (the
native app — the only way to get GPU acceleration on a Mac — or a server
elsewhere), instead of one app-managed container per model. Ollama then
schedules memory across models itself.

Its models are reported in the same shape as the managed containers
(ContainerManager.map_container), so the rest of the app and the frontend
don't need a second code path for listing them.
"""

import logging
import typing

import requests

from django_app import app_settings, ollama_api, progress_state

from .ContainerManager import ContainerManager

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 5


def url() -> str | None:
    return app_settings.external_ollama_url()


def version(base_url: str) -> str | None:
    """Ollama's version, or None if nothing answering like Ollama is there."""
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/version", timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return str(response.json().get("version") or "unknown")
    except (requests.exceptions.RequestException, ValueError, AttributeError):
        return None


def list_models(base_url: str) -> list[dict[str, typing.Any]]:
    sizes = ollama_api.installed_model_sizes(base_url)
    if sizes is None:
        return []

    listed = []
    for name in sizes:
        model, _, parameters = name.partition(":")
        parameters = parameters or "latest"
        container_name = ContainerManager.container_name(model, parameters)
        listed.append({
            "name": container_name,
            "status": "pulling_model" if progress_state.is_in_progress(container_name) else "running",
            "port": None,
            "environment": {"model": model, "parameters": parameters},
        })

    # A first-time pull hasn't shown up in /api/tags yet but is in progress.
    known = {entry["name"] for entry in listed}
    for name in progress_state.get_snapshot()["progress"]:
        if name not in known and progress_state.is_in_progress(name) and "_" in name:
            model, _, parameters = name.rpartition("_")
            listed.append({
                "name": name,
                "status": "pulling_model",
                "port": None,
                "environment": {"model": model, "parameters": parameters},
            })
    return listed


def register_models(base_url: str) -> int:
    """Makes every model already on the user's Ollama selectable: one that
    the ollama.com catalog doesn't list (or a tag it doesn't list) is added
    as a custom model. Returns how many entries were added."""
    from django_app.catalog import registry
    from django_app.models import AIModel, AIModelVersion

    added = 0
    for name, size in (ollama_api.installed_model_sizes(base_url) or {}).items():
        model, _, parameters = name.partition(":")
        ai_model, created = AIModel.objects.get_or_create(
            model=model,
            defaults={"name": model.rsplit("/", 1)[-1], "description": "Found on your Ollama.", "is_custom": True},
        )
        _, version_created = AIModelVersion.objects.get_or_create(
            ai_model=ai_model,
            parameters=parameters or "latest",
            defaults={"size": registry.format_size(size), "size_bytes": size},
        )
        added += int(created or version_created)
    return added


def has_model(base_url: str, model: str, parameters: str) -> bool | None:
    sizes = ollama_api.installed_model_sizes(base_url)
    if sizes is None:
        return None
    return f"{model}:{parameters}" in sizes


def delete_model(base_url: str, model: str, parameters: str) -> bool:
    try:
        response = requests.delete(
            f"{base_url}/api/delete", json={"model": f"{model}:{parameters}"}, timeout=REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.warning("Could not delete %s:%s from %s: %s", model, parameters, base_url, e)
        return False
    return True
