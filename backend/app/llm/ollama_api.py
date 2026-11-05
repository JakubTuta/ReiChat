"""Direct calls to an Ollama server's management API (/api/show, /api/ps,
...) — things LangChain's ChatOllama doesn't expose. Container lookup lives
in containers/manager.py; this module only speaks HTTP to a base URL it's
given.
"""

import logging
import typing

import httpx

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT_SECONDS = 5.0

# /api/show describes a model file, which never changes for a given tag on a
# given server — safe to keep for the life of the process.
_show_cache: dict[tuple[str, str], dict[str, typing.Any]] = {}


async def show_model(base_url: str, model_string: str) -> dict[str, typing.Any] | None:
    key = (base_url, model_string)
    if key in _show_cache:
        return _show_cache[key]

    try:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(f"{base_url}/api/show", json={"model": model_string})
        response.raise_for_status()
        info: dict[str, typing.Any] = response.json()
    except (httpx.HTTPError, ValueError) as e:
        logger.warning("Could not read model info for %s from %s: %s", model_string, base_url, e)
        return None

    _show_cache[key] = info
    return info


def context_length(show_info: dict[str, typing.Any]) -> int | None:
    """The model's trained maximum context, e.g. `qwen2.context_length` —
    the key is prefixed with the architecture name, so match the suffix."""
    for key, value in (show_info.get("model_info") or {}).items():
        if key.endswith(".context_length") and isinstance(value, int) and value > 0:
            return value
    return None


def capabilities(show_info: dict[str, typing.Any]) -> list[str]:
    return [c for c in show_info.get("capabilities") or [] if isinstance(c, str)]


class ThinkingControls(typing.TypedDict):
    # What `think` accepts for this model: booleans for on/off, strings for
    # named levels (e.g. gpt-oss's "low"/"medium"/"high").
    values: list[bool | str]
    default: bool | str


def thinking_controls(show_info: dict[str, typing.Any]) -> ThinkingControls | None:
    """None when the model can't think at all. Older Ollama versions list
    the "thinking" capability without the `thinking` metadata block — plain
    on/off is assumed then, defaulting to on like Ollama does."""
    if "thinking" not in capabilities(show_info):
        return None

    raw = show_info.get("thinking") or {}
    values = [v for v in raw.get("values") or [] if isinstance(v, (bool, str))] or [False, True]
    if values == [False]:
        return None

    default = raw.get("default")
    if not isinstance(default, (bool, str)):
        default = True

    return {"values": values, "default": default}


async def installed_model_sizes(base_url: str) -> dict[str, int] | None:
    """name:tag -> size on disk in bytes, per /api/tags."""
    try:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.get(f"{base_url}/api/tags")
        response.raise_for_status()
        listed = response.json().get("models") or []
    except (httpx.HTTPError, ValueError, AttributeError) as e:
        logger.warning("Could not list installed models at %s: %s", base_url, e)
        return None

    return {str(entry.get("name") or entry.get("model")): int(entry.get("size") or 0) for entry in listed}


async def unload_model(base_url: str, model_string: str) -> bool:
    """Frees the memory a loaded model holds right away, instead of after
    Ollama's 5-minute keep-alive. True when Ollama confirmed it."""
    try:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{base_url}/api/generate", json={"model": model_string, "keep_alive": 0}
            )
        response.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning("Could not unload %s at %s: %s", model_string, base_url, e)
        return False
    return True


class RunningModel(typing.TypedDict):
    name: str
    size: int
    size_vram: int
    context_length: int | None
    expires_at: str | None


async def running_models(base_url: str) -> list[RunningModel] | None:
    """What /api/ps reports as loaded in memory right now, or None if Ollama
    can't be asked."""
    try:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.get(f"{base_url}/api/ps")
        response.raise_for_status()
        listed = response.json().get("models") or []
    except (httpx.HTTPError, ValueError, AttributeError) as e:
        logger.warning("Could not list running models at %s: %s", base_url, e)
        return None

    return [
        {
            "name": str(entry.get("name") or entry.get("model") or ""),
            "size": int(entry.get("size") or 0),
            "size_vram": int(entry.get("size_vram") or 0),
            "context_length": entry.get("context_length"),
            "expires_at": entry.get("expires_at"),
        }
        for entry in listed
    ]
