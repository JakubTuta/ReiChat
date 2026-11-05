"""Direct calls to an Ollama server's management API (/api/show, /api/ps,
...) — things LangChain's ChatOllama doesn't expose. Container lookup lives
in containers/manager.py; this module only speaks HTTP to a base URL it's
given.
"""

import asyncio
import json
import logging
import time
import typing

import httpx

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT_SECONDS = 5.0
# For questions asked while a container may still be booting: a short answer
# or none at all.
_PROBE_TIMEOUT_SECONDS = 2.0
# Ollama streams a pull for as long as it takes; only silence this long means
# something is wrong.
_PULL_IDLE_TIMEOUT_SECONDS = 600.0

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


async def version(base_url: str) -> str | None:
    """Ollama's version, or None if nothing answering like Ollama is there."""
    try:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.get(f"{base_url.rstrip('/')}/api/version")
        response.raise_for_status()
        return str(response.json().get("version") or "unknown")
    except (httpx.HTTPError, ValueError, AttributeError):
        return None


async def wait_until_ready(base_url: str, max_wait_seconds: float = 60.0, poll_interval: float = 1.0) -> bool:
    """Polls until Ollama answers, e.g. in a container that was just started."""
    deadline = time.monotonic() + max_wait_seconds
    while time.monotonic() < deadline:
        try:
            async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_SECONDS) as client:
                response = await client.get(f"{base_url}/api/version")
            if response.status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        await asyncio.sleep(poll_interval)
    return False


async def is_model_installed(base_url: str, model: str, parameters: str) -> bool | None:
    """Whether Ollama at `base_url` holds `model:parameters`, or None if
    Ollama can't be asked right now (still booting, or unreachable)."""
    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_SECONDS) as client:
            response = await client.get(f"{base_url}/api/tags")
        response.raise_for_status()
        listed = response.json().get("models") or []
    except (httpx.HTTPError, ValueError, AttributeError):
        return None

    wanted = f"{model}:{parameters}"
    return any(wanted in (entry.get("name"), entry.get("model")) for entry in listed)


async def delete_model(base_url: str, model: str, parameters: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
            # `client.delete` takes no body, and Ollama wants one.
            response = await client.request(
                "DELETE", f"{base_url}/api/delete", json={"model": f"{model}:{parameters}"}
            )
        response.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning("Could not delete %s:%s from %s: %s", model, parameters, base_url, e)
        return False
    return True


ProgressCallback = typing.Callable[[int, str], None]


async def pull_model(
    base_url: str, model: str, parameters: str, progress_cb: ProgressCallback | None = None
) -> tuple[bool, str | None]:
    """Returns (success, error_message). Ollama's /api/pull streams
    `{"error": "..."}` on failure (bad tag, manifest not found, disk
    full) with a 200 status — the request never raises on its own, so
    every line has to be checked for an "error" key or a failed pull
    silently reports 100% / "Model ready" like a success.
    """
    layer_totals: dict[str, int] = {}
    layer_currents: dict[str, int] = {}
    current_digest = ""

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(_PULL_IDLE_TIMEOUT_SECONDS)) as client:
            async with client.stream(
                "POST", f"{base_url}/api/pull", json={"model": f"{model}:{parameters}", "stream": True}
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if error := event.get("error"):
                        logger.error("Ollama reported a pull error for %s:%s: %s", model, parameters, error)
                        return False, str(error)

                    status = event.get("status", "")
                    digest = event.get("digest", "")
                    total = event.get("total", 0)
                    completed = event.get("completed", 0)

                    if digest:
                        if total:
                            layer_totals[digest] = total
                            current_digest = digest
                        if digest in layer_totals:
                            layer_currents[digest] = completed

                    if layer_totals and progress_cb:
                        total_bytes = sum(layer_totals.values())
                        current_bytes = sum(layer_currents.get(d, 0) for d in layer_totals)
                        overall_pct = int(100 * current_bytes / total_bytes) if total_bytes else 0
                        if current_digest and layer_totals.get(current_digest):
                            layer_pct = int(
                                100 * layer_currents.get(current_digest, 0) / layer_totals[current_digest]
                            )
                            short_id = (
                                current_digest[7:19]
                                if current_digest.startswith("sha256:")
                                else current_digest[:12]
                            )
                            detail = f"Layer {short_id} — {layer_pct}%"
                        else:
                            detail = status or "Pulling…"
                        progress_cb(overall_pct, detail)
                    elif progress_cb and status:
                        progress_cb(0, status)
    except httpx.HTTPError as e:
        logger.error("Error pulling model %s:%s: %s", model, parameters, e)
        return False, str(e)

    if progress_cb:
        progress_cb(100, "Model ready")
    return True, None
