"""Verifies parsed model variants against Ollama's OCI-compatible registry.

registry.ollama.ai serves image manifests without authentication — verified
live: a GET to /v2/library/<name>/manifests/<tag> returns 200 with a layer
list whose summed `size` is the exact pull size in bytes. A 200 also proves
the tag is actually pullable, which is stronger than anything the library
page's size chips ("8b") tell us.

/v2/library/<name>/tags/list returns 404 — Ollama does not implement OCI tag
enumeration, so this module only *confirms* tags the parser already found;
it cannot be used for discovery.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import typing

import httpx

logger = logging.getLogger(__name__)

_MANIFEST_URL = "https://registry.ollama.ai/v2/library/{name}/manifests/{tag}"
_ACCEPT_HEADER = "application/vnd.docker.distribution.manifest.v2+json"

Status = typing.Literal["ok", "not_found", "error"]


@dataclasses.dataclass(frozen=True)
class VerifyResult:
    status: Status
    size_bytes: int | None = None


async def verify_tag(client: httpx.AsyncClient, name: str, tag: str) -> VerifyResult:
    url = _MANIFEST_URL.format(name=name, tag=tag)
    try:
        response = await client.get(url, headers={"Accept": _ACCEPT_HEADER}, timeout=10.0)
    except httpx.HTTPError as e:
        logger.warning("Registry request failed for %s:%s: %s", name, tag, e)
        return VerifyResult(status="error")

    if response.status_code == 404:
        return VerifyResult(status="not_found")
    if response.status_code != 200:
        logger.warning("Registry returned %d for %s:%s", response.status_code, name, tag)
        return VerifyResult(status="error")

    try:
        manifest = response.json()
    except ValueError:
        return VerifyResult(status="error")

    size_bytes = manifest.get("config", {}).get("size", 0)
    size_bytes += sum(layer.get("size", 0) for layer in manifest.get("layers", []))

    return VerifyResult(status="ok", size_bytes=size_bytes)


async def verify_tags(
    names_and_tags: list[tuple[str, str]],
    max_concurrent: int = 10,
    on_result: typing.Callable[[str, str, VerifyResult], None] | None = None,
) -> dict[tuple[str, str], VerifyResult]:
    """Verify many (name, tag) pairs concurrently over one connection pool."""
    results: dict[tuple[str, str], VerifyResult] = {}
    slots = asyncio.Semaphore(max_concurrent)

    async with httpx.AsyncClient(http2=False) as client:

        async def verify_one(name: str, tag: str) -> None:
            async with slots:
                result = await verify_tag(client, name, tag)
            results[(name, tag)] = result
            if on_result is not None:
                on_result(name, tag, result)

        async with asyncio.TaskGroup() as group:
            for name, tag in names_and_tags:
                group.create_task(verify_one(name, tag))

    return results


def format_size(size_bytes: int | None) -> str:
    if not size_bytes:
        return "unknown"

    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024.0

    return f"{value:.1f} PB"
