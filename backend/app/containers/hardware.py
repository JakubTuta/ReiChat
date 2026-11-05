"""Host VRAM detection for the "which model fits my machine" feature.

(RAM comes straight from the Docker daemon — see ContainerManager.host_memory_bytes
— and on Docker Desktop that is the memory given to the Docker VM, which is
actually the more relevant number than raw host RAM for "will this fit in a
container".) VRAM has no Docker equivalent; nvidia-smi is the only reasonably
portable way to read it without a vendor SDK, so non-NVIDIA machines (AMD,
Apple Silicon, CPU-only) get None rather than a guess.
"""

import asyncio
import logging
import shutil
import subprocess

from app.containers.manager import ContainerManager

logger = logging.getLogger(__name__)

NVIDIA_SMI_TIMEOUT_SECONDS = 5

NVIDIA_SMI_QUERY = ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]

# Total VRAM doesn't change while the app runs; once read, keep it.
_cached_vram_bytes: int | None = None


def _parse_nvidia_smi_mib(output: str) -> int:
    return int(output.strip().splitlines()[0]) * 1024 * 1024  # nvidia-smi reports MiB


def get_gpu_vram_bytes() -> int | None:
    """nvidia-smi on this machine — only works when the backend runs from
    source on the host. Under compose the backend container has neither the
    binary nor a GPU. Blocking: run it off the event loop."""
    if shutil.which("nvidia-smi") is None:
        return None

    try:
        result = subprocess.run(
            NVIDIA_SMI_QUERY,
            capture_output=True,
            text=True,
            timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
            check=True,
        )
        return _parse_nvidia_smi_mib(result.stdout)
    except Exception as e:
        logger.warning("Could not read GPU VRAM via nvidia-smi: %s", e)
        return None


async def get_gpu_vram_bytes_via_ollama_container(manager: ContainerManager) -> int | None:
    """Runs nvidia-smi inside a running GPU-enabled Ollama container — the
    NVIDIA runtime mounts the binary into any container started with a GPU
    request, so this is the one place that sees the GPU under compose."""
    output = await manager.exec_in_gpu_container(NVIDIA_SMI_QUERY)
    if output is None:
        return None

    try:
        return _parse_nvidia_smi_mib(output)
    except (ValueError, IndexError) as e:
        logger.warning("Could not read GPU VRAM from nvidia-smi's output %r: %s", output, e)
        return None


async def detect_vram_bytes(manager: ContainerManager) -> int | None:
    global _cached_vram_bytes

    if _cached_vram_bytes is None:
        _cached_vram_bytes = await asyncio.to_thread(
            get_gpu_vram_bytes
        ) or await get_gpu_vram_bytes_via_ollama_container(manager)

    return _cached_vram_bytes


def reset() -> None:
    """For tests, which share this module's state."""
    global _cached_vram_bytes
    _cached_vram_bytes = None
