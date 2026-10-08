"""Host RAM/VRAM detection for the "which model fits my machine" feature.

RAM comes from the Docker SDK (`client.info()["MemTotal"]`) — on Docker
Desktop this is the memory allocated to the Docker VM, which is actually the
more relevant number than raw host RAM for "will this fit in a container".
VRAM has no Docker SDK equivalent; nvidia-smi is the only reasonably
portable way to read it without a vendor SDK, so non-NVIDIA machines (AMD,
Apple Silicon, CPU-only) get None rather than a guess.
"""

import logging
import shutil
import subprocess

from .ContainerManager import LABEL_GPU, LABEL_MANAGED, ContainerManager

logger = logging.getLogger(__name__)

NVIDIA_SMI_TIMEOUT_SECONDS = 5


def get_host_ram_bytes(container_manager: ContainerManager) -> int | None:
    if container_manager.client is None:
        return None

    try:
        return container_manager.client.info().get("MemTotal")
    except Exception as e:
        logger.warning("Could not read host RAM from Docker: %s", e)
        return None


NVIDIA_SMI_QUERY = ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]

# Total VRAM doesn't change while the app runs; once read, keep it.
_cached_vram_bytes: int | None = None


def _parse_nvidia_smi_mib(output: str) -> int:
    return int(output.strip().splitlines()[0]) * 1024 * 1024  # nvidia-smi reports MiB


def get_gpu_vram_bytes() -> int | None:
    """nvidia-smi on this machine — only works when the backend runs from
    source on the host. Under compose the backend container has neither the
    binary nor a GPU."""
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


def get_gpu_vram_bytes_via_ollama_container(container_manager: ContainerManager) -> int | None:
    """Runs nvidia-smi inside a running GPU-enabled Ollama container — the
    NVIDIA runtime mounts the binary into any container started with a GPU
    request, so this is the one place that sees the GPU under compose."""
    if container_manager.client is None:
        return None

    try:
        candidates = container_manager.client.containers.list(
            filters={"label": [f"{LABEL_MANAGED}=true", f"{LABEL_GPU}=true"], "status": "running"}
        )
    except Exception as e:
        logger.warning("Could not list containers to read GPU VRAM: %s", e)
        return None

    for container in candidates:
        try:
            exit_code, output = container.exec_run(NVIDIA_SMI_QUERY)
            if exit_code == 0:
                return _parse_nvidia_smi_mib(output.decode(errors="replace"))
        except Exception as e:
            logger.warning("Could not read GPU VRAM inside %s: %s", container.name, e)

    return None


def detect_vram_bytes(container_manager: ContainerManager) -> int | None:
    global _cached_vram_bytes

    if _cached_vram_bytes is None:
        _cached_vram_bytes = get_gpu_vram_bytes() or get_gpu_vram_bytes_via_ollama_container(container_manager)

    return _cached_vram_bytes
