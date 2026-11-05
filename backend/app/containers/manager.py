"""Everything that talks to the Docker daemon, and the only module that imports
the Docker SDK.

The SDK is blocking, so each public method here is async and runs the SDK call
on a worker thread; callers never block the event loop. A `Container` handed
back is a snapshot: reading its attributes (`name`, `status`, `attrs`) is free,
but anything that acts on it goes back through this module.

Talking to Ollama inside a container is llm/ollama_api.py's job, and what
the install progress looks like is realtime/progress.py's.
"""

import asyncio
import logging
import re
import threading
import typing

import docker
import docker.errors
from docker.models.containers import Container
from docker.types.containers import DeviceRequest

from app.containers import ports
from app.core.config import get_settings

__all__ = ["Container", "ContainerError", "ContainerManager"]

logger = logging.getLogger(__name__)

OLLAMA_IMAGE = "ollama/ollama"
OLLAMA_DATA_PATH = "/root/.ollama"
OLLAMA_INTERNAL_PORT = 11434
NETWORK_NAME = "chatbot-network"

# Labels, not env vars, identify a container as ours and carry its metadata:
# the old `ancestor=ollama/ollama:latest` filter stopped matching every
# existing container as soon as that tag moved to a newer image, and env vars
# leak into the Ollama process for no reason.
LABEL_MANAGED = "reichat.managed"
LABEL_MODEL = "reichat.model"
LABEL_PARAMETERS = "reichat.parameters"
LABEL_PORT = "reichat.port"
LABEL_GPU = "reichat.gpu"

_METADATA_LABELS = {"model": LABEL_MODEL, "parameters": LABEL_PARAMETERS, "port": LABEL_PORT}

# What Docker says when a GPU is requested on a host without the NVIDIA
# container runtime (macOS, AMD/Intel, CPU-only Linux/Windows) or with the
# runtime but no working driver.
_GPU_UNAVAILABLE_MARKERS = ("could not select device driver", "nvidia", "gpu")

_UNSAFE_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_.-]")

ProgressCallback = typing.Callable[[int, str], None]


class ContainerError(Exception):
    """A container operation failed; the message is written for the UI."""


def _docker_error_text(error: docker.errors.APIError) -> str:
    explanation = error.explanation
    if isinstance(explanation, bytes):
        explanation = explanation.decode(errors="replace")
    return str(explanation or error)


def _is_gpu_unavailable_error(error: docker.errors.APIError) -> bool:
    text = _docker_error_text(error).lower()
    return any(marker in text for marker in _GPU_UNAVAILABLE_MARKERS)


class ContainerManager:
    CONTAINER_STATUS = {
        "RUNNING": "running",
        "EXITED": "exited",
        "PAUSED": "paused",
        "RESTARTING": "restarting",
        "PULLING_MODEL": "pulling_model",
        # Running, but Ollama inside hasn't answered yet.
        "STARTING": "starting",
        # Running and reachable, but the model isn't in it — a download that
        # failed or was interrupted (e.g. the backend restarted mid-pull).
        "INCOMPLETE": "incomplete",
    }

    # One client for the whole process, not one per ContainerManager(): the
    # manager is constructed on every request and every 2s snapshot tick, and
    # nothing ever closed a per-instance client, leaking a connection each time.
    _client: docker.DockerClient | None = None
    _client_lock = threading.Lock()

    # None until the first container start tells us either way.
    _gpu_supported: bool | None = None

    @staticmethod
    def container_name(model: str, parameters: str) -> str:
        # Docker names allow [a-zA-Z0-9_.-] only; custom models can be
        # namespaced ("hf.co/user/repo"), so anything else becomes "-".
        return _UNSAFE_NAME_CHARS.sub("-", f"{model}_{parameters}")

    @staticmethod
    def volume_name(container_name: str) -> str:
        return f"reichat-ollama-{container_name}"

    # Connection

    async def ensure_connected(self) -> bool:
        """True once the daemon answers, connecting first if needed."""
        return await asyncio.to_thread(self._ensure_connected)

    @classmethod
    def _ensure_connected(cls) -> bool:
        return cls._ping() or cls._connect()

    @classmethod
    def _ping(cls) -> bool:
        client = cls._client
        if client is None:
            return False

        try:
            return bool(client.ping())
        except Exception as e:
            # The client object existing doesn't mean the daemon behind it is
            # still alive — a stale client used to report connected forever
            # after Docker Desktop was quit.
            logger.warning("Docker client failed a health check, dropping it: %s", e)
            cls._client = None
            return False

    @classmethod
    def _connect(cls) -> bool:
        with cls._client_lock:
            if cls._client is not None:
                return True

            try:
                cls._client = docker.DockerClient()
                return True
            except Exception as e:
                logger.warning("Failed to connect to Docker: %s", e)
                return False

    @classmethod
    def _require_client(cls) -> docker.DockerClient:
        if cls._client is None:
            raise ContainerError("Docker isn't running. Start Docker Desktop and try again.")
        return cls._client

    # The Ollama image

    async def is_ollama_image_pulled(self) -> bool:
        if not await self.ensure_connected():
            return False
        return await asyncio.to_thread(self._is_ollama_image_pulled)

    @classmethod
    def _is_ollama_image_pulled(cls) -> bool:
        try:
            cls._require_client().images.get(OLLAMA_IMAGE)
            return True
        except (docker.errors.DockerException, ContainerError):
            return False

    async def pull_ollama_image(self) -> None:
        """Downloads the newest ollama/ollama. Raises ContainerError."""
        if not await self.ensure_connected():
            raise ContainerError("Docker isn't running. Start Docker Desktop and try again.")
        await asyncio.to_thread(self._pull_ollama_image)

    @classmethod
    def _pull_ollama_image(cls) -> None:
        try:
            cls._require_client().images.pull(OLLAMA_IMAGE, "latest")
        except docker.errors.DockerException as e:
            logger.error("Error pulling ollama image: %s", e)
            raise ContainerError(f"Couldn't download the Ollama image: {e}") from e

    async def ensure_ollama_image(self, progress_cb: ProgressCallback | None = None) -> bool:
        """Pulls the base image unless it is already there. `progress_cb` is
        called from a worker thread."""
        if not await self.ensure_connected():
            return False
        return await asyncio.to_thread(self._ensure_ollama_image, progress_cb)

    @classmethod
    def _ensure_ollama_image(cls, progress_cb: ProgressCallback | None) -> bool:
        if cls._is_ollama_image_pulled():
            if progress_cb:
                progress_cb(100, "Base image already present")
            return True

        try:
            client = cls._require_client()
            layer_totals: dict[str, int] = {}
            layer_currents: dict[str, int] = {}
            current_layer_id = ""

            for event in client.api.pull(OLLAMA_IMAGE, tag="latest", stream=True, decode=True):
                layer_id = event.get("id", "")
                progress_detail = event.get("progressDetail") or {}

                if layer_id and progress_detail:
                    total = progress_detail.get("total", 0)
                    current = progress_detail.get("current", 0)
                    if total:
                        layer_totals[layer_id] = total
                        current_layer_id = layer_id
                    if layer_id in layer_totals:
                        layer_currents[layer_id] = current

                if layer_totals and progress_cb:
                    total_bytes = sum(layer_totals.values())
                    current_bytes = sum(layer_currents.get(lid, 0) for lid in layer_totals)
                    overall_pct = int(100 * current_bytes / total_bytes) if total_bytes else 0
                    if current_layer_id and layer_totals.get(current_layer_id):
                        layer_pct = int(
                            100 * layer_currents.get(current_layer_id, 0) / layer_totals[current_layer_id]
                        )
                        detail = f"Layer {current_layer_id[:12]} — {layer_pct}%"
                    else:
                        detail = "Preparing layers…"
                    progress_cb(overall_pct, detail)
            if progress_cb:
                progress_cb(100, "Base image ready")
            return True

        except Exception as e:
            logger.error("Error pulling ollama image: %s", e)
            return False

    async def image_id(self) -> str | None:
        if not await self.ensure_connected():
            return None
        return await asyncio.to_thread(self._image_id)

    @classmethod
    def _image_id(cls) -> str | None:
        try:
            image_id: str | None = cls._require_client().images.get(f"{OLLAMA_IMAGE}:latest").id
            return image_id
        except (docker.errors.DockerException, ContainerError):
            return None

    # Finding containers

    @staticmethod
    def _labels(container: Container) -> dict[str, str]:
        return container.attrs.get("Config", {}).get("Labels") or {}

    @staticmethod
    def is_managed_container(container: Container) -> bool:
        if ContainerManager._labels(container).get(LABEL_MANAGED) == "true":
            return True

        # Containers created before labels existed: still ours if they run the
        # Ollama image and carry the model env var the old code set.
        image = container.attrs.get("Config", {}).get("Image", "") or ""
        return image.startswith(OLLAMA_IMAGE) and (
            ContainerManager.get_container_metadata(container, "model") is not None
        )

    @staticmethod
    def get_container_metadata(container: Container, key: str) -> str | None:
        """`key` is "model", "parameters" or "port" — read from our labels,
        falling back to the env vars containers created before labels used."""
        if (label := _METADATA_LABELS.get(key)) and (value := ContainerManager._labels(container).get(label)):
            return str(value)

        for env in container.attrs.get("Config", {}).get("Env") or []:
            name, sep, env_value = env.partition("=")
            if sep and name == key:
                return str(env_value)

        return None

    @staticmethod
    def has_weights_volume(container: Container) -> bool:
        expected = ContainerManager.volume_name(container.name or "")
        return any(mount.get("Name") == expected for mount in container.attrs.get("Mounts") or [])

    @staticmethod
    def map_container(container: Container, status: str | None = None) -> dict[str, typing.Any]:
        return {
            "name": container.name or "No name",
            "status": container.status if status is None else status,
            "port": ContainerManager.get_container_metadata(container, "port"),
            "environment": {
                "model": ContainerManager.get_container_metadata(container, "model"),
                "parameters": ContainerManager.get_container_metadata(container, "parameters"),
            },
        }

    @staticmethod
    def base_url_for(container: Container) -> str | None:
        """In Docker mode this server is itself a container on the same
        `chatbot-network` bridge, so it addresses the target directly by
        container name over Docker's internal DNS — no host port needed,
        and no `host.docker.internal`, which isn't resolvable on Linux
        Docker Engine without an `extra_hosts` entry docker-compose.yaml
        doesn't set.

        Outside Docker (the backend run from source on the host, container
        management still going through the Docker socket), there's no
        shared network to piggyback on, so it has to go through the
        allocated host-published port instead. An IP literal, not
        "localhost" — see database_url() in db/session.py.
        """
        if get_settings().docker:
            return f"http://{container.name}:{OLLAMA_INTERNAL_PORT}"

        host_port = ContainerManager.get_container_metadata(container, "port")
        if host_port is None:
            return None
        return f"http://127.0.0.1:{host_port}"

    async def get_container(self, container_name: str) -> Container | None:
        if not await self.ensure_connected():
            return None
        return await asyncio.to_thread(self._get_container, container_name)

    @classmethod
    def _get_container(cls, container_name: str) -> Container | None:
        try:
            return cls._require_client().containers.get(container_name)
        except docker.errors.NotFound:
            return None
        except (docker.errors.DockerException, ContainerError) as e:
            logger.error("Error retrieving container: %s", e)
            return None

    async def list_managed(self, *, running_only: bool = False) -> list[Container]:
        """Every container this app manages (running ones only, if asked)."""
        if not await self.ensure_connected():
            return []
        return await asyncio.to_thread(self._list_managed, running_only)

    @classmethod
    def _list_managed(cls, running_only: bool) -> list[Container]:
        client = cls._require_client()
        containers = (
            client.containers.list(filters={"status": "running"})
            if running_only
            else client.containers.list(all=True)
        )
        return [container for container in containers if cls.is_managed_container(container)]

    async def get_ollama_base_url(self, model: str, parameters: str) -> str | None:
        container = await self.get_container(self.container_name(model, parameters))
        if container is None or container.status != self.CONTAINER_STATUS["RUNNING"]:
            return None
        return self.base_url_for(container)

    async def gpu_status(self) -> bool | None:
        """True/False once a container start has shown whether GPUs work on
        this host, else whatever existing containers were created with, else
        None (unknown — nothing has been started yet)."""
        if ContainerManager._gpu_supported is not None:
            return ContainerManager._gpu_supported

        if not await self.ensure_connected():
            return None
        return await asyncio.to_thread(self._gpu_status)

    @classmethod
    def _gpu_status(cls) -> bool | None:
        for container in cls._require_client().containers.list(
            all=True, filters={"label": f"{LABEL_MANAGED}=true"}
        ):
            gpu_label = cls._labels(container).get(LABEL_GPU)
            if gpu_label in ("true", "false"):
                return gpu_label == "true"
        return None

    # Starting, stopping and replacing model containers

    async def create_and_start_container(self, model_name: str, parameters: str) -> Container:
        """Starts the model's container, creating it first if needed. Asks
        for every GPU, and if the host can't provide one, recreates the
        container CPU-only — most machines (every Mac, AMD/Intel GPUs, no
        GPU at all) have no NVIDIA runtime, and Docker refuses to even start
        a container that requests one there. Raises ContainerError.
        """
        if not await self.ensure_connected():
            raise ContainerError("Docker isn't running. Start Docker Desktop and try again.")

        container_name = self.container_name(model_name, parameters)

        if (container := await self.get_container(container_name)) is not None:
            await self.start_container(container)
            return container

        port = await ports.allocate_port(container_name)
        return await asyncio.to_thread(
            self._create_and_start_container, container_name, model_name, parameters, port
        )

    @classmethod
    def _create_and_start_container(
        cls, container_name: str, model_name: str, parameters: str, port: int
    ) -> Container:
        use_gpu = cls._gpu_supported is not False
        container = cls._create_container(container_name, model_name, parameters, port, use_gpu)

        try:
            container.start()
        except docker.errors.APIError as e:
            if not (use_gpu and _is_gpu_unavailable_error(e)):
                logger.error("Error starting container %s: %s", container_name, e)
                raise ContainerError(f"Couldn't start the model container: {_docker_error_text(e)}") from e

            logger.warning(
                "No usable GPU (%s) — recreating %s CPU-only", _docker_error_text(e), container_name
            )
            cls._gpu_supported = False
            container.remove(force=True)
            container = cls._create_container(container_name, model_name, parameters, port, use_gpu=False)
            cls._start_container(container)
            return container

        if use_gpu:
            cls._gpu_supported = True
        container.reload()
        return container

    @classmethod
    def _create_container(
        cls, container_name: str, model_name: str, parameters: str, port: int, use_gpu: bool
    ) -> Container:
        try:
            return cls._require_client().containers.create(
                name=container_name,
                image=f"{OLLAMA_IMAGE}:latest",
                detach=True,
                # Loopback only, like every port docker-compose.yaml
                # publishes — a bare port binds 0.0.0.0, putting an
                # unauthenticated Ollama API on the user's whole network.
                ports={f"{OLLAMA_INTERNAL_PORT}/tcp": ("127.0.0.1", port)},
                network=NETWORK_NAME,
                device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])] if use_gpu else [],
                hostname=container_name,
                # Model weights live in a named volume rather than the
                # container's writable layer, so recreating the container
                # (e.g. onto a newer Ollama image) doesn't re-download them.
                volumes={cls.volume_name(container_name): {"bind": OLLAMA_DATA_PATH, "mode": "rw"}},
                labels={
                    LABEL_MANAGED: "true",
                    LABEL_MODEL: model_name,
                    LABEL_PARAMETERS: parameters,
                    LABEL_PORT: str(port),
                    LABEL_GPU: "true" if use_gpu else "false",
                },
            )
        except docker.errors.APIError as e:
            logger.error("Error creating container %s: %s", container_name, e)
            raise ContainerError(f"Couldn't create the model container: {_docker_error_text(e)}") from e

    async def start_container(self, container: Container) -> None:
        await asyncio.to_thread(self._start_container, container)

    @classmethod
    def _start_container(cls, container: Container) -> None:
        if container.status == cls.CONTAINER_STATUS["RUNNING"]:
            return

        try:
            container.start()
        except docker.errors.APIError as e:
            logger.error("Error starting container %s: %s", container.name, e)
            if _is_gpu_unavailable_error(e):
                raise ContainerError(
                    "This model was set up to use a graphics card (GPU) that isn't available now. "
                    "Remove it on the Models page and create it again to run it on the CPU."
                ) from e
            raise ContainerError(f"Couldn't start the model container: {_docker_error_text(e)}") from e

        container.reload()

    async def recreate_on_current_image(self, container: Container) -> Container:
        """Replaces a container with one on the current ollama/ollama image,
        same name, port, labels and weights volume — how a model moves to a
        newer Ollama without downloading again. Raises ContainerError."""
        return await asyncio.to_thread(self._recreate_on_current_image, container)

    @classmethod
    def _recreate_on_current_image(cls, container: Container) -> Container:
        model = cls.get_container_metadata(container, "model")
        parameters = cls.get_container_metadata(container, "parameters")
        port = cls.get_container_metadata(container, "port")
        if not (model and parameters and port and cls.has_weights_volume(container)):
            raise ContainerError(
                f"{container.name} was created before model files were kept separately, so updating it would "
                "mean downloading the model again. Remove it and create it again to make it updatable."
            )

        was_running = container.status == cls.CONTAINER_STATUS["RUNNING"]
        try:
            container.remove(force=True)
        except docker.errors.APIError as e:
            raise ContainerError(f"Couldn't replace {container.name}: {_docker_error_text(e)}") from e

        use_gpu = cls._labels(container).get(LABEL_GPU) != "false"
        if cls._gpu_supported is False:
            use_gpu = False
        replacement = cls._create_container(container.name or "", model, parameters, int(port), use_gpu)
        if was_running:
            cls._start_container(replacement)
        return replacement

    async def stop_container(self, container_name: str) -> None:
        if not await self.ensure_connected():
            return
        await asyncio.to_thread(self._stop_container, container_name)

    @classmethod
    def _stop_container(cls, container_name: str) -> None:
        if (container := cls._get_container(container_name)) is not None:
            try:
                container.stop()
            except docker.errors.APIError as e:
                logger.warning("Error stopping container %s: %s", container_name, e)

    async def remove_container(self, container_name: str) -> str | None:
        """Removes the container and its weights volume. Returns the id of the
        container that was removed, None if there was none or it wouldn't go."""
        if not await self.ensure_connected():
            return None
        return await asyncio.to_thread(self._remove_container, container_name)

    @classmethod
    def _remove_container(cls, container_name: str) -> str | None:
        removed_id = None
        if (container := cls._get_container(container_name)) is not None:
            try:
                container.stop()
                container.remove()
            except docker.errors.APIError as e:
                logger.warning("Error removing container %s: %s", container_name, e)
                return None
            removed_id = container.id

        # `remove(v=True)` only deletes anonymous volumes — the named one
        # holding the weights has to go explicitly, or "Remove" frees nothing.
        try:
            cls._require_client().volumes.get(cls.volume_name(container_name)).remove()
        except docker.errors.NotFound:
            pass
        except docker.errors.APIError as e:
            logger.warning("Error removing volume for %s: %s", container_name, e)
        return removed_id

    # Other containers the app runs (web search) and what Docker knows about the host

    async def ensure_service_running(
        self,
        *,
        image: str,
        name: str,
        internal_port: int,
        host_port: int,
        environment: dict[str, str],
        labels: dict[str, str],
        config_dir: str,
        config_archive: bytes,
        failure: str,
    ) -> None:
        """Starts the named service container, creating it (pulling `image`,
        publishing `host_port` on loopback, unpacking `config_archive` into
        `config_dir`) if it doesn't exist yet. Raises ContainerError starting
        with `failure`, e.g. "Couldn't set up web search"."""
        if not await self.ensure_connected():
            raise ContainerError("Docker isn't running. Start Docker Desktop and try again.")
        await asyncio.to_thread(
            self._ensure_service_running,
            image,
            name,
            internal_port,
            host_port,
            environment,
            labels,
            config_dir,
            config_archive,
            failure,
        )

    @classmethod
    def _ensure_service_running(
        cls,
        image: str,
        name: str,
        internal_port: int,
        host_port: int,
        environment: dict[str, str],
        labels: dict[str, str],
        config_dir: str,
        config_archive: bytes,
        failure: str,
    ) -> None:
        if (existing := cls._get_container(name)) is not None:
            cls._start_container(existing)
            return

        client = cls._require_client()
        try:
            client.images.pull(image)
            container = client.containers.create(
                image,
                name=name,
                detach=True,
                network=NETWORK_NAME,
                ports={f"{internal_port}/tcp": ("127.0.0.1", host_port)},
                environment=environment,
                labels=labels,
                restart_policy={"Name": "unless-stopped"},
            )
            container.put_archive(config_dir, config_archive)
            container.start()
        except docker.errors.APIError as e:
            raise ContainerError(f"{failure}: {_docker_error_text(e)}") from e

    async def force_remove_container(self, container_name: str, *, failure: str) -> None:
        """Removes the container without touching any volume. Raises
        ContainerError starting with `failure`, e.g. "Couldn't turn web search off"."""
        if not await self.ensure_connected():
            return
        await asyncio.to_thread(self._force_remove_container, container_name, failure)

    @classmethod
    def _force_remove_container(cls, container_name: str, failure: str) -> None:
        container = cls._get_container(container_name)
        if container is None:
            return
        try:
            container.remove(force=True)
        except docker.errors.APIError as e:
            raise ContainerError(f"{failure}: {_docker_error_text(e)}") from e

    async def host_memory_bytes(self) -> int | None:
        if not await self.ensure_connected():
            return None
        return await asyncio.to_thread(self._host_memory_bytes)

    @classmethod
    def _host_memory_bytes(cls) -> int | None:
        try:
            memory: int | None = cls._require_client().info().get("MemTotal")
            return memory
        except Exception as e:
            logger.warning("Could not read host RAM from Docker: %s", e)
            return None

    async def exec_in_gpu_container(self, command: list[str]) -> str | None:
        """Output of `command` run inside a running GPU-enabled Ollama
        container, or None if there is none or the command failed."""
        if not await self.ensure_connected():
            return None
        return await asyncio.to_thread(self._exec_in_gpu_container, command)

    @classmethod
    def _exec_in_gpu_container(cls, command: list[str]) -> str | None:
        try:
            candidates = cls._require_client().containers.list(
                filters={"label": [f"{LABEL_MANAGED}=true", f"{LABEL_GPU}=true"], "status": "running"}
            )
        except Exception as e:
            logger.warning("Could not list containers to run %s in: %s", command[0], e)
            return None

        for container in candidates:
            try:
                exit_code, output = container.exec_run(command)
                if exit_code == 0:
                    text: str = output.decode(errors="replace")
                    return text
            except Exception as e:
                logger.warning("Could not run %s inside %s: %s", command[0], container.name, e)

        return None
