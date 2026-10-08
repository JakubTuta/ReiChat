import json
import logging
import os
import threading
import time
import typing

import docker
import docker.errors
import requests
from django.db import transaction
from docker.models.containers import Container
from docker.models.images import Image
from docker.models.networks import Network
from docker.types.containers import DeviceRequest

from django_app import progress_state

from .models import ContainerPort

logger = logging.getLogger(__name__)

PORT_RANGE_START = 11434

OLLAMA_IMAGE = "ollama/ollama"
OLLAMA_DATA_PATH = "/root/.ollama"

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

    # A module-level singleton, not a per-instance client — ContainerManager()
    # is constructed on every chat message, every 2s snapshot tick, and every
    # view, and nothing ever closed the client, leaking a connection each time.
    _client: docker.DockerClient | None = None
    _client_lock = threading.Lock()

    # None until the first container start tells us either way.
    _gpu_supported: bool | None = None

    # Container ids whose model Ollama has confirmed it holds. A model only
    # disappears when its container is removed, so once confirmed this never
    # needs asking again — it keeps /api/tags off the 2s snapshot path.
    _installed_ids: set[str] = set()
    _installed_lock = threading.Lock()

    def __init__(self) -> None:
        if ContainerManager._client is None:
            self.connect_to_docker()

    @property
    def client(self) -> docker.DockerClient | None:
        return ContainerManager._client

    @staticmethod
    def container_name(model: str, parameters: str) -> str:
        return f"{model}_{parameters}"

    @staticmethod
    def volume_name(container_name: str) -> str:
        return f"reichat-ollama-{container_name}"

    def is_connected(self) -> bool:
        if ContainerManager._client is None:
            return False

        try:
            return bool(ContainerManager._client.ping())
        except Exception as e:
            # The client object existing doesn't mean the daemon behind it is
            # still alive — a stale client used to report connected forever
            # after Docker Desktop was quit.
            logger.warning("Docker client failed a health check, dropping it: %s", e)
            ContainerManager._client = None
            return False

    def connect_to_docker(self) -> bool:
        with ContainerManager._client_lock:
            if ContainerManager._client is not None:
                return True

            try:
                ContainerManager._client = docker.DockerClient()

                return True

            except Exception as e:
                logger.warning("Failed to connect to Docker: %s", e)
                return False

    def is_ollama_image_pulled(self) -> bool:
        if not self.is_connected() or self.client is None:
            return False

        try:
            self.client.images.get(OLLAMA_IMAGE)

            return True

        except docker.errors.DockerException:
            return False

    def pull_ollama_image(self) -> Image | None:
        if not self.is_connected() or self.client is None:
            return None

        return self.client.images.pull(OLLAMA_IMAGE, "latest")

    def ensure_ollama_image(self, progress_cb: typing.Callable | None = None) -> bool:
        if not self.is_connected() or self.client is None:
            return False

        if self.is_ollama_image_pulled():
            if progress_cb:
                progress_cb(100, "Base image already present")
            return True

        try:
            layer_totals: dict[str, int] = {}
            layer_currents: dict[str, int] = {}
            current_layer_id: str = ""

            for event in self.client.api.pull(
                OLLAMA_IMAGE, tag="latest", stream=True, decode=True
            ):
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
                        lid_total = layer_totals[current_layer_id]
                        lid_current = layer_currents.get(current_layer_id, 0)
                        layer_pct = int(100 * lid_current / lid_total)
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

    def get_available_containers(self) -> list[dict[str, str]]:
        if not self.is_connected() or self.client is None:
            return []

        all_containers: list[Container] = self.client.containers.list(all=True)

        return [
            ContainerManager.map_container(container, status=self.resolve_status(container))
            for container in all_containers
            if ContainerManager.is_managed_container(container)
        ]

    def resolve_status(self, container: Container) -> str:
        if container.status != ContainerManager.CONTAINER_STATUS["RUNNING"]:
            return container.status

        if progress_state.is_in_progress(container.name or ""):
            return ContainerManager.CONTAINER_STATUS["PULLING_MODEL"]

        installed = self.check_model_installed(container)
        if installed is None:
            return ContainerManager.CONTAINER_STATUS["STARTING"]

        return ContainerManager.CONTAINER_STATUS["RUNNING" if installed else "INCOMPLETE"]

    def get_container(self, container_name: str) -> Container | None:
        if not self.is_connected() or self.client is None:
            return None

        try:
            return self.client.containers.get(container_name)

        except docker.errors.NotFound:
            return None

        except docker.errors.DockerException as e:
            logger.error("Error retrieving container: %s", e)
            return None

    @staticmethod
    def allocate_port(container_name: str) -> int:
        """Return the host port owned by `container_name`, allocating the
        lowest free port >= PORT_RANGE_START if it doesn't have one yet.
        Locked so two concurrent container creations can't both compute the
        same free port and collide.
        """
        with transaction.atomic():
            existing = (
                ContainerPort.objects.select_for_update()
                .filter(container_name=container_name)
                .first()
            )
            if existing is not None:
                return existing.port

            used_ports = set(
                ContainerPort.objects.select_for_update().values_list("port", flat=True)
            )
            port = PORT_RANGE_START
            while port in used_ports:
                port += 1

            ContainerPort.objects.create(container_name=container_name, port=port)
            return port

    def _create_container(
        self, container_name: str, model_name: str, parameters: str, port: int, use_gpu: bool
    ) -> Container:
        assert self.client is not None

        try:
            return self.client.containers.create(
                name=container_name,
                image=f"{OLLAMA_IMAGE}:latest",
                detach=True,
                ports={"11434/tcp": port},
                network="chatbot-network",
                device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])] if use_gpu else [],
                hostname=container_name,
                # Model weights live in a named volume rather than the
                # container's writable layer, so recreating the container
                # (e.g. onto a newer Ollama image) doesn't re-download them.
                volumes={
                    ContainerManager.volume_name(container_name): {"bind": OLLAMA_DATA_PATH, "mode": "rw"}
                },
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

    def start_container(self, container: Container) -> None:
        if container.status == ContainerManager.CONTAINER_STATUS["RUNNING"]:
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

    def create_and_start_container(self, model_name: str, parameters: str) -> Container:
        """Starts the model's container, creating it first if needed. Asks
        for every GPU, and if the host can't provide one, recreates the
        container CPU-only — most machines (every Mac, AMD/Intel GPUs, no
        GPU at all) have no NVIDIA runtime, and Docker refuses to even start
        a container that requests one there. Raises ContainerError.
        """
        if not self.is_connected() or self.client is None:
            raise ContainerError("Docker isn't running. Start Docker Desktop and try again.")

        container_name = ContainerManager.container_name(model_name, parameters)

        if (container := self.get_container(container_name)) is not None:
            self.start_container(container)
            return container

        port = ContainerManager.allocate_port(container_name)
        use_gpu = ContainerManager._gpu_supported is not False
        container = self._create_container(container_name, model_name, parameters, port, use_gpu)

        try:
            container.start()
        except docker.errors.APIError as e:
            if not (use_gpu and _is_gpu_unavailable_error(e)):
                logger.error("Error starting container %s: %s", container_name, e)
                raise ContainerError(f"Couldn't start the model container: {_docker_error_text(e)}") from e

            logger.warning("No usable GPU (%s) — recreating %s CPU-only", _docker_error_text(e), container_name)
            ContainerManager._gpu_supported = False
            container.remove(force=True)
            container = self._create_container(container_name, model_name, parameters, port, use_gpu=False)
            self.start_container(container)
            return container

        if use_gpu:
            ContainerManager._gpu_supported = True
        container.reload()
        return container

    def gpu_status(self) -> bool | None:
        """True/False once a container start has shown whether GPUs work on
        this host, else whatever existing containers were created with, else
        None (unknown — nothing has been started yet)."""
        if ContainerManager._gpu_supported is not None:
            return ContainerManager._gpu_supported

        if not self.is_connected() or self.client is None:
            return None

        for container in self.client.containers.list(all=True, filters={"label": f"{LABEL_MANAGED}=true"}):
            gpu_label = ContainerManager._labels(container).get(LABEL_GPU)
            if gpu_label in ("true", "false"):
                return gpu_label == "true"

        return None

    def wait_for_ollama_ready(self, base_url: str, timeout: int = 60) -> bool:
        url = f"{base_url}/api/version"

        start = time.time()
        while time.time() - start < timeout:
            try:
                r = requests.get(url, timeout=2)
                if r.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)

        return False

    @staticmethod
    def is_model_installed(base_url: str, model: str, parameters: str) -> bool | None:
        """Whether Ollama at `base_url` holds `model:parameters`, or None if
        Ollama can't be asked right now (still booting, or unreachable)."""
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=2)
            response.raise_for_status()
            listed = response.json().get("models") or []
        except (requests.exceptions.RequestException, ValueError, AttributeError):
            return None

        wanted = f"{model}:{parameters}"
        return any(wanted in (entry.get("name"), entry.get("model")) for entry in listed)

    def check_model_installed(self, container: Container) -> bool | None:
        if container.id in ContainerManager._installed_ids:
            return True

        base_url = self.base_url_for(container)
        model = ContainerManager.get_container_metadata(container, "model")
        parameters = ContainerManager.get_container_metadata(container, "parameters")
        if base_url is None or model is None or parameters is None:
            return None

        installed = ContainerManager.is_model_installed(base_url, model, parameters)
        if installed:
            ContainerManager.mark_installed(container)
        return installed

    @staticmethod
    def mark_installed(container: Container) -> None:
        with ContainerManager._installed_lock:
            if container.id:
                ContainerManager._installed_ids.add(container.id)

    @staticmethod
    def is_marked_installed(container: Container) -> bool:
        return container.id in ContainerManager._installed_ids

    def pull_model_with_progress(
        self,
        model: str,
        parameters: str,
        base_url: str,
        progress_cb: typing.Callable | None = None,
    ) -> tuple[bool, str | None]:
        """Returns (success, error_message). Ollama's /api/pull streams
        `{"error": "..."}` on failure (bad tag, manifest not found, disk
        full) with a 200 status — the request never raises on its own, so
        every line has to be checked for an "error" key or a failed pull
        silently reports 100% / "Model ready" like a success.
        """
        url = f"{base_url}/api/pull"

        try:
            layer_totals: dict[str, int] = {}
            layer_currents: dict[str, int] = {}
            current_digest: str = ""

            with requests.post(
                url,
                json={"model": f"{model}:{parameters}", "stream": True},
                stream=True,
                timeout=600,
            ) as resp:
                resp.raise_for_status()

                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if error := event.get("error"):
                        logger.error("Ollama reported a pull error for %s:%s: %s", model, parameters, error)
                        return False, error

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
                            d_total = layer_totals[current_digest]
                            d_current = layer_currents.get(current_digest, 0)
                            layer_pct = int(100 * d_current / d_total)
                            short_id = current_digest[7:19] if current_digest.startswith("sha256:") else current_digest[:12]
                            detail = f"Layer {short_id} — {layer_pct}%"
                        else:
                            detail = status or "Pulling…"
                        progress_cb(overall_pct, detail)
                    elif progress_cb and status:
                        progress_cb(0, status)

            if progress_cb:
                progress_cb(100, "Model ready")
            return True, None

        except requests.exceptions.RequestException as e:
            logger.error("Error pulling model %s:%s: %s", model, parameters, e)
            return False, str(e)

    def get_network(self, network_name: str) -> Network | None:
        if not self.is_connected() or self.client is None:
            return None

        try:
            return self.client.networks.get(network_name)

        except docker.errors.NotFound:
            return None

        except docker.errors.DockerException as e:
            logger.error("Error retrieving network: %s", e)
            return None

    def create_network(self, network_name: str) -> Network | None:
        if not self.is_connected() or self.client is None:
            return None

        try:
            return self.client.networks.create(network_name, driver="bridge")

        except docker.errors.DockerException as e:
            logger.error("Error creating network: %s", e)
            return None

    def stop_container(self, container_name: str) -> None:
        if not self.is_connected() or self.client is None:
            return

        if (container := self.get_container(container_name)) is not None:
            try:
                container.stop()
            except docker.errors.APIError as e:
                logger.warning("Error stopping container %s: %s", container_name, e)

    def remove_container(self, container_name: str) -> None:
        if not self.is_connected() or self.client is None:
            return

        if (container := self.get_container(container_name)) is not None:
            try:
                container.stop()
                container.remove()
            except docker.errors.APIError as e:
                logger.warning("Error removing container %s: %s", container_name, e)
                return

            with ContainerManager._installed_lock:
                ContainerManager._installed_ids.discard(container.id or "")

        # `remove(v=True)` only deletes anonymous volumes — the named one
        # holding the weights has to go explicitly, or "Remove" frees nothing.
        try:
            self.client.volumes.get(ContainerManager.volume_name(container_name)).remove()
        except docker.errors.NotFound:
            pass
        except docker.errors.APIError as e:
            logger.warning("Error removing volume for %s: %s", container_name, e)

    def base_url_for(self, container: Container) -> str | None:
        """In Docker mode this server is itself a container on the same
        `chatbot-network` bridge, so it addresses the target directly by
        container name over Docker's internal DNS — no host port needed,
        and no `host.docker.internal`, which isn't resolvable on Linux
        Docker Engine without an `extra_hosts` entry docker-compose.yaml
        doesn't set.

        Outside Docker (`python manage.py runserver` on the host, container
        management still going through the Docker socket), there's no
        shared network to piggyback on, so it has to go through the
        allocated host-published port instead. An IP literal, not
        "localhost" — see the DATABASES note in settings.py.
        """
        if os.getenv("DOCKER", "false") == "true":
            return f"http://{container.name}:11434"

        host_port = ContainerManager.get_container_metadata(container, "port")
        if host_port is None:
            return None
        return f"http://127.0.0.1:{host_port}"

    def get_ollama_base_url(self, model: str, parameters: str) -> str | None:
        container = self.get_container(ContainerManager.container_name(model, parameters))
        if container is None or container.status != "running":
            return None

        return self.base_url_for(container)

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
    def _labels(container: Container) -> dict[str, str]:
        return container.attrs.get("Config", {}).get("Labels") or {}

    @staticmethod
    def get_container_metadata(container: Container, key: str) -> str | None:
        """`key` is "model", "parameters" or "port" — read from our labels,
        falling back to the env vars containers created before labels used."""
        if (label := _METADATA_LABELS.get(key)) and (value := ContainerManager._labels(container).get(label)):
            return value

        for env in container.attrs.get("Config", {}).get("Env") or []:
            name, sep, value = env.partition("=")
            if sep and name == key:
                return value

        return None
