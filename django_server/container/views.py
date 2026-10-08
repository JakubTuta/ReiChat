import logging
import threading

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from django_app import app_settings, ollama_api
from django_app.models import AIModel, AIModelVersion

from . import external, hardware, install, searxng
from .ContainerManager import ContainerError, ContainerManager

logger = logging.getLogger(__name__)


class Docker(APIView):
    def get(self, request) -> Response:
        # With the user's own Ollama, Docker isn't needed for models at all —
        # what matters is whether that Ollama answers.
        if (external_url := external.url()) is not None:
            if external.version(external_url) is None:
                return Response(
                    {"error": f"Can't reach your Ollama at {external_url}. Start it, or switch back to "
                              "app-managed models on the Models page."},
                    status=status.HTTP_503_SERVICE_UNAVAILABLE,
                )
            return Response({"status": "Ollama is reachable"}, status=status.HTTP_200_OK)

        docker_client = ContainerManager()

        if not docker_client.is_connected() and not docker_client.connect_to_docker():
            return Response(
                {"error": "Error connecting to Docker"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return Response(
            {"status": "Docker is running"},
            status=status.HTTP_200_OK,
        )


class Containers(APIView):
    def get(self, request) -> Response:
        docker_client = ContainerManager()

        if (external_url := external.url()) is not None:
            external.register_models(external_url)
            return Response(docker_client.get_available_containers(), status=status.HTTP_200_OK)

        if not docker_client.is_connected() and not docker_client.connect_to_docker():
            return Response(
                {"error": "Error connecting to Docker"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        containers = docker_client.get_available_containers()

        return Response(containers, status=status.HTTP_200_OK)


class Hardware(APIView):
    def get(self, request) -> Response:
        docker_client = ContainerManager()

        # Docker's numbers describe Docker's VM, not where the user's own
        # Ollama runs — better to admit not knowing than to mislead.
        if external.url() is not None:
            return Response(
                {"ram_bytes": None, "vram_bytes": None, "gpu_available": None}, status=status.HTTP_200_OK
            )

        if not docker_client.is_connected() and not docker_client.connect_to_docker():
            return Response(
                {"error": "Error connecting to Docker"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return Response(
            {
                "ram_bytes": hardware.get_host_ram_bytes(docker_client),
                "vram_bytes": hardware.detect_vram_bytes(docker_client),
                "gpu_available": docker_client.gpu_status(),
            },
            status=status.HTTP_200_OK,
        )


class DiskUsage(APIView):
    def get(self, request) -> Response:
        # Real installed-model disk usage, not the registry's advertised
        # size — matched against the catalog by (model, parameters) since
        # Docker itself doesn't know which Ollama model a container holds.
        docker_client = ContainerManager()
        external_url = external.url()

        if external_url is None and not docker_client.is_connected() and not docker_client.connect_to_docker():
            return Response(
                {"error": "Error connecting to Docker"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        containers = docker_client.get_available_containers()
        external_sizes = (ollama_api.installed_model_sizes(external_url) or {}) if external_url else {}

        versions_by_key = {
            (v.ai_model.model, v.parameters): v.size_bytes
            for v in AIModelVersion.objects.select_related("ai_model").all()
        }

        entries = []
        total_bytes = 0
        for container in containers:
            model = container["environment"].get("model")
            parameters = container["environment"].get("parameters")
            # The real size on disk when the container can be asked (a
            # running one), the registry's advertised size otherwise.
            size_bytes = external_sizes.get(f"{model}:{parameters}")
            if size_bytes is None and not external_url and container["status"] in ("running", "incomplete") \
                    and model and parameters:
                base_url = docker_client.get_ollama_base_url(model, parameters)
                sizes = ollama_api.installed_model_sizes(base_url) if base_url else None
                size_bytes = (sizes or {}).get(f"{model}:{parameters}")
            if size_bytes is None:
                size_bytes = versions_by_key.get((model, parameters))
            if size_bytes:
                total_bytes += size_bytes

            entries.append({
                "model": model,
                "parameters": parameters,
                "status": container["status"],
                "size_bytes": size_bytes,
            })

        return Response({"total_bytes": total_bytes, "models": entries}, status=status.HTTP_200_OK)


class Runtime(APIView):
    """What each running model container has loaded in memory right now."""

    def get(self, request) -> Response:
        if (external_url := external.url()) is not None:
            loaded_external = [
                {
                    "container": None,
                    "model": entry["name"],
                    "size_bytes": entry["size"],
                    "vram_bytes": entry["size_vram"],
                    "context_length": entry["context_length"],
                    "expires_at": entry["expires_at"],
                }
                for entry in ollama_api.running_models(external_url) or []
            ]
            return Response({"loaded": loaded_external}, status=status.HTTP_200_OK)

        docker_client = ContainerManager()

        if not docker_client.is_connected() and not docker_client.connect_to_docker():
            return Response({"error": "Error connecting to Docker"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        assert docker_client.client is not None
        loaded = []
        for container in docker_client.client.containers.list(filters={"status": "running"}):
            if not ContainerManager.is_managed_container(container):
                continue
            base_url = docker_client.base_url_for(container)
            for entry in (ollama_api.running_models(base_url) if base_url else None) or []:
                loaded.append({
                    "container": container.name,
                    "model": entry["name"],
                    "size_bytes": entry["size"],
                    "vram_bytes": entry["size_vram"],
                    "context_length": entry["context_length"],
                    "expires_at": entry["expires_at"],
                })

        return Response({"loaded": loaded}, status=status.HTTP_200_OK)


class ContainerUnload(APIView):
    def post(self, request, model) -> Response:
        parameters = request.query_params.get("parameters")
        if parameters is None:
            return Response({"error": "Invalid request: missing 'parameters' query param"},
                            status=status.HTTP_400_BAD_REQUEST)

        docker_client = ContainerManager()
        base_url = docker_client.get_ollama_base_url(model, parameters)
        if base_url is None:
            return Response({"error": "That model isn't running."}, status=status.HTTP_404_NOT_FOUND)

        if not ollama_api.unload_model(base_url, f"{model}:{parameters}"):
            return Response({"error": "Ollama didn't respond — try again in a moment."},
                            status=status.HTTP_502_BAD_GATEWAY)

        return Response({"status": "Model unloaded"}, status=status.HTTP_200_OK)


class ContainerUpdate(APIView):
    def post(self, request, model) -> Response:
        parameters = request.query_params.get("parameters")
        if parameters is None:
            return Response({"error": "Invalid request: missing 'parameters' query param"},
                            status=status.HTTP_400_BAD_REQUEST)

        external_url = external.url()
        installed = (
            external.has_model(external_url, model, parameters)
            if external_url
            else ContainerManager().get_container(ContainerManager.container_name(model, parameters)) is not None
        )
        if not installed:
            return Response({"error": "That model isn't installed."}, status=status.HTTP_404_NOT_FOUND)

        install.start_model_update(model, parameters)
        return Response({"status": "Checking for updates"}, status=status.HTTP_202_ACCEPTED)


class OllamaUpdate(APIView):
    def post(self, request) -> Response:
        docker_client = ContainerManager()
        if not docker_client.is_connected() and not docker_client.connect_to_docker():
            return Response({"error": "Error connecting to Docker"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        install.start_ollama_update()
        return Response({"status": "Ollama update started"}, status=status.HTTP_202_ACCEPTED)


class OllamaConnection(APIView):
    """Where models run: app-managed containers (default) or the user's own
    Ollama at a URL."""

    def get(self, request) -> Response:
        external_url = external.url()
        return Response(
            {
                "mode": "external" if external_url else "managed",
                "url": external_url,
                "version": external.version(external_url) if external_url else None,
            },
            status=status.HTTP_200_OK,
        )

    def put(self, request) -> Response:
        mode = request.data.get("mode")
        if mode == "managed":
            app_settings.set_external_ollama_url(None)
            return Response({"mode": "managed", "url": None, "version": None}, status=status.HTTP_200_OK)

        if mode != "external":
            return Response({"error": "mode must be 'managed' or 'external'"}, status=status.HTTP_400_BAD_REQUEST)

        url = str(request.data.get("url") or "").strip().rstrip("/")
        if not url.startswith(("http://", "https://")):
            return Response(
                {"error": "Enter the address with http:// in front, e.g. http://host.docker.internal:11434"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Checked before saving: switching to an address where nothing
        # answers would make every model disappear at once.
        found = external.version(url)
        if found is None:
            return Response(
                {"error": f"Nothing answered like Ollama at {url}. Make sure Ollama is running — and if it runs "
                          "on this computer while ReiChat runs in Docker, use http://host.docker.internal:11434."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        app_settings.set_external_ollama_url(url)
        external.register_models(url)
        return Response({"mode": "external", "url": url, "version": found}, status=status.HTTP_200_OK)


class WebSearch(APIView):
    def get(self, request) -> Response:
        return Response({"status": searxng.status()}, status=status.HTTP_200_OK)

    def post(self, request) -> Response:
        docker_client = ContainerManager()
        if not docker_client.is_connected() and not docker_client.connect_to_docker():
            return Response({"error": "Error connecting to Docker"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        searxng.start_enable()
        return Response({"status": "installing"}, status=status.HTTP_202_ACCEPTED)

    def delete(self, request) -> Response:
        try:
            searxng.disable()
        except ContainerError as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response({"status": "off"}, status=status.HTTP_200_OK)


class OllamaImage(APIView):
    def get(self, request) -> Response:
        docker_client = ContainerManager()

        if not docker_client.is_connected() and not docker_client.connect_to_docker():
            return Response(
                {"error": "Error connecting to Docker"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        if docker_client.is_ollama_image_pulled():
            return Response(status=status.HTTP_200_OK)

        return Response(status=status.HTTP_404_NOT_FOUND)

    def post(self, request) -> Response:
        docker_client = ContainerManager()

        if not docker_client.is_connected() and not docker_client.connect_to_docker():
            return Response(
                {"error": "Error connecting to Docker"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        threading.Thread(target=docker_client.pull_ollama_image, daemon=True).start()

        return Response(
            {"status": "Pulling ollama/ollama image in background"},
            status=status.HTTP_202_ACCEPTED,
        )


class Container(APIView):
    def get(self, request, model) -> Response:
        docker_client = ContainerManager()

        if not docker_client.is_connected() and not docker_client.connect_to_docker():
            return Response(
                {"error": "Error connecting to Docker"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        if not (container := docker_client.get_container(model)):
            return Response(
                {"error": "Container not found"}, status=status.HTTP_404_NOT_FOUND
            )

        return Response(
            {
                "status": "Container found",
                "container": ContainerManager.map_container(container),
            },
            status=status.HTTP_200_OK,
        )

    def post(self, request, model) -> Response:
        # 200 when the model is already confirmed running; otherwise 202 and
        # progress arrives over the container socket (see container/install.py).
        query_model_params = request.query_params.get("parameters", None)
        if query_model_params is None:
            return Response(
                {"error": "Invalid request: missing 'parameters' query param"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        docker_client = ContainerManager()
        external_url = external.url()
        if external_url is None and not docker_client.is_connected() and not docker_client.connect_to_docker():
            return Response(
                {"error": "Error connecting to Docker"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        if not (ai_model := AIModel.objects.filter(model=model).first()):
            return Response(
                {"error": "Model not found"}, status=status.HTTP_404_NOT_FOUND
            )

        if not (
            ai_model_version := next(
                (
                    version
                    for version in ai_model.versions.all()
                    if version.parameters == query_model_params
                ),
                None,
            )
        ):
            return Response(
                {"error": "Invalid model version"}, status=status.HTTP_404_NOT_FOUND
            )

        container_name = ContainerManager.container_name(model, query_model_params)

        if external_url is not None:
            if external.has_model(external_url, model, query_model_params):
                return Response({"status": "Model is installed"}, status=status.HTTP_200_OK)
            install.start_install(ai_model.model, ai_model_version.parameters)
            return Response({"status": "Model setup started"}, status=status.HTTP_202_ACCEPTED)

        existing = docker_client.get_container(container_name)

        if existing is not None:
            try:
                docker_client.start_container(existing)
            except ContainerError as e:
                return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Already confirmed to hold its model — nothing left to do. Any
            # other case (first start since the backend restarted, or a pull
            # that never finished) goes through the install worker, which
            # just checks /api/tags when the model is there and resumes the
            # download when it isn't.
            if ContainerManager.is_marked_installed(existing):
                return Response(
                    {"status": "Container is running", "container": ContainerManager.map_container(existing)},
                    status=status.HTTP_200_OK,
                )

        install.start_install(ai_model.model, ai_model_version.parameters)

        return Response({"status": "Model setup started"}, status=status.HTTP_202_ACCEPTED)

    def delete(self, request, model) -> Response:
        docker_client = ContainerManager()

        query_model_params = request.query_params.get("parameters", None)
        query_method = request.query_params.get("method", None)

        if query_model_params is None or query_method is None:
            return Response(
                {"error": "Invalid request: missing 'parameters' or 'method' query params"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if (external_url := external.url()) is not None:
            # No container to stop: "stop" frees the model's memory, "remove"
            # deletes it from the user's Ollama.
            if query_method == "stop":
                ollama_api.unload_model(external_url, f"{model}:{query_model_params}")
                return Response({"status": "Model unloaded"}, status=status.HTTP_200_OK)
            if query_method == "remove":
                if not external.delete_model(external_url, model, query_model_params):
                    return Response(
                        {"error": "Your Ollama didn't delete the model — try again."},
                        status=status.HTTP_502_BAD_GATEWAY,
                    )
                return Response({"status": "Model removed"}, status=status.HTTP_200_OK)

        if not docker_client.is_connected() and not docker_client.connect_to_docker():
            return Response(
                {"error": "Error connecting to Docker"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        container_name = ContainerManager.container_name(model, query_model_params)

        if query_method == "stop":
            docker_client.stop_container(container_name)

            return Response(
                {"status": "Container stopped"},
                status=status.HTTP_200_OK,
            )

        elif query_method == "remove":
            docker_client.remove_container(container_name)

            return Response(
                {"status": "Container removed"},
                status=status.HTTP_200_OK,
            )

        else:
            return Response(
                {"error": "Invalid method: use 'stop' or 'remove'"},
                status=status.HTTP_400_BAD_REQUEST,
            )
