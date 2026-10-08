import logging
import threading

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from django_app.models import AIModel, AIModelVersion

from . import hardware, install
from .ContainerManager import ContainerError, ContainerManager

logger = logging.getLogger(__name__)


class Docker(APIView):
    def get(self, request) -> Response:
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

        if not docker_client.is_connected() and not docker_client.connect_to_docker():
            return Response(
                {"error": "Error connecting to Docker"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        containers = docker_client.get_available_containers()

        versions_by_key = {
            (v.ai_model.model, v.parameters): v.size_bytes
            for v in AIModelVersion.objects.select_related("ai_model").all()
        }

        entries = []
        total_bytes = 0
        for container in containers:
            model = container["environment"].get("model")
            parameters = container["environment"].get("parameters")
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
        if not docker_client.is_connected() and not docker_client.connect_to_docker():
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
