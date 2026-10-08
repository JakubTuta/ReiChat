import logging
import threading

from asgiref.sync import async_to_sync
from django.db import close_old_connections
from django.db.models import Count, Q
from rest_framework import status
from rest_framework.parsers import JSONParser, MultiPartParser
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from helpers import decorators

from . import functions, models, serializers
from .catalog import sync as catalog_sync
from .mcp_integration import client as mcp_client
from .rag import extract as rag_extract
from .rag import ingest as rag_ingest

logger = logging.getLogger(__name__)


class AIModels(APIView):
    def get(self, request: Request) -> Response:
        all_models = models.AIModel.objects.prefetch_related("versions").order_by("-popularity")
        deserialized_models = serializers.AIModelSerializer(all_models, many=True).data
        return Response(deserialized_models, status=status.HTTP_200_OK)

    @decorators.required_query_params(["minPullCount"])
    def put(self, request: Request) -> Response:
        from . import progress_state

        raw_min_pull_count = request.query_params.get("minPullCount", "")
        try:
            min_pull_count = int(raw_min_pull_count) if raw_min_pull_count else 200_000
        except ValueError:
            return Response(
                {"error": "minPullCount must be a whole number"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not progress_state.try_start_scrape():
            return Response(
                {"error": "A catalog refresh is already running"},
                status=status.HTTP_409_CONFLICT,
            )

        logger.info("Starting catalog refresh (minPullCount=%d)", min_pull_count)
        progress_state.broadcast(
            "scrape_progress",
            {"type": "scrape_progress", "running": True, "total": None, "completed": 0, "current": None, "error": None},
            key="scrape",
            force=True,
        )

        def _refresh_worker():
            try:
                close_old_connections()

                def _cb(completed: int, total: int, title: str) -> None:
                    progress_state.set_scrape(total=total, completed=completed, current=title)
                    progress_state.broadcast(
                        "scrape_progress",
                        {"type": "scrape_progress", "running": True, "total": total, "completed": completed, "current": title, "error": None},
                        key="scrape",
                    )

                result = catalog_sync.refresh_catalog(min_pull_count, progress_cb=_cb)

                if result.error:
                    logger.error("Catalog refresh failed: %s", result.error)
                    progress_state.set_scrape(running=False, error=result.error)
                    progress_state.broadcast(
                        "scrape_progress",
                        {"type": "scrape_progress", "running": False, "total": None, "completed": 0, "current": None, "error": result.error},
                        key="scrape",
                        force=True,
                    )
                    return

                logger.info(
                    "Catalog refresh completed: %d created, %d updated, %d pruned, %d variants verified",
                    result.created, result.updated, result.pruned, result.variants_verified,
                )
                progress_state.set_scrape(running=False, error=None)
                progress_state.broadcast(
                    "scrape_progress",
                    {"type": "scrape_progress", "running": False, "total": None, "completed": 0, "current": None, "error": None},
                    key="scrape",
                    force=True,
                )
            except Exception as e:
                logger.exception("Catalog refresh crashed")
                progress_state.set_scrape(running=False, error=str(e))
                progress_state.broadcast(
                    "scrape_progress",
                    {"type": "scrape_progress", "running": False, "total": None, "completed": 0, "current": None, "error": str(e)},
                    key="scrape",
                    force=True,
                )

        threading.Thread(target=_refresh_worker, daemon=True).start()

        return Response(
            {"status": "Catalog refresh started"},
            status=status.HTTP_202_ACCEPTED,
        )


class Personas(APIView):
    def get(self, request: Request) -> Response:
        all_personas = models.Persona.objects.order_by("name")
        serialized = serializers.PersonaSerializer(all_personas, many=True).data
        return Response(serialized, status=status.HTTP_200_OK)

    @decorators.required_body_params(["name", "system_prompt"])
    def post(self, request: Request) -> Response:
        serializer = serializers.PersonaSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(
                {"error": "Invalid persona data", "details": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST,
            )

        serializer.save()

        return Response(serializer.data, status=status.HTTP_201_CREATED)

    @decorators.required_body_params(["id"])
    def put(self, request: Request) -> Response:
        request_data: dict[str, object] = request.data  # type: ignore

        if (persona := models.Persona.objects.filter(id=request_data["id"]).first()) is None:
            return Response({"error": "Persona not found"}, status=status.HTTP_404_NOT_FOUND)

        serializer = serializers.PersonaSerializer(persona, data=request.data, partial=True)

        if not serializer.is_valid():
            return Response(
                {"error": "Invalid persona data", "details": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST,
            )

        serializer.save()

        return Response(serializer.data, status=status.HTTP_200_OK)

    @decorators.required_body_params(["id"])
    def delete(self, request: Request) -> Response:
        request_data: dict[str, object] = request.data  # type: ignore

        persona = models.Persona.objects.filter(id=request_data["id"]).first()

        if persona is None:
            return Response({"error": "Persona not found"}, status=status.HTTP_404_NOT_FOUND)

        # Chats using this persona fall back to no persona (SET_NULL on the
        # FK) rather than losing their history.
        persona.delete()

        return Response({"status": "Persona deleted"}, status=status.HTTP_200_OK)


class PromptTemplates(APIView):
    def get(self, request: Request) -> Response:
        templates = models.PromptTemplate.objects.order_by("name")
        serialized = serializers.PromptTemplateSerializer(templates, many=True).data
        return Response(serialized, status=status.HTTP_200_OK)

    @decorators.required_body_params(["name", "content"])
    def post(self, request: Request) -> Response:
        serializer = serializers.PromptTemplateSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(
                {"error": "Invalid prompt template data", "details": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST,
            )

        serializer.save()

        return Response(serializer.data, status=status.HTTP_201_CREATED)

    @decorators.required_body_params(["id"])
    def put(self, request: Request) -> Response:
        request_data: dict[str, object] = request.data  # type: ignore

        if (template := models.PromptTemplate.objects.filter(id=request_data["id"]).first()) is None:
            return Response({"error": "Prompt template not found"}, status=status.HTTP_404_NOT_FOUND)

        serializer = serializers.PromptTemplateSerializer(template, data=request.data, partial=True)

        if not serializer.is_valid():
            return Response(
                {"error": "Invalid prompt template data", "details": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST,
            )

        serializer.save()

        return Response(serializer.data, status=status.HTTP_200_OK)

    @decorators.required_body_params(["id"])
    def delete(self, request: Request) -> Response:
        request_data: dict[str, object] = request.data  # type: ignore

        template = models.PromptTemplate.objects.filter(id=request_data["id"]).first()

        if template is None:
            return Response({"error": "Prompt template not found"}, status=status.HTTP_404_NOT_FOUND)

        template.delete()

        return Response({"status": "Prompt template deleted"}, status=status.HTTP_200_OK)


class Collections(APIView):
    def get(self, request: Request) -> Response:
        collections = (
            models.DocumentCollection.objects.select_related("embedding_model")
            .annotate(document_count=Count("documents"))
            .order_by("-created_at")
        )

        chat_id = request.query_params.get("chat_id")
        if chat_id:
            # A chat sees its own collections plus every global one — global
            # collections (chat=None) are usable from any chat.
            collections = collections.filter(Q(chat_id=chat_id) | Q(chat__isnull=True))

        serialized = serializers.DocumentCollectionSerializer(collections, many=True).data
        return Response(serialized, status=status.HTTP_200_OK)

    @decorators.required_body_params(["name", "embedding_model", "embedding_parameters"])
    def post(self, request: Request) -> Response:
        request_data: dict[str, object] = request.data  # type: ignore

        if (ai_model := models.AIModel.objects.filter(id=request_data["embedding_model"]).first()) is None:
            return Response({"error": "Embedding model not found"}, status=status.HTTP_404_NOT_FOUND)

        if not ai_model.is_embedding:
            return Response(
                {"error": "That model isn't an embedding model"}, status=status.HTTP_400_BAD_REQUEST
            )

        serializer = serializers.DocumentCollectionSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(
                {"error": "Invalid collection data", "details": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST,
            )

        serializer.save()

        return Response(serializer.data, status=status.HTTP_201_CREATED)

    @decorators.required_body_params(["id"])
    def delete(self, request: Request) -> Response:
        request_data: dict[str, object] = request.data  # type: ignore

        collection = models.DocumentCollection.objects.filter(id=request_data["id"]).first()

        if collection is None:
            return Response({"error": "Collection not found"}, status=status.HTTP_404_NOT_FOUND)

        # Cascades to its Documents/DocumentChunks; any chat that had it
        # active just loses that M2M row (nothing else references it).
        collection.delete()

        return Response({"status": "Collection deleted"}, status=status.HTTP_200_OK)


class Documents(APIView):
    # The only endpoint that takes a file upload, so the only one that opts
    # back into multipart over the JSON-only default (settings.REST_FRAMEWORK).
    parser_classes = [JSONParser, MultiPartParser]

    @decorators.required_query_params(["collection_id"])
    def get(self, request: Request) -> Response:
        rag_ingest.mark_interrupted_documents(request.query_params["collection_id"])

        documents = models.Document.objects.filter(
            collection_id=request.query_params["collection_id"]
        ).order_by("-uploaded_at")

        serialized = serializers.DocumentSerializer(documents, many=True).data
        return Response(serialized, status=status.HTTP_200_OK)

    @decorators.required_body_params(["collection_id", "file"])
    def post(self, request: Request) -> Response:
        request_data = request.data

        collection = models.DocumentCollection.objects.filter(
            id=request_data.get("collection_id")
        ).first()
        if collection is None:
            return Response({"error": "Collection not found"}, status=status.HTTP_404_NOT_FOUND)

        uploaded_file = request.FILES.get("file")
        if uploaded_file is None:
            return Response({"error": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)

        raw = uploaded_file.read()

        try:
            text = rag_extract.extract_text(uploaded_file.name, raw)
        except rag_extract.UnsupportedFileTypeError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

        if not text.strip():
            return Response(
                {"error": "No extractable text found in this file."}, status=status.HTTP_400_BAD_REQUEST
            )

        document = models.Document.objects.create(collection=collection, filename=uploaded_file.name)

        # Chunking + embedding involves real HTTP calls to the embedding
        # model and would block the request for a while on anything but a
        # tiny file — same background-worker pattern as container creation
        # and catalog refresh (see container/install.py / AIModels.put).
        rag_ingest.start_ingest(document.id, text)

        return Response(serializers.DocumentSerializer(document).data, status=status.HTTP_202_ACCEPTED)

    @decorators.required_body_params(["id"])
    def delete(self, request: Request) -> Response:
        request_data: dict[str, object] = request.data  # type: ignore

        document = models.Document.objects.filter(id=request_data["id"]).first()

        if document is None:
            return Response({"error": "Document not found"}, status=status.HTTP_404_NOT_FOUND)

        document.delete()

        return Response({"status": "Document deleted"}, status=status.HTTP_200_OK)


class MCPServers(APIView):
    def get(self, request: Request) -> Response:
        servers = models.MCPServer.objects.order_by("name")
        serialized = serializers.MCPServerSerializer(servers, many=True).data

        return Response(serialized, status=status.HTTP_200_OK)

    @decorators.required_body_params(["name", "transport"])
    def post(self, request: Request) -> Response:
        request_data: dict[str, object] = request.data  # type: ignore

        error = self._validate_transport_config(request_data)
        if error:
            return Response({"error": error}, status=status.HTTP_400_BAD_REQUEST)

        serializer = serializers.MCPServerSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(
                {"error": "Invalid MCP server data", "details": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST,
            )

        serializer.save()

        return Response(serializer.data, status=status.HTTP_201_CREATED)

    @decorators.required_body_params(["id"])
    def put(self, request: Request) -> Response:
        request_data: dict[str, object] = request.data  # type: ignore

        server = models.MCPServer.objects.filter(id=request_data["id"]).first()
        if server is None:
            return Response({"error": "MCP server not found"}, status=status.HTTP_404_NOT_FOUND)

        error = self._validate_transport_config({
            "transport": request_data.get("transport", server.transport),
            "command": request_data.get("command", server.command),
            "url": request_data.get("url", server.url),
        })
        if error:
            return Response({"error": error}, status=status.HTTP_400_BAD_REQUEST)

        serializer = serializers.MCPServerSerializer(server, data=request.data, partial=True)

        if not serializer.is_valid():
            return Response(
                {"error": "Invalid MCP server data", "details": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST,
            )

        serializer.save()

        return Response(serializer.data, status=status.HTTP_200_OK)

    @decorators.required_body_params(["id"])
    def delete(self, request: Request) -> Response:
        request_data: dict[str, object] = request.data  # type: ignore

        server = models.MCPServer.objects.filter(id=request_data["id"]).first()

        if server is None:
            return Response({"error": "MCP server not found"}, status=status.HTTP_404_NOT_FOUND)

        server.delete()

        return Response({"status": "MCP server deleted"}, status=status.HTTP_200_OK)

    @staticmethod
    def _validate_transport_config(data: dict[str, object]) -> str | None:
        transport = data.get("transport")

        if transport not in (models.MCPServer.TRANSPORT_STDIO, models.MCPServer.TRANSPORT_HTTP):
            return "transport must be 'stdio' or 'http'"

        if transport == models.MCPServer.TRANSPORT_STDIO and not str(data.get("command") or "").strip():
            return "command is required for a stdio server"

        if transport == models.MCPServer.TRANSPORT_HTTP and not str(data.get("url") or "").strip():
            return "url is required for an http server"

        return None


class MCPServerTest(APIView):
    @decorators.required_body_params(["id"])
    def post(self, request: Request) -> Response:
        request_data: dict[str, object] = request.data  # type: ignore

        server = models.MCPServer.objects.filter(id=request_data["id"]).first()
        if server is None:
            return Response({"error": "MCP server not found"}, status=status.HTTP_404_NOT_FOUND)

        try:
            tools = async_to_sync(mcp_client.probe_server_tools)(server)
        except mcp_client.MCPToolError as e:
            return Response({"error": str(e)}, status=status.HTTP_502_BAD_GATEWAY)

        return Response(
            {"tools": [{"name": t["name"], "description": t["description"]} for t in tools]},
            status=status.HTTP_200_OK,
        )


class Search(APIView):
    def get(self, request: Request, model: str) -> Response:
        if (ai_model := models.AIModel.objects.filter(model=model).first()) is None:
            return Response({"error": "AI model not found"}, status=status.HTTP_404_NOT_FOUND)

        query = request.query_params.get("q", "")
        results = functions.search_messages(ai_model, query)

        return Response(results, status=status.HTTP_200_OK)


class ChatHistory(APIView):
    def get(self, request: Request, model: str, chat_id: str) -> Response:
        if (ai_model := models.AIModel.objects.filter(model=model).first()) is None:
            return Response(
                {"error": "AI model not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        if (
            chat_history := functions.get_chat_history(ai_model, chat_id)
        ) is None:
            return Response([], status=status.HTTP_200_OK)

        messages = functions.get_messages_for_chat(chat_history)
        serialized_messages = functions.serialize_messages_for_display(messages)

        return Response(serialized_messages, status=status.HTTP_200_OK)

    @decorators.required_body_params(["index", "sibling_index"])
    def patch(self, request: Request, model: str, chat_id: str) -> Response:
        """Switches the active branch at a given message position — no
        generation involved, so this is a plain synchronous endpoint rather
        than another WebSocket message type.
        """
        request_data: dict[str, object] = request.data  # type: ignore

        if (ai_model := models.AIModel.objects.filter(model=model).first()) is None:
            return Response({"error": "AI model not found"}, status=status.HTTP_404_NOT_FOUND)

        if (chat_history := functions.get_chat_history(ai_model, chat_id)) is None:
            return Response({"error": "Chat not found"}, status=status.HTTP_404_NOT_FOUND)

        index = request_data["index"]
        sibling_index = request_data["sibling_index"]

        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not isinstance(sibling_index, int)
            or isinstance(sibling_index, bool)
        ):
            return Response(
                {"error": "index and sibling_index must be integers"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        new_path = functions.switch_branch(chat_history, index, sibling_index)

        if new_path is None:
            return Response(
                {"error": "That branch no longer exists — the conversation may have changed."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        return Response(new_path, status=status.HTTP_200_OK)


class AllChats(APIView):
    def get(self, request: Request, model: str) -> Response:
        if (ai_model := models.AIModel.objects.filter(model=model).first()) is None:
            return Response(
                {
                    "error": "AI model not found",
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        # Only these fields ever reach the client — .only() avoids Django
        # building the full nested AIModel+versions payload
        # (ChatHistorySerializer) for every chat just to throw it away.
        # active_collections is M2M — .only() can't limit it, it's always a
        # separate prefetch query regardless.
        chat_histories = (
            functions.get_chats(ai_model, sort=True)
            .only(
                "id", "title", "persona__id", "persona__name",
                "temperature", "num_ctx", "top_p", "seed", "tools_enabled",
            )
            .select_related("persona")
            .prefetch_related("active_collections")
        )

        response_data = [
            {
                "id": chat.id,
                "title": chat.title,
                "persona": {"id": chat.persona.id, "name": chat.persona.name} if chat.persona else None,
                "temperature": chat.temperature,
                "num_ctx": chat.num_ctx,
                "top_p": chat.top_p,
                "seed": chat.seed,
                "active_collections": [
                    {"id": c.id, "name": c.name} for c in chat.active_collections.all()
                ],
                "tools_enabled": chat.tools_enabled,
            }
            for chat in chat_histories
        ]

        return Response(
            response_data,
            status=status.HTTP_200_OK,
        )

    def post(self, request: Request, model: str) -> Response:
        if (ai_model := models.AIModel.objects.filter(model=model).first()) is None:
            return Response(
                {
                    "error": "AI model not found",
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        serializer = serializers.ChatHistorySerializer(
            data=request.data, context={"ai_model": ai_model}
        )

        if not serializer.is_valid():
            return Response(
                {
                    "error": "Invalid chat data",
                    "details": serializer.errors,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        serializer.save()

        return Response(
            serializer.data,
            status=status.HTTP_201_CREATED,
        )

    @decorators.required_body_params(["chat_id"])
    def delete(self, request: Request, model: str) -> Response:
        request_data: dict[str, str] = request.data  # type: ignore

        if (ai_model := models.AIModel.objects.filter(model=model).first()) is None:
            return Response(
                {
                    "error": "AI model not found",
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        chat_history = functions.get_chat_history(ai_model, request_data["chat_id"])

        if not chat_history:
            return Response(
                {
                    "error": "Chat not found",
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        chat_history.delete()
        return Response(
            {
                "status": "Chat deleted",
            },
            status=status.HTTP_200_OK,
        )

    # (python type, min, max) — None bound means unbounded. Ollama doesn't
    # itself enforce these ranges, but they're the sane UI bounds for a
    # slider and catch obvious mistakes (e.g. temperature=200).
    _GENERATION_PARAM_BOUNDS: dict[str, tuple[type, float | None, float | None]] = {
        "temperature": (float, 0.0, 2.0),
        "top_p": (float, 0.0, 1.0),
        "num_ctx": (int, 1, None),
        "seed": (int, None, None),
    }

    _UPDATABLE_CHAT_FIELDS = {
        "title", "persona_id", "collection_ids", "tools_enabled", *_GENERATION_PARAM_BOUNDS,
    }

    @decorators.required_body_params(["id"])
    def put(self, request: Request, model: str) -> Response:
        request_data: dict[str, object] = request.data  # type: ignore

        if not self._UPDATABLE_CHAT_FIELDS & request_data.keys():
            return Response(
                {
                    "error": "Nothing to update — provide title, persona_id, collection_ids, "
                    "tools_enabled, and/or a generation parameter (temperature, num_ctx, top_p, seed)",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        if (ai_model := models.AIModel.objects.filter(model=model).first()) is None:
            return Response(
                {
                    "error": "AI model not found",
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        if (
            chat_history := functions.get_chat_history(ai_model, request_data["id"])  # type: ignore[arg-type]
        ) is None:
            return Response(
                {
                    "error": "Chat not found",
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        update_fields = []

        if "title" in request_data:
            chat_history.title = request_data["title"]
            update_fields.append("title")

        if "persona_id" in request_data:
            persona_id = request_data["persona_id"]

            if persona_id is None:
                chat_history.persona = None
            else:
                if (persona := models.Persona.objects.filter(id=persona_id).first()) is None:
                    return Response({"error": "Persona not found"}, status=status.HTTP_404_NOT_FOUND)
                chat_history.persona = persona

            update_fields.append("persona")

        if "collection_ids" in request_data:
            collection_ids = request_data["collection_ids"]

            if not isinstance(collection_ids, list) or not all(
                isinstance(c, int) and not isinstance(c, bool) for c in collection_ids
            ):
                return Response(
                    {"error": "collection_ids must be a list of integers"}, status=status.HTTP_400_BAD_REQUEST
                )

            collections = list(models.DocumentCollection.objects.filter(id__in=collection_ids))
            if len(collections) != len(set(collection_ids)):
                return Response({"error": "One or more collections not found"}, status=status.HTTP_404_NOT_FOUND)

            # M2M — set() writes through immediately, it's not part of the
            # deferred save(update_fields=...) below.
            chat_history.active_collections.set(collections)

        if "tools_enabled" in request_data:
            tools_enabled = request_data["tools_enabled"]

            if not isinstance(tools_enabled, bool):
                return Response({"error": "tools_enabled must be a boolean"}, status=status.HTTP_400_BAD_REQUEST)

            chat_history.tools_enabled = tools_enabled
            update_fields.append("tools_enabled")

        for field, (expected_type, min_value, max_value) in self._GENERATION_PARAM_BOUNDS.items():
            if field not in request_data:
                continue

            value = request_data[field]

            if value is None:
                setattr(chat_history, field, None)
                update_fields.append(field)
                continue

            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return Response({"error": f"{field} must be a number or null"}, status=status.HTTP_400_BAD_REQUEST)

            value = expected_type(value)

            if min_value is not None and value < min_value:
                return Response(
                    {"error": f"{field} must be >= {min_value}"}, status=status.HTTP_400_BAD_REQUEST
                )
            if max_value is not None and value > max_value:
                return Response(
                    {"error": f"{field} must be <= {max_value}"}, status=status.HTTP_400_BAD_REQUEST
                )

            setattr(chat_history, field, value)
            update_fields.append(field)

        chat_history.save(update_fields=update_fields)

        return Response(
            {
                "status": "Chat updated",
            },
            status=status.HTTP_200_OK,
        )


