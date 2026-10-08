import asyncio
import base64
import binascii
import json
import logging
import typing

from asgiref.sync import sync_to_async
from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer

from . import context_window, functions, models
from .functions import ModelUnavailableError
from .rag import retrieval as rag_retrieval

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 10_000
MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB, measured on the decoded binary
MAX_STRUCTURED_FIELD_DESCRIPTION_LENGTH = 500


class ChatConsumer(AsyncWebsocketConsumer):
    generation_task: asyncio.Task | None = None

    async def connect(self):
        self.room_name = self.scope["url_route"]["kwargs"]["chat_id"]
        self.room_group_name = f"chat_{self.room_name}"
        self.generation_task = None

        await self.channel_layer.group_add(self.room_group_name, self.channel_name)  # type: ignore
        await self.accept()

    async def disconnect(self, code):
        # Best-effort: releases the model/DB work promptly instead of letting
        # it keep streaming into a socket nothing is reading from anymore.
        if self._is_generating():
            self.generation_task.cancel()  # type: ignore[union-attr]

        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)  # type: ignore

    async def _send_error(self, message: str) -> None:
        await self.send(text_data=json.dumps({"error": message, "done": True}))

    def _is_generating(self) -> bool:
        return self.generation_task is not None and not self.generation_task.done()

    async def receive(self, text_data=None, bytes_data=None):
        if text_data is None:
            await self.close(1000)
            return

        try:
            text_data_json = json.loads(text_data)
        except json.JSONDecodeError:
            logger.warning("Received invalid JSON over WebSocket")
            await self._send_error("Invalid message format.")
            return

        message_type = text_data_json.get("type", "prompt")

        if message_type == "stop":
            await self._handle_stop()
        elif message_type == "prompt":
            await self._handle_prompt(text_data_json)
        elif message_type == "regenerate":
            await self._handle_regenerate(text_data_json)
        elif message_type == "edit_resend":
            await self._handle_edit_resend(text_data_json)
        else:
            await self._send_error(f"Unknown message type: {message_type!r}")

    async def _handle_stop(self) -> None:
        # asyncio.Task.cancel() only *requests* cancellation — it's delivered
        # at the task's next await point, handled in _generate's except
        # asyncio.CancelledError block, which sends the "stopped" response.
        # Nothing to do here if there's no in-flight generation to cancel.
        if self._is_generating():
            self.generation_task.cancel()  # type: ignore[union-attr]

    async def _handle_prompt(self, text_data_json: dict) -> None:
        # dispatch() (channels/consumer.py) awaits each incoming message's
        # handler to completion before processing the next one, so without
        # running generation in a background task, a "stop" sent while a
        # response is streaming would queue behind it instead of cancelling
        # it — the whole point of Stop generation would be unreachable.
        if self._is_generating():
            await self._send_error("A response is already generating. Stop it first.")
            return

        try:
            required_keys = ["message", "ai_model", "ai_model_parameters"]
            if not all(key in text_data_json for key in required_keys):
                raise ValueError("Missing required keys")

            user_prompt = text_data_json.get("message", "")
            ai_model_value = text_data_json.get("ai_model", "")
            ai_model_parameters = text_data_json.get("ai_model_parameters", "")
            image = text_data_json.get("image", "")
            structured_output = text_data_json.get("structured_output", None)

            if not user_prompt or not isinstance(user_prompt, str):
                raise ValueError("Message must be a non-empty string.")

            if len(user_prompt) > MAX_MESSAGE_LENGTH:
                raise ValueError(
                    f"Message too long. Maximum length is {MAX_MESSAGE_LENGTH} characters."
                )

            if image:
                self._validate_image(image)

            if structured_output is not None:
                self._validate_structured_output(structured_output)

            ai_model = await database_sync_to_async(functions.get_ai_model)(
                ai_model_value
            )
            chat_history = await database_sync_to_async(functions.get_chat_history)(
                ai_model, self.room_name
            )

            if ai_model is None or chat_history is None:
                raise ValueError("Invalid model or chat history.")

            history_messages = await database_sync_to_async(self._load_history)(
                chat_history
            )
        except ValueError as e:
            logger.warning("Validation error in WebSocket receive: %s", e)
            await self._send_error(str(e))
            return

        self.generation_task = asyncio.create_task(
            self._generate(
                ai_model,
                chat_history,
                ai_model_parameters,
                user_prompt,
                image,
                structured_output,
                history_messages,
            )
        )

    async def _handle_regenerate(self, text_data_json: dict) -> None:
        if self._is_generating():
            await self._send_error("A response is already generating. Stop it first.")
            return

        try:
            required_keys = ["ai_model", "ai_model_parameters"]
            if not all(key in text_data_json for key in required_keys):
                raise ValueError("Missing required keys")

            ai_model_value = text_data_json.get("ai_model", "")
            ai_model_parameters = text_data_json.get("ai_model_parameters", "")
            structured_output = text_data_json.get("structured_output", None)

            if structured_output is not None:
                self._validate_structured_output(structured_output)

            ai_model = await database_sync_to_async(functions.get_ai_model)(
                ai_model_value
            )
            chat_history = await database_sync_to_async(functions.get_chat_history)(
                ai_model, self.room_name
            )

            if ai_model is None or chat_history is None:
                raise ValueError("Invalid model or chat history.")

            regenerate_data = await database_sync_to_async(
                functions.prepare_regenerate
            )(chat_history)
            if regenerate_data is None:
                raise ValueError("Nothing to regenerate yet — send a message first.")

            parent_user_message, user_prompt, image, history_messages = regenerate_data
        except ValueError as e:
            logger.warning("Validation error in WebSocket receive: %s", e)
            await self._send_error(str(e))
            return

        self.generation_task = asyncio.create_task(
            self._generate(
                ai_model,
                chat_history,
                ai_model_parameters,
                user_prompt,
                image,
                structured_output,
                history_messages,
                regenerate_from_user_message=parent_user_message,
            )
        )

    async def _handle_edit_resend(self, text_data_json: dict) -> None:
        if self._is_generating():
            await self._send_error("A response is already generating. Stop it first.")
            return

        try:
            required_keys = ["index", "message", "ai_model", "ai_model_parameters"]
            if not all(key in text_data_json for key in required_keys):
                raise ValueError("Missing required keys")

            index = text_data_json.get("index")
            if not isinstance(index, int) or isinstance(index, bool):
                raise ValueError("index must be an integer.")

            user_prompt = text_data_json.get("message", "")
            ai_model_value = text_data_json.get("ai_model", "")
            ai_model_parameters = text_data_json.get("ai_model_parameters", "")
            image = text_data_json.get("image", "")
            structured_output = text_data_json.get("structured_output", None)

            if not user_prompt or not isinstance(user_prompt, str):
                raise ValueError("Message must be a non-empty string.")

            if len(user_prompt) > MAX_MESSAGE_LENGTH:
                raise ValueError(
                    f"Message too long. Maximum length is {MAX_MESSAGE_LENGTH} characters."
                )

            if image:
                self._validate_image(image)

            if structured_output is not None:
                self._validate_structured_output(structured_output)

            ai_model = await database_sync_to_async(functions.get_ai_model)(
                ai_model_value
            )
            chat_history = await database_sync_to_async(functions.get_chat_history)(
                ai_model, self.room_name
            )

            if ai_model is None or chat_history is None:
                raise ValueError("Invalid model or chat history.")

            edit_data = await database_sync_to_async(functions.prepare_edit_resend)(
                chat_history, index
            )
            if edit_data is None:
                raise ValueError(
                    "That message can no longer be edited — the conversation may have changed."
                )

            edit_resend_parent, history_messages = edit_data
        except ValueError as e:
            logger.warning("Validation error in WebSocket receive: %s", e)
            await self._send_error(str(e))
            return

        self.generation_task = asyncio.create_task(
            self._generate(
                ai_model,
                chat_history,
                ai_model_parameters,
                user_prompt,
                image,
                structured_output,
                history_messages,
                is_edit_resend=True,
                edit_resend_parent=edit_resend_parent,
            )
        )

    async def _generate(
        self,
        ai_model: models.AIModel,
        chat_history: models.ChatHistory,
        ai_model_parameters: str,
        user_prompt: str,
        image: str,
        structured_output: list[dict] | None,
        history_messages: list[dict[str, str]],
        regenerate_from_user_message: models.ChatMessage | None = None,
        is_edit_resend: bool = False,
        edit_resend_parent: models.ChatMessage | None = None,
    ) -> None:
        full_response = ""
        stats: functions.GenerationStats = {}
        citations: list[rag_retrieval.Citation] = []
        context_report: context_window.ContextReport | None = None
        # Plain columns on the already-fetched row — no query, safe to read
        # directly in this async method (unlike get_persona_system_prompt's
        # `.persona` access, which needs a real query when a persona is set).
        generation_params = functions.get_generation_params(chat_history)

        def finalize_stats() -> dict | None:
            return dict(stats) if stats else None

        def finalize_context() -> dict | None:
            if context_report is None:
                return None
            # The prompt side is our estimate — Ollama's prompt_eval_count
            # only counts tokens it didn't have cached, so it undercounts
            # every turn after the first. The reply side is exact.
            return {**context_report, "used": context_report["used"] + stats.get("completion_tokens", 0)}

        async def persist(content: str) -> None:
            # Regenerate and edit-resend both branch a new leaf onto the
            # message tree instead of touching what's already there — old
            # branches are never deleted, they're just no longer
            # chat_history.active_leaf. A normal prompt appends a fresh
            # user+assistant pair under the current leaf. Called only once a
            # real response exists, which is why a failed or cancelled
            # regenerate/edit-resend never creates anything and the previous
            # branch stays active — see the docstrings on
            # commit_new_assistant_branch / commit_new_exchange_branch.
            #
            # edit_resend_parent can legitimately be None (editing a chat's
            # very first message makes the new exchange a new root), so
            # is_edit_resend is the actual mode flag — it can't double as
            # "was a parent found".
            if regenerate_from_user_message is not None:
                await database_sync_to_async(functions.commit_new_assistant_branch)(
                    chat_history, regenerate_from_user_message, content
                )
            elif is_edit_resend:
                await database_sync_to_async(functions.commit_new_exchange_branch)(
                    chat_history, edit_resend_parent, user_prompt, content, image
                )
            else:
                await database_sync_to_async(functions.add_messages_to_history)(
                    chat_history, user_prompt, content, image
                )

        # Structured output and tool calling are separate response shapes —
        # a schema-constrained JSON reply has nowhere to put an intermediate
        # tool call, so structured_output always wins when both are set.
        use_tools = (
            structured_output is None
            and chat_history.tools_enabled
            and ai_model.can_use_tools
        )

        # Everything that can fail — including this setup — lives inside the
        # try: an exception escaping here would end the task without ever
        # sending a done/error frame, leaving the UI on "generating" forever
        # with nothing left for Stop to cancel.
        try:
            persona_prompt = await database_sync_to_async(
                functions.get_persona_system_prompt
            )(chat_history)
            # Fetching active collections and running the pgvector search are
            # both blocking (DB + Docker SDK + an HTTP call to the embedding
            # model), same as get_ollama_url — run off the event loop so one
            # chat's retrieval can't stall every other concurrent chat.
            rag_context, citations, rag_warnings = await sync_to_async(
                rag_retrieval.retrieve_context_for_chat, thread_sensitive=False
            )(chat_history, user_prompt)
            # Sent before the answer starts, so the user knows up front that
            # it's being written without some of their documents.
            for warning in rag_warnings:
                await self._send_notice(warning)
            system_prompt, history_messages, context_report = await self._fit_to_memory(
                ai_model,
                ai_model_parameters,
                chat_history,
                persona_prompt,
                rag_context,
                history_messages,
                {"role": "user", "content": user_prompt, "image": image},
                generation_params,
            )
            # MCP servers are global, not per-chat (see models.MCPServer) — only
            # worth a query when the tool path is actually going to run.
            mcp_servers = (
                await database_sync_to_async(list)(
                    models.MCPServer.objects.filter(enabled=True)
                )
                if use_tools
                else []
            )

            if structured_output is not None:
                full_response = await functions.astream_structured_bot_response(
                    ai_model,
                    ai_model_parameters,
                    user_prompt,
                    image,
                    history_messages,
                    structured_output,
                    system_prompt,
                    generation_params,
                    stats,
                )

                if full_response is None:
                    await self._send_error(
                        "Unable to parse the structured response. Try again or adjust the schema."
                    )
                    return

            elif use_tools:
                async for chunk in functions.astream_tool_response(
                    ai_model,
                    ai_model_parameters,
                    user_prompt,
                    image,
                    history_messages,
                    system_prompt,
                    generation_params,
                    stats,
                    mcp_servers,
                ):
                    if isinstance(chunk, dict):
                        await self.send(
                            text_data=json.dumps({"tool_call": chunk, "done": False})
                        )
                    else:
                        full_response += chunk
                        await self.send(
                            text_data=json.dumps({"message": chunk, "done": False})
                        )

            else:
                async for chunk in functions.astream_bot_response(
                    ai_model,
                    ai_model_parameters,
                    user_prompt,
                    image,
                    history_messages,
                    system_prompt,
                    generation_params,
                    stats,
                ):
                    full_response += chunk
                    await self.send(
                        text_data=json.dumps({"message": chunk, "done": False})
                    )

            if not full_response:
                await self._send_error(
                    "The model returned an empty response. Try again."
                )
                return

            await persist(full_response)
            if context_report is not None:
                # Asked after the reply, when Ollama has actually loaded the
                # model at this context size.
                context_report["gpu_share"] = await sync_to_async(
                    functions.get_gpu_share, thread_sensitive=False
                )(ai_model.model, ai_model_parameters)
            done_payload = {"message": full_response, "done": True}
            if (final_stats := finalize_stats()) is not None:
                done_payload["stats"] = final_stats
            if (final_context := finalize_context()) is not None:
                done_payload["context"] = final_context
            if citations:
                done_payload["citations"] = citations
            await self.send(text_data=json.dumps(done_payload))

        except asyncio.CancelledError:
            logger.info("Generation stopped by client for chat %s", self.room_name)

            # Streamed plain-text responses build up token by token, so a
            # cancellation mid-stream still has a real partial answer worth
            # keeping. Structured output only ever assigns full_response
            # once, after the whole (uninterruptible from the caller's view)
            # call returns — cancelling it always leaves full_response "".
            # For regenerate specifically, an empty full_response here means
            # `persist` is never called, so the original assistant message
            # this was trying to replace is left untouched.
            if full_response:
                await persist(full_response)

            stopped_payload = {"message": "", "done": True, "stopped": True}
            if (final_stats := finalize_stats()) is not None:
                stopped_payload["stats"] = final_stats
            if (final_context := finalize_context()) is not None:
                stopped_payload["context"] = final_context
            await self.send(text_data=json.dumps(stopped_payload))
        except ModelUnavailableError as e:
            logger.warning("Model unavailable in WebSocket receive: %s", e)
            await self._send_error(str(e))
        except Exception:
            logger.exception("Unexpected error in ChatConsumer._generate")
            await self._send_error("An unexpected error occurred. Please try again.")

    async def _fit_to_memory(
        self,
        ai_model: models.AIModel,
        parameters: str,
        chat_history: models.ChatHistory,
        persona_prompt: str | None,
        rag_context: str,
        history: list[dict[str, typing.Any]],
        new_message: dict[str, typing.Any],
        generation_params: functions.GenerationParams,
    ) -> tuple[str | None, list[dict[str, typing.Any]], context_window.ContextReport]:
        """Decides the context size and what history fits in it (see
        context_window.py). Sets generation_params["num_ctx"] explicitly so the
        window the UI reports is the one Ollama actually uses. In summarize
        mode, messages that no longer fit are folded into the chat's running
        summary instead of silently vanishing.

        Summaries go between the persona and the retrieved documents, so the
        persona leads and the documents sit closest to the question.
        """
        model_max = await sync_to_async(functions.get_model_context_length, thread_sensitive=False)(
            ai_model.model, parameters
        )
        num_ctx = context_window.resolve_num_ctx(chat_history.num_ctx, model_max)
        generation_params["num_ctx"] = num_ctx
        summarize_mode = chat_history.memory_mode == models.ChatHistory.MEMORY_MODE_SUMMARIZE

        if chat_history.seed_source_id and not chat_history.seed_summary:
            source_history = await database_sync_to_async(functions.get_seed_source_history)(chat_history)
            if source_history:
                await self._send_notice("Summarizing your previous chat so this one can pick up where it left off…")
                chat_history.seed_summary = await functions.asummarize_conversation(
                    ai_model, parameters, "", source_history, num_ctx
                )
                await database_sync_to_async(functions.save_seed_summary)(chat_history, chat_history.seed_summary)

        memory_summary = ""
        history_ids = [m.get("id") for m in history]
        if summarize_mode and chat_history.memory_summary and chat_history.memory_summary_until_id in history_ids:
            memory_summary = chat_history.memory_summary
            history = history[history_ids.index(chat_history.memory_summary_until_id) + 1:]

        def system_prompt() -> str | None:
            blocks = [persona_prompt]
            if chat_history.seed_summary:
                blocks.append(context_window.summary_system_block(chat_history.seed_summary, carried_over=True))
            if memory_summary:
                blocks.append(context_window.summary_system_block(memory_summary, carried_over=False))
            blocks.append(rag_context)
            return "\n\n".join(b for b in blocks if b) or None

        fit = context_window.fit_history(system_prompt(), history, new_message, num_ctx)

        if fit.dropped and summarize_mode:
            to_fold = history[:fit.dropped]
            await self._send_notice(
                "This conversation is longer than the model's memory, so older messages are being summarized first…"
            )
            memory_summary = await functions.asummarize_conversation(
                ai_model, parameters, memory_summary, to_fold, num_ctx
            )
            await database_sync_to_async(functions.save_memory_summary)(chat_history, memory_summary, to_fold[-1]["id"])
            fit = context_window.fit_history(system_prompt(), history[fit.dropped:], new_message, num_ctx)

        report: context_window.ContextReport = {
            "limit": num_ctx,
            "used": fit.estimated_prompt_tokens,
            "model_max": model_max,
            "preset": context_window.preset_for(num_ctx),
            "dropped": fit.dropped,
            "summarized": bool(memory_summary),
        }
        return system_prompt(), fit.history, report

    async def _send_notice(self, notice: str) -> None:
        await self.send(text_data=json.dumps({"notice": notice, "done": False}))

    @staticmethod
    def _load_history(chat_history) -> list[dict[str, str]]:
        return functions.deserialize_messages(
            functions.get_messages_for_chat(chat_history)
        )

    @staticmethod
    def _validate_image(image: object) -> None:
        if not isinstance(image, str) or not image.startswith("data:image/"):
            raise ValueError("Invalid image format. Expected a base64 data URI.")

        image_parts = image.split(",", 1)
        if len(image_parts) < 2 or not image_parts[1]:
            raise ValueError("Invalid image format. Expected a base64 data URI.")

        try:
            decoded_size = len(base64.b64decode(image_parts[1], validate=True))
        except (binascii.Error, ValueError) as e:
            raise ValueError("Invalid image data — could not decode base64.") from e

        if decoded_size > MAX_IMAGE_SIZE_BYTES:
            max_mb = MAX_IMAGE_SIZE_BYTES // (1024 * 1024)
            raise ValueError(f"Image exceeds the maximum allowed size of {max_mb} MB.")

    @staticmethod
    def _validate_structured_output(structured_output: object) -> None:
        if not isinstance(structured_output, list):
            raise ValueError("structured_output must be a list.")

        for i, spec in enumerate(structured_output):
            if not isinstance(spec, dict):
                raise ValueError(f"structured_output[{i}] must be an object.")
            if not isinstance(spec.get("field"), str) or not spec["field"].strip():
                raise ValueError(
                    f"structured_output[{i}].field must be a non-empty string."
                )
            if not isinstance(spec.get("type"), str) or not spec["type"].strip():
                raise ValueError(
                    f"structured_output[{i}].type must be a non-empty string."
                )

            description = spec.get("description")
            if description is not None and (
                not isinstance(description, str)
                or len(description) > MAX_STRUCTURED_FIELD_DESCRIPTION_LENGTH
            ):
                raise ValueError(
                    f"structured_output[{i}].description must be a string of at most "
                    f"{MAX_STRUCTURED_FIELD_DESCRIPTION_LENGTH} characters."
                )

            array_type = spec.get("arrayType")
            if array_type is not None and not isinstance(array_type, str):
                raise ValueError(f"structured_output[{i}].arrayType must be a string.")


MAX_COMPARE_TARGETS = 4


class CompareConsumer(AsyncWebsocketConsumer):
    """Runs one prompt across several models concurrently so their answers
    and timing can be read side by side. Ephemeral — nothing here touches
    ChatHistory; this is a comparison tool, not a chat, so there's no
    persistence and no branching to coordinate.
    """

    active_tasks: list[asyncio.Task]
    stopping: bool

    async def connect(self):
        self.active_tasks = []
        self.stopping = False
        await self.accept()

    async def disconnect(self, code):
        for task in self.active_tasks:
            if not task.done():
                task.cancel()

    async def _send_error(self, target_key: str | None, message: str) -> None:
        await self.send(
            text_data=json.dumps({"target": target_key, "error": message, "done": True})
        )

    async def _handle_stop(self) -> None:
        # Same asyncio.Task.cancel() + CancelledError-in-_run_one pattern
        # ChatConsumer uses for Stop — the difference here is there's one
        # task per target instead of one generation_task, so every pending
        # one needs cancelling. `stopping` tells _run_one it's safe (and
        # expected) to send a "stopped" frame back, as opposed to a
        # cancellation from disconnect(), where nothing is listening anymore.
        pending = [task for task in self.active_tasks if not task.done()]
        if not pending:
            return

        self.stopping = True
        for task in pending:
            task.cancel()

    async def receive(self, text_data=None, bytes_data=None):
        if text_data is None:
            await self.close(1000)
            return

        try:
            data = json.loads(text_data)
        except json.JSONDecodeError:
            logger.warning("Received invalid JSON over WebSocket")
            await self._send_error(None, "Invalid message format.")
            return

        if data.get("type") == "stop":
            await self._handle_stop()
            return

        if any(not task.done() for task in self.active_tasks):
            await self._send_error(
                None, "A comparison is already running. Wait for it to finish."
            )
            return

        prompt = data.get("prompt", "")
        targets = data.get("targets", [])

        if not prompt or not isinstance(prompt, str):
            await self._send_error(None, "Prompt must be a non-empty string.")
            return

        if len(prompt) > MAX_MESSAGE_LENGTH:
            await self._send_error(
                None,
                f"Prompt too long. Maximum length is {MAX_MESSAGE_LENGTH} characters.",
            )
            return

        if not isinstance(targets, list) or not (
            1 <= len(targets) <= MAX_COMPARE_TARGETS
        ):
            await self._send_error(
                None, f"Provide between 1 and {MAX_COMPARE_TARGETS} models to compare."
            )
            return

        parsed_targets: list[tuple[str, str]] = []
        for target in targets:
            if (
                not isinstance(target, dict)
                or not isinstance(target.get("model"), str)
                or not isinstance(target.get("parameters"), str)
            ):
                await self._send_error(
                    None, "Each target needs a model and parameters."
                )
                return
            parsed_targets.append((target["model"], target["parameters"]))

        # Fire-and-forget, same reasoning as ChatConsumer's generation_task:
        # dispatch() awaits receive() to completion before it'll process the
        # next incoming frame, so awaiting the fan-out here would make the
        # "already running" guard above unreachable — no message could ever
        # arrive while a comparison was still in flight to trigger it.
        self.stopping = False
        self.active_tasks = [
            asyncio.create_task(self._run_one(model_name, parameters, prompt))
            for model_name, parameters in parsed_targets
        ]

    async def _run_one(self, model_name: str, parameters: str, prompt: str) -> None:
        target_key = f"{model_name}:{parameters}"

        ai_model = await database_sync_to_async(functions.get_ai_model)(model_name)
        if ai_model is None:
            await self._send_error(target_key, "Model not found.")
            return

        stats: functions.GenerationStats = {}
        try:
            async for chunk in functions.astream_bot_response(
                ai_model, parameters, prompt, "", [], None, None, stats
            ):
                await self.send(
                    text_data=json.dumps(
                        {"target": target_key, "message": chunk, "done": False}
                    )
                )

            done_payload = {"target": target_key, "message": "", "done": True}
            if stats:
                done_payload["stats"] = dict(stats)
            await self.send(text_data=json.dumps(done_payload))
        except asyncio.CancelledError:
            logger.info("Comparison stopped for target %s", target_key)

            # `stopping` means a client-initiated Stop message cancelled
            # this — the socket is still open and expecting a reply. A plain
            # disconnect cancels the same way but the socket is already
            # gone, so sending here would be pointless (nothing listening)
            # or unsafe (writing to a closed connection).
            if self.stopping:
                await self.send(
                    text_data=json.dumps(
                        {
                            "target": target_key,
                            "message": "",
                            "done": True,
                            "stopped": True,
                        }
                    )
                )
        except ModelUnavailableError as e:
            logger.warning("Model unavailable in CompareConsumer: %s", e)
            await self._send_error(target_key, str(e))
        except Exception:
            logger.exception("Unexpected error in CompareConsumer for %s", target_key)
            await self._send_error(
                target_key, "An unexpected error occurred. Please try again."
            )
