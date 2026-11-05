import asyncio
import dataclasses
import json
import logging
import typing
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, WebSocket
from sqlalchemy import select

from app.chat import context_window, history, memory, validation
from app.core.websocket import Client
from app.db import models
from app.db.session import SessionLocal
from app.llm import generation, runtime
from app.llm.runtime import ModelUnavailableError
from app.rag import retrieval

logger = logging.getLogger(__name__)

router = APIRouter()

TOOL_APPROVAL_TIMEOUT_SECONDS = 300

ALREADY_GENERATING_MESSAGE = "A response is already generating. Stop it first."
UNEXPECTED_ERROR_MESSAGE = "An unexpected error occurred. Please try again."


@dataclasses.dataclass(frozen=True)
class Turn:
    """One request to the model, and where its reply joins the message tree."""

    ai_model: models.AIModel
    chat: models.ChatHistory
    parameters: str
    prompt: str
    image: str
    structured_output: list[dict[str, str | None]] | None
    history: list[dict[str, typing.Any]]
    # Regenerate: the user message the new reply is added under.
    regenerate_under: int | None = None
    # Edit-resend: the new exchange is added under edit_parent. None is a
    # legitimate parent (editing a chat's first message makes a new root), so
    # this flag, not edit_parent, says which mode it is.
    is_edit_resend: bool = False
    edit_parent: int | None = None

    @property
    def model_label(self) -> str:
        return f"{self.ai_model.model}:{self.parameters}"

    @property
    def is_first_exchange(self) -> bool:
        return not self.history and self.regenerate_under is None and not self.is_edit_resend


@dataclasses.dataclass
class Reply:
    text: str = ""
    thinking: str = ""
    stats: generation.GenerationStats = dataclasses.field(default_factory=generation.GenerationStats)
    context: context_window.ContextReport | None = None
    citations: list[retrieval.Citation] = dataclasses.field(default_factory=list)

    def with_report(self, payload: dict[str, typing.Any]) -> dict[str, typing.Any]:
        if self.stats:
            payload["stats"] = dict(self.stats)
        if self.context is not None:
            # The prompt side is our estimate — Ollama's prompt_eval_count
            # only counts tokens it didn't have cached, so it undercounts
            # every turn after the first. The reply side is exact.
            payload["context"] = {
                **self.context,
                "used": self.context["used"] + self.stats.get("completion_tokens", 0),
            }
        return payload


class ChatConnection:
    def __init__(self, websocket: WebSocket, chat_id: int) -> None:
        self.client = Client(websocket)
        self.chat_id = chat_id
        self.generation_task: asyncio.Task[None] | None = None
        self.title_task: asyncio.Task[None] | None = None
        self.pending_approvals: dict[str, asyncio.Future[bool]] = {}
        self._stop_requested = False
        # Once the finished reply is being saved, a Stop would only cancel the
        # save and then save the same reply a second time.
        self._saving = False

    async def serve(self) -> None:
        await self.client.accept()
        try:
            async for text in self.client.text_frames():
                await self._receive(text)
        finally:
            await self._shutdown()

    async def _shutdown(self) -> None:
        self.client.gone = True
        self._request_stop()
        if self.title_task is not None and not self.title_task.done():
            self.title_task.cancel()
        # Awaited, not just cancelled: a cancelled reply still saves what it
        # has, and that has to finish before this connection's database work
        # is considered over.
        running = [task for task in (self.generation_task, self.title_task) if task is not None]
        await asyncio.gather(*running, return_exceptions=True)

    async def _send_error(self, message: str) -> None:
        await self.client.send({"error": message, "done": True})

    def _is_generating(self) -> bool:
        return self.generation_task is not None and not self.generation_task.done()

    def _request_stop(self) -> None:
        # Cancelling twice would interrupt the first cancellation's own
        # clean-up, which is where the partial reply gets saved.
        if (
            self.generation_task is not None
            and self._is_generating()
            and not self._stop_requested
            and not self._saving
        ):
            self._stop_requested = True
            self.generation_task.cancel()

    async def _receive(self, text: str) -> None:
        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                raise json.JSONDecodeError("not an object", text, 0)
        except json.JSONDecodeError:
            logger.warning("Received invalid JSON over WebSocket")
            await self._send_error("Invalid message format.")
            return

        try:
            await self._dispatch(data)
        except validation.InvalidMessage as e:
            logger.warning("Rejected a chat socket message: %s", e)
            await self._send_error(str(e))
        except Exception:
            logger.exception("Unexpected error handling a chat socket message")
            await self._send_error(UNEXPECTED_ERROR_MESSAGE)

    async def _dispatch(self, data: dict[str, typing.Any]) -> None:
        kind = data.get("type", "prompt")

        # Stop and tool approvals have to be answered while a reply streams,
        # which is why generation runs in its own task and never blocks this
        # receive loop.
        if kind == "stop":
            self._request_stop()
            return
        if kind == "tool_approval":
            self._resolve_tool_approval(data)
            return

        builders = {
            "prompt": self._prompt_turn,
            "regenerate": self._regenerate_turn,
            "edit_resend": self._edit_resend_turn,
        }
        build = builders.get(kind) if isinstance(kind, str) else None
        if build is None:
            await self._send_error(f"Unknown message type: {kind!r}")
            return

        if self._is_generating():
            await self._send_error(ALREADY_GENERATING_MESSAGE)
            return

        turn = await build(data)
        self._stop_requested = False
        self._saving = False
        self.generation_task = asyncio.create_task(self._generate(turn), name=f"chat-{self.chat_id}-reply")

    def _resolve_tool_approval(self, data: dict[str, typing.Any]) -> None:
        future = self.pending_approvals.get(str(data.get("id", "")))
        if future is not None and not future.done():
            future.set_result(data.get("approved") is True)

    async def _request_tool_approval(self, request: generation.ToolApprovalRequest) -> bool:
        """Shows the user what an MCP tool is about to do and waits for
        Allow/Deny. Unanswered (tab closed, user walked away) counts as denied."""
        if self.client.gone:
            return False

        approval_id = uuid.uuid4().hex
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self.pending_approvals[approval_id] = future

        await self.client.send({"tool_approval": {"id": approval_id, **request}, "done": False})
        try:
            return await asyncio.wait_for(future, TOOL_APPROVAL_TIMEOUT_SECONDS)
        except TimeoutError:
            return False
        finally:
            self.pending_approvals.pop(approval_id, None)

    async def _load_target(self, model_value: object) -> tuple[models.AIModel, models.ChatHistory]:
        name = validation.check_model_name(model_value)
        async with SessionLocal() as session:
            ai_model = await history.get_ai_model(session, name)
            chat = await history.get_chat_history(session, ai_model, self.chat_id)
        if ai_model is None or chat is None:
            raise validation.InvalidMessage(validation.INVALID_TARGET_MESSAGE)
        return ai_model, chat

    async def _prompt_turn(self, data: dict[str, typing.Any]) -> Turn:
        validation.require_keys(data, "message", "ai_model", "ai_model_parameters")
        prompt = validation.check_text(data["message"], "Message")
        image = validation.check_image(data.get("image"))
        structured_output = validation.check_structured_output(data.get("structured_output"))
        parameters = validation.check_parameters(data["ai_model_parameters"])

        ai_model, chat = await self._load_target(data["ai_model"])
        async with SessionLocal() as session:
            past = history.deserialize_messages(await history.get_messages_for_chat(session, chat))

        return Turn(ai_model, chat, parameters, prompt, image, structured_output, past)

    async def _regenerate_turn(self, data: dict[str, typing.Any]) -> Turn:
        validation.require_keys(data, "ai_model", "ai_model_parameters")
        structured_output = validation.check_structured_output(data.get("structured_output"))
        parameters = validation.check_parameters(data["ai_model_parameters"])

        ai_model, chat = await self._load_target(data["ai_model"])
        async with SessionLocal() as session:
            prepared = await history.prepare_regenerate(session, chat)
        if prepared is None:
            raise validation.InvalidMessage("Nothing to regenerate yet — send a message first.")

        user_message_id, prompt, image, past = prepared
        return Turn(
            ai_model,
            chat,
            parameters,
            prompt,
            image,
            structured_output,
            past,
            regenerate_under=user_message_id,
        )

    async def _edit_resend_turn(self, data: dict[str, typing.Any]) -> Turn:
        validation.require_keys(data, "index", "message", "ai_model", "ai_model_parameters")
        index = validation.check_index(data["index"])
        prompt = validation.check_text(data["message"], "Message")
        image = validation.check_image(data.get("image"))
        structured_output = validation.check_structured_output(data.get("structured_output"))
        parameters = validation.check_parameters(data["ai_model_parameters"])

        ai_model, chat = await self._load_target(data["ai_model"])
        async with SessionLocal() as session:
            prepared = await history.prepare_edit_resend(session, chat, index)
        if prepared is None:
            raise validation.InvalidMessage(
                "That message can no longer be edited — the conversation may have changed."
            )

        edit_parent, past = prepared
        return Turn(
            ai_model,
            chat,
            parameters,
            prompt,
            image,
            structured_output,
            past,
            is_edit_resend=True,
            edit_parent=edit_parent,
        )

    async def _save(self, turn: Turn, reply: Reply) -> None:
        """Regenerate and edit-resend both branch a new leaf onto the message
        tree instead of touching what's already there — old branches are never
        deleted, they just stop being the chat's active leaf. A normal prompt
        appends a user+assistant pair under the current leaf. Only called once
        a real reply exists, which is why a failed or cancelled regenerate or
        edit-resend creates nothing and the previous branch stays active.
        """
        async with SessionLocal() as session:
            if turn.regenerate_under is not None:
                await history.commit_new_assistant_branch(
                    session, turn.chat, turn.regenerate_under, reply.text, reply.thinking, turn.model_label
                )
            elif turn.is_edit_resend:
                await history.commit_new_exchange_branch(
                    session,
                    turn.chat,
                    turn.edit_parent,
                    turn.prompt,
                    reply.text,
                    turn.image,
                    reply.thinking,
                    turn.model_label,
                )
            else:
                await history.add_messages_to_history(
                    session, turn.chat, turn.prompt, reply.text, turn.image, reply.thinking, turn.model_label
                )

    async def _generate(self, turn: Turn) -> None:
        reply = Reply()

        # Everything that can fail — including the setup — lives inside the
        # try: an exception escaping would end the task without a done or
        # error frame, leaving the UI on "generating" forever with nothing
        # left for Stop to cancel.
        try:
            await self._produce(turn, reply)
        except asyncio.CancelledError:
            logger.info("Generation stopped by client for chat %s", self.chat_id)
            await self._finish_stopped(turn, reply)
        except ModelUnavailableError as e:
            logger.warning("Model unavailable in the chat socket: %s", e)
            await self._send_error(str(e))
        except Exception:
            logger.exception("Unexpected error generating a reply for chat %s", self.chat_id)
            await self._send_error(UNEXPECTED_ERROR_MESSAGE)

    async def _finish_stopped(self, turn: Turn, reply: Reply) -> None:
        # Streamed plain-text replies build up token by token, so a
        # cancellation mid-stream leaves a real partial answer worth keeping.
        # Structured output only ever assigns the reply once, after the whole
        # call returns, so cancelling it always leaves nothing to keep — and
        # for a regenerate that means the reply being replaced stays as it was.
        try:
            if reply.text:
                await self._save(turn, reply)
        except Exception:
            logger.exception("Could not save the partial reply for chat %s", self.chat_id)
            await self._send_error("Your reply was stopped, but the part written so far couldn't be saved.")
            return
        await self.client.send(reply.with_report({"message": "", "done": True, "stopped": True}))

    async def _produce(self, turn: Turn, reply: Reply) -> None:
        ai_model, chat = turn.ai_model, turn.chat
        generation_params = history.get_generation_params(chat)

        previous_question = next(
            (m["content"] for m in reversed(turn.history) if m.get("role") == "user"), None
        )
        async with SessionLocal() as session:
            persona_prompt = await history.get_persona_system_prompt(session, chat)
            retrieved = await retrieval.retrieve_context_for_chat(
                session, chat, turn.prompt, previous_question
            )
        reply.citations = retrieved.citations
        # Sent before the answer starts, so the user knows up front that it's
        # being written without some of their documents.
        for warning in retrieved.warnings:
            await self._send_notice(warning)

        model_info = await runtime.get_model_info(ai_model.model, turn.parameters)
        reasoning = runtime.resolve_reasoning(chat.think_level, model_info)
        # Structured output and tool calling are separate response shapes — a
        # schema-constrained JSON reply has nowhere to put an intermediate tool
        # call, so structured_output always wins when both are set.
        use_tools = (
            turn.structured_output is None
            and chat.tools_enabled
            and runtime.model_can("tools", model_info, ai_model.can_use_tools)
        )
        system_prompt, past, reply.context = await memory.fit_to_memory(
            model_info=model_info,
            ai_model=ai_model,
            parameters=turn.parameters,
            chat=chat,
            persona_prompt=persona_prompt,
            rag_context=retrieved.context,
            history_messages=turn.history,
            new_message={"role": "user", "content": turn.prompt, "image": turn.image},
            generation_params=generation_params,
            notify=self._send_notice,
        )

        if turn.structured_output is not None:
            structured = await generation.astream_structured_bot_response(
                ai_model.model,
                turn.parameters,
                turn.prompt,
                turn.image,
                past,
                turn.structured_output,
                system_prompt,
                generation_params,
                reply.stats,
            )
            if structured is None:
                await self._send_error(
                    "Unable to parse the structured response. Try again or adjust the schema."
                )
                return
            reply.text = structured

        elif use_tools:
            # MCP servers are global, not per-chat — only worth a query when the
            # tool path is actually going to run.
            async with SessionLocal() as session:
                mcp_servers = list(
                    (
                        await session.execute(
                            select(models.MCPServer)
                            .where(models.MCPServer.enabled)
                            .order_by(models.MCPServer.id)
                        )
                    ).scalars()
                )
            await self._relay(
                generation.astream_tool_response(
                    ai_model.model,
                    turn.parameters,
                    turn.prompt,
                    turn.image,
                    past,
                    system_prompt,
                    generation_params,
                    reply.stats,
                    mcp_servers,
                    reasoning,
                    self._request_tool_approval,
                ),
                reply,
            )

        else:
            await self._relay(
                generation.astream_bot_response(
                    ai_model.model,
                    turn.parameters,
                    turn.prompt,
                    turn.image,
                    past,
                    system_prompt,
                    generation_params,
                    reply.stats,
                    reasoning,
                ),
                reply,
            )

        if not reply.text:
            await self._send_error("The model returned an empty response. Try again.")
            return

        self._saving = True
        await self._save(turn, reply)
        if reply.context is not None:
            # Asked after the reply, when Ollama has actually loaded the model
            # at this context size.
            reply.context["gpu_share"] = await runtime.get_gpu_share(ai_model.model, turn.parameters)

        done = reply.with_report({"message": reply.text, "done": True})
        if reply.citations:
            done["citations"] = reply.citations
        await self.client.send(done)

        # Its own task, after the reply is out: a title is cosmetic and must
        # neither delay the answer nor keep _is_generating() true.
        if turn.is_first_exchange and chat.title == models.DEFAULT_CHAT_TITLE:
            self.title_task = asyncio.create_task(
                self._name_chat(turn, reply.text, reasoning), name=f"chat-{self.chat_id}-title"
            )

    async def _relay(
        self,
        chunks: AsyncIterator[str | generation.ThinkingDelta | generation.ToolCallInfo],
        reply: Reply,
    ) -> None:
        async for chunk in chunks:
            if isinstance(chunk, generation.ThinkingDelta):
                reply.thinking += chunk.text
                await self.client.send({"thinking": chunk.text, "done": False})
            elif isinstance(chunk, dict):
                await self.client.send({"tool_call": chunk, "done": False})
            else:
                reply.text += chunk
                await self.client.send({"message": chunk, "done": False})

    async def _name_chat(self, turn: Turn, answer: str, reasoning: bool | str | None) -> None:
        try:
            title = await generation.agenerate_title(
                turn.ai_model.model, turn.parameters, turn.prompt, answer, reasoning
            )
            if not title:
                return
            async with SessionLocal() as session:
                renamed = await history.save_title_if_untitled(session, turn.chat, title)
            if renamed:
                await self.client.send({"chat_title": title})
        except Exception as e:
            # Only a cosmetic default title is lost — the chat itself is fine.
            logger.warning("Could not generate a title for chat %s: %s", self.chat_id, e)

    async def _send_notice(self, notice: str) -> None:
        await self.client.send({"notice": notice, "done": False})


@router.websocket("/ws/chat/{chat_id:int}/")
async def chat_socket(websocket: WebSocket, chat_id: int) -> None:
    await ChatConnection(websocket, chat_id).serve()
