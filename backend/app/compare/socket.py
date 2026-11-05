import asyncio
import json
import logging
import typing

from fastapi import APIRouter, WebSocket

from app.chat import history, validation
from app.core.websocket import Client
from app.db.session import SessionLocal
from app.llm import generation
from app.llm.runtime import ModelUnavailableError

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_COMPARE_TARGETS = 4


class CompareConnection:
    """Runs one prompt across several models at once so their answers and
    timing can be read side by side. Ephemeral — nothing here touches a chat;
    this is a comparison tool, so there is nothing to save or branch."""

    def __init__(self, websocket: WebSocket) -> None:
        self.client = Client(websocket)
        self.tasks: list[asyncio.Task[None]] = []

    async def serve(self) -> None:
        await self.client.accept()
        try:
            async for text in self.client.text_frames():
                await self._receive(text)
        finally:
            await self._shutdown()

    async def _shutdown(self) -> None:
        self.client.gone = True
        self._cancel_running()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    def _running(self) -> list[asyncio.Task[None]]:
        return [task for task in self.tasks if not task.done()]

    def _cancel_running(self) -> None:
        for task in self._running():
            task.cancel()

    async def _send_error(self, target: str | None, message: str) -> None:
        await self.client.send({"target": target, "error": message, "done": True})

    async def _receive(self, text: str) -> None:
        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                raise json.JSONDecodeError("not an object", text, 0)
        except json.JSONDecodeError:
            logger.warning("Received invalid JSON over WebSocket")
            await self._send_error(None, "Invalid message format.")
            return

        if data.get("type") == "stop":
            # Each target's task answers with its own "stopped" frame.
            self._cancel_running()
            return

        if self._running():
            await self._send_error(None, "A comparison is already running. Wait for it to finish.")
            return

        try:
            prompt, targets = self._parse(data)
        except validation.InvalidMessage as e:
            await self._send_error(None, str(e))
            return

        # Fire-and-forget: awaiting the fan-out here would stop this loop from
        # seeing the Stop message that is meant to cancel it.
        self.tasks = [
            asyncio.create_task(self._run_one(model_name, parameters, prompt), name=f"compare-{model_name}")
            for model_name, parameters in targets
        ]

    @staticmethod
    def _parse(data: dict[str, typing.Any]) -> tuple[str, list[tuple[str, str]]]:
        prompt = validation.check_text(data.get("prompt", ""), "Prompt")
        targets = data.get("targets", [])

        if not isinstance(targets, list) or not 1 <= len(targets) <= MAX_COMPARE_TARGETS:
            raise validation.InvalidMessage(f"Provide between 1 and {MAX_COMPARE_TARGETS} models to compare.")

        parsed: list[tuple[str, str]] = []
        for target in targets:
            if (
                not isinstance(target, dict)
                or not isinstance(target.get("model"), str)
                or not isinstance(target.get("parameters"), str)
            ):
                raise validation.InvalidMessage("Each target needs a model and parameters.")
            parsed.append((target["model"], target["parameters"]))
        return prompt, parsed

    async def _run_one(self, model_name: str, parameters: str, prompt: str) -> None:
        target = f"{model_name}:{parameters}"

        try:
            async with SessionLocal() as session:
                ai_model = await history.get_ai_model(session, model_name)
            if ai_model is None:
                await self._send_error(target, "Model not found.")
                return

            stats = generation.GenerationStats()
            async for chunk in generation.astream_bot_response(
                ai_model.model, parameters, prompt, "", [], None, None, stats
            ):
                # Reasoning isn't requested here, so only text arrives.
                if isinstance(chunk, str):
                    await self.client.send({"target": target, "message": chunk, "done": False})

            done: dict[str, typing.Any] = {"target": target, "message": "", "done": True}
            if stats:
                done["stats"] = dict(stats)
            await self.client.send(done)
        except asyncio.CancelledError:
            logger.info("Comparison stopped for target %s", target)
            # Nothing to say if the whole client left; send() is then a no-op.
            await self.client.send({"target": target, "message": "", "done": True, "stopped": True})
        except ModelUnavailableError as e:
            logger.warning("Model unavailable in the compare socket: %s", e)
            await self._send_error(target, str(e))
        except Exception:
            logger.exception("Unexpected error comparing %s", target)
            await self._send_error(target, "An unexpected error occurred. Please try again.")


@router.websocket("/ws/compare/")
async def compare_socket(websocket: WebSocket) -> None:
    await CompareConnection(websocket).serve()
