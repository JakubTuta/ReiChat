import typing
from collections.abc import Awaitable, Callable

from app.chat import context_window, history
from app.db import models
from app.db.session import SessionLocal
from app.llm import generation, runtime

Notify = Callable[[str], Awaitable[None]]

SEED_SUMMARY_NOTICE = "Summarizing your previous chat so this one can pick up where it left off…"
FOLDING_NOTICE = (
    "This conversation is longer than the model's memory, so older messages are being summarized first…"
)


async def fit_to_memory(
    *,
    model_info: runtime.ModelInfo | None,
    ai_model: models.AIModel,
    parameters: str,
    chat: models.ChatHistory,
    persona_prompt: str | None,
    rag_context: str,
    history_messages: list[dict[str, typing.Any]],
    new_message: dict[str, typing.Any],
    generation_params: generation.GenerationParams,
    notify: Notify,
) -> tuple[str | None, list[dict[str, typing.Any]], context_window.ContextReport]:
    """Decides the context size and what history fits in it (see
    context_window.py). Sets generation_params["num_ctx"] explicitly so the
    window the UI reports is the one Ollama actually uses. In summarize
    mode, messages that no longer fit are folded into the chat's running
    summary instead of silently vanishing.

    Summaries go between the persona and the retrieved documents, so the
    persona leads and the documents sit closest to the question.
    """
    model_max = model_info["context_length"] if model_info else None
    num_ctx = context_window.resolve_num_ctx(chat.num_ctx, model_max)
    generation_params["num_ctx"] = num_ctx
    summarize_mode = chat.memory_mode == models.ChatHistory.MEMORY_MODE_SUMMARIZE

    if chat.seed_source_id and not chat.seed_summary:
        async with SessionLocal() as session:
            source_history = await history.get_seed_source_history(session, chat)
        if source_history:
            await notify(SEED_SUMMARY_NOTICE)
            seed_summary = await generation.asummarize_conversation(
                ai_model.model, parameters, "", source_history, num_ctx
            )
            async with SessionLocal() as session:
                await history.save_seed_summary(session, chat, seed_summary)

    memory_summary = ""
    history_ids = [m.get("id") for m in history_messages]
    if summarize_mode and chat.memory_summary and chat.memory_summary_until_id in history_ids:
        memory_summary = chat.memory_summary
        history_messages = history_messages[history_ids.index(chat.memory_summary_until_id) + 1 :]

    def system_prompt() -> str | None:
        blocks = [persona_prompt]
        if chat.seed_summary:
            blocks.append(context_window.summary_system_block(chat.seed_summary, carried_over=True))
        if memory_summary:
            blocks.append(context_window.summary_system_block(memory_summary, carried_over=False))
        blocks.append(rag_context)
        return "\n\n".join(b for b in blocks if b) or None

    fit = context_window.fit_history(system_prompt(), history_messages, new_message, num_ctx)

    if fit.dropped and summarize_mode:
        to_fold = history_messages[: fit.dropped]
        await notify(FOLDING_NOTICE)
        memory_summary = await generation.asummarize_conversation(
            ai_model.model, parameters, memory_summary, to_fold, num_ctx
        )
        async with SessionLocal() as session:
            await history.save_memory_summary(session, chat, memory_summary, to_fold[-1]["id"])
        fit = context_window.fit_history(
            system_prompt(), history_messages[fit.dropped :], new_message, num_ctx
        )

    report: context_window.ContextReport = {
        "limit": num_ctx,
        "used": fit.estimated_prompt_tokens,
        "model_max": model_max,
        "preset": context_window.preset_for(num_ctx),
        "dropped": fit.dropped,
        "summarized": bool(memory_summary),
    }
    return system_prompt(), fit.history, report
