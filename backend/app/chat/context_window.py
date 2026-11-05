"""How much of a conversation fits in the model's context window ("memory"
in the UI), and what to leave out when it doesn't.

Ollama silently drops the oldest messages once a prompt outgrows num_ctx —
the model then "forgets" without anyone being told. Fitting the history
here first makes that visible (the reply says how many messages were left
out) and lets the chat choose what happens instead: forget the oldest
messages, or replace them with a running summary.
"""

from __future__ import annotations

import dataclasses
import math
import typing

# Plain-language presets the UI offers instead of a raw num_ctx. "Pages"
# assume ~500 words ≈ 650 tokens of English prose.
MEMORY_PRESETS: dict[str, int] = {
    "short": 4096,
    "standard": 8192,
    "long": 16384,
    "very_long": 32768,
}
DEFAULT_PRESET = "standard"
TOKENS_PER_PAGE = 650

# No tokenizer is reachable through Ollama's API, so this estimates.
# 3.5 characters per token is deliberately pessimistic for English (closer
# to 4) so the estimate errs towards "doesn't fit" rather than letting
# Ollama truncate behind our back; other languages run denser anyway.
CHARS_PER_TOKEN = 3.5
IMAGE_TOKENS = 800
PER_MESSAGE_OVERHEAD_TOKENS = 4

MEMORY_MODE_TRIM = "trim"
MEMORY_MODE_SUMMARIZE = "summarize"


def resolve_num_ctx(chat_num_ctx: int | None, model_max: int | None) -> int:
    """The context size actually requested: the chat's own choice or the
    "standard" preset, never above what the model was trained for."""
    requested = chat_num_ctx or MEMORY_PRESETS[DEFAULT_PRESET]
    return min(requested, model_max) if model_max else requested


def preset_for(num_ctx: int) -> str | None:
    return next((name for name, size in MEMORY_PRESETS.items() if size == num_ctx), None)


def estimate_text_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def estimate_message_tokens(message: typing.Mapping[str, typing.Any]) -> int:
    tokens = estimate_text_tokens(str(message.get("content") or "")) + PER_MESSAGE_OVERHEAD_TOKENS
    if message.get("image"):
        tokens += IMAGE_TOKENS
    return tokens


def response_reserve(num_ctx: int) -> int:
    """Room left for the reply itself — a prompt that fills the whole window
    leaves the model nothing to answer with."""
    return max(512, min(2048, num_ctx // 4))


@dataclasses.dataclass
class FitResult:
    history: list[dict[str, typing.Any]]
    # How many of the oldest history messages were left out.
    dropped: int
    # Images removed from older turns to make room (the turn itself kept).
    images_dropped: int
    estimated_prompt_tokens: int
    # Even the system prompt + newest message alone exceed the window.
    overflow: bool


def fit_history(
    system_prompt: str | None,
    history: list[dict[str, typing.Any]],
    new_message: dict[str, typing.Any],
    num_ctx: int,
) -> FitResult:
    """Keeps as much recent history as fits in num_ctx minus the reply
    reserve. Images go first (one costs as much as ~3 pages of text and the
    model rarely needs an old one again), then whole messages from the
    oldest end. Never drops the system prompt or the new message.
    """
    budget = num_ctx - response_reserve(num_ctx)
    fixed = estimate_message_tokens(new_message)
    if system_prompt:
        fixed += estimate_message_tokens({"content": system_prompt})

    kept = [dict(m) for m in history]
    costs = [estimate_message_tokens(m) for m in kept]
    images_dropped = 0

    for i, message in enumerate(kept):
        if fixed + sum(costs) <= budget:
            break
        if message.get("image"):
            message["image"] = ""
            costs[i] -= IMAGE_TOKENS
            images_dropped += 1

    dropped = 0
    while kept and fixed + sum(costs) > budget:
        kept.pop(0)
        costs.pop(0)
        dropped += 1

    # A history that now starts with an assistant reply reads as the model
    # talking to itself; drop the orphan too.
    while kept and kept[0].get("role") == "assistant":
        kept.pop(0)
        costs.pop(0)
        dropped += 1

    estimated = fixed + sum(costs)
    return FitResult(
        history=kept,
        dropped=dropped,
        images_dropped=images_dropped,
        estimated_prompt_tokens=estimated,
        overflow=estimated > budget,
    )


class ContextReport(typing.TypedDict):
    limit: int
    used: int
    model_max: int | None
    preset: str | None
    dropped: int
    summarized: bool
    # Fraction of the model in GPU memory after this reply; None when no
    # GPU is involved. See llm/runtime.get_gpu_share.
    gpu_share: typing.NotRequired[float | None]


SUMMARY_INSTRUCTIONS = (
    "Summarize the conversation below so it can be continued later without the original messages. "
    "Keep every fact, name, number, decision, open question and preference the user stated. "
    "Write in the same language as the conversation, in at most 250 words, as plain prose — "
    "no preamble, just the summary."
)
MAX_TRANSCRIPT_MESSAGE_CHARS = 2000


def build_summary_prompt(previous_summary: str, messages: list[dict[str, typing.Any]], num_ctx: int) -> str:
    """The transcript is capped so the summarizing call itself fits in the
    same window it's trying to free up."""
    max_chars = int((num_ctx - response_reserve(num_ctx)) * CHARS_PER_TOKEN) - len(SUMMARY_INSTRUCTIONS) - 200
    lines: list[str] = []
    for message in messages:
        speaker = "User" if message.get("role") == "user" else "Assistant"
        content = str(message.get("content") or "")[:MAX_TRANSCRIPT_MESSAGE_CHARS]
        lines.append(f"{speaker}: {content}")

    transcript = "\n\n".join(lines)
    earlier = f"Summary of what came before:\n{previous_summary}\n\n" if previous_summary else ""
    budget_for_transcript = max(0, max_chars - len(earlier))
    if len(transcript) > budget_for_transcript:
        transcript = transcript[-budget_for_transcript:]

    return f"{SUMMARY_INSTRUCTIONS}\n\n{earlier}Conversation:\n{transcript}"


def summary_system_block(summary: str, carried_over: bool) -> str:
    if carried_over:
        return f"This chat continues an earlier conversation. Summary of that conversation:\n{summary}"
    return (
        "Summary of the earlier part of this conversation (those messages no longer fit in memory):\n"
        f"{summary}"
    )
