"""Reading and writing a chat's messages: the branching conversation tree,
persisting a finished exchange, searching, and the per-chat settings the model
call needs.

A chat is a tree of messages, not a list. `ChatHistory.active_leaf_id` points
at the tip of the branch being shown; regenerate, edit-resend and branch
switches just move that pointer, so an old branch is never deleted.

Functions that write take the session to write in and commit it themselves —
they are the unit that has to succeed or fail as a whole. They take the chat
as a row loaded earlier (possibly in a session that has since closed) and only
use its id and its `active_leaf_id` as it was loaded: a reply belongs under the
message the conversation was built from, whatever else happened meanwhile.
"""

import typing

from sqlalchemy import Select, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import set_committed_value

from app.db import models
from app.db.models import utcnow
from app.llm.generation import GenerationParams

MIN_SEARCH_QUERY_LENGTH = 2
SEARCH_RESULT_LIMIT = 50
SEARCH_SNIPPET_CONTEXT_CHARS = 60


def deserialize_messages(messages: typing.Iterable[models.ChatMessage]) -> list[dict[str, typing.Any]]:
    # "id" rides along for context_window bookkeeping (which message a
    # summary covers up to); generation.map_history ignores it when building
    # the actual LLM payload.
    return [
        {
            "id": message.id,
            "role": str(message.role),
            "content": str(message.content),
            "image": str(message.image) if message.image else "",
        }
        for message in messages
    ]


async def get_ai_model(session: AsyncSession, value: str) -> models.AIModel | None:
    return (
        await session.execute(select(models.AIModel).where(models.AIModel.model == value))
    ).scalar_one_or_none()


def chats_of_model(model_id: int, *, newest_first: bool = False) -> Select[models.ChatHistory]:
    statement = select(models.ChatHistory).where(models.ChatHistory.ai_model_id == model_id)
    if newest_first:
        statement = statement.order_by(
            models.ChatHistory.last_update_time.desc(), models.ChatHistory.id.desc()
        )
    return statement


async def get_chat_history(
    session: AsyncSession, model: models.AIModel | None, chat_id: int
) -> models.ChatHistory | None:
    if model is None:
        return None

    return (
        await session.execute(chats_of_model(model.id).where(models.ChatHistory.id == chat_id))
    ).scalar_one_or_none()


def get_generation_params(chat: models.ChatHistory) -> GenerationParams:
    return GenerationParams(
        temperature=chat.temperature,
        num_ctx=chat.num_ctx,
        top_p=chat.top_p,
        seed=chat.seed,
    )


async def get_persona_system_prompt(session: AsyncSession, chat: models.ChatHistory) -> str | None:
    if not chat.persona_id:
        return None

    return (
        await session.execute(
            select(models.Persona.system_prompt).where(models.Persona.id == chat.persona_id)
        )
    ).scalar_one_or_none()


async def get_messages_for_chat(session: AsyncSession, chat: models.ChatHistory) -> list[models.ChatMessage]:
    """Root-to-leaf path of the currently active branch — everything the
    user sees and everything sent to the model as history. Walking up via
    `parent_id` from `active_leaf_id`, rather than a flat created_at-ordered
    scan, is what makes regenerate/edit-resend/branch-switch safe: an old
    branch's messages are never deleted, just left off this path.
    """
    return (await _load_active_path(session, chat))[0]


async def _load_active_path(
    session: AsyncSession, chat: models.ChatHistory
) -> tuple[list[models.ChatMessage], dict[int | None, list[int]]]:
    """(active path, parent id -> child ids oldest first) in two queries,
    however long the chat — following parents one hop at a time would cost a
    query per message, plus one more per message for its siblings. The tree
    pass reads only ids (no content, no base64 images); full rows are fetched
    just for the messages actually on the path.
    """
    parent_of: dict[int, int | None] = {}
    children_of: dict[int | None, list[int]] = {}

    tree_rows = await session.execute(
        select(models.ChatMessage.id, models.ChatMessage.parent_id)
        .where(models.ChatMessage.chat_id == chat.id)
        .order_by(models.ChatMessage.created_at, models.ChatMessage.id)
    )
    for message_id, parent_id in tree_rows:
        parent_of[message_id] = parent_id
        children_of.setdefault(parent_id, []).append(message_id)

    path_ids: list[int] = []
    node_id = chat.active_leaf_id
    while node_id is not None and node_id in parent_of:
        path_ids.append(node_id)
        node_id = parent_of[node_id]
    path_ids.reverse()

    if not path_ids:
        return [], children_of

    rows = {
        message.id: message
        for message in (
            await session.execute(select(models.ChatMessage).where(models.ChatMessage.id.in_(path_ids)))
        ).scalars()
    }
    return [rows[i] for i in path_ids], children_of


async def get_branch_siblings(session: AsyncSession, message: models.ChatMessage) -> list[models.ChatMessage]:
    """All messages competing for the same slot as `message` — same parent,
    same chat — oldest first. What a branch switcher cycles through."""
    return list(
        (
            await session.execute(
                select(models.ChatMessage)
                .where(
                    models.ChatMessage.chat_id == message.chat_id,
                    models.ChatMessage.parent_id == message.parent_id,
                )
                .order_by(models.ChatMessage.created_at, models.ChatMessage.id)
            )
        ).scalars()
    )


async def get_branch_leaf(session: AsyncSession, message: models.ChatMessage) -> models.ChatMessage:
    """Descends to the tip of whatever was most recently built on top of
    `message` — where switching to its branch should land."""
    node = message
    while True:
        child = (
            await session.execute(
                select(models.ChatMessage)
                .where(models.ChatMessage.parent_id == node.id)
                .order_by(models.ChatMessage.created_at.desc(), models.ChatMessage.id.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if child is None:
            return node
        node = child


async def serialize_active_path_for_display(
    session: AsyncSession, chat: models.ChatHistory
) -> list[dict[str, str | int]]:
    """Like deserialize_messages, but for the REST payloads the frontend
    renders from — adds branch position so a message with siblings can show
    a "◀ 2/3 ▶" switcher. Not used for LLM history, which has no use for it.
    """
    path, children_of = await _load_active_path(session, chat)
    result: list[dict[str, str | int]] = []

    for message in path:
        sibling_ids = children_of.get(message.parent_id, [message.id])
        result.append(
            {
                "role": str(message.role),
                "content": str(message.content),
                "image": str(message.image) if message.image else "",
                "thinking": str(message.thinking),
                "model": str(message.model_label),
                "sibling_count": len(sibling_ids),
                "sibling_index": sibling_ids.index(message.id),
            }
        )

    return result


async def _point_chat_at(
    session: AsyncSession, chat: models.ChatHistory, leaf_id: int, *, touch: bool
) -> None:
    """Moves the chat's active leaf, and — for the writes that add or replace
    messages — its last-update time, which orders the chat list. A rename or
    a branch switch must not touch it, or the chat would jump to the top."""
    values: dict[str, typing.Any] = {"active_leaf_id": leaf_id}
    if touch:
        values["last_update_time"] = utcnow()

    await session.execute(update(models.ChatHistory).where(models.ChatHistory.id == chat.id).values(**values))
    for attribute, value in values.items():
        set_committed_value(chat, attribute, value)


async def add_messages_to_history(
    session: AsyncSession,
    chat: models.ChatHistory,
    user_content: str,
    assistant_content: str,
    image: str = "",
    thinking: str = "",
    model_label: str = "",
) -> None:
    user_message = models.ChatMessage(
        chat_id=chat.id, parent_id=chat.active_leaf_id, role="user", content=user_content, image=image
    )
    session.add(user_message)
    await session.flush()

    assistant_message = models.ChatMessage(
        chat_id=chat.id,
        parent_id=user_message.id,
        role="assistant",
        content=assistant_content,
        thinking=thinking,
        model_label=model_label,
    )
    session.add(assistant_message)
    await session.flush()

    await _point_chat_at(session, chat, assistant_message.id, touch=True)
    await session.commit()


async def commit_new_assistant_branch(
    session: AsyncSession,
    chat: models.ChatHistory,
    parent_user_message_id: int,
    assistant_content: str,
    thinking: str = "",
    model_label: str = "",
) -> None:
    """Regenerate: adds a new sibling reply under the same user message
    instead of touching the old one. Only ever called once a new response
    has actually been generated — a failed or cancelled regenerate simply
    never calls this, so the old branch is untouched and still active_leaf.
    """
    new_assistant = models.ChatMessage(
        chat_id=chat.id,
        parent_id=parent_user_message_id,
        role="assistant",
        content=assistant_content,
        thinking=thinking,
        model_label=model_label,
    )
    session.add(new_assistant)
    await session.flush()

    await _point_chat_at(session, chat, new_assistant.id, touch=True)
    await session.commit()


async def commit_new_exchange_branch(
    session: AsyncSession,
    chat: models.ChatHistory,
    parent_message_id: int | None,
    user_content: str,
    assistant_content: str,
    image: str = "",
    thinking: str = "",
    model_label: str = "",
) -> None:
    """Edit-resend: branches a new user+assistant pair under the same parent
    the edited message had, leaving the old branch — and everything after
    it — exactly where it was. Only called once a new response actually
    exists, for the same reason as commit_new_assistant_branch.
    """
    new_user = models.ChatMessage(
        chat_id=chat.id, parent_id=parent_message_id, role="user", content=user_content, image=image
    )
    session.add(new_user)
    await session.flush()

    new_assistant = models.ChatMessage(
        chat_id=chat.id,
        parent_id=new_user.id,
        role="assistant",
        content=assistant_content,
        thinking=thinking,
        model_label=model_label,
    )
    session.add(new_assistant)
    await session.flush()

    await _point_chat_at(session, chat, new_assistant.id, touch=True)
    await session.commit()


async def prepare_edit_resend(
    session: AsyncSession, chat: models.ChatHistory, index: int
) -> tuple[int | None, list[dict[str, typing.Any]]] | None:
    """Returns (id of the message to branch the new exchange from, history
    before it) for the user message at `index` in the active path — an index
    rather than a message id because the frontend only ever tracks messages by
    their position in the array it already has loaded, and that array's order
    matches the active path by construction (see ChatCard.vue). The edited
    message's own old content/image aren't returned: the caller already has
    the new ones from the client, which is the whole point of an edit. None if
    the index is out of range or doesn't land on a user message.
    """
    path = await get_messages_for_chat(session, chat)

    if index < 0 or index >= len(path) or path[index].role != "user":
        return None

    return path[index].parent_id, deserialize_messages(path[:index])


async def prepare_regenerate(
    session: AsyncSession, chat: models.ChatHistory
) -> tuple[int, str, str, list[dict[str, typing.Any]]] | None:
    """Returns (id of the user message to branch a new reply from, its
    content, its image, history before that turn) for the active path's most
    recent exchange, or None if it doesn't end in one (nothing to regenerate —
    e.g. empty chat, or the last turn is still an un-answered user message).
    """
    path = await get_messages_for_chat(session, chat)

    if len(path) < 2 or path[-1].role != "assistant" or path[-2].role != "user":
        return None

    last_user = path[-2]
    return (
        last_user.id,
        str(last_user.content),
        str(last_user.image) if last_user.image else "",
        deserialize_messages(path[:-2]),
    )


async def switch_branch(
    session: AsyncSession, chat: models.ChatHistory, index: int, sibling_index: int
) -> list[dict[str, str | int]] | None:
    """Moves the active path's branch point at `index` to a different
    sibling, jumping active_leaf to the tip of whatever was built on that
    sibling. Returns the new active path for display, or None if `index` or
    `sibling_index` don't resolve to a real message.
    """
    path = await get_messages_for_chat(session, chat)

    if index < 0 or index >= len(path):
        return None

    siblings = await get_branch_siblings(session, path[index])

    if sibling_index < 0 or sibling_index >= len(siblings):
        return None

    leaf = await get_branch_leaf(session, siblings[sibling_index])
    await _point_chat_at(session, chat, leaf.id, touch=False)
    await session.commit()

    return await serialize_active_path_for_display(session, chat)


async def save_title_if_untitled(session: AsyncSession, chat: models.ChatHistory, title: str) -> bool:
    """Only replaces the default title — never one the user typed, even if
    they renamed the chat while the title was being generated."""
    updated = (
        await session.execute(
            update(models.ChatHistory)
            .where(models.ChatHistory.id == chat.id, models.ChatHistory.title == models.DEFAULT_CHAT_TITLE)
            .values(title=title)
            .returning(models.ChatHistory.id)
        )
    ).scalar_one_or_none() is not None
    await session.commit()
    if updated:
        set_committed_value(chat, "title", title)
    return updated


async def save_memory_summary(
    session: AsyncSession, chat: models.ChatHistory, summary: str, until_message_id: int
) -> None:
    await session.execute(
        update(models.ChatHistory)
        .where(models.ChatHistory.id == chat.id)
        .values(memory_summary=summary, memory_summary_until_id=until_message_id)
    )
    await session.commit()
    set_committed_value(chat, "memory_summary", summary)
    set_committed_value(chat, "memory_summary_until_id", until_message_id)


async def save_seed_summary(session: AsyncSession, chat: models.ChatHistory, summary: str) -> None:
    await session.execute(
        update(models.ChatHistory).where(models.ChatHistory.id == chat.id).values(seed_summary=summary)
    )
    await session.commit()
    set_committed_value(chat, "seed_summary", summary)


async def get_seed_source_history(
    session: AsyncSession, chat: models.ChatHistory
) -> list[dict[str, typing.Any]]:
    if chat.seed_source_id is None:
        return []
    source = (
        await session.execute(select(models.ChatHistory).where(models.ChatHistory.id == chat.seed_source_id))
    ).scalar_one_or_none()
    return deserialize_messages(await get_messages_for_chat(session, source)) if source else []


def _build_search_snippet(content: str, query: str) -> str:
    lower_content = content.lower()
    match_index = lower_content.find(query.lower())

    if match_index == -1:
        # Shouldn't happen given the ILIKE filter that gets us here, but
        # fail into a plain truncation rather than an IndexError.
        return content[: SEARCH_SNIPPET_CONTEXT_CHARS * 2]

    start = max(0, match_index - SEARCH_SNIPPET_CONTEXT_CHARS)
    end = min(len(content), match_index + len(query) + SEARCH_SNIPPET_CONTEXT_CHARS)

    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(content) else ""

    return f"{prefix}{content[start:end]}{suffix}"


async def search_messages(
    session: AsyncSession, ai_model: models.AIModel, query: str
) -> list[dict[str, str]]:
    """Searches every message across every chat for this model — including
    inactive branches, not just each chat's currently active path, since
    finding something said down an old branch is still useful. Capped and
    ordered newest-first rather than paginated: this is a quick-jump search
    box, not a full chat browser.
    """
    query = query.strip()

    if len(query) < MIN_SEARCH_QUERY_LENGTH:
        return []

    matches = await session.execute(
        select(
            models.ChatMessage.chat_id,
            models.ChatHistory.title,
            models.ChatMessage.role,
            models.ChatMessage.content,
        )
        .join(models.ChatHistory, models.ChatHistory.id == models.ChatMessage.chat_id)
        .where(
            models.ChatHistory.ai_model_id == ai_model.id,
            models.ChatMessage.content.icontains(query, autoescape=True),
        )
        .order_by(models.ChatMessage.created_at.desc(), models.ChatMessage.id.desc())
        .limit(SEARCH_RESULT_LIMIT)
    )

    return [
        {
            "chat_id": str(chat_id),
            "chat_title": title,
            "role": role,
            "snippet": _build_search_snippet(content, query),
        }
        for chat_id, title, role, content in matches
    ]
