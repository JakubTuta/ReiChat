import typing
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete

from app.chat import history, service
from app.chat.schemas import (
    UPDATABLE_CHAT_FIELDS,
    BranchSwitch,
    ChatContinue,
    ChatCreate,
    ChatCreatedOut,
    ChatFromCompare,
    ChatListItem,
    ChatRef,
    ChatRemove,
    ChatUpdate,
)
from app.core.security import require_json_body
from app.db import models
from app.db.session import SessionDep

router = APIRouter(dependencies=[Depends(require_json_body)])


async def _model_from_path(model: str, session: SessionDep) -> models.AIModel:
    found = await history.get_ai_model(session, model)
    if found is None:
        raise HTTPException(status_code=404, detail="AI model not found")
    return found


# The `{model:path}` in every route below: model names can contain slashes
# ("hf.co/user/repo").
ModelDep = Annotated[models.AIModel, Depends(_model_from_path)]


async def _chat_or_404(session: SessionDep, model: models.AIModel, chat_id: int) -> models.ChatHistory:
    chat = await history.get_chat_history(session, model, chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@router.get("/search/{model:path}")
async def search(model: ModelDep, session: SessionDep, q: str = "") -> list[dict[str, str]]:
    return await history.search_messages(session, model, q)


@router.get("/chat-history/{model:path}/{chat_id}")
async def get_chat_history(model: ModelDep, chat_id: int, session: SessionDep) -> list[dict[str, str | int]]:
    chat = await history.get_chat_history(session, model, chat_id)
    if chat is None:
        return []
    return await history.serialize_active_path_for_display(session, chat)


@router.patch("/chat-history/{model:path}/{chat_id}")
async def switch_branch(
    model: ModelDep, chat_id: int, body: BranchSwitch, session: SessionDep
) -> list[dict[str, str | int]]:
    """Switches the active branch at a given message position — no
    generation involved, so this is a plain endpoint rather than another
    WebSocket message type."""
    chat = await _chat_or_404(session, model, chat_id)

    new_path = await history.switch_branch(session, chat, body.index, body.sibling_index)
    if new_path is None:
        raise HTTPException(
            status_code=400, detail="That branch no longer exists — the conversation may have changed."
        )
    return new_path


@router.get("/all-chats/{model:path}")
async def list_chats(model: ModelDep, session: SessionDep) -> list[ChatListItem]:
    return await service.list_chats(session, model)


@router.post("/all-chats/{model:path}", status_code=201)
async def create_chat(model: ModelDep, session: SessionDep, body: ChatCreate | None = None) -> ChatCreatedOut:
    chat = await service.create_chat(session, model, body or ChatCreate())
    return ChatCreatedOut.model_validate(chat)


@router.delete("/all-chats/{model:path}")
async def delete_chat(model: ModelDep, body: ChatRemove, session: SessionDep) -> dict[str, str]:
    chat = await _chat_or_404(session, model, body.chat_id)

    await session.execute(delete(models.ChatHistory).where(models.ChatHistory.id == chat.id))
    await session.commit()
    return {"status": "Chat deleted"}


@router.put("/all-chats/{model:path}")
async def update_chat(model: ModelDep, body: ChatUpdate, session: SessionDep) -> dict[str, str]:
    if not UPDATABLE_CHAT_FIELDS & body.model_fields_set:
        raise HTTPException(
            status_code=400,
            detail="Nothing to update — provide title, persona_id, collection_ids, "
            "tools_enabled, and/or a generation parameter (temperature, num_ctx, top_p, seed)",
        )

    chat = await _chat_or_404(session, model, body.id)
    await service.update_chat_settings(session, chat, body)
    return {"status": "Chat updated"}


@router.post("/chat-continue/{model:path}", status_code=201)
async def continue_chat(model: ModelDep, body: ChatContinue, session: SessionDep) -> ChatRef:
    source = await _chat_or_404(session, model, body.chat_id)
    continued = await service.continue_chat(session, source)
    return ChatRef(id=continued.id, title=continued.title)


@router.post("/chat-from-compare/{model:path}", status_code=201)
async def chat_from_compare(model: ModelDep, body: ChatFromCompare, session: SessionDep) -> ChatRef:
    chat = await service.chat_from_compare(session, model, body.parameters, body.prompt, body.response)
    return ChatRef(id=chat.id, title=chat.title)


@router.get("/chat-context/{model:path}/{chat_id}")
async def chat_context(
    model: ModelDep, chat_id: int, session: SessionDep, parameters: str = ""
) -> dict[str, typing.Any]:
    chat = await _chat_or_404(session, model, chat_id)
    return await service.context_report(session, model, chat, parameters)
