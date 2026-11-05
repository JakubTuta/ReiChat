import re

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import delete, select
from sqlalchemy.orm import selectinload

from app.catalog import library, refresh
from app.catalog.schemas import AIModelOut, CustomModelIn, CustomModelRemove
from app.core.security import require_json_body
from app.db.models import AIModel
from app.db.session import SessionDep

router = APIRouter(dependencies=[Depends(require_json_body)])

# Any Ollama model reference: a library name ("llama3.2"), a namespaced
# one ("user/model"), or a Hugging Face GGUF ("hf.co/user/repo"), plus an
# optional ":tag" (a size or a quantization like Q4_K_M).
_MODEL_REFERENCE_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*(/[A-Za-z0-9._-]+)*(:[A-Za-z0-9._-]+)?$")
_MAX_REFERENCE_LENGTH = 200


@router.get("/ai-models/")
async def list_models(session: SessionDep) -> list[AIModelOut]:
    models = (
        await session.execute(
            select(AIModel)
            .options(selectinload(AIModel.versions))
            .order_by(AIModel.popularity.desc(), AIModel.id)
        )
    ).scalars()
    return [AIModelOut.model_validate(model) for model in models]


@router.put("/ai-models/", status_code=202)
async def refresh_models(minPullCount: str) -> dict[str, str]:
    try:
        min_pull_count = int(minPullCount) if minPullCount else refresh.DEFAULT_MIN_PULL_COUNT
    except ValueError:
        raise HTTPException(status_code=400, detail="minPullCount must be a whole number") from None

    if not refresh.start_refresh(min_pull_count):
        raise HTTPException(status_code=409, detail="A catalog refresh is already running")

    return {"status": "Catalog refresh started"}


@router.post("/custom-models/", response_model=AIModelOut)
async def add_custom_model(body: CustomModelIn, response: Response, session: SessionDep) -> AIModel:
    reference = str(body.reference).strip()
    if reference.startswith(("https://", "http://")):
        reference = reference.split("://", 1)[1]

    if not _MODEL_REFERENCE_RE.match(reference) or len(reference) > _MAX_REFERENCE_LENGTH:
        raise HTTPException(
            status_code=400,
            detail="That doesn't look like a model name. Use something like llama3.2:3b or "
            "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M.",
        )

    model_name, _, tag = reference.partition(":")
    tag = tag or "latest"

    model_id, created = await library.get_or_create_model(
        session,
        model=model_name,
        name=model_name.rsplit("/", 1)[-1],
        description="Added by you by name.",
        is_custom=True,
    )
    await library.get_or_create_version(session, ai_model_id=model_id, parameters=tag, size="unknown")
    await session.commit()

    response.status_code = 201 if created else 200
    return (
        await session.execute(
            select(AIModel).where(AIModel.id == model_id).options(selectinload(AIModel.versions))
        )
    ).scalar_one()


@router.delete("/custom-models/")
async def remove_custom_model(body: CustomModelRemove, session: SessionDep) -> dict[str, str]:
    is_custom = (
        await session.execute(select(AIModel.is_custom).where(AIModel.id == body.id))
    ).scalar_one_or_none()
    if not is_custom:
        raise HTTPException(status_code=404, detail="Custom model not found")

    # AIModel cascades into ChatHistory — never take a user's chats with it.
    usage = await library.lock_and_check_usage(session, body.id)
    if usage is None:
        raise HTTPException(status_code=404, detail="Custom model not found")
    if usage is library.ModelUsage.HAS_CHATS:
        raise HTTPException(
            status_code=409,
            detail="This model still has chats. Delete them or move them to another model first.",
        )
    if usage is library.ModelUsage.USED_BY_COLLECTION:
        raise HTTPException(
            status_code=409,
            detail="This model is still used by a document collection. Delete that collection first.",
        )

    await session.execute(delete(AIModel).where(AIModel.id == body.id))
    await session.commit()
    return {"status": "Custom model removed"}
