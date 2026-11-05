from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete, select

from app.core.security import require_json_body
from app.db.models import PromptTemplate
from app.db.session import SessionDep
from app.prompt_templates.schemas import (
    PromptTemplateCreate,
    PromptTemplateOut,
    PromptTemplateRemove,
    PromptTemplateUpdate,
)

router = APIRouter(prefix="/prompt-templates", dependencies=[Depends(require_json_body)])


@router.get("/")
async def list_prompt_templates(session: SessionDep) -> list[PromptTemplateOut]:
    templates = (
        await session.execute(select(PromptTemplate).order_by(PromptTemplate.name, PromptTemplate.id))
    ).scalars()
    return [PromptTemplateOut.model_validate(template) for template in templates]


@router.post("/", status_code=201)
async def create_prompt_template(body: PromptTemplateCreate, session: SessionDep) -> PromptTemplateOut:
    template = PromptTemplate(name=body.name, description=body.description, content=body.content)
    session.add(template)
    await session.commit()
    return PromptTemplateOut.model_validate(template)


@router.put("/")
async def update_prompt_template(body: PromptTemplateUpdate, session: SessionDep) -> PromptTemplateOut:
    template = (
        await session.execute(select(PromptTemplate).where(PromptTemplate.id == body.id))
    ).scalar_one_or_none()
    if template is None:
        raise HTTPException(status_code=404, detail="Prompt template not found")

    for field in body.model_fields_set - {"id"}:
        setattr(template, field, getattr(body, field))
    await session.commit()
    return PromptTemplateOut.model_validate(template)


@router.delete("/")
async def delete_prompt_template(body: PromptTemplateRemove, session: SessionDep) -> dict[str, str]:
    deleted = (
        await session.execute(
            delete(PromptTemplate).where(PromptTemplate.id == body.id).returning(PromptTemplate.id)
        )
    ).first()
    if deleted is None:
        raise HTTPException(status_code=404, detail="Prompt template not found")

    await session.commit()
    return {"status": "Prompt template deleted"}
