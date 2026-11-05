from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete, select

from app.core.security import require_json_body
from app.db.models import Persona
from app.db.session import SessionDep
from app.personas.schemas import PersonaCreate, PersonaOut, PersonaRemove, PersonaUpdate

router = APIRouter(prefix="/personas", dependencies=[Depends(require_json_body)])


@router.get("/")
async def list_personas(session: SessionDep) -> list[PersonaOut]:
    personas = (await session.execute(select(Persona).order_by(Persona.name, Persona.id))).scalars()
    return [PersonaOut.model_validate(persona) for persona in personas]


@router.post("/", status_code=201)
async def create_persona(body: PersonaCreate, session: SessionDep) -> PersonaOut:
    persona = Persona(name=body.name, system_prompt=body.system_prompt)
    session.add(persona)
    await session.commit()
    return PersonaOut.model_validate(persona)


@router.put("/")
async def update_persona(body: PersonaUpdate, session: SessionDep) -> PersonaOut:
    persona = (await session.execute(select(Persona).where(Persona.id == body.id))).scalar_one_or_none()
    if persona is None:
        raise HTTPException(status_code=404, detail="Persona not found")

    for field in body.model_fields_set - {"id"}:
        setattr(persona, field, getattr(body, field))
    await session.commit()
    return PersonaOut.model_validate(persona)


@router.delete("/")
async def delete_persona(body: PersonaRemove, session: SessionDep) -> dict[str, str]:
    # Chats using this persona fall back to no persona (SET NULL on the
    # foreign key) rather than losing their history.
    deleted = (
        await session.execute(delete(Persona).where(Persona.id == body.id).returning(Persona.id))
    ).first()
    if deleted is None:
        raise HTTPException(status_code=404, detail="Persona not found")

    await session.commit()
    return {"status": "Persona deleted"}
