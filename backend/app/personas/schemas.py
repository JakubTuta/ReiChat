import datetime

from pydantic import BaseModel, ConfigDict, model_validator

from app.core.types import NonBlankStr, reject_explicit_nulls


class PersonaOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    system_prompt: str
    created_at: datetime.datetime


class PersonaCreate(BaseModel):
    name: NonBlankStr
    system_prompt: NonBlankStr


class PersonaUpdate(BaseModel):
    id: int
    name: NonBlankStr | None = None
    system_prompt: NonBlankStr | None = None

    @model_validator(mode="after")
    def _only_sent_fields_may_change(self) -> "PersonaUpdate":
        return reject_explicit_nulls(self)


class PersonaRemove(BaseModel):
    id: int
