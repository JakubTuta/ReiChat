import datetime

from pydantic import BaseModel, ConfigDict, model_validator

from app.core.types import NonBlankStr, TrimmedStr, reject_explicit_nulls


class PromptTemplateOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str
    content: str
    created_at: datetime.datetime


class PromptTemplateCreate(BaseModel):
    name: NonBlankStr
    # The frontend always sends this, possibly as "".
    description: TrimmedStr = ""
    content: NonBlankStr


class PromptTemplateUpdate(BaseModel):
    id: int
    name: NonBlankStr | None = None
    description: TrimmedStr | None = None
    content: NonBlankStr | None = None

    @model_validator(mode="after")
    def _only_sent_fields_may_change(self) -> "PromptTemplateUpdate":
        return reject_explicit_nulls(self)


class PromptTemplateRemove(BaseModel):
    id: int
