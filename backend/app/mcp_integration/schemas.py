import datetime

from pydantic import BaseModel, ConfigDict, model_validator

from app.core.types import NonBlankStr, TrimmedStr, reject_explicit_nulls


class MCPServerOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    transport: str
    command: str
    url: str
    enabled: bool
    auto_approve: bool
    created_at: datetime.datetime


class MCPServerCreate(BaseModel):
    name: NonBlankStr
    # Checked in the route, which names exactly what to fix.
    transport: str
    command: TrimmedStr = ""
    url: TrimmedStr = ""
    enabled: bool = True
    auto_approve: bool = False


class MCPServerUpdate(BaseModel):
    id: int
    name: NonBlankStr | None = None
    transport: str | None = None
    command: TrimmedStr | None = None
    url: TrimmedStr | None = None
    enabled: bool | None = None
    auto_approve: bool | None = None

    @model_validator(mode="after")
    def _only_sent_fields_may_change(self) -> "MCPServerUpdate":
        return reject_explicit_nulls(self)


class MCPServerRef(BaseModel):
    id: int


class ToolOut(BaseModel):
    name: str
    description: str


class ToolListOut(BaseModel):
    tools: list[ToolOut]
