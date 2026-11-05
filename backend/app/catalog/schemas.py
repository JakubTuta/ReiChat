import datetime

from pydantic import BaseModel, ConfigDict


class AIModelVersionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    parameters: str
    size: str
    size_bytes: int | None
    verified_at: datetime.datetime | None


class AIModelOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    model: str
    description: str
    popularity: int
    can_process_image: bool
    is_embedding: bool
    can_use_tools: bool
    is_custom: bool
    versions: list[AIModelVersionOut]
    index: int


class CustomModelIn(BaseModel):
    # Anything a client sends is read as text, as it always was.
    reference: object


class CustomModelRemove(BaseModel):
    id: int
