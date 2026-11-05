import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.core.types import NonBlankStr


class CollectionOut(BaseModel):
    id: int
    name: str
    chat: int | None
    embedding_model: int
    embedding_model_name: str
    embedding_parameters: str
    document_count: int
    created_at: datetime.datetime


class CollectionCreate(BaseModel):
    name: NonBlankStr
    embedding_model: int
    embedding_parameters: NonBlankStr
    # Set when the collection is created from one chat's document panel.
    chat: int | None = None


class CollectionRemove(BaseModel):
    id: int


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    collection: int = Field(validation_alias="collection_id")
    filename: str
    status: str
    error_message: str
    chunk_count: int
    progress: int
    uploaded_at: datetime.datetime


class DocumentRemove(BaseModel):
    id: int


class ChunkOut(BaseModel):
    filename: str
    page: int | None
    content: str
