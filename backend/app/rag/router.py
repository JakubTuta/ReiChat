import asyncio
import logging
import posixpath
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from sqlalchemy import delete, func, select

from app.core.security import require_json_body
from app.db import models
from app.db.session import SessionDep
from app.rag import extract, ingest
from app.rag.schemas import (
    ChunkOut,
    CollectionCreate,
    CollectionOut,
    CollectionRemove,
    DocumentOut,
    DocumentRemove,
)

logger = logging.getLogger(__name__)

router = APIRouter()
json_only = [Depends(require_json_body)]

_MAX_FILENAME_LENGTH = 255


def _collection_out(
    collection: models.DocumentCollection, embedding_model_name: str, document_count: int
) -> CollectionOut:
    return CollectionOut(
        id=collection.id,
        name=collection.name,
        chat=collection.chat_id,
        embedding_model=collection.embedding_model_id,
        embedding_model_name=embedding_model_name,
        embedding_parameters=collection.embedding_parameters,
        document_count=document_count,
        created_at=collection.created_at,
    )


@router.get("/collections/", dependencies=json_only)
async def list_collections(session: SessionDep, chat_id: int | None = None) -> list[CollectionOut]:
    statement = (
        select(models.DocumentCollection, models.AIModel.model, func.count(models.Document.id))
        .join(models.AIModel, models.AIModel.id == models.DocumentCollection.embedding_model_id)
        .outerjoin(models.Document, models.Document.collection_id == models.DocumentCollection.id)
        .group_by(models.DocumentCollection.id, models.AIModel.model)
        .order_by(models.DocumentCollection.created_at.desc(), models.DocumentCollection.id.desc())
    )
    if chat_id:
        # A chat sees its own collections plus every global one — global
        # collections (no chat) are usable from any chat.
        statement = statement.where(
            (models.DocumentCollection.chat_id == chat_id) | models.DocumentCollection.chat_id.is_(None)
        )

    return [
        _collection_out(collection, name, count)
        for collection, name, count in await session.execute(statement)
    ]


@router.post("/collections/", status_code=201, dependencies=json_only)
async def create_collection(body: CollectionCreate, session: SessionDep) -> CollectionOut:
    embedding_model = (
        await session.execute(select(models.AIModel).where(models.AIModel.id == body.embedding_model))
    ).scalar_one_or_none()
    if embedding_model is None:
        raise HTTPException(status_code=404, detail="Embedding model not found")
    if not embedding_model.is_embedding:
        raise HTTPException(status_code=400, detail="That model isn't an embedding model")

    if body.chat is not None:
        chat = (
            await session.execute(select(models.ChatHistory.id).where(models.ChatHistory.id == body.chat))
        ).scalar_one_or_none()
        if chat is None:
            raise HTTPException(status_code=404, detail="Chat not found")

    collection = models.DocumentCollection(
        name=body.name,
        chat_id=body.chat,
        embedding_model_id=embedding_model.id,
        embedding_parameters=body.embedding_parameters,
    )
    session.add(collection)
    await session.commit()
    return _collection_out(collection, embedding_model.model, 0)


@router.delete("/collections/", dependencies=json_only)
async def delete_collection(body: CollectionRemove, session: SessionDep) -> dict[str, str]:
    # Cascades to its documents and their chunks; any chat that had it active
    # just loses that link (nothing else references it).
    deleted = (
        await session.execute(
            delete(models.DocumentCollection)
            .where(models.DocumentCollection.id == body.id)
            .returning(models.DocumentCollection.id)
        )
    ).first()
    if deleted is None:
        raise HTTPException(status_code=404, detail="Collection not found")

    await session.commit()
    return {"status": "Collection deleted"}


@router.get("/documents/", dependencies=json_only)
async def list_documents(collection_id: int, session: SessionDep) -> list[DocumentOut]:
    await ingest.mark_interrupted_documents(collection_id)

    documents = (
        await session.execute(
            select(models.Document)
            .where(models.Document.collection_id == collection_id)
            .order_by(models.Document.uploaded_at.desc(), models.Document.id.desc())
        )
    ).scalars()
    return [DocumentOut.model_validate(document) for document in documents]


def _safe_filename(name: str | None) -> str:
    """Just the file's own name, as Django's uploads reduced it: a client can
    send a full path, and the name is shown in the UI and read for its extension."""
    base = posixpath.basename((name or "").replace(chr(92), "/"))
    return base[:_MAX_FILENAME_LENGTH]


@router.post("/documents/", status_code=202)
async def upload_document(
    collection_id: Annotated[int, Form()], file: UploadFile, session: SessionDep
) -> DocumentOut:
    # The one route that takes a file, so the one that is not JSON-only.
    collection = (
        await session.execute(
            select(models.DocumentCollection.id).where(models.DocumentCollection.id == collection_id)
        )
    ).scalar_one_or_none()
    if collection is None:
        raise HTTPException(status_code=404, detail="Collection not found")

    filename = _safe_filename(file.filename)
    raw = await file.read()

    try:
        # Reading a PDF or Word file is CPU work.
        text, page_starts = await asyncio.to_thread(extract.extract_document, filename, raw)
    except extract.UnsupportedFileTypeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.warning("Could not read the uploaded file %r: %s", filename, e)
        raise HTTPException(
            status_code=400,
            detail="Couldn't read that file — it may be damaged, or protected with a password.",
        ) from e

    if not text.strip():
        raise HTTPException(status_code=400, detail="No extractable text found in this file.")

    document = models.Document(collection_id=collection_id, filename=filename)
    session.add(document)
    await session.commit()

    # Chunking + embedding involves real HTTP calls to the embedding model and
    # would block the request for a while on anything but a tiny file — same
    # background-task pattern as container creation and catalog refresh.
    ingest.start_ingest(document.id, text, page_starts)

    return DocumentOut.model_validate(document)


@router.delete("/documents/", dependencies=json_only)
async def delete_document(body: DocumentRemove, session: SessionDep) -> dict[str, str]:
    deleted = (
        await session.execute(
            delete(models.Document).where(models.Document.id == body.id).returning(models.Document.id)
        )
    ).first()
    if deleted is None:
        raise HTTPException(status_code=404, detail="Document not found")

    await session.commit()
    return {"status": "Document deleted"}


@router.get("/documents/chunk/", dependencies=json_only)
async def get_cited_passage(document_id: int, chunk_index: int, session: SessionDep) -> ChunkOut:
    """The passage behind a citation, so the user can check what the model was
    actually shown."""
    row = (
        await session.execute(
            select(models.Document.filename, models.DocumentChunk.page, models.DocumentChunk.content)
            .join(models.Document, models.Document.id == models.DocumentChunk.document_id)
            .where(
                models.DocumentChunk.document_id == document_id,
                models.DocumentChunk.chunk_index == chunk_index,
            )
            .limit(1)
        )
    ).one_or_none()
    if row is None:
        raise HTTPException(
            status_code=404, detail="That passage no longer exists — the file may have been removed."
        )

    return ChunkOut(filename=row.filename, page=row.page, content=row.content)
