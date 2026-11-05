import asyncio
import dataclasses
import datetime
import json
import logging
import typing

from fastapi import APIRouter, HTTPException, Response, UploadFile
from sqlalchemy.exc import StatementError

from app.backup import service
from app.chat import history
from app.db.session import SessionDep

logger = logging.getLogger(__name__)

# The uploads take files, not JSON, so they are not behind the JSON-only guard
# the rest of the API sits behind.
router = APIRouter()

MAX_BACKUP_BYTES = 500 * 1024 * 1024


async def _read_json_upload(file: UploadFile) -> typing.Any:
    if file.size and file.size > MAX_BACKUP_BYTES:
        raise HTTPException(status_code=400, detail="That file is too large to import.")

    raw = await file.read()
    try:
        # Parsing a file this large would stall every other request.
        return await asyncio.to_thread(json.loads, raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(
            status_code=400, detail="That file couldn't be read — it isn't valid JSON."
        ) from None


@router.get("/backup/")
async def download_backup(session: SessionDep, documents: str = "") -> Response:
    data = await service.export_data(session, include_documents=documents == "true")
    payload = await asyncio.to_thread(json.dumps, data)
    filename = f"reichat-backup-{datetime.datetime.now(datetime.UTC):%Y-%m-%d}.json"

    return Response(
        payload,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/backup/")
async def restore_backup(file: UploadFile, session: SessionDep) -> dict[str, typing.Any]:
    data = await _read_json_upload(file)

    try:
        report = await service.import_data(session, data)
    except service.BackupFormatError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except (KeyError, TypeError, ValueError, StatementError) as e:
        # StatementError (and the database errors it parents): a value that can't be stored.
        logger.warning("Backup import failed on malformed data: %s", e)
        raise HTTPException(
            status_code=400, detail="Part of that backup file is damaged, so nothing was imported."
        ) from e

    return dataclasses.asdict(report)


@router.post("/chat-import/{model:path}", status_code=201)
async def import_chat(model: str, file: UploadFile, session: SessionDep) -> dict[str, typing.Any]:
    ai_model = await history.get_ai_model(session, model)
    if ai_model is None:
        raise HTTPException(status_code=404, detail="AI model not found")

    data = await _read_json_upload(file)

    try:
        chat, chat_model = await service.import_single_chat(session, data, ai_model)
    except service.BackupFormatError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return {"id": chat.id, "title": chat.title, "model": chat_model.model}
