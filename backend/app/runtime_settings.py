"""App-wide settings the user changes at runtime (stored in AppSetting),
as opposed to core/config.py, which is fixed at startup."""

from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.db.models import AppSetting
from app.db.session import SessionLocal

EXTERNAL_OLLAMA_URL = "external_ollama_url"


async def get(key: str, default: Any = None) -> Any:
    async with SessionLocal() as session:
        row = (await session.execute(select(AppSetting).where(AppSetting.key == key))).scalar_one_or_none()
    return row.value if row is not None else default


async def put(key: str, value: Any) -> None:
    statement = insert(AppSetting).values(key=key, value=value)
    statement = statement.on_conflict_do_update(
        index_elements=[AppSetting.key], set_={"value": statement.excluded.value}
    )
    async with SessionLocal() as session:
        await session.execute(statement)
        # Committed here, not left to the caller: the next request has to see it.
        await session.commit()


async def external_ollama_url() -> str | None:
    """Set when the user runs their own Ollama instead of letting the app
    manage one container per model — every model then lives there."""
    value = await get(EXTERNAL_OLLAMA_URL)
    return value.rstrip("/") if isinstance(value, str) and value else None


async def set_external_ollama_url(url: str | None) -> None:
    await put(EXTERNAL_OLLAMA_URL, url or "")
