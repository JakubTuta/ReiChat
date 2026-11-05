from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends
from sqlalchemy import URL
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import get_settings


def database_url() -> URL:
    settings = get_settings()
    # Credentials are the ones docker-compose.yaml gives the postgres service —
    # both sides have to agree, so changing one means changing the other. The
    # database only ever listens on 127.0.0.1, so they are not a secret.
    #
    # The local address is an IP literal, not "localhost" — Windows resolves
    # "localhost" to ::1 first and Docker only publishes on IPv4, so a hostname
    # hangs for a long OS-level timeout before falling back.
    return URL.create(
        "postgresql+asyncpg",
        username="admin",
        password="password",
        host="postgres" if settings.docker else "127.0.0.1",
        port=5432,
        database=settings.db_name,
    )


def create_engine() -> AsyncEngine:
    if get_settings().db_null_pool:
        return create_async_engine(database_url(), poolclass=NullPool)
    # Check a pooled connection is still alive before using it, and retire it
    # after ten minutes: Postgres or Docker may have dropped it while idle.
    return create_async_engine(database_url(), pool_pre_ping=True, pool_recycle=600)


engine = create_engine()

# expire_on_commit=False: after a commit the objects stay readable. With the
# default, touching one afterwards would reload it implicitly, which async
# sessions can't do.
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    # Never commits: a service commits on its success path, so a request that
    # ends in a handled 4xx can't persist half a change.
    async with SessionLocal() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]
