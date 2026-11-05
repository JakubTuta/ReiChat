import os
import subprocess
import sys
from pathlib import Path

import asyncpg
from sqlalchemy import URL

from app.db.session import database_url

BACKEND_DIR = Path(__file__).resolve().parent.parent


def server_dsn(url: URL) -> str:
    return url.set(drivername="postgresql").render_as_string(hide_password=False)


async def recreate_database(name: str) -> None:
    """Drop and create `name` on the app's Postgres server."""
    maintenance_dsn = server_dsn(database_url().set(database="postgres"))
    try:
        connection = await asyncpg.connect(maintenance_dsn)
    except OSError as error:
        raise RuntimeError(
            "PostgreSQL isn't reachable on 127.0.0.1:5432, and the tests need it. "
            "Start it with `docker compose up -d postgres`."
        ) from error
    try:
        await connection.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        await connection.execute(f'CREATE DATABASE "{name}"')
    finally:
        await connection.close()


async def drop_database(name: str) -> None:
    connection = await asyncpg.connect(server_dsn(database_url().set(database="postgres")))
    try:
        await connection.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    finally:
        await connection.close()


def alembic(*args: str, database: str) -> subprocess.CompletedProcess[str]:
    """Run the alembic CLI against `database`, as the container entrypoint does."""
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=BACKEND_DIR,
        env={**os.environ, "DB_NAME": database},
        capture_output=True,
        text=True,
        check=False,
    )


def migrate(database: str) -> None:
    result = alembic("upgrade", "head", database=database)
    if result.returncode != 0:
        raise RuntimeError(f"alembic upgrade head failed:\n{result.stdout}\n{result.stderr}")
