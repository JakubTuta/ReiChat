import asyncio
import json
from collections.abc import Iterator

import asyncpg
import pytest

from app.db import models  # noqa: F401  (registers the tables on Base.metadata)
from app.db.base import Base
from app.db.session import database_url
from tests.database import BACKEND_DIR, alembic, drop_database, migrate, recreate_database, server_dsn

SCRATCH = "chatbot_migration_test"
CATALOG = json.loads((BACKEND_DIR / "migrations" / "data" / "model_catalog.json").read_text(encoding="utf-8"))


async def query(sql: str) -> list[asyncpg.Record]:
    connection = await asyncpg.connect(server_dsn(database_url().set(database=SCRATCH)))
    try:
        return await connection.fetch(sql)
    finally:
        await connection.close()


@pytest.fixture(scope="module")
def scratch_database() -> Iterator[None]:
    asyncio.run(recreate_database(SCRATCH))
    migrate(SCRATCH)
    yield
    asyncio.run(drop_database(SCRATCH))


@pytest.mark.usefixtures("scratch_database")
def test_every_table_the_models_declare_exists():
    rows = asyncio.run(query("SELECT tablename FROM pg_tables WHERE schemaname = 'public'"))

    assert {row["tablename"] for row in rows} - {"alembic_version"} == set(Base.metadata.tables)


@pytest.mark.usefixtures("scratch_database")
def test_models_and_migrations_describe_the_same_schema():
    result = alembic("check", database=SCRATCH)

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.usefixtures("scratch_database")
def test_the_bundled_catalog_is_seeded():
    models_rows = asyncio.run(
        query('SELECT model, "index", can_use_tools, is_custom FROM ai_model ORDER BY id')
    )
    version_count = asyncio.run(query("SELECT count(*) AS n FROM ai_model_version"))[0]["n"]

    assert len(models_rows) == len(CATALOG) == 40
    assert [row["model"] for row in models_rows] == [entry["model"] for entry in CATALOG]
    assert "llama3.1" in {row["model"] for row in models_rows}
    assert [row["index"] for row in models_rows] == list(range(1, len(CATALOG) + 1))
    assert not any(row["can_use_tools"] or row["is_custom"] for row in models_rows)
    assert version_count == sum(len(entry["versions"]) for entry in CATALOG)


@pytest.mark.usefixtures("scratch_database")
def test_rows_added_after_the_seed_get_fresh_ids():
    next_id = asyncio.run(
        query(
            "INSERT INTO ai_model (name, model, description, popularity, can_process_image, is_embedding, "
            "can_use_tools, \"index\", is_custom) VALUES ('x', 'x', 'x', 0, false, false, false, 0, false) "
            "RETURNING id"
        )
    )[0]["id"]

    assert next_id == len(CATALOG) + 1


def test_migrations_apply_in_both_directions():
    database = SCRATCH + "_roundtrip"
    asyncio.run(recreate_database(database))
    try:
        migrate(database)
        down = alembic("downgrade", "base", database=database)
        assert down.returncode == 0, down.stdout + down.stderr
        migrate(database)

        async def model_count() -> int:
            connection = await asyncpg.connect(server_dsn(database_url().set(database=database)))
            try:
                return await connection.fetchval("SELECT count(*) FROM ai_model")
            finally:
                await connection.close()

        assert asyncio.run(model_count()) == len(CATALOG)
    finally:
        asyncio.run(drop_database(database))
