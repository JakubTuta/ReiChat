import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import models
from tests.factories import make_chat

pytestmark = pytest.mark.usefixtures("db")


async def add_persona(db: AsyncSession, name: str, system_prompt: str = "x") -> models.Persona:
    persona = models.Persona(name=name, system_prompt=system_prompt)
    db.add(persona)
    await db.commit()
    return persona


async def stored(db: AsyncSession, persona: models.Persona) -> models.Persona:
    return (
        await db.execute(
            select(models.Persona)
            .where(models.Persona.id == persona.id)
            .execution_options(populate_existing=True)
        )
    ).scalar_one()


async def test_get_personas_lists_alphabetically(client: httpx.AsyncClient, db: AsyncSession):
    await add_persona(db, "Zeta")
    await add_persona(db, "Alpha")

    response = await client.get("/personas/")

    assert response.status_code == 200
    assert [p["name"] for p in response.json()] == ["Alpha", "Zeta"]


async def test_a_persona_is_listed_with_exactly_the_fields_the_ui_reads(
    client: httpx.AsyncClient, db: AsyncSession
):
    persona = await add_persona(db, "Pirate", "Arr.")

    [listed] = (await client.get("/personas/")).json()

    assert set(listed) == {"id", "name", "system_prompt", "created_at"}
    assert listed["id"] == persona.id
    assert listed["created_at"].endswith("Z")


async def test_no_personas_is_an_empty_list(client: httpx.AsyncClient):
    assert (await client.get("/personas/")).json() == []


async def test_post_persona_creates_it(client: httpx.AsyncClient, db: AsyncSession):
    response = await client.post(
        "/personas/", json={"name": "Pirate", "system_prompt": "Talk like a pirate."}
    )

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Pirate"
    assert body["system_prompt"] == "Talk like a pirate."
    assert (await db.execute(select(models.Persona).where(models.Persona.name == "Pirate"))).scalar_one()


async def test_a_persona_is_trimmed_like_any_text_field(client: httpx.AsyncClient):
    response = await client.post("/personas/", json={"name": "  Pirate  ", "system_prompt": "\nArr.\n"})

    assert response.json()["name"] == "Pirate"
    assert response.json()["system_prompt"] == "Arr."


async def test_post_persona_missing_fields_returns_400(client: httpx.AsyncClient):
    response = await client.post("/personas/", json={"name": "Pirate"})

    assert response.status_code == 400
    assert response.json() == {
        "error": "Missing required body parameters",
        "missing_params": ["system_prompt"],
    }


@pytest.mark.parametrize("body", [{"name": "", "system_prompt": "x"}, {"name": "x", "system_prompt": "   "}])
async def test_post_persona_blank_fields_return_400(client: httpx.AsyncClient, body):
    response = await client.post("/personas/", json=body)

    assert response.status_code == 400
    assert "not valid" in response.json()["error"]


async def test_put_persona_updates_it(client: httpx.AsyncClient, db: AsyncSession):
    persona = await add_persona(db, "Pirate", "Arr.")

    response = await client.put(
        "/personas/", json={"id": persona.id, "system_prompt": "Talk like a pirate, matey."}
    )

    assert response.status_code == 200
    assert response.json()["system_prompt"] == "Talk like a pirate, matey."
    refreshed = await stored(db, persona)
    assert (refreshed.name, refreshed.system_prompt) == ("Pirate", "Talk like a pirate, matey.")


async def test_put_persona_can_rename_it(client: httpx.AsyncClient, db: AsyncSession):
    persona = await add_persona(db, "Pirate", "Arr.")

    await client.put("/personas/", json={"id": persona.id, "name": "Captain"})

    refreshed = await stored(db, persona)
    assert (refreshed.name, refreshed.system_prompt) == ("Captain", "Arr.")


@pytest.mark.parametrize("body", [{"name": None}, {"name": "  "}, {"system_prompt": ""}])
async def test_put_persona_refuses_to_blank_a_field(client: httpx.AsyncClient, db: AsyncSession, body):
    persona = await add_persona(db, "Pirate", "Arr.")

    response = await client.put("/personas/", json={"id": persona.id, **body})

    assert response.status_code == 400
    assert (await stored(db, persona)).name == "Pirate"


async def test_put_persona_not_found_returns_404(client: httpx.AsyncClient):
    response = await client.put("/personas/", json={"id": 999999, "name": "x"})

    assert response.status_code == 404
    assert response.json() == {"error": "Persona not found"}


async def test_put_persona_needs_an_id(client: httpx.AsyncClient):
    response = await client.put("/personas/", json={"name": "x"})

    assert response.status_code == 400
    assert response.json()["missing_params"] == ["id"]


async def test_put_persona_with_a_non_numeric_id_is_a_400_not_a_500(client: httpx.AsyncClient):
    response = await client.put("/personas/", json={"id": "abc", "name": "x"})

    assert response.status_code == 400


async def test_delete_persona_removes_it_and_unsets_it_on_chats(client: httpx.AsyncClient, db: AsyncSession):
    persona = await add_persona(db, "Pirate", "Arr.")
    chat = await make_chat(db, persona_id=persona.id)

    response = await client.request("DELETE", "/personas/", json={"id": persona.id})

    assert response.status_code == 200
    assert response.json() == {"status": "Persona deleted"}
    assert (await db.execute(select(func.count()).select_from(models.Persona))).scalar_one() == 0
    await db.refresh(chat)
    assert chat.persona_id is None
    assert (await db.execute(select(func.count()).select_from(models.ChatHistory))).scalar_one() == 1


async def test_delete_persona_not_found_returns_404(client: httpx.AsyncClient):
    response = await client.request("DELETE", "/personas/", json={"id": 999999})

    assert response.status_code == 404
    assert response.json() == {"error": "Persona not found"}
