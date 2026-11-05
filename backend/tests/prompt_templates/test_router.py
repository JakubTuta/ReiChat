import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import models

pytestmark = pytest.mark.usefixtures("db")


async def add_template(
    db: AsyncSession, name: str, content: str = "x", description: str = ""
) -> models.PromptTemplate:
    template = models.PromptTemplate(name=name, content=content, description=description)
    db.add(template)
    await db.commit()
    return template


async def stored(db: AsyncSession, template: models.PromptTemplate) -> models.PromptTemplate:
    return (
        await db.execute(
            select(models.PromptTemplate)
            .where(models.PromptTemplate.id == template.id)
            .execution_options(populate_existing=True)
        )
    ).scalar_one()


async def test_get_prompt_templates_lists_alphabetically(client: httpx.AsyncClient, db: AsyncSession):
    await add_template(db, "Zeta", "z")
    await add_template(db, "Alpha", "a")

    response = await client.get("/prompt-templates/")

    assert response.status_code == 200
    assert [t["name"] for t in response.json()] == ["Alpha", "Zeta"]


async def test_a_template_is_listed_with_exactly_the_fields_the_ui_reads(
    client: httpx.AsyncClient, db: AsyncSession
):
    await add_template(db, "Bug report", "Bug: {{summary}}", "A starter")

    [listed] = (await client.get("/prompt-templates/")).json()

    assert set(listed) == {"id", "name", "description", "content", "created_at"}
    assert (listed["description"], listed["content"]) == ("A starter", "Bug: {{summary}}")


async def test_post_prompt_template_creates_it(client: httpx.AsyncClient, db: AsyncSession):
    response = await client.post(
        "/prompt-templates/",
        json={
            "name": "Bug report",
            "description": "Structured bug report starter",
            "content": "Bug: {{summary}}\nSteps to reproduce: {{steps}}",
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Bug report"
    assert body["content"] == "Bug: {{summary}}\nSteps to reproduce: {{steps}}"
    assert (
        await db.execute(select(models.PromptTemplate).where(models.PromptTemplate.name == "Bug report"))
    ).scalar_one()


async def test_post_prompt_template_defaults_description_to_empty(client: httpx.AsyncClient):
    response = await client.post("/prompt-templates/", json={"name": "Plain", "content": "Just some text."})

    assert response.status_code == 201
    assert response.json()["description"] == ""


async def test_post_prompt_template_missing_fields_returns_400(client: httpx.AsyncClient):
    response = await client.post("/prompt-templates/", json={"name": "Incomplete"})

    assert response.status_code == 400
    assert response.json() == {"error": "Missing required body parameters", "missing_params": ["content"]}


async def test_post_prompt_template_blank_required_fields_return_400(client: httpx.AsyncClient):
    body = {"name": " ", "content": "x"}

    assert (await client.post("/prompt-templates/", json=body)).status_code == 400


async def test_put_prompt_template_updates_it(client: httpx.AsyncClient, db: AsyncSession):
    template = await add_template(db, "Draft", "v1")

    response = await client.put("/prompt-templates/", json={"id": template.id, "content": "v2"})

    assert response.status_code == 200
    refreshed = await stored(db, template)
    assert (refreshed.name, refreshed.content) == ("Draft", "v2")


@pytest.mark.parametrize("body", [{"name": None}, {"description": None}])
async def test_put_prompt_template_refuses_to_blank_a_required_field(
    client: httpx.AsyncClient, db: AsyncSession, body
):
    template = await add_template(db, "Draft", "v1")

    response = await client.put("/prompt-templates/", json={"id": template.id, **body})

    assert response.status_code == 400
    assert (await stored(db, template)).content == "v1"


async def test_delete_prompt_template_removes_it(client: httpx.AsyncClient, db: AsyncSession):
    template = await add_template(db, "Draft", "v1")

    response = await client.request("DELETE", "/prompt-templates/", json={"id": template.id})

    assert response.status_code == 200
    assert response.json() == {"status": "Prompt template deleted"}
    assert (await db.execute(select(func.count()).select_from(models.PromptTemplate))).scalar_one() == 0
