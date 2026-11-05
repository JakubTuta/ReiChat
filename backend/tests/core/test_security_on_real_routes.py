"""The origin and body guards, exercised through real routes (the same
scenarios the Django middleware tests used) rather than test probes."""

import httpx
import pytest

pytestmark = pytest.mark.usefixtures("db")

PERSONA = {"name": "Pirate", "system_prompt": "Talk like a pirate."}


async def test_frontend_origin_is_allowed(client: httpx.AsyncClient):
    response = await client.post("/personas/", json=PERSONA, headers={"Origin": "http://localhost:3000"})

    assert response.status_code == 201


async def test_a_foreign_site_cannot_write_a_persona(client: httpx.AsyncClient):
    response = await client.post("/personas/", json=PERSONA, headers={"Origin": "https://evil.example"})

    assert response.status_code == 403
    assert (await client.get("/personas/")).json() == []
