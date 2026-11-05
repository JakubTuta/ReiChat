"""The origin and body guards, exercised through real routes (the same
scenarios the Django middleware tests used) rather than test probes."""

import httpx
import pytest

pytestmark = pytest.mark.usefixtures("db")

PERSONA = {"name": "Pirate", "system_prompt": "Talk like a pirate."}


async def test_null_origin_is_rejected(client: httpx.AsyncClient):
    response = await client.post(
        "/personas/", json={"name": "x", "system_prompt": "y"}, headers={"Origin": "null"}
    )

    assert response.status_code == 403


async def test_frontend_origin_is_allowed(client: httpx.AsyncClient):
    response = await client.post("/personas/", json=PERSONA, headers={"Origin": "http://localhost:3000"})

    assert response.status_code == 201


@pytest.mark.parametrize("origin", ["http://localhost:3000", "http://127.0.0.1:3000"])
async def test_frontend_origin_gets_cors_header_under_either_loopback_name(
    client: httpx.AsyncClient, origin: str
):
    response = await client.get("/personas/", headers={"Origin": origin})

    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == origin


async def test_frontend_on_ip_loopback_can_write(client: httpx.AsyncClient):
    response = await client.post("/personas/", json=PERSONA, headers={"Origin": "http://127.0.0.1:3000"})

    assert response.status_code == 201


async def test_request_without_origin_is_allowed_for_non_browser_clients(client: httpx.AsyncClient):
    response = await client.post("/personas/", json=PERSONA)

    assert response.status_code == 201


async def test_safe_methods_ignore_origin(client: httpx.AsyncClient):
    response = await client.get("/personas/", headers={"Origin": "https://evil.example"})

    assert response.status_code == 200


async def test_a_foreign_site_cannot_write_a_persona(client: httpx.AsyncClient):
    response = await client.post("/personas/", json=PERSONA, headers={"Origin": "https://evil.example"})

    assert response.status_code == 403
    assert (await client.get("/personas/")).json() == []
