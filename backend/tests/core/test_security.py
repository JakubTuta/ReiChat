import httpx
import pytest
from httpx_ws import WebSocketDisconnect

from tests.conftest import ConnectWs

FRONTEND = "http://localhost:3000"
FRONTEND_IP = "http://127.0.0.1:3000"
BODY = {"name": "Pirate", "count": 1}


async def test_cross_origin_form_post_is_rejected_before_it_reaches_the_route(client: httpx.AsyncClient):
    response = await client.post(
        "/probe/echo",
        content="name=evil&count=1",
        headers={"Content-Type": "application/x-www-form-urlencoded", "Origin": "https://evil.example"},
    )

    assert response.status_code == 403
    assert response.json() == {"error": "Requests from other websites are not allowed."}


async def test_null_origin_is_rejected(client: httpx.AsyncClient):
    response = await client.post("/probe/echo", json=BODY, headers={"Origin": "null"})

    assert response.status_code == 403


@pytest.mark.parametrize("origin", [FRONTEND, FRONTEND_IP])
async def test_frontend_origin_can_write_and_gets_cors_header_under_either_loopback_name(
    client: httpx.AsyncClient, origin: str
):
    response = await client.post("/probe/echo", json=BODY, headers={"Origin": origin})

    assert response.status_code == 201
    assert response.headers["Access-Control-Allow-Origin"] == origin


async def test_request_without_origin_is_allowed_for_non_browser_clients(client: httpx.AsyncClient):
    response = await client.post("/probe/echo", json=BODY)

    assert response.status_code == 201


async def test_same_origin_form_encoded_body_is_unsupported(client: httpx.AsyncClient):
    response = await client.post(
        "/probe/echo",
        content="name=x&count=1",
        headers={"Content-Type": "application/x-www-form-urlencoded", "Origin": FRONTEND},
    )

    assert response.status_code == 415
    assert "JSON" in response.json()["error"]


async def test_responses_carry_the_hardening_headers(client: httpx.AsyncClient):
    response = await client.get("/health/")

    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "same-origin"
    assert response.headers["Cross-Origin-Opener-Policy"] == "same-origin"


async def test_error_responses_carry_the_hardening_headers_too(client: httpx.AsyncClient):
    response = await client.get("/no-such-route/")

    assert response.status_code == 404
    assert response.headers["X-Content-Type-Options"] == "nosniff"


async def test_foreign_host_header_is_rejected_against_dns_rebinding(client: httpx.AsyncClient):
    response = await client.get("/health/", headers={"Host": "evil.example"})

    assert response.status_code == 400


async def test_loopback_host_headers_are_accepted(client: httpx.AsyncClient):
    host = "127.0.0.1:8000"

    response = await client.get("/health/", headers={"Host": host})

    assert response.status_code == 200


async def test_preflight_from_the_frontend_is_answered(client: httpx.AsyncClient):
    response = await client.options(
        "/probe/echo",
        headers={
            "Origin": FRONTEND,
            "Access-Control-Request-Method": "PATCH",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == FRONTEND
    assert "PATCH" in response.headers["Access-Control-Allow-Methods"]
    assert response.headers["Access-Control-Max-Age"] == "86400"


async def test_preflight_from_another_site_is_refused(client: httpx.AsyncClient):
    response = await client.options(
        "/probe/echo",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )

    assert response.status_code == 400
    assert "Access-Control-Allow-Origin" not in response.headers


async def test_websocket_from_a_loopback_origin_connects(connect_ws: ConnectWs):
    origin = FRONTEND

    async with connect_ws("/probe/ws", origin=origin) as ws:
        assert await ws.receive_text() == "hi"


@pytest.mark.parametrize("origin", [None, "https://evil.example"])
async def test_websocket_without_a_loopback_origin_is_refused(connect_ws: ConnectWs, origin: str | None):
    # The transport's task group wraps what the handshake raises in an ExceptionGroup.
    with pytest.RaisesGroup(WebSocketDisconnect):
        async with connect_ws("/probe/ws", origin=origin):
            pass
