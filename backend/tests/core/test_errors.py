import logging

import httpx
import pytest
from starlette.requests import ClientDisconnect
from starlette.types import Message, Receive, Scope, Send

from app.core.errors import UnhandledErrorMiddleware

FRONTEND = "http://localhost:3000"


async def test_missing_body_fields_are_listed(client: httpx.AsyncClient):
    response = await client.post("/probe/echo", json={})

    assert response.status_code == 400
    assert response.json() == {
        "error": "Missing required body parameters",
        "missing_params": ["name", "count"],
    }


async def test_a_request_with_no_body_names_the_fields_it_needed(client: httpx.AsyncClient):
    response = await client.post("/probe/echo")

    assert response.status_code == 400
    assert response.json() == {
        "error": "Missing required body parameters",
        "missing_params": ["name", "count"],
    }


async def test_missing_query_parameter_is_listed(client: httpx.AsyncClient):
    response = await client.get("/probe/query")

    assert response.status_code == 400
    assert response.json() == {
        "error": "Missing required query parameters",
        "missing_params": ["limit"],
    }


async def test_invalid_value_gets_a_plain_message_and_details(client: httpx.AsyncClient):
    response = await client.post("/probe/echo", json={"name": "x", "count": "many"})

    body = response.json()
    assert response.status_code == 400
    assert body["error"].startswith("Some values in the request are not valid.")
    assert "count" in body["error"]
    assert body["details"][0]["field"] == "count"
    assert "integer" in body["details"][0]["message"]


async def test_non_numeric_path_id_is_a_400_not_a_500(client: httpx.AsyncClient):
    response = await client.get("/probe/items/abc")

    assert response.status_code == 400
    assert response.json()["details"][0]["field"] == "item_id"


async def test_malformed_json_is_reported_as_such(client: httpx.AsyncClient):
    response = await client.post(
        "/probe/echo", content="{not json", headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 400
    assert response.json() == {"error": "The request body is not valid JSON."}


async def test_http_exception_becomes_an_error_object(client: httpx.AsyncClient):
    response = await client.get("/probe/conflict")

    assert response.status_code == 409
    assert response.json() == {"error": "Already exists."}


async def test_http_exception_can_carry_extra_keys(client: httpx.AsyncClient):
    response = await client.get("/probe/conflict-extra")

    assert response.status_code == 409
    assert response.json() == {"error": "Still in use.", "used_by": ["a", "b"]}


async def test_unknown_route_is_json(client: httpx.AsyncClient):
    response = await client.get("/nope/")

    assert response.status_code == 404
    assert response.json() == {"error": "Not Found"}


async def test_wrong_method_is_json_and_keeps_the_allow_header(client: httpx.AsyncClient):
    response = await client.delete("/health/")

    assert response.status_code == 405
    assert response.json() == {"error": "Method Not Allowed"}
    assert "GET" in response.headers["Allow"]


async def test_unhandled_error_is_a_json_500_with_cors_headers(
    client: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
):
    with caplog.at_level(logging.ERROR, logger="app"):
        response = await client.get("/probe/boom", headers={"Origin": FRONTEND})

    assert response.status_code == 500
    assert "server logs" in response.json()["error"]
    # Without this the browser hides the 500 and the UI reports "backend unreachable".
    assert response.headers["Access-Control-Allow-Origin"] == FRONTEND
    assert any("kaboom" in str(record.exc_info) for record in caplog.records)


async def test_error_after_the_response_started_is_not_swallowed(client: httpx.AsyncClient):
    with pytest.raises(RuntimeError, match="died mid-stream"):
        await client.get("/probe/boom-mid-stream")


async def test_a_client_that_hangs_up_mid_request_is_not_reported_as_an_error(
    caplog: pytest.LogCaptureFixture,
):
    async def hangs_up(scope: Scope, receive: Receive, send: Send) -> None:
        raise ClientDisconnect

    async def receive() -> Message:
        return {"type": "http.request"}

    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    with caplog.at_level(logging.ERROR, logger="app"):
        await UnhandledErrorMiddleware(hangs_up)(
            {"type": "http", "method": "POST", "path": "/x"}, receive, send
        )

    assert sent == []
    assert caplog.records == []
