import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import models
from app.mcp_integration import client as mcp_client
from app.mcp_integration import router as mcp_router

pytestmark = pytest.mark.usefixtures("db")


async def add_server(db: AsyncSession, name: str, transport: str = "stdio", **fields) -> models.MCPServer:
    if transport == "stdio":
        fields.setdefault("command", "x")
    server = models.MCPServer(name=name, transport=transport, **fields)
    db.add(server)
    await db.commit()
    return server


async def stored(db: AsyncSession, server: models.MCPServer) -> models.MCPServer:
    return (
        await db.execute(
            select(models.MCPServer)
            .where(models.MCPServer.id == server.id)
            .execution_options(populate_existing=True)
        )
    ).scalar_one()


async def server_count(db: AsyncSession) -> int:
    return (await db.execute(select(func.count()).select_from(models.MCPServer))).scalar_one()


async def test_post_mcp_server_stdio_creates_it(client: httpx.AsyncClient, db: AsyncSession):
    response = await client.post(
        "/mcp-servers/", json={"name": "Local Tools", "transport": "stdio", "command": "python server.py"}
    )

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Local Tools"
    assert body["enabled"] is True
    assert body["auto_approve"] is False
    assert set(body) == {"id", "name", "transport", "command", "url", "enabled", "auto_approve", "created_at"}
    assert await server_count(db) == 1


async def test_post_mcp_server_http_creates_it(client: httpx.AsyncClient, db: AsyncSession):
    response = await client.post(
        "/mcp-servers/",
        json={"name": "Remote Tools", "transport": "http", "url": "http://localhost:9000/mcp"},
    )

    assert response.status_code == 201
    assert response.json()["url"] == "http://localhost:9000/mcp"
    assert response.json()["command"] == ""


async def test_the_settings_are_stored_trimmed_and_as_sent(client: httpx.AsyncClient):
    response = await client.post(
        "/mcp-servers/",
        json={
            "name": "  Trusted  ",
            "transport": "stdio",
            "command": "  python server.py  ",
            "enabled": False,
            "auto_approve": True,
        },
    )

    body = response.json()
    assert (body["name"], body["command"], body["enabled"], body["auto_approve"]) == (
        "Trusted",
        "python server.py",
        False,
        True,
    )


async def test_post_mcp_server_stdio_without_command_returns_400(client: httpx.AsyncClient, db: AsyncSession):
    response = await client.post("/mcp-servers/", json={"name": "Local Tools", "transport": "stdio"})

    assert response.status_code == 400
    assert response.json() == {"error": "command is required for a stdio server"}
    assert await server_count(db) == 0


async def test_a_blank_command_counts_as_no_command(client: httpx.AsyncClient):
    response = await client.post("/mcp-servers/", json={"name": "L", "transport": "stdio", "command": "   "})

    assert response.status_code == 400


async def test_post_mcp_server_http_without_url_returns_400(client: httpx.AsyncClient):
    response = await client.post("/mcp-servers/", json={"name": "Remote Tools", "transport": "http"})

    assert response.status_code == 400
    assert response.json() == {"error": "url is required for an http server"}


async def test_post_mcp_server_invalid_transport_returns_400(client: httpx.AsyncClient):
    response = await client.post("/mcp-servers/", json={"name": "Bad", "transport": "carrier-pigeon"})

    assert response.status_code == 400
    assert response.json() == {"error": "transport must be 'stdio' or 'http'"}


async def test_post_mcp_server_needs_a_name_and_transport(client: httpx.AsyncClient):
    response = await client.post("/mcp-servers/", json={})

    assert response.status_code == 400
    assert response.json() == {
        "error": "Missing required body parameters",
        "missing_params": ["name", "transport"],
    }


async def test_a_second_server_cannot_reuse_a_name(client: httpx.AsyncClient, db: AsyncSession):
    await add_server(db, "Local")

    response = await client.post(
        "/mcp-servers/", json={"name": "Local", "transport": "stdio", "command": "y"}
    )

    assert response.status_code == 400
    assert response.json() == {"error": "A tool server with that name already exists."}
    assert await server_count(db) == 1


async def test_get_mcp_servers_lists_alphabetically(client: httpx.AsyncClient, db: AsyncSession):
    await add_server(db, "Zeta")
    await add_server(db, "Alpha")

    response = await client.get("/mcp-servers/")

    assert response.status_code == 200
    assert [s["name"] for s in response.json()] == ["Alpha", "Zeta"]


async def test_put_mcp_server_toggles_enabled(client: httpx.AsyncClient, db: AsyncSession):
    server = await add_server(db, "Local", enabled=True)

    response = await client.put("/mcp-servers/", json={"id": server.id, "enabled": False})

    assert response.status_code == 200
    assert response.json()["enabled"] is False
    assert (await stored(db, server)).enabled is False


async def test_put_mcp_server_trusts_a_server_only_when_told_to(client: httpx.AsyncClient, db: AsyncSession):
    server = await add_server(db, "Local")

    await client.put("/mcp-servers/", json={"id": server.id, "auto_approve": True})

    refreshed = await stored(db, server)
    assert (refreshed.auto_approve, refreshed.enabled, refreshed.name) == (True, True, "Local")


async def test_put_mcp_server_can_rename_it_to_a_free_name(client: httpx.AsyncClient, db: AsyncSession):
    server = await add_server(db, "Local")

    assert (await client.put("/mcp-servers/", json={"id": server.id, "name": "Renamed"})).status_code == 200
    assert (await stored(db, server)).name == "Renamed"


async def test_put_mcp_server_keeping_its_own_name_is_fine(client: httpx.AsyncClient, db: AsyncSession):
    server = await add_server(db, "Local")

    assert (
        await client.put("/mcp-servers/", json={"id": server.id, "name": "Local", "enabled": False})
    ).status_code == 200


async def test_put_mcp_server_cannot_take_another_servers_name(client: httpx.AsyncClient, db: AsyncSession):
    await add_server(db, "Taken")
    server = await add_server(db, "Local")

    response = await client.put("/mcp-servers/", json={"id": server.id, "name": "Taken"})

    assert response.status_code == 400
    assert (await stored(db, server)).name == "Local"


async def test_put_mcp_server_not_found_returns_404(client: httpx.AsyncClient):
    response = await client.put("/mcp-servers/", json={"id": 999999, "enabled": False})

    assert response.status_code == 404
    assert response.json() == {"error": "MCP server not found"}


async def test_put_mcp_server_switching_to_http_without_url_returns_400(
    client: httpx.AsyncClient, db: AsyncSession
):
    server = await add_server(db, "Local")

    response = await client.put("/mcp-servers/", json={"id": server.id, "transport": "http"})

    assert response.status_code == 400
    assert response.json() == {"error": "url is required for an http server"}
    assert (await stored(db, server)).transport == "stdio"


async def test_put_mcp_server_switching_to_http_with_a_url_works(client: httpx.AsyncClient, db: AsyncSession):
    server = await add_server(db, "Local")

    response = await client.put(
        "/mcp-servers/", json={"id": server.id, "transport": "http", "url": "http://x/mcp"}
    )

    assert response.status_code == 200
    assert (await stored(db, server)).transport == "http"


async def test_put_mcp_server_clearing_the_command_of_a_stdio_server_is_refused(
    client: httpx.AsyncClient, db: AsyncSession
):
    server = await add_server(db, "Local")

    response = await client.put("/mcp-servers/", json={"id": server.id, "command": ""})

    assert response.status_code == 400
    assert (await stored(db, server)).command == "x"


async def test_put_mcp_server_refuses_null_fields(client: httpx.AsyncClient, db: AsyncSession):
    server = await add_server(db, "Local")

    assert (await client.put("/mcp-servers/", json={"id": server.id, "enabled": None})).status_code == 400


async def test_delete_mcp_server_removes_it(client: httpx.AsyncClient, db: AsyncSession):
    server = await add_server(db, "Local")

    response = await client.request("DELETE", "/mcp-servers/", json={"id": server.id})

    assert response.status_code == 200
    assert response.json() == {"status": "MCP server deleted"}
    assert await server_count(db) == 0


async def test_delete_mcp_server_not_found_returns_404(client: httpx.AsyncClient):
    response = await client.request("DELETE", "/mcp-servers/", json={"id": 999999})

    assert response.status_code == 404
    assert response.json() == {"error": "MCP server not found"}


async def test_mcp_server_test_lists_tools(client: httpx.AsyncClient, db: AsyncSession, monkeypatch):
    server = await add_server(db, "Probe OK", "http", url="http://x/mcp")
    seen = []

    async def probe(srv):
        seen.append(srv.name)
        return [{"name": "search", "description": "Search", "input_schema": {}}]

    monkeypatch.setattr(mcp_router.mcp_client, "probe_server_tools", probe)

    response = await client.post("/mcp-servers/test/", json={"id": server.id})

    assert response.status_code == 200
    assert response.json() == {"tools": [{"name": "search", "description": "Search"}]}
    assert seen == ["Probe OK"]


async def test_mcp_server_test_reports_the_reason_it_failed(
    client: httpx.AsyncClient, db: AsyncSession, monkeypatch
):
    server = await add_server(db, "Probe bad", command="npx x")

    async def probe(srv):
        raise mcp_client.MCPToolError("The command 'npx' isn't available.")

    monkeypatch.setattr(mcp_router.mcp_client, "probe_server_tools", probe)

    response = await client.post("/mcp-servers/test/", json={"id": server.id})

    assert response.status_code == 502
    assert response.json() == {"error": "The command 'npx' isn't available."}


async def test_mcp_server_test_of_an_unknown_server_is_a_404(client: httpx.AsyncClient):
    response = await client.post("/mcp-servers/test/", json={"id": 999999})

    assert response.status_code == 404


async def test_a_form_encoded_body_is_unsupported(client: httpx.AsyncClient, db: AsyncSession):
    response = await client.post(
        "/mcp-servers/",
        content="name=x&transport=stdio&command=echo",
        headers={"Content-Type": "application/x-www-form-urlencoded", "Origin": "http://localhost:3000"},
    )

    assert response.status_code == 415
    assert await server_count(db) == 0


async def test_a_cross_origin_form_post_never_reaches_the_route(client: httpx.AsyncClient, db: AsyncSession):
    # A web page can send this without a CORS preflight; the stdio command in it
    # would run inside the backend container, which holds the Docker socket.
    response = await client.post(
        "/mcp-servers/",
        content="name=evil&transport=stdio&command=sh+-c+id",
        headers={"Content-Type": "application/x-www-form-urlencoded", "Origin": "https://evil.example"},
    )

    assert response.status_code == 403
    assert await server_count(db) == 0


async def test_a_cross_origin_json_post_is_refused_too(client: httpx.AsyncClient, db: AsyncSession):
    response = await client.post(
        "/mcp-servers/",
        json={"name": "evil", "transport": "stdio", "command": "sh -c id"},
        headers={"Origin": "https://evil.example"},
    )

    assert response.status_code == 403
    assert await server_count(db) == 0
