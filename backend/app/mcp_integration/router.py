from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import require_json_body
from app.db import models
from app.db.session import SessionDep
from app.mcp_integration import client as mcp_client
from app.mcp_integration.schemas import (
    MCPServerCreate,
    MCPServerOut,
    MCPServerRef,
    MCPServerUpdate,
    ToolListOut,
    ToolOut,
)

router = APIRouter(prefix="/mcp-servers", dependencies=[Depends(require_json_body)])

NAME_TAKEN = "A tool server with that name already exists."


def _validate_transport_config(transport: str, command: str, url: str) -> str | None:
    if transport not in (models.MCPServer.TRANSPORT_STDIO, models.MCPServer.TRANSPORT_HTTP):
        return "transport must be 'stdio' or 'http'"

    if transport == models.MCPServer.TRANSPORT_STDIO and not command.strip():
        return "command is required for a stdio server"

    if transport == models.MCPServer.TRANSPORT_HTTP and not url.strip():
        return "url is required for an http server"

    return None


async def _require_free_name(session: AsyncSession, name: str, except_id: int | None = None) -> None:
    statement = select(models.MCPServer.id).where(models.MCPServer.name == name)
    if except_id is not None:
        statement = statement.where(models.MCPServer.id != except_id)
    if (await session.execute(statement)).first() is not None:
        raise HTTPException(status_code=400, detail=NAME_TAKEN)


async def _server_or_404(session: AsyncSession, server_id: int) -> models.MCPServer:
    server = (
        await session.execute(select(models.MCPServer).where(models.MCPServer.id == server_id))
    ).scalar_one_or_none()
    if server is None:
        raise HTTPException(status_code=404, detail="MCP server not found")
    return server


@router.get("/")
async def list_servers(session: SessionDep) -> list[MCPServerOut]:
    servers = (
        await session.execute(select(models.MCPServer).order_by(models.MCPServer.name, models.MCPServer.id))
    ).scalars()
    return [MCPServerOut.model_validate(server) for server in servers]


@router.post("/", status_code=201)
async def create_server(body: MCPServerCreate, session: SessionDep) -> MCPServerOut:
    if error := _validate_transport_config(body.transport, body.command, body.url):
        raise HTTPException(status_code=400, detail=error)
    await _require_free_name(session, body.name)

    server = models.MCPServer(
        name=body.name,
        transport=body.transport,
        command=body.command,
        url=body.url,
        enabled=body.enabled,
        auto_approve=body.auto_approve,
    )
    session.add(server)
    try:
        await session.commit()
    except IntegrityError:
        # Two requests with the same name raced past the check above.
        raise HTTPException(status_code=400, detail=NAME_TAKEN) from None
    return MCPServerOut.model_validate(server)


@router.put("/")
async def update_server(body: MCPServerUpdate, session: SessionDep) -> MCPServerOut:
    server = await _server_or_404(session, body.id)

    sent = body.model_fields_set
    error = _validate_transport_config(
        body.transport if "transport" in sent and body.transport is not None else server.transport,
        body.command if "command" in sent and body.command is not None else server.command,
        body.url if "url" in sent and body.url is not None else server.url,
    )
    if error:
        raise HTTPException(status_code=400, detail=error)

    if "name" in sent and body.name is not None:
        await _require_free_name(session, body.name, except_id=server.id)

    for field in sent - {"id"}:
        setattr(server, field, getattr(body, field))
    try:
        await session.commit()
    except IntegrityError:
        raise HTTPException(status_code=400, detail=NAME_TAKEN) from None
    return MCPServerOut.model_validate(server)


@router.delete("/")
async def delete_server(body: MCPServerRef, session: SessionDep) -> dict[str, str]:
    deleted = (
        await session.execute(
            delete(models.MCPServer).where(models.MCPServer.id == body.id).returning(models.MCPServer.id)
        )
    ).first()
    if deleted is None:
        raise HTTPException(status_code=404, detail="MCP server not found")

    await session.commit()
    return {"status": "MCP server deleted"}


@router.post("/test/")
async def test_server(body: MCPServerRef, session: SessionDep) -> ToolListOut:
    server = await _server_or_404(session, body.id)

    try:
        tools = await mcp_client.probe_server_tools(server)
    except mcp_client.MCPToolError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    return ToolListOut(tools=[ToolOut(name=t["name"], description=t["description"]) for t in tools])
