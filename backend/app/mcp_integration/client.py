"""Talks to external Model Context Protocol servers the user has
configured (models.MCPServer). Each call here opens a fresh connection —
a stdio server means spawning a subprocess, an HTTP server means opening a
streamable-HTTP connection — and closes it again once done. That trades
some latency (a subprocess launch per tool call) for not having to manage
long-lived MCP sessions across unrelated WebSocket requests, which would
need its own lifecycle manager for comparatively little benefit at the
scale this app runs at.
"""

import asyncio
import contextlib
import logging
import re
import shlex
import typing

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from app.db import models

logger = logging.getLogger(__name__)


class MCPToolError(Exception):
    pass


class MCPToolInfo(typing.TypedDict):
    name: str
    description: str
    input_schema: dict[str, typing.Any]


def _connect(server: models.MCPServer) -> contextlib.AbstractAsyncContextManager[tuple[typing.Any, ...]]:
    if server.transport == models.MCPServer.TRANSPORT_STDIO:
        parts = shlex.split(server.command)
        if not parts:
            raise MCPToolError(f"MCP server {server.name!r} has no command configured.")

        return stdio_client(StdioServerParameters(command=parts[0], args=parts[1:]))

    if not server.url:
        raise MCPToolError(f"MCP server {server.name!r} has no URL configured.")

    return streamable_http_client(server.url)


# A server that hangs on connect (a stdio command waiting on input, an HTTP
# URL that never answers) must not stall every tools-enabled chat turn.
LIST_TOOLS_TIMEOUT_SECONDS = 15


@contextlib.asynccontextmanager
async def _session(server: models.MCPServer) -> typing.AsyncIterator[ClientSession]:
    async with _connect(server) as streams:
        # Streamable HTTP yields a third item (a session-id getter) that stdio
        # doesn't; only the read and write streams are needed here.
        read, write = streams[0], streams[1]
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def _fetch_tools(server: models.MCPServer) -> list[MCPToolInfo]:
    async with _session(server) as session:
        result = await session.list_tools()

        return [
            {"name": t.name, "description": t.description or "", "input_schema": t.inputSchema or {}}
            for t in result.tools
        ]


def _leaf_exceptions(error: BaseException) -> list[BaseException]:
    # The MCP client runs on anyio task groups, which wrap the real cause
    # (a missing binary, a refused connection) in nested ExceptionGroups.
    if isinstance(error, BaseExceptionGroup):
        return [leaf for inner in error.exceptions for leaf in _leaf_exceptions(inner)]
    return [error]


def describe_connection_error(server: models.MCPServer, error: BaseException) -> str:
    leaves = _leaf_exceptions(error)

    for leaf in leaves:
        if isinstance(leaf, FileNotFoundError):
            command = shlex.split(server.command)[0] if server.command.strip() else server.command
            return (
                f"The command {command!r} isn't available inside the ReiChat server. Local commands run in "
                "the backend container, which only has Python — tools like npx or uvx aren't installed there. "
                "Run the MCP server yourself and connect it over HTTP instead."
            )
        if isinstance(leaf, TimeoutError):
            return f"{server.name!r} didn't respond within {LIST_TOOLS_TIMEOUT_SECONDS} seconds."
        if isinstance(leaf, (ConnectionError, OSError)) or type(leaf).__name__ in (
            "ConnectError",
            "ConnectTimeout",
        ):
            target = server.url or server.command
            return f"Couldn't connect to {target!r}: {leaf}. Check that the server is running."

    return f"Couldn't talk to {server.name!r}: {leaves[0]}"


async def probe_server_tools(server: models.MCPServer) -> list[MCPToolInfo]:
    """Like list_server_tools, but raises MCPToolError with a plain-language
    reason — for the Integrations page's "Test" button, where the user needs
    to see *why* a server contributes no tools."""
    try:
        return await asyncio.wait_for(_fetch_tools(server), LIST_TOOLS_TIMEOUT_SECONDS)
    except MCPToolError:
        raise
    except Exception as e:
        raise MCPToolError(describe_connection_error(server, e)) from e


async def list_server_tools(server: models.MCPServer) -> list[MCPToolInfo]:
    """Best-effort — a server that's down or misconfigured returns an empty
    list (logged) rather than raising, so one broken integration doesn't
    stop a chat response. The Integrations page's Test button
    (probe_server_tools) is where the user finds out why.
    """
    try:
        return await probe_server_tools(server)
    except MCPToolError as e:
        logger.warning("Could not list tools from MCP server %r: %s", server.name, e)
        return []


async def call_server_tool(server: models.MCPServer, tool_name: str, arguments: dict[str, typing.Any]) -> str:
    async with _session(server) as session:
        result = await session.call_tool(tool_name, arguments)
        text_parts = [block.text for block in result.content if hasattr(block, "text")]
        output = "\n".join(text_parts) if text_parts else str(result.content)

        return f"Error: {output}" if result.isError else output


_NAME_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_-]")


class ToolRegistry(typing.NamedTuple):
    # OpenAI-format tool specs — what ChatOllama.bind_tools() expects
    # alongside the @tool-decorated builtins (see llm/generation.py).
    tool_specs: list[dict[str, typing.Any]]
    # Prefixed name -> (server, original tool name). Built once so a tool
    # call never has to parse the prefix back apart — the prefix only
    # exists to keep identically-named tools on different servers from
    # colliding in bind_tools()'s flat tool list.
    lookup: dict[str, tuple[models.MCPServer, str]]


def build_tool_registry(servers_with_tools: list[tuple[models.MCPServer, list[MCPToolInfo]]]) -> ToolRegistry:
    tool_specs: list[dict[str, typing.Any]] = []
    lookup: dict[str, tuple[models.MCPServer, str]] = {}

    for server, tools in servers_with_tools:
        server_slug = _NAME_SANITIZE_RE.sub("_", server.name)

        for tool_info in tools:
            tool_slug = _NAME_SANITIZE_RE.sub("_", tool_info["name"])
            prefixed_name = f"mcp_{server_slug}_{tool_slug}"

            tool_specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": prefixed_name,
                        "description": tool_info["description"],
                        "parameters": tool_info["input_schema"] or {"type": "object", "properties": {}},
                    },
                }
            )
            lookup[prefixed_name] = (server, tool_info["name"])

    return ToolRegistry(tool_specs=tool_specs, lookup=lookup)
