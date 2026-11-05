"""list_server_tools/call_server_tool never touch a real MCP server here —
_connect and ClientSession are both mocked. build_tool_registry is pure
logic and tested directly. Servers are never saved: nothing in the client
reads the database.
"""

import asyncio
import contextlib

import pytest

from app.db import models
from app.mcp_integration import client as mcp_client


@pytest.fixture
def stdio_server():
    return models.MCPServer(
        name="Local Tools", transport=models.MCPServer.TRANSPORT_STDIO, command="python server.py"
    )


@pytest.fixture
def http_server():
    return models.MCPServer(
        name="Remote Tools", transport=models.MCPServer.TRANSPORT_HTTP, url="http://localhost:9000/mcp"
    )


class _FakeTool:
    def __init__(self, name, description, input_schema):
        self.name = name
        self.description = description
        self.inputSchema = input_schema


class _FakeListToolsResult:
    def __init__(self, tools):
        self.tools = tools


class _FakeTextContent:
    def __init__(self, text):
        self.text = text


class _FakeCallToolResult:
    def __init__(self, text, is_error=False):
        self.content = [_FakeTextContent(text)]
        self.isError = is_error


class _FakeSession:
    def __init__(self, list_tools_result=None, call_tool_result=None):
        self._list_tools_result = list_tools_result
        self._call_tool_result = call_tool_result

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def initialize(self):
        return None

    async def list_tools(self):
        return self._list_tools_result

    async def call_tool(self, name, arguments):
        return self._call_tool_result


def _fake_connect_returning(session: _FakeSession):
    @contextlib.asynccontextmanager
    async def _connect(server):
        yield (None, None)

    return _connect


async def test_list_server_tools_returns_tool_info(stdio_server, monkeypatch):
    session = _FakeSession(
        list_tools_result=_FakeListToolsResult(
            [
                _FakeTool(
                    "search", "Search the web", {"type": "object", "properties": {"q": {"type": "string"}}}
                ),
            ]
        ),
    )
    monkeypatch.setattr(mcp_client, "_connect", _fake_connect_returning(session))
    monkeypatch.setattr(mcp_client, "ClientSession", lambda read, write: session)

    result = await mcp_client.list_server_tools(stdio_server)

    assert result == [
        {
            "name": "search",
            "description": "Search the web",
            "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    ]


async def test_list_server_tools_returns_empty_list_on_connection_failure(stdio_server, monkeypatch):
    @contextlib.asynccontextmanager
    async def _raise_connect(server):
        raise ConnectionError("no such server")
        yield  # pragma: no cover — unreachable, keeps this a generator

    monkeypatch.setattr(mcp_client, "_connect", _raise_connect)

    result = await mcp_client.list_server_tools(stdio_server)

    assert result == []


async def test_call_server_tool_returns_text_content(stdio_server, monkeypatch):
    session = _FakeSession(call_tool_result=_FakeCallToolResult("42"))
    monkeypatch.setattr(mcp_client, "_connect", _fake_connect_returning(session))
    monkeypatch.setattr(mcp_client, "ClientSession", lambda read, write: session)

    result = await mcp_client.call_server_tool(stdio_server, "search", {"q": "life the universe"})

    assert result == "42"


async def test_call_server_tool_prefixes_error_results(stdio_server, monkeypatch):
    session = _FakeSession(call_tool_result=_FakeCallToolResult("not found", is_error=True))
    monkeypatch.setattr(mcp_client, "_connect", _fake_connect_returning(session))
    monkeypatch.setattr(mcp_client, "ClientSession", lambda read, write: session)

    result = await mcp_client.call_server_tool(stdio_server, "search", {})

    assert result == "Error: not found"


def _http_transport_yielding_three_items(opened):
    # What the MCP SDK's streamable HTTP transport really yields: the two
    # streams plus a session-id getter. The stdio transport yields only two.
    @contextlib.asynccontextmanager
    async def streamable_http_client(url):
        opened.append(url)
        yield ("read-stream", "write-stream", lambda: "session-1")

    return streamable_http_client


async def test_http_server_tools_can_be_listed_although_the_transport_yields_three_items(
    http_server, monkeypatch
):
    opened = []
    session = _FakeSession(list_tools_result=_FakeListToolsResult([_FakeTool("ping", "Ping", {})]))
    streams_given_to_session = []

    def client_session(read, write):
        streams_given_to_session.append((read, write))
        return session

    monkeypatch.setattr(mcp_client, "streamable_http_client", _http_transport_yielding_three_items(opened))
    monkeypatch.setattr(mcp_client, "ClientSession", client_session)

    result = await mcp_client.probe_server_tools(http_server)

    assert [tool["name"] for tool in result] == ["ping"]
    assert opened == ["http://localhost:9000/mcp"]
    assert streams_given_to_session == [("read-stream", "write-stream")]


async def test_http_server_tools_can_be_called_although_the_transport_yields_three_items(
    http_server, monkeypatch
):
    session = _FakeSession(call_tool_result=_FakeCallToolResult("pong"))
    monkeypatch.setattr(mcp_client, "streamable_http_client", _http_transport_yielding_three_items([]))
    monkeypatch.setattr(mcp_client, "ClientSession", lambda read, write: session)

    assert await mcp_client.call_server_tool(http_server, "ping", {}) == "pong"


def test_connect_stdio_raises_when_command_missing(stdio_server):
    stdio_server.command = "   "

    with pytest.raises(mcp_client.MCPToolError):
        mcp_client._connect(stdio_server)


def test_connect_http_raises_when_url_missing(http_server):
    http_server.url = ""

    with pytest.raises(mcp_client.MCPToolError):
        mcp_client._connect(http_server)


def test_build_tool_registry_prefixes_names_and_builds_lookup(stdio_server, http_server):
    registry = mcp_client.build_tool_registry(
        [
            (stdio_server, [{"name": "search", "description": "Search", "input_schema": {"type": "object"}}]),
            (
                http_server,
                [{"name": "search", "description": "Different search", "input_schema": {"type": "object"}}],
            ),
        ]
    )

    names = [spec["function"]["name"] for spec in registry.tool_specs]
    assert len(names) == 2
    assert len(set(names)) == 2  # same tool name on two servers doesn't collide

    for spec in registry.tool_specs:
        assert spec["type"] == "function"
        server, original_name = registry.lookup[spec["function"]["name"]]
        assert original_name == "search"
        assert server in (stdio_server, http_server)


def test_build_tool_registry_defaults_missing_schema_to_empty_object(stdio_server):
    registry = mcp_client.build_tool_registry(
        [
            (stdio_server, [{"name": "ping", "description": "", "input_schema": {}}]),
        ]
    )

    assert registry.tool_specs[0]["function"]["parameters"] == {"type": "object", "properties": {}}


def test_build_tool_registry_empty_input_returns_empty_registry():
    registry = mcp_client.build_tool_registry([])

    assert registry.tool_specs == []
    assert registry.lookup == {}


async def test_probe_explains_a_command_missing_from_the_backend_image(stdio_server, monkeypatch):
    stdio_server.command = "npx -y @modelcontextprotocol/server-everything"

    @contextlib.asynccontextmanager
    async def _missing_binary(server):
        raise ExceptionGroup("task group failed", [FileNotFoundError(2, "No such file or directory")])
        yield  # pragma: no cover

    monkeypatch.setattr(mcp_client, "_connect", _missing_binary)

    with pytest.raises(mcp_client.MCPToolError, match="'npx' isn't available.*over HTTP"):
        await mcp_client.probe_server_tools(stdio_server)


async def test_probe_explains_a_refused_http_connection(http_server, monkeypatch):
    @contextlib.asynccontextmanager
    async def _refused(server):
        raise ConnectionRefusedError("connection refused")
        yield  # pragma: no cover

    monkeypatch.setattr(mcp_client, "_connect", _refused)

    with pytest.raises(mcp_client.MCPToolError, match="Couldn't connect to 'http://localhost:9000/mcp'"):
        await mcp_client.probe_server_tools(http_server)


async def test_list_server_tools_gives_up_on_a_hanging_server(stdio_server, monkeypatch):
    monkeypatch.setattr(mcp_client, "LIST_TOOLS_TIMEOUT_SECONDS", 0.05)

    async def _hang(server):
        await asyncio.sleep(10)

    monkeypatch.setattr(mcp_client, "_fetch_tools", _hang)

    assert await mcp_client.list_server_tools(stdio_server) == []
