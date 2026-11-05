from unittest.mock import AsyncMock

import httpx
import pytest
import respx

from app.tools import web_search

SEARX = "http://searx"


@pytest.fixture
def searx_running(monkeypatch):
    monkeypatch.setattr(web_search.searxng, "base_url", AsyncMock(return_value=SEARX))


@respx.mock
async def test_web_search_formats_top_results(searx_running):
    payload = {
        "results": [{"title": f"T{i}", "url": f"https://e.com/{i}", "content": "snippet"} for i in range(8)]
    }
    route = respx.get(f"{SEARX}/search").respond(200, json=payload)

    result = await web_search.web_search.ainvoke({"query": "ollama"})

    assert result.count("https://e.com/") == web_search.MAX_RESULTS
    assert result.startswith("T0\nhttps://e.com/0\nsnippet")
    assert dict(route.calls.last.request.url.params) == {"q": "ollama", "format": "json"}


@respx.mock
async def test_web_search_says_so_when_nothing_was_found(searx_running):
    respx.get(f"{SEARX}/search").respond(200, json={"results": []})

    assert await web_search.web_search.ainvoke({"query": "zzzz"}) == "No web results for 'zzzz'."


async def test_web_search_reports_when_turned_off(monkeypatch):
    monkeypatch.setattr(web_search.searxng, "base_url", AsyncMock(return_value=None))

    result = await web_search.web_search.ainvoke({"query": "x"})

    assert result.startswith("Error: web search is turned off")


@respx.mock
async def test_web_search_reports_network_failures(searx_running):
    respx.get(f"{SEARX}/search").mock(side_effect=httpx.ConnectError("down"))

    assert "failed" in await web_search.web_search.ainvoke({"query": "x"})


@pytest.mark.parametrize(("status", "expected"), [("running", True), ("stopped", False)])
async def test_the_tool_is_only_offered_while_web_search_runs(monkeypatch, status, expected):
    monkeypatch.setattr(web_search.searxng, "status", AsyncMock(return_value=status))

    assert await web_search.available() is expected
