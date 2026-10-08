from __future__ import annotations

import requests

from django_app.tools import web_search


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_web_search_formats_top_results(monkeypatch):
    monkeypatch.setattr(web_search.searxng, "base_url", lambda: "http://searx")
    payload = {"results": [{"title": f"T{i}", "url": f"https://e.com/{i}", "content": "snippet"} for i in range(8)]}
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(payload))

    result = web_search.web_search.invoke({"query": "ollama"})

    assert result.count("https://e.com/") == web_search.MAX_RESULTS
    assert result.startswith("T0\nhttps://e.com/0\nsnippet")


def test_web_search_reports_when_turned_off(monkeypatch):
    monkeypatch.setattr(web_search.searxng, "base_url", lambda: None)

    assert web_search.web_search.invoke({"query": "x"}).startswith("Error: web search is turned off")


def test_web_search_reports_network_failures(monkeypatch):
    monkeypatch.setattr(web_search.searxng, "base_url", lambda: "http://searx")

    def _fail(*a, **k):
        raise requests.exceptions.ConnectionError("down")

    monkeypatch.setattr(requests, "get", _fail)

    assert "failed" in web_search.web_search.invoke({"query": "x"})
