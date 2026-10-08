from __future__ import annotations

import requests

from django_app import ollama_api


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_context_length_matches_any_architecture_prefix():
    assert ollama_api.context_length({"model_info": {"qwen2.context_length": 32768}}) == 32768
    assert ollama_api.context_length({"model_info": {"general.architecture": "llama"}}) is None
    assert ollama_api.context_length({}) is None


def test_show_model_is_cached_per_server_and_tag(monkeypatch):
    calls = []

    def _post(url, json, timeout):
        calls.append(json["model"])
        return _FakeResponse({"model_info": {"llama.context_length": 8192}, "capabilities": ["completion"]})

    monkeypatch.setattr(ollama_api, "_show_cache", {})
    monkeypatch.setattr(requests, "post", _post)

    first = ollama_api.show_model("http://o", "llama3.2:1b")
    second = ollama_api.show_model("http://o", "llama3.2:1b")

    assert first is second
    assert calls == ["llama3.2:1b"]
    assert ollama_api.capabilities(first) == ["completion"]


def test_show_model_returns_none_when_unreachable(monkeypatch):
    def _refuse(*a, **k):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(ollama_api, "_show_cache", {})
    monkeypatch.setattr(requests, "post", _refuse)

    assert ollama_api.show_model("http://o", "llama3.2:1b") is None


def test_running_models_normalizes_ps_output(monkeypatch):
    payload = {"models": [{"name": "qwen2.5:0.5b", "size": 1000, "size_vram": 600, "context_length": 8192}]}
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(payload))

    assert ollama_api.running_models("http://o") == [
        {"name": "qwen2.5:0.5b", "size": 1000, "size_vram": 600, "context_length": 8192}
    ]
