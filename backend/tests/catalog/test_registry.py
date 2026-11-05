"""registry.ollama.ai returns 200 with a manifest for a pullable tag, 404 for
one that doesn't exist, and has no `tags/list` endpoint — verify_tag must
turn each of those into the right VerifyResult without ever raising.
"""

import asyncio

import httpx
import pytest
import respx

from app.catalog import registry


class _FakeResponse:
    def __init__(self, status_code: int, json_body: dict | None = None):
        self.status_code = status_code
        self._json_body = json_body or {}

    def json(self):
        return self._json_body


def _client_returning(response: _FakeResponse):
    class _Client:
        async def get(self, url, headers=None, **kwargs):
            return response

    return _Client()


async def test_ok_manifest_sums_config_and_layer_sizes():
    manifest = {
        "config": {"size": 500},
        "layers": [{"size": 1000}, {"size": 2500}],
    }
    result = await registry.verify_tag(_client_returning(_FakeResponse(200, manifest)), "llama3.1", "8b")

    assert result.status == "ok"
    assert result.size_bytes == 4000


async def test_not_found_tag():
    result = await registry.verify_tag(_client_returning(_FakeResponse(404)), "llama3.1", "bogus-tag")

    assert result.status == "not_found"
    assert result.size_bytes is None


async def test_unexpected_status_is_error_not_a_crash():
    result = await registry.verify_tag(_client_returning(_FakeResponse(500)), "llama3.1", "8b")

    assert result.status == "error"


async def test_network_failure_is_error_not_a_crash():
    class _RaisingClient:
        async def get(self, url, headers=None, **kwargs):
            raise httpx.ConnectError("connection refused")

    result = await registry.verify_tag(_RaisingClient(), "llama3.1", "8b")

    assert result.status == "error"


async def test_malformed_json_is_error_not_a_crash():
    class _BadJsonResponse:
        status_code = 200

        def json(self):
            raise ValueError("not json")

    result = await registry.verify_tag(_client_returning(_BadJsonResponse()), "llama3.1", "8b")

    assert result.status == "error"


@pytest.mark.parametrize(
    ("size_bytes", "expected"),
    [
        (None, "unknown"),
        (0, "unknown"),
        (512, "512 B"),
        (2048, "2.0 KB"),
        (4_700_000_000, "4.4 GB"),
    ],
)
def test_format_size(size_bytes, expected):
    assert registry.format_size(size_bytes) == expected


MANIFEST_URL = "https://registry.ollama.ai/v2/library/{name}/manifests/{tag}"


@respx.mock
async def test_verify_tags_reports_every_pair_and_calls_back_for_each():
    respx.get(MANIFEST_URL.format(name="llama3.1", tag="8b")).respond(200, json={"layers": [{"size": 10}]})
    respx.get(MANIFEST_URL.format(name="llama3.1", tag="bogus")).respond(404)
    seen = []

    results = await registry.verify_tags(
        [("llama3.1", "8b"), ("llama3.1", "bogus")],
        on_result=lambda name, tag, result: seen.append((name, tag)),
    )

    assert results[("llama3.1", "8b")] == registry.VerifyResult(status="ok", size_bytes=10)
    assert results[("llama3.1", "bogus")].status == "not_found"
    assert sorted(seen) == [("llama3.1", "8b"), ("llama3.1", "bogus")]


@respx.mock
async def test_verify_tags_never_runs_more_requests_at_once_than_asked():
    in_flight = 0
    peak = 0

    async def slow_manifest(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return httpx.Response(200, json={})

    respx.get(url__startswith="https://registry.ollama.ai/").mock(side_effect=slow_manifest)

    results = await registry.verify_tags([("m", str(i)) for i in range(12)], max_concurrent=3)

    assert len(results) == 12
    assert peak == 3
