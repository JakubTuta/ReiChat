import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from app.core.rate_limit import RateLimitMiddleware

LIMIT = 120


async def test_requests_over_the_limit_get_a_429(client: httpx.AsyncClient):
    for _ in range(LIMIT):
        assert (await client.get("/probe/ok")).status_code == 200

    response = await client.get("/probe/ok")

    assert response.status_code == 429
    assert response.json() == {"error": "Too many requests. Wait a minute and try again."}
    assert 1 <= int(response.headers["Retry-After"]) <= 60


async def test_the_limit_is_shared_across_routes(client: httpx.AsyncClient):
    for _ in range(LIMIT):
        await client.get("/probe/ok")

    response = await client.get("/probe/query", params={"limit": 1})

    assert response.status_code == 429


async def test_throttled_responses_still_carry_cors_headers(client: httpx.AsyncClient):
    for _ in range(LIMIT):
        await client.get("/probe/ok")

    response = await client.get("/probe/ok", headers={"Origin": "http://localhost:3000"})

    assert response.status_code == 429
    assert response.headers["Access-Control-Allow-Origin"] == "http://localhost:3000"


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def build_limited_app(clock: FakeClock, limit: int = 3) -> httpx.AsyncClient:
    async def ping(request: Request) -> JSONResponse:
        return JSONResponse({"ok": True})

    app = Starlette(
        routes=[Route("/ping", ping), Route("/v1/ping", ping), Route("/free", ping)],
        middleware=[],
    )
    limited = RateLimitMiddleware(
        app,
        limit=limit,
        window_seconds=60,
        exempt_paths=["/free"],
        exempt_prefixes=["/v1/"],
        clock=clock,
    )
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=limited), base_url="http://localhost")


async def test_the_budget_comes_back_when_the_window_ends():
    clock = FakeClock()
    async with build_limited_app(clock) as client:
        assert [(await client.get("/ping")).status_code for _ in range(4)] == [200, 200, 200, 429]

        clock.now += 59
        assert (await client.get("/ping")).status_code == 429

        clock.now += 1
        assert (await client.get("/ping")).status_code == 200


async def test_exempt_paths_and_prefixes_are_not_counted():
    clock = FakeClock()
    async with build_limited_app(clock, limit=1) as client:
        for _ in range(5):
            assert (await client.get("/free")).status_code == 200
            assert (await client.get("/v1/ping")).status_code == 200

        assert (await client.get("/ping")).status_code == 200
