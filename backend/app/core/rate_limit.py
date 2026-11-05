import math
import time
from collections.abc import Callable, Iterable

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class RateLimitMiddleware:
    """One fixed-window counter shared by every throttled request.

    Every request is anonymous (no auth exists), so a single bucket is the whole
    throttle — a floor against a malicious page firing blind requests (no CORS
    read access needed to send them) or a runaway client loop, not a real
    per-user limit.

    Exemptions are explicit paths: a limiter that finds the route handler to
    decide this (slowapi) silently stops limiting anything FastAPI mounts through
    include_router, because those routes aren't in app.routes as handlers.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        limit: int,
        window_seconds: float,
        exempt_paths: Iterable[str] = (),
        exempt_prefixes: Iterable[str] = (),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.app = app
        self.limit = limit
        self.window_seconds = window_seconds
        self.exempt_paths = frozenset(exempt_paths)
        self.exempt_prefixes = tuple(exempt_prefixes)
        self.clock = clock
        self._window_start = clock()
        self._count = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or self._is_exempt(scope["path"]):
            await self.app(scope, receive, send)
            return

        now = self.clock()
        if now - self._window_start >= self.window_seconds:
            self._window_start = now
            self._count = 0
        self._count += 1

        if self._count > self.limit:
            retry_after = max(1, math.ceil(self._window_start + self.window_seconds - now))
            response = JSONResponse(
                {"error": "Too many requests. Wait a minute and try again."},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)

    def _is_exempt(self, path: str) -> bool:
        return path in self.exempt_paths or path.startswith(self.exempt_prefixes)
