from collections.abc import Iterable
from urllib.parse import urlsplit

from fastapi import HTTPException, Request
from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})

_RESPONSE_HEADERS = (
    ("X-Content-Type-Options", "nosniff"),
    ("X-Frame-Options", "DENY"),
    ("Referrer-Policy", "same-origin"),
    ("Cross-Origin-Opener-Policy", "same-origin"),
)


class SecurityHeadersMiddleware:
    """Adds the browser-hardening headers Django's SecurityMiddleware and
    XFrameOptionsMiddleware used to add. A header a route set itself wins."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in _RESPONSE_HEADERS:
                    headers.setdefault(name, value)
            await send(message)

        await self.app(scope, receive, send_with_headers)


class OriginGuardMiddleware:
    """Rejects state-changing requests sent by a browser page on any other
    origin, and WebSocket handshakes from any other host. There is no auth, and
    CORS alone doesn't stop a write: a form-encoded or multipart POST is a
    "simple request" that the browser sends without a preflight — only the
    *response* is hidden from the page. Any website the user visits could
    otherwise blind-POST e.g. a stdio MCP server whose command then runs in the
    backend container, which holds the Docker socket (= host root).

    Browsers always attach `Origin` to cross-origin POST/PUT/PATCH/DELETE, so
    a missing header on an HTTP request means a non-browser client (curl, the
    OpenAI SDK against /v1/), which a web page can't impersonate. `Origin: null`
    (sandboxed iframes, file://) is rejected like any other foreign origin.

    WebSocket handshakes are not subject to CORS at all, so the Origin check is
    the only thing standing between any open web page and the chat sockets;
    there a missing Origin is refused too. Only the hostname is compared (any
    port, any scheme), which is the rule Channels'
    AllowedHostsOriginValidator applied.
    """

    def __init__(self, app: ASGIApp, allowed_origins: Iterable[str], allowed_hosts: Iterable[str]) -> None:
        self.app = app
        self.allowed_origins = frozenset(allowed_origins)
        self.allowed_hosts = frozenset(allowed_hosts)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            origin = Headers(scope=scope).get("origin")
            if (
                scope["method"] not in _SAFE_METHODS
                and origin is not None
                and origin not in self.allowed_origins
            ):
                response = JSONResponse(
                    {"error": "Requests from other websites are not allowed."}, status_code=403
                )
                await response(scope, receive, send)
                return
        elif scope["type"] == "websocket" and not self._websocket_origin_allowed(
            Headers(scope=scope).get("origin")
        ):
            # The connect event must be consumed before a close is accepted;
            # closing before accept makes the server answer the handshake 403.
            await receive()
            await send({"type": "websocket.close", "code": 1008})
            return

        await self.app(scope, receive, send)

    def _websocket_origin_allowed(self, origin: str | None) -> bool:
        if origin is None:
            return False
        try:
            hostname = urlsplit(origin).hostname
        except ValueError:
            return False
        return hostname in self.allowed_hosts


async def require_json_body(request: Request) -> None:
    """JSON only — form/multipart bodies are what a cross-origin page can send
    without a CORS preflight. The one upload route opts out by not using this."""
    if not await request.body():
        return
    media_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if media_type != "application/json" and not media_type.endswith("+json"):
        raise HTTPException(
            status_code=415, detail="Send the request body as JSON (Content-Type: application/json)."
        )
