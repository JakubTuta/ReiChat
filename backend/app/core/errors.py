import logging
from collections.abc import Sequence
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import ClientDisconnect
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

_NO_BODY_STATUSES = frozenset({204, 304})
_MAX_MESSAGES_SHOWN = 3


def _field_name(loc: Sequence[int | str]) -> str:
    return ".".join(str(part) for part in loc[1:]) or str(loc[0])


def _required_body_fields(request: Request) -> list[str]:
    # With no body at all FastAPI reports one "body is missing" error, not one
    # per field; the caller still wants to know which fields were expected.
    body_field = getattr(request.scope.get("route"), "body_field", None)
    model = getattr(getattr(body_field, "field_info", None), "annotation", None)
    if isinstance(model, type) and issubclass(model, BaseModel):
        return [name for name, field in model.model_fields.items() if field.is_required()]
    return []


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> Response:
    if exc.status_code in _NO_BODY_STATUSES:
        return Response(status_code=exc.status_code, headers=exc.headers)
    detail: Any = exc.detail
    # A dict carrying its own "error" lets a route attach extra keys.
    body = detail if isinstance(detail, dict) and "error" in detail else {"error": str(detail)}
    return JSONResponse(body, status_code=exc.status_code, headers=exc.headers)


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = exc.errors()

    if any(error["type"] == "json_invalid" for error in errors):
        return JSONResponse({"error": "The request body is not valid JSON."}, status_code=400)

    missing = [error for error in errors if error["type"] == "missing"]
    if missing:
        where = str(missing[0]["loc"][0])
        if len(missing[0]["loc"]) == 1:
            names = _required_body_fields(request)
        else:
            names = [_field_name(error["loc"]) for error in missing]
        return JSONResponse(
            {"error": f"Missing required {where} parameters", "missing_params": names},
            status_code=400,
        )

    details: list[dict[str, Any]] = [
        {"field": _field_name(error["loc"]), "message": error["msg"]} for error in errors
    ]
    shown = "; ".join(f"{d['field']}: {d['message']}" for d in details[:_MAX_MESSAGES_SHOWN])
    return JSONResponse(
        {"error": f"Some values in the request are not valid. {shown}", "details": details},
        status_code=400,
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.exception_handler(StarletteHTTPException)(http_exception_handler)
    app.exception_handler(RequestValidationError)(validation_error_handler)


class UnhandledErrorMiddleware:
    """Turns an uncaught exception into a JSON 500.

    Starlette's own handler for a bare Exception lives in ServerErrorMiddleware,
    which sits *outside* every user middleware — including CORS. Its 500 would
    carry no Access-Control-Allow-Origin, so the browser hides it and the app
    misreports the backend as unreachable instead of showing a server error.
    Placed inside CORS, this one's response gets the headers.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = False

        async def tracking_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, receive, tracking_send)
        except ClientDisconnect:
            logger.debug("Client went away mid-request: %s %s", scope["method"], scope["path"])
        except Exception:
            if started:
                # Headers are already on the wire; all that is left is to let
                # the server cut the connection and log it.
                raise
            logger.exception("Unhandled error on %s %s", scope["method"], scope["path"])
            response = JSONResponse(
                {"error": "Something went wrong on the server. Check the server logs for details."},
                status_code=500,
            )
            await response(scope, receive, send)
