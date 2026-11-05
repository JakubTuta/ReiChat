from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.core.config import ALLOWED_HOSTS, CORS_ALLOWED_ORIGINS, get_settings
from app.core.errors import UnhandledErrorMiddleware, register_exception_handlers
from app.core.rate_limit import RateLimitMiddleware
from app.core.security import OriginGuardMiddleware, SecurityHeadersMiddleware
from app.db.session import engine


async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    await engine.dispose()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="ReiChat",
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.debug else None,
        # First in the list is outermost. CORS must wrap everything that can
        # produce a response (including errors) or those responses silently
        # lose their CORS headers.
        middleware=[
            Middleware(TrustedHostMiddleware, allowed_hosts=list(ALLOWED_HOSTS)),
            Middleware(
                CORSMiddleware,
                allow_origins=list(CORS_ALLOWED_ORIGINS),
                allow_methods=["DELETE", "GET", "OPTIONS", "PATCH", "POST", "PUT"],
                allow_headers=[
                    "accept",
                    "authorization",
                    "content-type",
                    "user-agent",
                    "x-csrftoken",
                    "x-requested-with",
                ],
                max_age=86400,
            ),
            Middleware(SecurityHeadersMiddleware),
            Middleware(
                OriginGuardMiddleware,
                allowed_origins=CORS_ALLOWED_ORIGINS,
                allowed_hosts=ALLOWED_HOSTS,
            ),
            Middleware(UnhandledErrorMiddleware),
            # /v1/ is the OpenAI-compatible API other tools drive in loops, and
            # /health/ is polled by the container healthcheck: neither is the
            # runaway-page case this guards against.
            Middleware(
                RateLimitMiddleware,
                limit=120,
                window_seconds=60,
                exempt_paths=["/health/"],
                exempt_prefixes=["/v1/"],
            ),
        ],
    )
    register_exception_handlers(app)
    app.add_api_route("/health/", health_check, methods=["GET"])
    return app
