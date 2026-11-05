from functools import lru_cache
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings

# This app is single-user, unauthenticated and bound to 127.0.0.1 (see
# docker-compose.yaml). There is no deployment mode: nothing here is meant to
# be reachable from another machine, so hosts and origins are constants rather
# than knobs. The only things that genuinely differ between running in compose
# and running from source are the Settings fields below.

# Not "*": the Origin check on WebSocket handshakes (core/security.py) reuses
# this list, so a permissive value would accept a handshake from *any* web page
# the user has open and let it drive their local chat sockets. Binding ports to
# 127.0.0.1 (docker-compose.yaml) stops other machines on the network, but it
# does nothing against a malicious page running in the user's own browser,
# which can always reach localhost — this is the actual defense for that case,
# and TrustedHostMiddleware uses it to stop DNS rebinding.
ALLOWED_HOSTS = ("localhost", "127.0.0.1")

# The frontend, on the port compose publishes for it, under both names the
# user may type. A browser treats them as different origins: missing one here
# makes every response to that tab unreadable (no Access-Control-Allow-Origin),
# so the app reports the backend as unreachable while the server logs 200s.
CORS_ALLOWED_ORIGINS = ("http://localhost:3000", "http://127.0.0.1:3000")


class Settings(BaseSettings):
    # Renders the OpenAPI docs, so it stays off unless you deliberately ask for
    # it while running from source.
    debug: bool = False

    @field_validator("debug", mode="before")
    @classmethod
    def _only_true_enables(cls, value: Any) -> bool:
        # DEBUG is a shared environment name (Node's `debug` package uses
        # DEBUG=*). A stray value must neither crash startup nor switch this on.
        return str(value).lower() == "true"


@lru_cache
def get_settings() -> Settings:
    return Settings()
