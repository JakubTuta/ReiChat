FROM python:3.13-slim-trixie AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /bin/

# Bytecode is compiled once here instead of on every container start. Files are
# copied rather than hard-linked because the cache mount is another filesystem.
# No Python is downloaded: the venv has to point at the interpreter the final
# stage also has.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev


FROM python:3.13-slim-trixie

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DOCKER=true

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY . .

RUN chmod +x entrypoint.sh

EXPOSE 8000

# Runs as root on purpose: the mounted Docker socket is only usable by root (or
# the host's docker group, whose id differs per machine).
CMD ["./entrypoint.sh"]
