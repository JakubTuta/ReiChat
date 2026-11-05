#!/bin/sh
set -e

alembic upgrade head

exec uvicorn app.asgi:app \
    --host 0.0.0.0 --port 8000 \
    --ws-max-size 8388608 \
    --timeout-graceful-shutdown 5 \
    --no-access-log
