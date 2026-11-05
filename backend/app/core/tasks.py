import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger(__name__)

# The event loop only keeps a weak reference to a task, so a fire-and-forget
# task nobody else holds can be garbage-collected mid-run.
_running: set[asyncio.Task[Any]] = set()


def spawn[T](coroutine: Coroutine[Any, Any, T], name: str) -> asyncio.Task[T]:
    """Run `coroutine` in the background (model installs, catalog refreshes,
    document ingestion — work that outlives the request that started it).

    Failures are logged with a traceback rather than vanishing: nothing awaits
    these tasks, so an unlogged exception would be a silent no-op for the user.
    """
    task = asyncio.create_task(coroutine, name=name)
    _running.add(task)
    task.add_done_callback(_finished)
    return task


def _finished(task: asyncio.Task[Any]) -> None:
    _running.discard(task)
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        logger.error("Background task %r failed", task.get_name(), exc_info=error)


async def cancel_all() -> None:
    """Stop everything still running; called when the server shuts down."""
    tasks = list(_running)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
