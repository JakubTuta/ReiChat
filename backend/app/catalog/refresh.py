"""Runs a catalog refresh in the background and reports its progress over the
container socket. The refresh itself is catalog/sync.py."""

import logging
from typing import Any

from app.catalog import sync
from app.core import tasks
from app.realtime import progress

logger = logging.getLogger(__name__)

DEFAULT_MIN_PULL_COUNT = 200_000


def _announce(
    *,
    running: bool,
    total: int | None = None,
    completed: int = 0,
    current: str | None = None,
    error: str | None = None,
    force: bool = False,
) -> None:
    payload: dict[str, Any] = {
        "type": "scrape_progress",
        "running": running,
        "total": total,
        "completed": completed,
        "current": current,
        "error": error,
    }
    progress.broadcast("scrape_progress", payload, key="scrape", force=force)


def start_refresh(min_pull_count: int) -> bool:
    """False if a refresh is already running."""
    if not progress.try_start_scrape():
        return False

    logger.info("Starting catalog refresh (minPullCount=%d)", min_pull_count)
    _announce(running=True, force=True)
    tasks.spawn(_refresh_worker(min_pull_count), name="catalog-refresh")
    return True


async def _refresh_worker(min_pull_count: int) -> None:
    def on_progress(completed: int, total: int, title: str) -> None:
        progress.set_scrape(total=total, completed=completed, current=title)
        _announce(running=True, total=total, completed=completed, current=title)

    try:
        result = await sync.refresh_catalog(min_pull_count, progress_cb=on_progress)

        if result.error:
            logger.error("Catalog refresh failed: %s", result.error)
            progress.set_scrape(running=False, error=result.error)
            _announce(running=False, error=result.error, force=True)
            return

        logger.info(
            "Catalog refresh completed: %d created, %d updated, %d pruned, %d variants verified",
            result.created,
            result.updated,
            result.pruned,
            result.variants_verified,
        )
        progress.set_scrape(running=False, error=None)
        _announce(running=False, force=True)
    except Exception as e:
        logger.exception("Catalog refresh crashed")
        progress.set_scrape(running=False, error=str(e))
        _announce(running=False, error=str(e), force=True)
