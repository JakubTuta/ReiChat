import asyncio

import pytest

from app.realtime import progress
from app.realtime.hub import hub


@pytest.fixture
async def events() -> asyncio.Queue[dict]:
    hub.bind(asyncio.get_running_loop())
    return hub.subscribe()


async def received(queue: asyncio.Queue[dict]) -> list[dict]:
    await asyncio.sleep(0)
    out = []
    while not queue.empty():
        out.append(queue.get_nowait())
    return out


def test_only_one_install_per_name_can_be_claimed():
    assert progress.try_claim("llama3.1_8b", "Starting…") is True
    assert progress.try_claim("llama3.1_8b", "Starting…") is False
    assert progress.is_in_progress("llama3.1_8b")


def test_only_one_catalog_refresh_can_run_at_a_time():
    assert progress.try_start_scrape() is True
    assert progress.try_start_scrape() is False

    progress.set_scrape(running=False)

    assert progress.try_start_scrape() is True


async def test_broadcast_publishes_the_payload_to_subscribers(events):
    progress.broadcast("model_pull_progress", {"type": "model_pull_progress", "percent": 5}, key="m")

    assert await received(events) == [{"type": "model_pull_progress", "percent": 5}]


async def test_broadcasts_for_one_key_are_throttled(events):
    for percent in range(5):
        progress.broadcast("model_pull_progress", {"percent": percent}, key="m")

    assert await received(events) == [{"percent": 0}]


async def test_force_skips_the_throttle(events):
    progress.broadcast("model_pull_progress", {"n": 1}, key="m")
    progress.broadcast("model_pull_progress", {"n": 2}, key="m", force=True)

    assert [event["n"] for event in await received(events)] == [1, 2]
