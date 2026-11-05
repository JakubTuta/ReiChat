import asyncio
from types import SimpleNamespace

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


def test_a_cleared_name_can_be_claimed_again():
    progress.try_claim("llama3.1_8b", "Starting…")

    progress.clear_progress("llama3.1_8b")

    assert not progress.is_in_progress("llama3.1_8b")
    assert progress.try_claim("llama3.1_8b", "Starting…") is True


def test_a_fresh_claim_starts_at_zero_percent_with_the_detail_given():
    progress.try_claim("llama3.1_8b", "Checking for a newer version…")

    assert progress.get_snapshot()["progress"]["llama3.1_8b"] == {
        "phase": "model_pull_progress",
        "percent": 0,
        "detail": "Checking for a newer version…",
    }


def test_set_progress_merges_into_what_is_already_there():
    progress.try_claim("m", "Starting…")

    progress.set_progress("m", percent=40)
    progress.set_progress("m", detail="Downloading")

    assert progress.get_snapshot()["progress"]["m"] == {
        "phase": "model_pull_progress",
        "percent": 40,
        "detail": "Downloading",
    }


def test_snapshot_is_a_copy_not_a_live_view():
    progress.try_claim("m", "Starting…")
    snapshot = progress.get_snapshot()

    progress.clear_progress("m")

    assert "m" in snapshot["progress"]


def test_only_one_catalog_refresh_can_run_at_a_time():
    assert progress.try_start_scrape() is True
    assert progress.try_start_scrape() is False

    progress.set_scrape(running=False)

    assert progress.try_start_scrape() is True


def test_starting_a_refresh_resets_the_previous_outcome():
    progress.set_scrape(running=False, total=10, completed=10, current="x", error="boom")

    progress.try_start_scrape()

    assert progress.get_snapshot()["scrape"] == {
        "running": True,
        "total": None,
        "completed": 0,
        "current": None,
        "error": None,
    }


async def test_broadcast_publishes_the_payload_to_subscribers(events):
    progress.broadcast("model_pull_progress", {"type": "model_pull_progress", "percent": 5}, key="m")

    assert await received(events) == [{"type": "model_pull_progress", "percent": 5}]


async def test_broadcasts_for_one_key_are_throttled(events):
    for percent in range(5):
        progress.broadcast("model_pull_progress", {"percent": percent}, key="m")

    assert await received(events) == [{"percent": 0}]


async def test_throttling_is_per_event_type_and_key(events):
    progress.broadcast("model_pull_progress", {"id": "a"}, key="a")
    progress.broadcast("model_pull_progress", {"id": "b"}, key="b")
    progress.broadcast("image_pull_progress", {"id": "c"}, key="a")

    assert [event["id"] for event in await received(events)] == ["a", "b", "c"]


async def test_force_skips_the_throttle(events):
    progress.broadcast("model_pull_progress", {"n": 1}, key="m")
    progress.broadcast("model_pull_progress", {"n": 2}, key="m", force=True)

    assert [event["n"] for event in await received(events)] == [1, 2]


async def test_the_throttle_opens_again_after_half_a_second(events, monkeypatch):
    now = [100.0]
    # Replacing the module attribute, not time.monotonic itself: asyncio reads that.
    monkeypatch.setattr(progress, "time", SimpleNamespace(monotonic=lambda: now[0]))

    progress.broadcast("model_pull_progress", {"n": 1}, key="m")
    now[0] += 0.4
    progress.broadcast("model_pull_progress", {"n": 2}, key="m")
    now[0] += 0.2
    progress.broadcast("model_pull_progress", {"n": 3}, key="m")

    assert [event["n"] for event in await received(events)] == [1, 3]


async def test_clearing_progress_lets_the_next_event_through_at_once(events):
    progress.try_claim("m", "Starting…")
    progress.broadcast("model_pull_progress", {"n": 1}, key="m")

    progress.clear_progress("m")
    progress.broadcast("model_pull_progress", {"n": 2}, key="m")

    assert [event["n"] for event in await received(events)] == [1, 2]
