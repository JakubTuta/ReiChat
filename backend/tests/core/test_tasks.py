import asyncio
import gc
import logging

import pytest

from app.core import tasks


async def test_a_spawned_coroutine_runs_to_completion():
    done = asyncio.Event()

    async def work() -> str:
        done.set()
        return "finished"

    task = tasks.spawn(work(), name="work")

    assert await task == "finished"
    assert done.is_set()


async def test_a_task_nobody_holds_is_not_garbage_collected_mid_run():
    finished = asyncio.Event()

    async def work() -> None:
        await asyncio.sleep(0.05)
        finished.set()

    tasks.spawn(work(), name="orphan")
    gc.collect()

    await asyncio.wait_for(finished.wait(), timeout=2)


async def test_a_failing_task_is_logged_with_its_traceback(caplog: pytest.LogCaptureFixture):
    async def explode() -> None:
        raise RuntimeError("model pull blew up")

    with caplog.at_level(logging.ERROR, logger="app"):
        task = tasks.spawn(explode(), name="install")
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)

    [record] = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert "install" in record.getMessage()
    assert record.exc_info is not None
    assert "model pull blew up" in str(record.exc_info[1])


async def test_cancel_all_stops_everything_still_running():
    started = asyncio.Event()

    async def forever() -> None:
        started.set()
        await asyncio.sleep(60)

    running = [tasks.spawn(forever(), name=f"forever-{n}") for n in range(3)]
    await started.wait()

    await tasks.cancel_all()

    assert all(task.cancelled() for task in running)
