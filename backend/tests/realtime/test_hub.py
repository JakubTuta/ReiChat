import asyncio
import threading

from app.realtime.hub import QUEUE_SIZE, Hub


async def drain(queue: asyncio.Queue[dict]) -> list[dict]:
    await asyncio.sleep(0)
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


async def test_every_subscriber_receives_every_event():
    hub = Hub()
    hub.bind(asyncio.get_running_loop())
    first, second = hub.subscribe(), hub.subscribe()

    hub.publish({"type": "snapshot", "n": 1})

    assert await drain(first) == [{"type": "snapshot", "n": 1}]
    assert await drain(second) == [{"type": "snapshot", "n": 1}]


async def test_events_arrive_in_the_order_they_were_published():
    hub = Hub()
    hub.bind(asyncio.get_running_loop())
    queue = hub.subscribe()

    for n in range(5):
        hub.publish({"n": n})

    assert [event["n"] for event in await drain(queue)] == [0, 1, 2, 3, 4]


async def test_an_unsubscribed_queue_stops_receiving():
    hub = Hub()
    hub.bind(asyncio.get_running_loop())
    queue = hub.subscribe()
    hub.unsubscribe(queue)

    hub.publish({"n": 1})

    assert await drain(queue) == []
    assert hub.subscriber_count == 0


async def test_unsubscribing_twice_is_harmless():
    hub = Hub()
    queue = hub.subscribe()

    hub.unsubscribe(queue)
    hub.unsubscribe(queue)


async def test_publishing_before_the_hub_is_bound_is_dropped_not_an_error():
    hub = Hub()
    queue = hub.subscribe()

    hub.publish({"n": 1})

    assert await drain(queue) == []


async def test_a_worker_thread_can_publish():
    # The Docker image-pull progress callback runs off the event loop.
    hub = Hub()
    hub.bind(asyncio.get_running_loop())
    queue = hub.subscribe()

    thread = threading.Thread(target=hub.publish, args=({"from": "thread"},))
    thread.start()
    thread.join()

    assert await asyncio.wait_for(queue.get(), timeout=2) == {"from": "thread"}


async def test_a_client_that_stops_reading_loses_events_without_blocking_the_others():
    hub = Hub()
    hub.bind(asyncio.get_running_loop())
    stalled, healthy = hub.subscribe(), hub.subscribe()

    received = []
    for n in range(QUEUE_SIZE + 20):
        hub.publish({"n": n})
        await asyncio.sleep(0)
        received.extend(await drain(healthy))

    assert stalled.qsize() == QUEUE_SIZE
    assert [event["n"] for event in received] == list(range(QUEUE_SIZE + 20))


async def test_publishing_after_the_loop_closed_does_not_raise():
    hub = Hub()
    loop = asyncio.new_event_loop()
    hub.bind(loop)
    loop.close()

    hub.publish({"n": 1})


async def test_unbinding_forgets_every_subscriber():
    hub = Hub()
    hub.bind(asyncio.get_running_loop())
    hub.subscribe()

    hub.unbind()

    assert hub.subscriber_count == 0
