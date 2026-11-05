import asyncio

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.containers.ports import PORT_RANGE_START, allocate_port
from app.db.models import ContainerPort

pytestmark = pytest.mark.usefixtures("db")


async def test_allocate_port_reuses_existing_allocation(db: AsyncSession):
    first = await allocate_port("llama3.1_8b")
    second = await allocate_port("llama3.1_8b")

    assert first == second
    count = (
        await db.execute(
            select(func.count())
            .select_from(ContainerPort)
            .where(ContainerPort.container_name == "llama3.1_8b")
        )
    ).scalar_one()
    assert count == 1


async def test_allocate_port_hands_out_the_lowest_free_port(db: AsyncSession):
    db.add_all(
        [
            ContainerPort(container_name="a", port=PORT_RANGE_START),
            ContainerPort(container_name="c", port=PORT_RANGE_START + 2),
        ]
    )
    await db.commit()

    assert await allocate_port("b") == PORT_RANGE_START + 1


async def test_simultaneous_allocations_never_share_a_port(db: AsyncSession):
    names = [f"model_{n}" for n in range(8)]

    ports = await asyncio.gather(*(allocate_port(name) for name in names))

    assert len(set(ports)) == len(names)
    stored = (await db.execute(select(func.count()).select_from(ContainerPort))).scalar_one()
    assert stored == len(names)
