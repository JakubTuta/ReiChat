from sqlalchemy import select, text

from app.db.models import ContainerPort
from app.db.session import SessionLocal

PORT_RANGE_START = 11434

# Any constant will do; it only has to be the same for every caller.
_ALLOCATION_LOCK_KEY = 11434


async def allocate_port(container_name: str) -> int:
    """Return the host port owned by `container_name`, allocating the lowest
    free port >= PORT_RANGE_START if it doesn't have one yet.

    A transaction-level advisory lock serializes allocations, so two concurrent
    container creations can't both compute the same free port. (Locking the
    existing rows would not do: the row about to be inserted isn't there to
    lock yet.)
    """
    async with SessionLocal() as session, session.begin():
        await session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": _ALLOCATION_LOCK_KEY})

        existing = (
            await session.execute(
                select(ContainerPort.port).where(ContainerPort.container_name == container_name)
            )
        ).scalar_one_or_none()
        if existing is not None:
            return existing

        used_ports = set((await session.execute(select(ContainerPort.port))).scalars())
        port = PORT_RANGE_START
        while port in used_ports:
            port += 1

        session.add(ContainerPort(container_name=container_name, port=port))
        return port
