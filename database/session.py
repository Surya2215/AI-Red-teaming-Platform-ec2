"""Async SQLAlchemy session setup."""

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.config import get_settings


settings = get_settings()
# pool_pre_ping: validates each pooled connection with a cheap "SELECT 1" before
# handing it to a request, transparently reconnecting if the remote (EC2 Postgres,
# reached over the open internet rather than a local socket) silently dropped an
# idle connection - without this, the *next* request to reuse that connection
# crashes with asyncpg.exceptions.ConnectionDoesNotExistError instead of retrying.
# pool_recycle: proactively retires connections older than 280s so they're never
# old enough to hit whatever idle-connection timeout the network path enforces.
engine = create_async_engine(settings.database_url, future=True, echo=False, pool_pre_ping=True, pool_recycle=280)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with AsyncSessionLocal() as session:
        yield session


async def init_db() -> None:
    from database.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
