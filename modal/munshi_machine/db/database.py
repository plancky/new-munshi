import logging
import os

from modal import is_local
from sqlalchemy.ext.asyncio.engine import create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import Session, create_engine
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.pool import NullPool

if is_local():
    from dotenv import load_dotenv

    load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")
DATABASE_URL_RO = os.environ.get("DATABASE_URL_RO")
# logging.basicConfig()
# logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)       # SQL statements + params
# logging.getLogger("sqlalchemy.pool").setLevel(logging.DEBUG)       # Pool events
# logging.getLogger("sqlalchemy.dialects").setLevel(logging.INFO)    # Driver-level logs
logging.getLogger("sqlalchemy.orm").setLevel(
    logging.DEBUG
)  # ORM events (lazy loads, flushes)
async_engine = create_async_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=250,
    pool_size=400,
    max_overflow=50,
    pool_timeout=30, # seconds to wait for a connection
    connect_args={
        "prepare_threshold": None,  # Disable for PGBouncer/NeonDB pooler
        # Note: sslmode and channel_binding should be in the DATABASE_URL, not here
        # For pooler connections, avoid keepalive params as they interfere with pooler
    },
)


engine_ro = create_engine(
    DATABASE_URL_RO,
    pool_pre_ping=True,
    poolclass=NullPool,
    pool_recycle=100,
    connect_args={
        "prepare_threshold": None,  # Disable for PGBouncer/NeonDB pooler
        "channel_binding": "require",
        # Note: sslmode and channel_binding should be in the DATABASE_URL, not here
        # For pooler connections, avoid keepalive params as they interfere with pooler
    },
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    poolclass=NullPool,
    pool_recycle=100,
    connect_args={
        "prepare_threshold": None,  # Disable for PGBouncer/NeonDB pooler
        "channel_binding": "require",
        # Note: sslmode and channel_binding should be in the DATABASE_URL, not here
        # For pooler connections, avoid keepalive params as they interfere with pooler
    },
)

AsyncSessionLocal = sessionmaker(
    class_=AsyncSession,
    bind=async_engine,
    expire_on_commit=False,
)


def connect():
    return Session(engine)


async def set_statement_timeout(session: AsyncSession, timeout_ms: int = 30000):
    """
    Set statement timeout for a session. Call this after creating a session.
    Neon requires this to be set per-session, not in connection options.

    Usage:
        async with AsyncSessionLocal() as session:
            await set_statement_timeout(session, 30000)
            # ... your queries
    """
    from sqlmodel import text

    await session.exec(text(f"SET statement_timeout = {timeout_ms}"))
    await session.commit()


if __name__ == "__main__":
    from sqlmodel import text

    from munshi_machine.models.private import PrivateSchemaBase

    def init_db():
        # ensure schema exists
        with connect() as session:
            session.exec(text("CREATE SCHEMA IF NOT EXISTS private"))
            session.commit()

        # create SQLModel tables in that schema
        bind = connect().get_bind()
        PrivateSchemaBase.metadata.create_all(bind)

    init_db()
    pass
