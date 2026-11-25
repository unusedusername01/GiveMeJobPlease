"""Async SQLAlchemy engine setup with SQLite WAL mode and foreign key enforcement."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine


def create_engine_and_sessionmaker(
    database_url: str,
) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    """Create an async SQLAlchemy engine configured cleanly for SQLite."""
    if database_url.startswith("sqlite+aiosqlite:///") and ":memory:" not in database_url:
        raw_path = database_url.replace("sqlite+aiosqlite:///", "", 1)
        Path(raw_path).parent.mkdir(parents=True, exist_ok=True)

    engine = create_async_engine(database_url, echo=False, future=True)

    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection: Any, _: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return engine, session_factory
