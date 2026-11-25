"""Alembic environment configured to read DATABASE_URL from Settings/environment."""

from __future__ import annotations

import os
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

from give_me_job_please.storage.orm import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_sync_database_url() -> str:
    """Resolve DATABASE_URL from env (or .env) and convert async aiosqlite URL to sync sqlite."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("DATABASE_URL="):
                    url = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not url:
        url = config.get_main_option("sqlalchemy.url") or "sqlite:///data/give_me_job_please.db"
    sync_url = url.replace("sqlite+aiosqlite://", "sqlite://")
    # Ensure parent folder exists for local sqlite paths
    if sync_url.startswith("sqlite:///"):
        db_file = Path(sync_url.replace("sqlite:///", "", 1))
        if str(db_file) != ":memory:":
            db_file.parent.mkdir(parents=True, exist_ok=True)
    return sync_url


def run_migrations_offline() -> None:
    url = get_sync_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = get_sync_database_url()
    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
