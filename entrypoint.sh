#!/usr/bin/env bash
set -euo pipefail

mkdir -p /app/data/logs /app/data/backups /app/data/model_cache

# Perform a safe online SQLite backup before running Alembic schema migrations
python - <<'PY'
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

db_url = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:////app/data/give_me_job_please.db")
if "sqlite" in db_url:
    raw_path = db_url.split(":///", 1)[-1]
    db_path = Path(raw_path)
    if db_path.exists() and db_path.stat().st_size > 0:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_dir = db_path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / f"give_me_job_please_{stamp}.db"
        with sqlite3.connect(db_path) as src, sqlite3.connect(backup_path) as dst:
            src.backup(dst)
        # Keep only the 10 most recent backups
        backups = sorted(backup_dir.glob("give_me_job_please_*.db"))
        for old in backups[:-10]:
            old.unlink(missing_ok=True)
        print(f"[entrypoint] Created pre-migration SQLite backup: {backup_path}")
PY

echo "[entrypoint] Running Alembic migrations..."
alembic upgrade head

echo "[entrypoint] Starting GiveMeJobPlease bot..."
exec give-me-job-please
