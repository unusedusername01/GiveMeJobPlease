"""Add nullable min_score_threshold column to user_config.

Revision ID: 0002_add_min_score_threshold
Revises: 0001_initial
Create Date: 2026-11-25 18:45:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_add_min_score_threshold"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("user_config") as batch_op:
        batch_op.add_column(sa.Column("min_score_threshold", sa.Float(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("user_config") as batch_op:
        batch_op.drop_column("min_score_threshold")
