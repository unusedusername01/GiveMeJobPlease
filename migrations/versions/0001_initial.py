"""Initial schema for GiveMeJobPlease V1.

Revision ID: 0001_initial
Revises:
Create Date: 2026-11-25 00:00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ingest_batches",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("file_type", sa.String(length=64), nullable=False),
        sa.Column("parser_version", sa.String(length=64), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("total_rows", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("new_rows", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("updated_rows", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "offers",
        sa.Column("uid", sa.String(length=128), primary_key=True),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("external_id", sa.String(length=64), nullable=False),
        sa.Column("stage_id", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("company", sa.Text(), nullable=False),
        sa.Column("company_hq", sa.Text(), nullable=False, server_default=""),
        sa.Column("assignment_location", sa.Text(), nullable=False, server_default=""),
        sa.Column("label_durable_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("format", sa.String(length=128), nullable=False, server_default=""),
        sa.Column("prof", sa.Text(), nullable=False, server_default=""),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("has_description", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.String(length=32), nullable=False, server_default=""),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_registered", sa.Integer(), nullable=True),
        sa.Column("last_places", sa.Integer(), nullable=True),
        sa.Column("in_latest_showroom", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("parser_version", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("embedding", sa.LargeBinary(), nullable=True),
        sa.Column("embedding_model", sa.String(length=128), nullable=True),
    )
    op.create_index("ix_offers_source", "offers", ["source"])
    op.create_index("ix_offers_external_id", "offers", ["external_id"])

    op.create_table(
        "offer_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "offer_uid",
            sa.String(length=128),
            sa.ForeignKey("offers.uid", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "batch_id",
            sa.Integer(),
            sa.ForeignKey("ingest_batches.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("snapshot_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("registered", sa.Integer(), nullable=True),
        sa.Column("places", sa.Integer(), nullable=True),
    )
    op.create_index("ix_offer_snapshots_offer_uid", "offer_snapshots", ["offer_uid"])

    op.create_table(
        "user_config",
        sa.Column("user_id", sa.BigInteger(), primary_key=True),
        sa.Column("target_roles", sa.Text(), nullable=False, server_default=""),
        sa.Column("preferred_locations", sa.Text(), nullable=False, server_default=""),
        sa.Column("accepted_formats_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("dealbreakers", sa.Text(), nullable=False, server_default=""),
        sa.Column("cv_text", sa.Text(), nullable=False, server_default=""),
        sa.Column("cv_filename", sa.String(length=255), nullable=False, server_default=""),
        sa.Column("profile_hash", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("cv_hash", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("profile_embedding", sa.LargeBinary(), nullable=True),
        sa.Column("profile_embedding_model", sa.String(length=128), nullable=True),
        sa.Column("openrouter_key_enc", sa.Text(), nullable=True),
        sa.Column("openrouter_key_hash", sa.String(length=64), nullable=True),
        sa.Column("openrouter_model", sa.String(length=128), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "user_offer_states",
        sa.Column("user_id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "offer_uid",
            sa.String(length=128),
            sa.ForeignKey("offers.uid", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="new"),
        sa.Column("is_starred", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("score_engine", sa.String(length=64), nullable=True),
        sa.Column("scoring_fingerprint", sa.String(length=64), nullable=True),
        sa.Column("location_match", sa.Boolean(), nullable=True),
        sa.Column("match_summary", sa.Text(), nullable=True),
        sa.Column("tags_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_user_offer_states_status", "user_offer_states", ["status"])
    op.create_index("ix_user_offer_states_is_starred", "user_offer_states", ["is_starred"])


def downgrade() -> None:
    op.drop_index("ix_user_offer_states_is_starred", table_name="user_offer_states")
    op.drop_index("ix_user_offer_states_status", table_name="user_offer_states")
    op.drop_table("user_offer_states")
    op.drop_table("user_config")
    op.drop_index("ix_offer_snapshots_offer_uid", table_name="offer_snapshots")
    op.drop_table("offer_snapshots")
    op.drop_index("ix_offers_external_id", table_name="offers")
    op.drop_index("ix_offers_source", table_name="offers")
    op.drop_table("offers")
    op.drop_table("ingest_batches")
