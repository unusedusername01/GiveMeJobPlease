"""SQLAlchemy 2.0 ORM definitions matching Alembic revision 0001_initial."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class IngestBatchORM(Base):
    __tablename__ = "ingest_batches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    file_type: Mapped[str] = mapped_column(String(64), nullable=False)
    parser_version: Mapped[str] = mapped_column(String(64), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    total_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    new_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class OfferORM(Base):
    __tablename__ = "offers"

    uid: Mapped[str] = mapped_column(String(128), primary_key=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    external_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    stage_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    title: Mapped[str] = mapped_column(Text, nullable=False)
    company: Mapped[str] = mapped_column(Text, nullable=False)
    company_hq: Mapped[str] = mapped_column(Text, nullable=False, default="")
    assignment_location: Mapped[str] = mapped_column(Text, nullable=False, default="")
    label_durable_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    format: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    prof: Mapped[str] = mapped_column(Text, nullable=False, default="")
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    has_description: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_registered: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_places: Mapped[int | None] = mapped_column(Integer, nullable=True)
    in_latest_showroom: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    parser_version: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(128), nullable=True)


class OfferSnapshotORM(Base):
    __tablename__ = "offer_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    offer_uid: Mapped[str] = mapped_column(
        String(128), ForeignKey("offers.uid", ondelete="CASCADE"), nullable=False, index=True
    )
    batch_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("ingest_batches.id", ondelete="SET NULL"), nullable=True
    )
    snapshot_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    registered: Mapped[int | None] = mapped_column(Integer, nullable=True)
    places: Mapped[int | None] = mapped_column(Integer, nullable=True)


class UserConfigORM(Base):
    __tablename__ = "user_config"

    user_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    target_roles: Mapped[str] = mapped_column(Text, nullable=False, default="")
    preferred_locations: Mapped[str] = mapped_column(Text, nullable=False, default="")
    accepted_formats_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    dealbreakers: Mapped[str] = mapped_column(Text, nullable=False, default="")
    cv_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    cv_filename: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    profile_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    cv_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    profile_embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    profile_embedding_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    openrouter_key_enc: Mapped[str | None] = mapped_column(Text, nullable=True)
    openrouter_key_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    openrouter_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class UserOfferStateORM(Base):
    __tablename__ = "user_offer_states"

    user_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    offer_uid: Mapped[str] = mapped_column(
        String(128), ForeignKey("offers.uid", ondelete="CASCADE"), primary_key=True
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="new", index=True)
    is_starred: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_engine: Mapped[str | None] = mapped_column(String(64), nullable=True)
    scoring_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    location_match: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    match_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
