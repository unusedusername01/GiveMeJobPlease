"""Async SQLite integration tests for ingestion, state transitions, starring, and cache invalidation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from sqlalchemy import create_engine, inspect

from give_me_job_please.bot.formatting import (
    DigestQuery,
    build_pagination_keyboard,
    decode_query_callback,
    encode_query_callback,
)
from give_me_job_please.core.diff import (
    PreScoringFilter,
    apply_pre_scoring_filters,
    filter_by_min_score,
    filter_eligible_pool,
    parse_digest_args,
    parse_min_score_threshold,
    sort_by_relevance,
)
from give_me_job_please.core.models import (
    EnrichedOffer,
    JobOffer,
    OfferScore,
    OfferStatus,
    RegistrationRecord,
    UserProfile,
)
from give_me_job_please.core.scoring.base import compute_scoring_fingerprint
from give_me_job_please.errors import ValidationError
from give_me_job_please.storage.db import create_engine_and_sessionmaker
from give_me_job_please.storage.orm import Base
from give_me_job_please.storage.repository import OfferRepository


async def _make_repo() -> tuple[OfferRepository, object]:
    engine, session_factory = create_engine_and_sessionmaker("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return OfferRepository(session_factory), engine


def _sample_offers() -> list[JobOffer]:
    return [
        JobOffer(
            uid="isacademia:10001",
            external_id="10001",
            stage_id="9001",
            title="AI Security Intern",
            company="CyberLab",
            assignment_location="Lausanne",
            format="Internship",
            registered=2,
            places=1,
            creation_date="2026-11-20",
            description="Red-teaming LLMs.",
            has_description=True,
        ),
        JobOffer(
            uid="isacademia:10002",
            external_id="10002",
            stage_id="9002",
            title="Frontend React Intern",
            company="WebCo",
            assignment_location="Paris",
            format="Internship",
            registered=10,
            places=1,
            creation_date="2026-11-19",
            description="Build dashboards.",
            has_description=True,
        ),
    ]


@pytest.mark.asyncio
async def test_lifecycle_states_starring_and_registrations_sync() -> None:
    repo, engine = await _make_repo()
    try:
        user_id = 12345
        summary = await repo.ingest_showroom_offers(
            user_id=user_id,
            offers=_sample_offers(),
            filename="showroom.htm",
            parser_version="isacademia_showroom:1.0",
        )
        assert summary.new_offers == 2
        assert summary.eligible_new_count == 2

        # Star offer 10001 and exclude 10002
        starred = await repo.toggle_star(user_id, "isacademia:10001")
        assert starred is not None and starred.is_starred is True
        await repo.set_offer_status(user_id, "isacademia:10002", OfferStatus.EXCLUDED)

        # Mark 10001 seen; it stays in Eligible Pool (include_seen=True), while excluded does not
        await repo.mark_offers_seen(user_id, ["isacademia:10001"])
        all_items = await repo.list_enriched_offers(user_id)
        assert [i.offer.uid for i in filter_eligible_pool(all_items, include_seen=False)] == []
        assert [i.offer.uid for i in filter_eligible_pool(all_items, include_seen=True)] == [
            "isacademia:10001"
        ]

        # Sync Registrations (stu).htm -> marks 10001 applied while preserving is_starred=True!
        reg_summary = await repo.ingest_registrations(
            user_id=user_id,
            records=[RegistrationRecord(internship_number="10001", stage_id="9001")],
            filename="registrations.htm",
            parser_version="isacademia_registrations:1.0",
        )
        assert reg_summary.auto_marked_applied == 1

        applied_item = await repo.get_enriched_offer(user_id, "isacademia:10001")
        assert applied_item is not None
        assert applied_item.status == OfferStatus.APPLIED
        assert applied_item.is_starred is True
    finally:
        await engine.dispose()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_score_persistence_sorting_and_fingerprint_invalidation() -> None:
    repo, engine = await _make_repo()
    try:
        user_id = 99
        await repo.ingest_showroom_offers(
            user_id=user_id,
            offers=_sample_offers(),
            filename="showroom.htm",
            parser_version="isacademia_showroom:1.0",
        )
        profile = await repo.save_profile(
            UserProfile(user_id=user_id, target_roles="AI Security", preferred_locations="Lausanne")
        )
        fp1 = compute_scoring_fingerprint(
            profile_hash=profile.profile_hash(),
            cv_hash=profile.cv_hash(),
            engine="openrouter",
            model_id="deepseek/deepseek-chat",
            prompt_version="openrouter_rubric:1.0",
        )
        await repo.save_scores(
            user_id,
            [
                OfferScore(
                    offer_uid="isacademia:10001",
                    score=9.4,
                    score_engine="openrouter",
                    scoring_fingerprint=fp1,
                    location_match=True,
                    match_summary="Great fit",
                    tags=["Security"],
                ),
                OfferScore(
                    offer_uid="isacademia:10002",
                    score=3.2,
                    score_engine="openrouter",
                    scoring_fingerprint=fp1,
                    location_match=False,
                    match_summary="Frontend mismatch",
                    tags=["React"],
                ),
            ],
        )

        ranked = sort_by_relevance(await repo.list_enriched_offers(user_id))
        assert [r.offer.uid for r in ranked] == ["isacademia:10001", "isacademia:10002"]

        # Updating profile changes fingerprint so stale scores are detected
        profile.target_roles = "Frontend React"
        updated_profile = await repo.save_profile(profile)
        fp2 = compute_scoring_fingerprint(
            profile_hash=updated_profile.profile_hash(),
            cv_hash=updated_profile.cv_hash(),
            engine="openrouter",
            model_id="deepseek/deepseek-chat",
            prompt_version="openrouter_rubric:1.0",
        )
        assert fp1 != fp2
    finally:
        await engine.dispose()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_min_score_threshold_persistence_and_hash_invariance() -> None:
    repo, engine = await _make_repo()
    try:
        user_id = 42
        base_profile = await repo.save_profile(
            UserProfile(
                user_id=user_id,
                target_roles="Systems Security",
                preferred_locations="Lausanne",
                min_score_threshold=None,
            )
        )
        hash_before = base_profile.profile_hash()

        updated = await repo.set_min_score_threshold(user_id, 7.5)
        assert updated.min_score_threshold == 7.5
        # Display cutoff changes must NEVER invalidate cached LLM scoring fingerprints
        assert updated.profile_hash() == hash_before

        reloaded = await repo.get_or_create_profile(user_id)
        assert reloaded.min_score_threshold == 7.5
        assert reloaded.profile_hash() == hash_before

        cleared = await repo.set_min_score_threshold(user_id, None)
        assert cleared.min_score_threshold is None
    finally:
        await engine.dispose()  # type: ignore[attr-defined]


def test_filter_by_min_score_and_threshold_parser() -> None:
    offers = _sample_offers()
    enriched = [
        EnrichedOffer(offer=offers[0], score=8.0),
        EnrichedOffer(offer=offers[1], score=6.5),
        EnrichedOffer(
            offer=JobOffer(
                uid="isacademia:10003",
                external_id="10003",
                title="Unscored Role",
                company="Mystery",
            ),
            score=None,
        ),
    ]

    assert len(filter_by_min_score(enriched, None)) == 3
    assert len(filter_by_min_score(enriched, 0.0)) == 3
    assert [i.offer.uid for i in filter_by_min_score(enriched, 6.5)] == [
        "isacademia:10001",
        "isacademia:10002",
    ]
    assert [i.offer.uid for i in filter_by_min_score(enriched, 7.0)] == ["isacademia:10001"]

    assert parse_min_score_threshold("7") == 7.0
    assert parse_min_score_threshold("6,5") == 6.5
    assert parse_min_score_threshold("8+") == 8.0
    assert parse_min_score_threshold("off") is None
    assert parse_min_score_threshold("0") is None
    for bad in ("-1", "10.5", "nan", "inf", "high"):
        with pytest.raises(ValidationError):
            parse_min_score_threshold(bad)


def test_alembic_0002_min_score_threshold_reversible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_file = tmp_path / "alembic_test.db"
    db_url = f"sqlite:///{db_file.as_posix()}"
    monkeypatch.setenv("DATABASE_URL", db_url)

    alembic_cfg = AlembicConfig("alembic.ini")
    alembic_command.upgrade(alembic_cfg, "head")

    sync_engine = create_engine(db_url)
    try:
        cols = {c["name"] for c in inspect(sync_engine).get_columns("user_config")}
        assert "min_score_threshold" in cols
    finally:
        sync_engine.dispose()

    alembic_command.downgrade(alembic_cfg, "0001_initial")
    sync_engine = create_engine(db_url)
    try:
        cols_after = {c["name"] for c in inspect(sync_engine).get_columns("user_config")}
        assert "min_score_threshold" not in cols_after
    finally:
        sync_engine.dispose()


def test_pre_scoring_recency_filter_posted_and_ingested() -> None:
    now = datetime(2026, 11, 25, 18, 0, tzinfo=UTC)
    recent_posted = EnrichedOffer(
        offer=JobOffer(
            uid="isacademia:1",
            external_id="1",
            title="Recent Posted",
            company="A",
            creation_date="2026-11-18",  # Exactly 7 calendar days ago -> included in 7d
        ),
        first_seen_at=now - timedelta(days=20),
    )
    old_posted_recently_uploaded = EnrichedOffer(
        offer=JobOffer(
            uid="isacademia:2",
            external_id="2",
            title="Old Posted New Upload",
            company="B",
            creation_date="2026-10-01",
        ),
        first_seen_at=now - timedelta(days=2),
    )
    missing_date_fallback = EnrichedOffer(
        offer=JobOffer(
            uid="isacademia:3",
            external_id="3",
            title="No Creation Date",
            company="C",
            creation_date="",
        ),
        first_seen_at=now - timedelta(days=3),
    )
    no_dates = EnrichedOffer(
        offer=JobOffer(
            uid="isacademia:4",
            external_id="4",
            title="No Dates At All",
            company="D",
            creation_date="",
        ),
        first_seen_at=None,
    )
    pool = [recent_posted, old_posted_recently_uploaded, missing_date_fallback, no_dates]

    # No filter returns all
    assert len(apply_pre_scoring_filters(pool, PreScoringFilter())) == 4

    # Posted basis (7d): includes exact 7d creation_date + fallback via first_seen_at
    posted_7d = apply_pre_scoring_filters(
        pool, PreScoringFilter(max_age_days=7, recency_basis="posted"), now=now
    )
    assert [i.offer.uid for i in posted_7d] == ["isacademia:1", "isacademia:3"]

    # Ingested basis (7d): uses first_seen_at strictly
    ingested_7d = apply_pre_scoring_filters(
        pool, PreScoringFilter(max_age_days=7, recency_basis="ingested"), now=now
    )
    assert [i.offer.uid for i in ingested_7d] == ["isacademia:2", "isacademia:3"]


def test_parse_digest_args_and_query_callbacks() -> None:
    parsed = parse_digest_args("local 14d ingested")
    assert parsed.max_age_days == 14
    assert parsed.recency_basis == "ingested"
    assert parsed.force_local is True

    assert parse_digest_args("2w").max_age_days == 14
    assert parse_digest_args("1m posted").max_age_days == 30
    assert parse_digest_args("all").max_age_days is None

    for bad in ("0d", "bogus", "99999d"):
        with pytest.raises(ValidationError):
            parse_digest_args(bad)

    q = DigestQuery(
        view="starred",
        page=3,
        max_age_days=30,
        recency_basis="ingested",
        ignore_threshold=True,
        force_local=True,
    )
    encoded = encode_query_callback("pg", q)
    assert len(encoded.encode("utf-8")) < 64
    prefix, decoded, extra = decode_query_callback(encoded)
    assert prefix == "pg"
    assert decoded == q
    assert extra == ""

    # Legacy 3-part callback remains supported
    lp, lq, _ = decode_query_callback("pg:all:2")
    assert lp == "pg" and lq.view == "all" and lq.page == 2

    kb = build_pagination_keyboard(query=q, total_pages=5, seen_count=12, profile_threshold=7.0)
    for row in kb.inline_keyboard:
        for btn in row:
            assert btn.callback_data is not None
            assert len(btn.callback_data.encode("utf-8")) < 64


@pytest.mark.asyncio
async def test_repository_exposes_first_seen_at() -> None:
    repo, engine = await _make_repo()
    try:
        await repo.ingest_showroom_offers(
            user_id=7,
            offers=_sample_offers(),
            filename="showroom.htm",
            parser_version="isacademia_showroom:1.0",
        )
        items = await repo.list_enriched_offers(7)
        assert items
        assert all(item.first_seen_at is not None for item in items)
    finally:
        await engine.dispose()  # type: ignore[attr-defined]

