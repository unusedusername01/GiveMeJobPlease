"""Async SQLite integration tests for ingestion, state transitions, starring, and cache invalidation."""

from __future__ import annotations

import pytest

from give_me_job_please.core.diff import filter_eligible_pool, sort_by_relevance
from give_me_job_please.core.models import (
    JobOffer,
    OfferScore,
    OfferStatus,
    RegistrationRecord,
    UserProfile,
)
from give_me_job_please.core.scoring.base import compute_scoring_fingerprint
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
