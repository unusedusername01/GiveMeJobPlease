"""Async Repository encapsulating all SQLite persistence and state transitions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from give_me_job_please.core.auth.crypto import SecretBox
from give_me_job_please.core.models import (
    EnrichedOffer,
    IngestSummary,
    JobOffer,
    OfferScore,
    OfferStatus,
    RegistrationRecord,
    UserProfile,
)
from give_me_job_please.core.scoring.local_embedder import bytes_to_vector, vector_to_bytes
from give_me_job_please.errors import AuthError
from give_me_job_please.storage.orm import (
    IngestBatchORM,
    OfferORM,
    OfferSnapshotORM,
    UserConfigORM,
    UserOfferStateORM,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# Marks a cv_text value as AES-256-GCM ciphertext rather than legacy plaintext.
_ENC_PREFIX = "enc:v1:"


@dataclass(slots=True)
class OpenRouterUserState:
    key_enc: str | None
    key_hash: str | None
    model_id: str | None


@dataclass(slots=True)
class UserStats:
    total_in_showroom: int
    new_count: int
    seen_count: int
    applied_count: int
    excluded_count: int
    starred_count: int
    last_ingest_at: datetime | None
    last_parser_version: str


class OfferRepository:
    """Data-access layer decoupling core domain models from SQLAlchemy ORM tables."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        secret_box: SecretBox | None = None,
    ) -> None:
        self._sessions = session_factory
        self._secret_box = secret_box

    # ------------------------------------------------------------------
    # CV encryption helpers (at-rest protection for resume text)
    # ------------------------------------------------------------------

    def _encrypt_cv(self, user_id: int, plaintext: str) -> str:
        """Encrypt CV text for storage; legacy plaintext is left readable on read."""
        if not plaintext or self._secret_box is None:
            return plaintext
        token = self._secret_box.encrypt(plaintext, associated_data=f"cv:{user_id}")
        return f"{_ENC_PREFIX}{token}"

    def _decrypt_cv(self, user_id: int, stored: str) -> str:
        """Decrypt a stored CV value, transparently accepting pre-encryption rows."""
        if not stored or not stored.startswith(_ENC_PREFIX):
            return stored
        if self._secret_box is None:
            raise AuthError(
                "Cannot decrypt the stored CV without the master encryption key. "
                "Restart the bot with a valid ENCRYPTION_KEY.",
                detail="SecretBox unavailable while reading encrypted cv_text.",
            )
        return self._secret_box.decrypt(
            stored[len(_ENC_PREFIX) :], associated_data=f"cv:{user_id}"
        )

    # ------------------------------------------------------------------
    # User Config / Profile / OpenRouter Credentials
    # ------------------------------------------------------------------

    async def get_or_create_profile(self, user_id: int) -> UserProfile:
        async with self._sessions() as session:
            row = await session.get(UserConfigORM, user_id)
            if row is None:
                row = UserConfigORM(user_id=user_id, updated_at=_utcnow())
                session.add(row)
                await session.commit()
            return self._orm_to_profile(row)

    async def save_profile(self, profile: UserProfile) -> UserProfile:
        async with self._sessions() as session:
            row = await session.get(UserConfigORM, profile.user_id)
            now = _utcnow()
            p_hash = profile.profile_hash()
            c_hash = profile.cv_hash()
            if row is None:
                row = UserConfigORM(user_id=profile.user_id, updated_at=now)
                session.add(row)

            # Invalidate cached profile embedding if profile or CV changed
            if row.profile_hash != p_hash or row.cv_hash != c_hash:
                row.profile_embedding = None
                row.profile_embedding_model = None

            row.target_roles = profile.target_roles
            row.preferred_locations = profile.preferred_locations
            row.accepted_formats_json = json.dumps(profile.accepted_formats, ensure_ascii=False)
            row.dealbreakers = profile.dealbreakers
            row.min_score_threshold = profile.min_score_threshold
            row.cv_text = self._encrypt_cv(profile.user_id, profile.cv_text)
            row.cv_filename = profile.cv_filename
            row.profile_hash = p_hash
            row.cv_hash = c_hash
            row.updated_at = now
            await session.commit()
            return self._orm_to_profile(row)

    async def set_min_score_threshold(
        self,
        user_id: int,
        threshold: float | None,
    ) -> UserProfile:
        """Update only the display score cutoff without touching profile hashes or embeddings."""
        async with self._sessions() as session:
            row = await session.get(UserConfigORM, user_id)
            now = _utcnow()
            if row is None:
                row = UserConfigORM(user_id=user_id, updated_at=now)
                session.add(row)
            row.min_score_threshold = threshold
            row.updated_at = now
            await session.commit()
            return self._orm_to_profile(row)

    async def get_openrouter_state(self, user_id: int) -> OpenRouterUserState:
        async with self._sessions() as session:
            row = await session.get(UserConfigORM, user_id)
            if row is None:
                return OpenRouterUserState(key_enc=None, key_hash=None, model_id=None)
            return OpenRouterUserState(
                key_enc=row.openrouter_key_enc,
                key_hash=row.openrouter_key_hash,
                model_id=row.openrouter_model,
            )

    async def save_openrouter_credentials(
        self,
        user_id: int,
        *,
        key_enc: str,
        key_hash: str,
        default_model: str,
    ) -> str:
        async with self._sessions() as session:
            row = await session.get(UserConfigORM, user_id)
            now = _utcnow()
            if row is None:
                row = UserConfigORM(user_id=user_id, updated_at=now)
                session.add(row)
            row.openrouter_key_enc = key_enc
            row.openrouter_key_hash = key_hash
            if not row.openrouter_model:
                row.openrouter_model = default_model
            row.updated_at = now
            await session.commit()
            return row.openrouter_model

    async def set_openrouter_model(self, user_id: int, model_id: str) -> None:
        async with self._sessions() as session:
            row = await session.get(UserConfigORM, user_id)
            now = _utcnow()
            if row is None:
                row = UserConfigORM(user_id=user_id, updated_at=now)
                session.add(row)
            row.openrouter_model = model_id
            row.updated_at = now
            await session.commit()

    async def clear_openrouter_credentials(self, user_id: int) -> None:
        async with self._sessions() as session:
            row = await session.get(UserConfigORM, user_id)
            if row is not None:
                row.openrouter_key_enc = None
                row.openrouter_key_hash = None
                row.updated_at = _utcnow()
                await session.commit()

    # ------------------------------------------------------------------
    # Ingestion (Showroom & Registrations)
    # ------------------------------------------------------------------

    async def ingest_showroom_offers(
        self,
        *,
        user_id: int,
        offers: list[JobOffer],
        filename: str,
        parser_version: str,
    ) -> IngestSummary:
        """Upsert showroom offers, record competition snapshots, and initialize user states."""
        now = _utcnow()
        new_count = 0
        updated_count = 0

        async with self._sessions() as session:
            batch = IngestBatchORM(
                source="isacademia",
                file_type="showroom",
                parser_version=parser_version,
                filename=filename,
                total_rows=len(offers),
                new_rows=0,
                updated_rows=0,
                ingested_at=now,
            )
            session.add(batch)
            await session.flush()

            # Mark previous showroom offers as not in latest before flagging current ones
            await session.execute(
                update(OfferORM)
                .where(OfferORM.source == "isacademia")
                .values(in_latest_showroom=False)
            )

            existing_offers_res = await session.execute(
                select(OfferORM).where(OfferORM.uid.in_([o.uid for o in offers]))
            )
            existing_offers = {o.uid: o for o in existing_offers_res.scalars().all()}

            existing_states_res = await session.execute(
                select(UserOfferStateORM).where(
                    UserOfferStateORM.user_id == user_id,
                    UserOfferStateORM.offer_uid.in_([o.uid for o in offers]),
                )
            )
            existing_states = {s.offer_uid: s for s in existing_states_res.scalars().all()}

            for offer in offers:
                db_offer = existing_offers.get(offer.uid)
                if db_offer is None:
                    new_count += 1
                    db_offer = OfferORM(
                        uid=offer.uid,
                        source=offer.source,
                        external_id=offer.external_id,
                        stage_id=offer.stage_id,
                        title=offer.title,
                        company=offer.company,
                        company_hq=offer.company_hq,
                        assignment_location=offer.assignment_location,
                        label_durable_json=json.dumps(offer.label_durable, ensure_ascii=False),
                        format=offer.format,
                        prof=offer.prof,
                        description=offer.description,
                        has_description=offer.has_description,
                        created_at=offer.creation_date,
                        first_seen_at=now,
                        last_seen_at=now,
                        last_registered=offer.registered,
                        last_places=offer.places,
                        in_latest_showroom=True,
                        parser_version=parser_version,
                    )
                    session.add(db_offer)
                else:
                    updated_count += 1
                    # Invalidate cached offer embedding if title or description changed
                    if db_offer.title != offer.title or db_offer.description != offer.description:
                        db_offer.embedding = None
                        db_offer.embedding_model = None
                    db_offer.stage_id = offer.stage_id or db_offer.stage_id
                    db_offer.title = offer.title
                    db_offer.company = offer.company
                    db_offer.company_hq = offer.company_hq
                    db_offer.assignment_location = offer.assignment_location
                    db_offer.label_durable_json = json.dumps(offer.label_durable, ensure_ascii=False)
                    db_offer.format = offer.format
                    db_offer.prof = offer.prof
                    db_offer.description = offer.description
                    db_offer.has_description = offer.has_description
                    db_offer.created_at = offer.creation_date
                    db_offer.last_seen_at = now
                    db_offer.last_registered = offer.registered
                    db_offer.last_places = offer.places
                    db_offer.in_latest_showroom = True
                    db_offer.parser_version = parser_version

                session.add(
                    OfferSnapshotORM(
                        offer_uid=offer.uid,
                        batch_id=batch.id,
                        snapshot_at=now,
                        registered=offer.registered,
                        places=offer.places,
                    )
                )

                if offer.uid not in existing_states:
                    session.add(
                        UserOfferStateORM(
                            user_id=user_id,
                            offer_uid=offer.uid,
                            status=OfferStatus.NEW.value,
                            is_starred=False,
                            updated_at=now,
                        )
                    )

            batch.new_rows = new_count
            batch.updated_rows = updated_count
            await session.commit()

        stats = await self.get_user_stats(user_id)
        return IngestSummary(
            file_type="showroom",
            parser_version=parser_version,
            total_parsed=len(offers),
            new_offers=new_count,
            updated_offers=updated_count,
            eligible_new_count=stats.new_count,
            eligible_seen_count=stats.seen_count,
        )

    async def ingest_registrations(
        self,
        *,
        user_id: int,
        records: list[RegistrationRecord],
        filename: str,
        parser_version: str,
    ) -> IngestSummary:
        """Auto-mark matching offers in SQLite as APPLIED based on Registrations (stu).htm."""
        now = _utcnow()
        marked = 0
        ext_ids = {r.internship_number for r in records if r.internship_number}
        stage_ids = {r.stage_id for r in records if r.stage_id}

        async with self._sessions() as session:
            batch = IngestBatchORM(
                source="isacademia",
                file_type="registrations",
                parser_version=parser_version,
                filename=filename,
                total_rows=len(records),
                new_rows=0,
                updated_rows=0,
                ingested_at=now,
            )
            session.add(batch)

            if ext_ids or stage_ids:
                stmt = select(OfferORM).where(
                    (OfferORM.external_id.in_(ext_ids)) | (OfferORM.stage_id.in_(stage_ids))
                )
                matched_offers = list((await session.execute(stmt)).scalars().all())
                for offer in matched_offers:
                    state = await session.get(UserOfferStateORM, (user_id, offer.uid))
                    if state is None:
                        session.add(
                            UserOfferStateORM(
                                user_id=user_id,
                                offer_uid=offer.uid,
                                status=OfferStatus.APPLIED.value,
                                is_starred=False,
                                updated_at=now,
                            )
                        )
                        marked += 1
                    elif state.status != OfferStatus.APPLIED.value:
                        state.status = OfferStatus.APPLIED.value
                        state.updated_at = now
                        marked += 1

            batch.updated_rows = marked
            await session.commit()

        stats = await self.get_user_stats(user_id)
        return IngestSummary(
            file_type="registrations",
            parser_version=parser_version,
            total_parsed=len(records),
            auto_marked_applied=marked,
            eligible_new_count=stats.new_count,
            eligible_seen_count=stats.seen_count,
        )

    # ------------------------------------------------------------------
    # Querying, State Transitions & Score Caching
    # ------------------------------------------------------------------

    async def list_enriched_offers(
        self,
        user_id: int,
        *,
        only_latest_showroom: bool = True,
        statuses: set[OfferStatus] | None = None,
        starred_only: bool = False,
    ) -> list[EnrichedOffer]:
        async with self._sessions() as session:
            stmt = (
                select(OfferORM, UserOfferStateORM)
                .outerjoin(
                    UserOfferStateORM,
                    (UserOfferStateORM.offer_uid == OfferORM.uid)
                    & (UserOfferStateORM.user_id == user_id),
                )
            )
            if only_latest_showroom:
                stmt = stmt.where(OfferORM.in_latest_showroom.is_(True))

            rows = (await session.execute(stmt)).all()
            result: list[EnrichedOffer] = []
            for offer_orm, state_orm in rows:
                status = (
                    OfferStatus(state_orm.status) if state_orm is not None else OfferStatus.NEW
                )
                is_starred = bool(state_orm.is_starred) if state_orm is not None else False
                if statuses is not None and status not in statuses:
                    continue
                if starred_only and not is_starred:
                    continue
                result.append(self._to_enriched(offer_orm, state_orm))
            return result

    async def get_enriched_offer(self, user_id: int, offer_uid: str) -> EnrichedOffer | None:
        async with self._sessions() as session:
            offer_orm = await session.get(OfferORM, offer_uid)
            if offer_orm is None:
                return None
            state_orm = await session.get(UserOfferStateORM, (user_id, offer_uid))
            return self._to_enriched(offer_orm, state_orm)

    async def set_offer_status(
        self,
        user_id: int,
        offer_uid: str,
        status: OfferStatus,
    ) -> EnrichedOffer | None:
        async with self._sessions() as session:
            offer_orm = await session.get(OfferORM, offer_uid)
            if offer_orm is None:
                return None
            state_orm = await session.get(UserOfferStateORM, (user_id, offer_uid))
            now = _utcnow()
            if state_orm is None:
                state_orm = UserOfferStateORM(
                    user_id=user_id,
                    offer_uid=offer_uid,
                    status=status.value,
                    is_starred=False,
                    updated_at=now,
                )
                session.add(state_orm)
            else:
                state_orm.status = status.value
                state_orm.updated_at = now
            await session.commit()
            return self._to_enriched(offer_orm, state_orm)

    async def toggle_star(self, user_id: int, offer_uid: str) -> EnrichedOffer | None:
        async with self._sessions() as session:
            offer_orm = await session.get(OfferORM, offer_uid)
            if offer_orm is None:
                return None
            state_orm = await session.get(UserOfferStateORM, (user_id, offer_uid))
            now = _utcnow()
            if state_orm is None:
                state_orm = UserOfferStateORM(
                    user_id=user_id,
                    offer_uid=offer_uid,
                    status=OfferStatus.NEW.value,
                    is_starred=True,
                    updated_at=now,
                )
                session.add(state_orm)
            else:
                state_orm.is_starred = not state_orm.is_starred
                state_orm.updated_at = now
            await session.commit()
            return self._to_enriched(offer_orm, state_orm)

    async def mark_offers_seen(self, user_id: int, offer_uids: list[str]) -> int:
        """Transition displayed `new` offers to `seen` without overwriting applied/excluded."""
        if not offer_uids:
            return 0
        now = _utcnow()
        async with self._sessions() as session:
            res = await session.execute(
                update(UserOfferStateORM)
                .where(
                    UserOfferStateORM.user_id == user_id,
                    UserOfferStateORM.offer_uid.in_(offer_uids),
                    UserOfferStateORM.status == OfferStatus.NEW.value,
                )
                .values(status=OfferStatus.SEEN.value, updated_at=now)
            )
            await session.commit()
            return int(res.rowcount or 0)

    async def save_scores(self, user_id: int, scores: list[OfferScore]) -> None:
        if not scores:
            return
        now = _utcnow()
        async with self._sessions() as session:
            for item in scores:
                state_orm = await session.get(UserOfferStateORM, (user_id, item.offer_uid))
                if state_orm is None:
                    state_orm = UserOfferStateORM(
                        user_id=user_id,
                        offer_uid=item.offer_uid,
                        status=OfferStatus.NEW.value,
                        is_starred=False,
                        updated_at=now,
                    )
                    session.add(state_orm)
                state_orm.score = item.score
                state_orm.score_engine = item.score_engine
                state_orm.scoring_fingerprint = item.scoring_fingerprint
                state_orm.location_match = item.location_match
                state_orm.match_summary = item.match_summary
                state_orm.tags_json = json.dumps(item.tags, ensure_ascii=False)
                state_orm.updated_at = now
            await session.commit()

    # ------------------------------------------------------------------
    # Vector Cache Helpers (Mode A Local FastEmbed)
    # ------------------------------------------------------------------

    async def load_embedding_cache(
        self,
        user_id: int,
        offer_uids: list[str],
        model_name: str,
    ) -> tuple[np.ndarray | None, dict[str, np.ndarray]]:
        async with self._sessions() as session:
            user_row = await session.get(UserConfigORM, user_id)
            profile_vec: np.ndarray | None = None
            if (
                user_row is not None
                and user_row.profile_embedding is not None
                and user_row.profile_embedding_model == model_name
            ):
                profile_vec = bytes_to_vector(user_row.profile_embedding)

            offer_vecs: dict[str, np.ndarray] = {}
            if offer_uids:
                rows = (
                    await session.execute(
                        select(OfferORM.uid, OfferORM.embedding).where(
                            OfferORM.uid.in_(offer_uids),
                            OfferORM.embedding.is_not(None),
                            OfferORM.embedding_model == model_name,
                        )
                    )
                ).all()
                for uid, blob in rows:
                    if blob:
                        offer_vecs[uid] = bytes_to_vector(blob)
            return profile_vec, offer_vecs

    async def save_embedding_cache(
        self,
        user_id: int,
        *,
        model_name: str,
        profile_vec: np.ndarray,
        new_offer_vecs: dict[str, np.ndarray],
    ) -> None:
        now = _utcnow()
        async with self._sessions() as session:
            user_row = await session.get(UserConfigORM, user_id)
            if user_row is None:
                user_row = UserConfigORM(user_id=user_id, updated_at=now)
                session.add(user_row)
            user_row.profile_embedding = vector_to_bytes(profile_vec)
            user_row.profile_embedding_model = model_name
            user_row.updated_at = now

            for uid, vec in new_offer_vecs.items():
                offer_row = await session.get(OfferORM, uid)
                if offer_row is not None:
                    offer_row.embedding = vector_to_bytes(vec)
                    offer_row.embedding_model = model_name

            await session.commit()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    async def get_user_stats(self, user_id: int) -> UserStats:
        async with self._sessions() as session:
            total = int(
                (
                    await session.execute(
                        select(func.count(OfferORM.uid)).where(OfferORM.in_latest_showroom.is_(True))
                    )
                ).scalar_one()
                or 0
            )
            counts_rows = (
                await session.execute(
                    select(UserOfferStateORM.status, func.count(UserOfferStateORM.offer_uid))
                    .join(OfferORM, OfferORM.uid == UserOfferStateORM.offer_uid)
                    .where(
                        UserOfferStateORM.user_id == user_id,
                        OfferORM.in_latest_showroom.is_(True),
                    )
                    .group_by(UserOfferStateORM.status)
                )
            ).all()
            by_status = {status: int(cnt) for status, cnt in counts_rows}

            starred = int(
                (
                    await session.execute(
                        select(func.count(UserOfferStateORM.offer_uid)).where(
                            UserOfferStateORM.user_id == user_id,
                            UserOfferStateORM.is_starred.is_(True),
                        )
                    )
                ).scalar_one()
                or 0
            )

            last_batch = (
                await session.execute(
                    select(IngestBatchORM).order_by(IngestBatchORM.ingested_at.desc()).limit(1)
                )
            ).scalar_one_or_none()

            return UserStats(
                total_in_showroom=total,
                new_count=by_status.get(OfferStatus.NEW.value, 0),
                seen_count=by_status.get(OfferStatus.SEEN.value, 0),
                applied_count=by_status.get(OfferStatus.APPLIED.value, 0),
                excluded_count=by_status.get(OfferStatus.EXCLUDED.value, 0),
                starred_count=starred,
                last_ingest_at=last_batch.ingested_at if last_batch else None,
                last_parser_version=last_batch.parser_version if last_batch else "none",
            )

    # ------------------------------------------------------------------
    # Converters
    # ------------------------------------------------------------------

    def _orm_to_profile(self, row: UserConfigORM) -> UserProfile:
        return UserProfile(
            user_id=row.user_id,
            target_roles=row.target_roles,
            preferred_locations=row.preferred_locations,
            accepted_formats=json.loads(row.accepted_formats_json or "[]"),
            dealbreakers=row.dealbreakers,
            min_score_threshold=row.min_score_threshold,
            cv_text=self._decrypt_cv(row.user_id, row.cv_text),
            cv_filename=row.cv_filename,
        )

    @staticmethod
    def _to_enriched(offer_orm: OfferORM, state_orm: UserOfferStateORM | None) -> EnrichedOffer:
        offer = JobOffer(
            uid=offer_orm.uid,
            source=offer_orm.source,
            external_id=offer_orm.external_id,
            stage_id=offer_orm.stage_id,
            title=offer_orm.title,
            company=offer_orm.company,
            company_hq=offer_orm.company_hq,
            assignment_location=offer_orm.assignment_location,
            label_durable=json.loads(offer_orm.label_durable_json or "[]"),
            format=offer_orm.format,
            registered=offer_orm.last_registered,
            places=offer_orm.last_places,
            prof=offer_orm.prof,
            creation_date=offer_orm.created_at,
            description=offer_orm.description,
            has_description=offer_orm.has_description,
            parser_version=offer_orm.parser_version,
        )
        if state_orm is None:
            return EnrichedOffer(offer=offer, first_seen_at=offer_orm.first_seen_at)
        return EnrichedOffer(
            offer=offer,
            first_seen_at=offer_orm.first_seen_at,
            status=OfferStatus(state_orm.status),
            is_starred=bool(state_orm.is_starred),
            score=state_orm.score,
            score_engine=state_orm.score_engine,
            scoring_fingerprint=state_orm.scoring_fingerprint,
            location_match=state_orm.location_match,
            match_summary=state_orm.match_summary or "",
            tags=json.loads(state_orm.tags_json or "[]"),
        )
