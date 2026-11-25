"""Scorer Protocol and deterministic score-cache fingerprinting."""

from __future__ import annotations

import hashlib
from typing import Protocol

from give_me_job_please.core.models import JobOffer, OfferScore, UserProfile


def compute_scoring_fingerprint(
    *,
    profile_hash: str,
    cv_hash: str,
    engine: str,
    model_id: str,
    prompt_version: str,
) -> str:
    """
    Deterministic cache key for offer scores.
    Changing profile, CV, scorer engine, model ID, or prompt version invalidates stale scores.
    """
    raw = "|".join([profile_hash, cv_hash, engine, model_id, prompt_version])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


class OfferScorer(Protocol):
    """Unified interface for both local embedding and remote LLM scorers."""

    engine_name: str
    model_id: str
    prompt_version: str

    async def score_offers(
        self,
        profile: UserProfile,
        offers: list[JobOffer],
        fingerprint: str,
    ) -> list[OfferScore]: ...
