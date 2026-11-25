"""Deterministic Eligible Pool, threshold filtering, and ranking logic."""

from __future__ import annotations

import math

from give_me_job_please.core.models import EnrichedOffer, OfferStatus
from give_me_job_please.errors import ValidationError


def filter_eligible_pool(
    offers: list[EnrichedOffer],
    *,
    include_seen: bool = False,
) -> list[EnrichedOffer]:
    """
    Compute the active recommendation pool:
        Eligible Pool = Current Showroom - Applied - Excluded

    When `include_seen=False` (default weekly digest), returns only `new` offers.
    When `include_seen=True`, returns both `new` and `seen` offers together.
    """
    allowed = {OfferStatus.NEW, OfferStatus.SEEN} if include_seen else {OfferStatus.NEW}
    return [item for item in offers if item.status in allowed]


def filter_by_min_score(
    offers: list[EnrichedOffer],
    min_score: float | None,
) -> list[EnrichedOffer]:
    """Keep only offers whose score meets or exceeds `min_score` (no-op when None or <= 0)."""
    if min_score is None or min_score <= 0.0:
        return list(offers)
    return [item for item in offers if item.score is not None and item.score >= min_score]


def parse_min_score_threshold(raw: str) -> float | None:
    """
    Parse user input for a minimum score threshold on the 0..10 scale.

    Accepts values like '7', '7.5', '7,5', '7+', or 'off'/'none'/'clear'/'0' to disable.
    """
    cleaned = raw.strip().lower().rstrip("+").replace(",", ".")
    if not cleaned or cleaned in {"off", "none", "clear", "disable", "disabled", "all"}:
        return None
    try:
        value = float(cleaned)
    except ValueError as exc:
        raise ValidationError(
            "Invalid score threshold. Enter a number between 0 and 10 (e.g. 6.5 or 7), "
            "or 'off' to show all offers."
        ) from exc
    if not math.isfinite(value) or value < 0.0 or value > 10.0:
        raise ValidationError("Score threshold must be between 0 and 10 (or 'off').")
    if value == 0.0:
        return None
    return round(value, 2)


def sort_by_relevance(offers: list[EnrichedOffer]) -> list[EnrichedOffer]:
    """Sort offers by descending score, then starred first, then newest creation date."""
    return sorted(
        offers,
        key=lambda item: (
            item.score if item.score is not None else -1.0,
            1 if item.is_starred else 0,
            item.offer.creation_date or "",
            item.offer.external_id,
        ),
        reverse=True,
    )
