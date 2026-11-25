"""Deterministic Eligible Pool & diff logic."""

from __future__ import annotations

from give_me_job_please.core.models import EnrichedOffer, OfferStatus


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
