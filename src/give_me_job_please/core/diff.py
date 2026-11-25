"""Deterministic Eligible Pool, pre-scoring filters, threshold filtering, and ranking logic."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Literal

from give_me_job_please.core.models import EnrichedOffer, OfferStatus
from give_me_job_please.errors import ValidationError

RecencyBasis = Literal["posted", "ingested"]

_DURATION_RE = re.compile(r"^(\d+)(d|w|m)$")


@dataclass(frozen=True, slots=True)
class PreScoringFilter:
    """
    Deterministic filter specification executed BEFORE OpenRouter or local scoring.

    Architecture note (future extensibility):
    Keep cheap, deterministic pre-LLM filters here so excluded offers never consume
    paid LLM tokens or embedding compute. Future filters — such as minimum salary,
    company allow/block lists, contract format, or location allowlists — should be
    added as optional fields on this dataclass and evaluated inside
    `apply_pre_scoring_filters`.
    """

    max_age_days: int | None = None
    recency_basis: RecencyBasis = "posted"


@dataclass(frozen=True, slots=True)
class DigestCommandArgs:
    """Parsed CLI options for /digest and /browse commands."""

    max_age_days: int | None = None
    recency_basis: RecencyBasis = "posted"
    force_local: bool = False


def parse_digest_args(raw_args: str) -> DigestCommandArgs:
    """
    Parse order-independent /digest arguments such as:
        '', 'local', '7d', '3d local', '14d ingested', '30d posted local', 'all'
    """
    max_age_days: int | None = None
    recency_basis: RecencyBasis = "posted"
    force_local = False

    tokens = [t.strip().lower() for t in raw_args.split() if t.strip()]
    for token in tokens:
        if token == "local":
            force_local = True
            continue
        if token in {"all", "any"}:
            max_age_days = None
            continue
        if token in {"posted", "created", "date"}:
            recency_basis = "posted"
            continue
        if token in {"ingested", "uploaded", "synced", "seen_at"}:
            recency_basis = "ingested"
            continue
        match = _DURATION_RE.match(token)
        if match:
            amount = int(match.group(1))
            unit = match.group(2)
            multiplier = 1 if unit == "d" else (7 if unit == "w" else 30)
            days = amount * multiplier
            if days < 1 or days > 3650:
                raise ValidationError("Recency window must be between 1d and 3650d (or 'all').")
            max_age_days = days
            continue
        raise ValidationError(
            f"Unknown option '{token}'. Examples: /digest 7d, /digest 3d local, "
            "/digest 14d ingested, /digest all."
        )

    return DigestCommandArgs(
        max_age_days=max_age_days,
        recency_basis=recency_basis,
        force_local=force_local,
    )


def _normalize_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _parse_iso_date_only(raw: str) -> date | None:
    cleaned = raw.strip()
    if not cleaned:
        return None
    try:
        return date.fromisoformat(cleaned[:10])
    except ValueError:
        return None


def _matches_recency(
    item: EnrichedOffer,
    *,
    max_age_days: int,
    basis: RecencyBasis,
    now_utc: datetime,
) -> bool:
    cutoff_dt = now_utc - timedelta(days=max_age_days)
    if basis == "posted":
        posted_date = _parse_iso_date_only(item.offer.creation_date)
        if posted_date is not None:
            # Calendar-date granularity: avoid dropping offers posted exactly N days ago.
            return (now_utc.date() - posted_date).days <= max_age_days
        # Graceful fallback to upload timestamp when creation_date is missing/unparseable.
        if item.first_seen_at is not None:
            return _normalize_utc(item.first_seen_at) >= cutoff_dt
        return False

    if item.first_seen_at is not None:
        return _normalize_utc(item.first_seen_at) >= cutoff_dt
    return False


def apply_pre_scoring_filters(
    offers: list[EnrichedOffer],
    flt: PreScoringFilter,
    *,
    now: datetime | None = None,
) -> list[EnrichedOffer]:
    """
    Apply deterministic filters BEFORE LLM or embedding scoring.

    Currently supports recency filtering (`max_age_days` against `posted` creation_date
    or `ingested` first_seen_at). Extend here for future pre-LLM filters (e.g., salary,
    company, or format rules).
    """
    result = list(offers)
    if flt.max_age_days is not None:
        now_utc = _normalize_utc(now) if now is not None else datetime.now(UTC)
        result = [
            item
            for item in result
            if _matches_recency(
                item,
                max_age_days=flt.max_age_days,
                basis=flt.recency_basis,
                now_utc=now_utc,
            )
        ]
    return result


def format_offer_age(
    item: EnrichedOffer,
    *,
    basis: RecencyBasis = "posted",
    now: datetime | None = None,
) -> str:
    """Return a compact relative age label like 'today', '3d', '2w', or '' if unknown."""
    now_utc = _normalize_utc(now) if now is not None else datetime.now(UTC)
    days: int | None = None
    if basis == "posted":
        posted_date = _parse_iso_date_only(item.offer.creation_date)
        if posted_date is not None:
            days = max(0, (now_utc.date() - posted_date).days)
        elif item.first_seen_at is not None:
            days = max(0, (now_utc - _normalize_utc(item.first_seen_at)).days)
    elif item.first_seen_at is not None:
        days = max(0, (now_utc - _normalize_utc(item.first_seen_at)).days)

    if days is None:
        return ""
    if days == 0:
        return "today"
    if days < 14:
        return f"{days}d"
    if days < 60:
        return f"{days // 7}w"
    return f"{days // 30}mo"


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
