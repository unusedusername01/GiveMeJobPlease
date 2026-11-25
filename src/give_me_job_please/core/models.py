"""Canonical source-agnostic domain schemas shared across parsers, scorers, and storage."""

from __future__ import annotations

import hashlib
from enum import StrEnum

from pydantic import BaseModel, Field


class OfferStatus(StrEnum):
    """Mutually exclusive lifecycle state for a job/internship offer."""

    NEW = "new"
    SEEN = "seen"
    APPLIED = "applied"
    EXCLUDED = "excluded"


class JobOffer(BaseModel):
    """Canonical offer model decoupled from any specific university portal or job board."""

    uid: str  # Source-prefixed unique key, e.g., "isacademia:60677"
    source: str = "isacademia"
    external_id: str  # e.g., internship_number "60677"
    stage_id: str = ""  # Internal portal ID, e.g., "5608120730"
    title: str
    company: str
    company_hq: str = ""
    assignment_location: str = ""
    label_durable: list[str] = Field(default_factory=list)
    format: str = ""
    registered: int | None = None
    places: int | None = None
    prof: str = ""
    creation_date: str = ""  # ISO YYYY-MM-DD when parseable
    description: str = ""
    has_description: bool = False
    parser_version: str = ""

    def embedding_text(self) -> str:
        """Build rich text representation for multilingual embedding or LLM prompts."""
        parts = [
            f"Title: {self.title}",
            f"Company: {self.company}",
            f"Location: {self.assignment_location or self.company_hq or 'Unspecified'}",
            f"Format: {self.format or 'Unspecified'}",
        ]
        if self.prof:
            parts.append(f"Supervisor: {self.prof}")
        if self.label_durable:
            parts.append(f"Sustainability: {', '.join(self.label_durable)}")
        if self.description:
            parts.append(f"Description: {self.description}")
        else:
            parts.append("Description: [No description embedded in portal export]")
        return "\n".join(parts)

    @property
    def competition_ratio(self) -> float | None:
        if self.registered is None or self.places is None or self.places <= 0:
            return None
        return round(self.registered / self.places, 2)


class RegistrationRecord(BaseModel):
    """Parsed entry from IS-Academia Registrations (stu).htm used to auto-mark applied offers."""

    internship_number: str = ""
    stage_id: str = ""
    title: str = ""
    company: str = ""
    status_text: str = ""


class UserProfile(BaseModel):
    """Structured user matching criteria + extracted CV context."""

    user_id: int
    target_roles: str = ""
    preferred_locations: str = ""
    accepted_formats: list[str] = Field(default_factory=list)
    dealbreakers: str = ""
    min_score_threshold: float | None = Field(default=None, ge=0.0, le=10.0)
    cv_text: str = ""
    cv_filename: str = ""

    def is_configured(self) -> bool:
        return bool(
            self.target_roles.strip()
            or self.preferred_locations.strip()
            or self.dealbreakers.strip()
            or self.cv_text.strip()
        )

    def profile_hash(self) -> str:
        # Intentionally excludes `min_score_threshold`: changing a post-scoring
        # display cutoff must never invalidate cached LLM scores or trigger re-billing.
        payload = "|".join(
            [
                self.target_roles.strip(),
                self.preferred_locations.strip(),
                ",".join(sorted(f.strip().lower() for f in self.accepted_formats)),
                self.dealbreakers.strip(),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def cv_hash(self) -> str:
        return hashlib.sha256(self.cv_text.strip().encode("utf-8")).hexdigest()[:16]

    def render_for_matching(self, max_cv_chars: int = 4500) -> str:
        """Render combined structured profile + CV text for embedding or LLM judging."""
        sections: list[str] = []
        if self.target_roles.strip():
            sections.append(f"Target roles & domains: {self.target_roles.strip()}")
        if self.preferred_locations.strip():
            sections.append(f"Preferred locations & remote policy: {self.preferred_locations.strip()}")
        if self.accepted_formats:
            sections.append(f"Accepted formats: {', '.join(self.accepted_formats)}")
        if self.dealbreakers.strip():
            sections.append(f"Languages & hard dealbreakers: {self.dealbreakers.strip()}")
        if self.cv_text.strip():
            cv_snippet = self.cv_text.strip()[:max_cv_chars]
            sections.append(f"Candidate CV:\n{cv_snippet}")
        if not sections:
            return "General engineering / computer science student looking for relevant internships."
        return "\n\n".join(sections)


class OfferScore(BaseModel):
    """Normalized scoring result produced by either Mode A (Local) or Mode B (OpenRouter)."""

    offer_uid: str
    score: float  # Normalized 0.0..10.0 scale (Mode A maps cosine similarity to 0..10)
    score_engine: str  # "openrouter" or "local_fastembed"
    scoring_fingerprint: str
    location_match: bool | None = None
    match_summary: str = ""
    tags: list[str] = Field(default_factory=list)


class EnrichedOffer(BaseModel):
    """Combined offer + user state + latest score for Telegram rendering and exports."""

    offer: JobOffer
    status: OfferStatus = OfferStatus.NEW
    is_starred: bool = False
    score: float | None = None
    score_engine: str | None = None
    scoring_fingerprint: str | None = None
    location_match: bool | None = None
    match_summary: str = ""
    tags: list[str] = Field(default_factory=list)


class IngestSummary(BaseModel):
    """Summary returned after ingesting a showroom or registrations .htm file."""

    file_type: str  # "showroom" or "registrations"
    parser_version: str
    total_parsed: int
    new_offers: int = 0
    updated_offers: int = 0
    eligible_new_count: int = 0
    eligible_seen_count: int = 0
    auto_marked_applied: int = 0
