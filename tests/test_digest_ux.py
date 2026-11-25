"""Unit tests for single-message in-place Telegram formatting, keyboards, and context cache."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

from give_me_job_please.bot.formatting import (
    DigestQuery,
    build_description_keyboard,
    build_digest_list_keyboard,
    build_offer_keyboard,
    decode_query_callback,
    format_digest_list,
    format_offer_card,
    format_offer_description_page,
)
from give_me_job_please.bot.handlers import (
    _MAX_MESSAGE_CONTEXTS,
    BotDependencies,
    DigestMessageContext,
)
from give_me_job_please.config import Settings
from give_me_job_please.core.auth.crypto import SecretBox
from give_me_job_please.core.auth.openrouter_pkce import OpenRouterPkceManager
from give_me_job_please.core.models import EnrichedOffer, JobOffer, OfferStatus
from give_me_job_please.core.scoring.local_embedder import LocalEmbedderScorer
from give_me_job_please.storage.repository import OfferRepository


def _make_enriched(
    idx: int,
    *,
    status: OfferStatus = OfferStatus.NEW,
    starred: bool = False,
    score: float = 8.4,
    description: str = "Build distributed systems in Python and Rust.",
) -> EnrichedOffer:
    return EnrichedOffer(
        offer=JobOffer(
            uid=f"isacademia:{60000 + idx}",
            external_id=str(60000 + idx),
            title=f"<Offer #{idx}> " + ("Very Long Internship Title " * 4),
            company=f"Company & Co #{idx}",
            assignment_location="Lausanne, Switzerland",
            format="Internship",
            registered=3,
            places=1,
            creation_date="2026-11-23",
            description=description,
            has_description=bool(description),
        ),
        first_seen_at=datetime.now(UTC) - timedelta(days=1),
        status=status,
        is_starred=starred,
        score=score,
        location_match=True,
        match_summary="Strong systems match",
        tags=["Python", "Rust"],
    )


def test_compact_digest_list_under_telegram_limit_and_escapes_html() -> None:
    items = [_make_enriched(i, starred=(i == 1)) for i in range(1, 16)]
    query = DigestQuery(view="new", page=0, max_age_days=7, recency_basis="posted")
    text = format_digest_list(
        title="🆕 New Eligible Offers",
        items=items,
        total_count=42,
        page=0,
        page_size=15,
        total_pages=3,
        query=query,
        engine="openrouter",
        model_id="deepseek/deepseek-chat",
        filtered_out_pre=10,
        active_threshold=7.0,
        hidden_below_threshold=5,
    )

    assert len(text) < 4096
    assert "&lt;Offer #1&gt;" in text
    assert "Company &amp; Co #1" in text
    assert "10 skipped" in text
    assert "5 hidden" in text


def test_all_inplace_keyboards_respect_64_byte_callback_limit_and_mark_seen_visibility() -> None:
    query = DigestQuery(
        view="new",
        page=1,
        max_age_days=14,
        recency_basis="ingested",
        ignore_threshold=False,
        force_local=True,
    )
    mixed_items = [
        _make_enriched(1, status=OfferStatus.NEW),
        _make_enriched(2, status=OfferStatus.SEEN),
        _make_enriched(3, status=OfferStatus.NEW),
    ]

    list_kb = build_digest_list_keyboard(
        query=query,
        page_items=mixed_items,
        page_size=5,
        total_pages=4,
        seen_count=9,
        profile_threshold=7.0,
    )
    callbacks = [
        btn.callback_data
        for row in list_kb.inline_keyboard
        for btn in row
        if btn.callback_data is not None
    ]
    assert all(len(cb.encode("utf-8")) < 64 for cb in callbacks)
    assert any(cb.startswith("ms:") for cb in callbacks)

    # First slot on page 1 (page_size=5) maps to global index 5
    prefix, decoded_q, extra = decode_query_callback(callbacks[0])
    assert prefix == "vw"
    assert extra == "5"
    assert decoded_q == query

    # When no NEW items are on the page, 'Mark page seen' button is omitted
    seen_only_kb = build_digest_list_keyboard(
        query=query,
        page_items=[_make_enriched(2, status=OfferStatus.SEEN)],
        page_size=5,
        total_pages=1,
        seen_count=1,
        profile_threshold=None,
    )
    assert not any(
        (btn.callback_data or "").startswith("ms:")
        for row in seen_only_kb.inline_keyboard
        for btn in row
    )

    # Card keyboard with contextual Prev/Next/Back
    card_kb = build_offer_keyboard(mixed_items[0], index=5, total=18, back_query=query)
    card_cbs = [btn.callback_data or "" for row in card_kb.inline_keyboard for btn in row]
    assert all(len(cb.encode("utf-8")) < 64 for cb in card_cbs)
    assert any(cb.startswith("oc:4:") for cb in card_cbs)
    assert any(cb.startswith("oc:6:") for cb in card_cbs)
    assert any(cb.startswith("pg:") for cb in card_cbs)
    assert any(cb.startswith("od:0:") for cb in card_cbs)
    assert "[6/18]" in format_offer_card(mixed_items[0], position_label="6/18")


def test_inplace_description_pagination_and_empty_description() -> None:
    long_desc = ("Paragraph with technical requirements and skills.\n" * 140).strip()
    long_item = _make_enriched(10, description=long_desc)

    page0, clamped0, total = format_offer_description_page(long_item, part=0)
    assert total >= 2
    assert clamped0 == 0
    assert len(page0) < 4096
    assert f"Part 1/{total}" in page0

    # Out-of-bounds part clamps cleanly to final chunk
    page_last, clamped_last, total_last = format_offer_description_page(long_item, part=99)
    assert total_last == total
    assert clamped_last == total - 1
    assert len(page_last) < 4096

    desc_kb = build_description_keyboard(
        long_item,
        part=0,
        total_parts=total,
        back_query=DigestQuery(view="all", page=2, max_age_days=30),
    )
    desc_cbs = [btn.callback_data or "" for row in desc_kb.inline_keyboard for btn in row]
    assert all(len(cb.encode("utf-8")) < 64 for cb in desc_cbs)
    assert any(cb.startswith("od:1:") for cb in desc_cbs)
    assert any(cb.startswith("of:s:0:") for cb in desc_cbs)
    assert any(cb.startswith("ob:") for cb in desc_cbs)

    empty_item = _make_enriched(11, description="")
    empty_html, clamped_empty, total_empty = format_offer_description_page(empty_item, part=0)
    assert clamped_empty == 0 and total_empty == 1
    assert "No description embedded" in empty_html


def test_digest_message_context_lru_is_bounded() -> None:
    deps = BotDependencies(
        settings=cast(Settings, object()),
        repo=cast(OfferRepository, object()),
        secret_box=cast(SecretBox, object()),
        pkce_manager=cast(OpenRouterPkceManager, object()),
        local_scorer=cast(LocalEmbedderScorer, object()),
        error_log_path=Path("errors.jsonl"),
    )
    for msg_id in range(_MAX_MESSAGE_CONTEXTS + 10):
        deps.set_digest_context(
            1,
            msg_id,
            DigestMessageContext(
                query=DigestQuery(view="new", page=0),
                ordered_uids=[f"isacademia:{msg_id}"],
                page_size=5,
            ),
        )

    assert len(deps.digest_contexts) == _MAX_MESSAGE_CONTEXTS
    assert deps.get_digest_context(1, 0) is None
    latest = deps.get_digest_context(1, _MAX_MESSAGE_CONTEXTS + 9)
    assert latest is not None
    assert latest.ordered_uids == [f"isacademia:{_MAX_MESSAGE_CONTEXTS + 9}"]
