"""Safe HTML rendering and inline keyboards for Telegram (respecting 4,096-char & 64-byte limits)."""

from __future__ import annotations

import html
from dataclasses import dataclass, replace

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from give_me_job_please.core.diff import RecencyBasis, format_offer_age
from give_me_job_please.core.models import EnrichedOffer, OfferStatus, UserProfile
from give_me_job_please.errors import ValidationError

_STATUS_BADGE = {
    OfferStatus.NEW: "🆕",
    OfferStatus.SEEN: "👀",
    OfferStatus.APPLIED: "✅",
    OfferStatus.EXCLUDED: "🚫",
}

_VIEW_TO_CODE = {
    "new": "n",
    "all": "a",
    "seen": "s",
    "starred": "st",
    "applied": "ap",
    "excluded": "ex",
}
_CODE_TO_VIEW = {v: k for k, v in _VIEW_TO_CODE.items()}

RECENCY_PRESETS: tuple[tuple[str, int | None], ...] = (
    ("All", None),
    ("3d", 3),
    ("7d", 7),
    ("14d", 14),
    ("30d", 30),
)


@dataclass(frozen=True, slots=True)
class DigestQuery:
    """Stateless query parameters encoded into Telegram callback_data (< 64 bytes)."""

    view: str = "new"
    page: int = 0
    max_age_days: int | None = None
    recency_basis: RecencyBasis = "posted"
    ignore_threshold: bool = False
    force_local: bool = False


def encode_query_callback(prefix: str, query: DigestQuery, extra: str = "") -> str:
    """Encode a DigestQuery (and optional short extra field) into compact callback_data."""
    vc = _VIEW_TO_CODE.get(query.view, query.view)
    days = query.max_age_days if query.max_age_days is not None else 0
    basis = "i" if query.recency_basis == "ingested" else "p"
    thresh = "a" if query.ignore_threshold else "p"
    local = "1" if query.force_local else "0"
    tail = f"{vc}:{max(0, query.page)}:{days}:{basis}:{thresh}:{local}"
    if extra:
        return f"{prefix}:{extra}:{tail}"
    return f"{prefix}:{tail}"


def decode_query_callback(data: str) -> tuple[str, DigestQuery, str]:
    """
    Decode compact callback_data created by `encode_query_callback`.

    Returns (prefix, query, extra). Also accepts legacy `pg:<view>:<page>` callbacks.
    """
    parts = data.split(":")
    if len(parts) == 3 and parts[0] == "pg":
        try:
            page = max(0, int(parts[2]))
        except ValueError as exc:
            raise ValidationError("Invalid pagination data. Please run the command again.") from exc
        view = _CODE_TO_VIEW.get(parts[1], parts[1])
        return "pg", DigestQuery(view=view, page=page), ""

    if len(parts) == 7:
        prefix, vc, raw_page, raw_days, raw_basis, raw_thresh, raw_local = parts
        extra = ""
    elif len(parts) == 8:
        prefix, extra, vc, raw_page, raw_days, raw_basis, raw_thresh, raw_local = parts
    else:
        raise ValidationError("Invalid button state. Please run /digest again.")

    try:
        page = max(0, int(raw_page))
        days_int = int(raw_days)
    except ValueError as exc:
        raise ValidationError("Invalid button state. Please run /digest again.") from exc

    view = _CODE_TO_VIEW.get(vc, vc)
    recency_basis: RecencyBasis = "ingested" if raw_basis == "i" else "posted"
    return (
        prefix,
        DigestQuery(
            view=view,
            page=page,
            max_age_days=days_int if days_int > 0 else None,
            recency_basis=recency_basis,
            ignore_threshold=(raw_thresh == "a"),
            force_local=(raw_local == "1"),
        ),
        extra,
    )


def format_recency_label(max_age_days: int | None, basis: RecencyBasis) -> str:
    """Human-readable label for the active recency filter."""
    if max_age_days is None:
        return "All dates"
    basis_txt = "uploaded" if basis == "ingested" else "posted"
    return f"≤{max_age_days}d ({basis_txt})"


def esc(text: str) -> str:
    """Escape dynamic text for Telegram HTML parse mode."""
    return html.escape(text or "", quote=False)


def _truncate(text: str, max_len: int) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max(1, max_len - 1)].rstrip() + "…"


def format_digest_list(
    *,
    title: str,
    items: list[EnrichedOffer],
    total_count: int,
    page: int,
    page_size: int,
    total_pages: int,
    query: DigestQuery,
    engine: str,
    model_id: str,
    filtered_out_pre: int = 0,
    active_threshold: float | None = None,
    hidden_below_threshold: int = 0,
) -> str:
    """Render a compact, single-message ranked list of offers for in-place Telegram browsing."""
    start_num = page * page_size + 1
    end_num = start_num + len(items) - 1
    engine_badge = (
        f"🤖 <code>{esc(model_id)}</code>"
        if engine == "openrouter"
        else f"🧠 <code>{esc(model_id)}</code>"
    )

    meta_parts: list[str] = [engine_badge]
    if query.max_age_days is not None:
        rec = format_recency_label(query.max_age_days, query.recency_basis)
        if filtered_out_pre > 0:
            rec += f" · {filtered_out_pre} skipped"
        meta_parts.append(f"📅 {esc(rec)}")
    if active_threshold is not None:
        t_txt = f"🎯 ≥{active_threshold:g}"
        if hidden_below_threshold > 0:
            t_txt += f" ({hidden_below_threshold} hidden)"
        meta_parts.append(t_txt)

    lines = [
        f"<b>{title}</b> — <b>{start_num}–{end_num}</b> of <b>{total_count}</b> "
        f"(Page {page + 1}/{total_pages})",
        " · ".join(meta_parts),
        "",
    ]

    for idx, item in enumerate(items, start=1):
        o = item.offer
        badge = _STATUS_BADGE.get(item.status, "")
        star = "⭐" if item.is_starred else ""
        score_txt = f"<b>{item.score:.1f}</b>" if item.score is not None else "<i>—</i>"
        loc = _truncate(o.assignment_location or o.company_hq or "Unspecified", 24)
        loc_flag = " ✓" if item.location_match is True else (" ⚠️" if item.location_match is False else "")
        age = format_offer_age(item, basis=query.recency_basis)
        age_part = f" · {esc(age)}" if age else ""
        comp_part = ""
        if o.registered is not None and o.places is not None:
            comp_part = f" · 👥 {o.registered}/{o.places}"

        lines.append(
            f"<b>{idx}.</b> {badge}{star} [{score_txt}] <b>{esc(_truncate(o.title, 58))}</b>\n"
            f"    {esc(_truncate(o.company, 28))} · {esc(loc)}{loc_flag}{age_part}{comp_part}"
        )

    lines.append("")
    lines.append("<i>Tap a number below to inspect an offer in-place.</i>")
    return "\n".join(lines)


def format_offer_card(
    item: EnrichedOffer,
    *,
    position_label: str = "",
) -> str:
    """Format a single internship offer as a compact, scannable Telegram HTML card."""
    o = item.offer
    badge = _STATUS_BADGE.get(item.status, "")
    star = "⭐ " if item.is_starred else ""
    score_txt = f"<b>{item.score:.1f}/10</b>" if item.score is not None else "<i>unscored</i>"

    loc_parts = [p for p in [o.assignment_location, o.company_hq] if p]
    location_line = loc_parts[0] if loc_parts else "Unspecified"
    if len(loc_parts) > 1 and loc_parts[0].lower() not in loc_parts[1].lower():
        location_line = f"{loc_parts[0]} (HQ: {loc_parts[1]})"

    loc_flag = ""
    if item.location_match is True:
        loc_flag = " ✓"
    elif item.location_match is False:
        loc_flag = " ⚠️"

    comp = ""
    if o.registered is not None and o.places is not None:
        ratio = o.competition_ratio
        ratio_txt = f" ({ratio:.1f}x)" if ratio is not None else ""
        comp = f" | 👥 {o.registered}/{o.places}{ratio_txt}"

    pos_prefix = f"<code>[{esc(position_label)}]</code> " if position_label else ""
    lines = [
        f"{pos_prefix}{badge} {star}<b>{esc(o.title)}</b>",
        f"🏢 {esc(o.company)} · 📍 {esc(location_line)}{loc_flag}",
        f"🎯 {score_txt} · 🏷 {esc(o.format or 'N/A')} · #{esc(o.external_id)}{comp}",
    ]
    if o.creation_date:
        lines[-1] += f" · 📅 {esc(o.creation_date)}"
    if o.prof:
        lines.append(f"🎓 {esc(o.prof)}")
    if item.match_summary:
        lines.append(f"💬 <i>{esc(item.match_summary)}</i>")
    if item.tags:
        tag_str = " ".join(f"<code>{esc(t)}</code>" for t in item.tags[:4])
        lines.append(tag_str)
    if not o.has_description:
        lines.append("⚠️ <i>No description embedded in IS-Academia export (check portal).</i>")

    return "\n".join(lines)


def build_offer_keyboard(
    item: EnrichedOffer,
    *,
    index: int | None = None,
    total: int = 0,
    back_query: DigestQuery | None = None,
) -> InlineKeyboardMarkup:
    """Build per-offer inline action buttons (all callback_data strictly < 64 bytes)."""
    uid = item.offer.uid
    star_label = "★ Unstar" if item.is_starred else "☆ Star"

    row1 = [
        InlineKeyboardButton(text=star_label, callback_data=f"o:s:{uid}"),
        InlineKeyboardButton(text="📄 Full Description", callback_data=f"od:0:{uid}"),
    ]

    if item.status in {OfferStatus.EXCLUDED, OfferStatus.APPLIED}:
        row2 = [
            InlineKeyboardButton(text="↩️ Restore to Seen", callback_data=f"o:r:{uid}"),
        ]
        if item.status != OfferStatus.APPLIED:
            row2.append(InlineKeyboardButton(text="✅ Applied", callback_data=f"o:a:{uid}"))
    else:
        row2 = [
            InlineKeyboardButton(text="✅ Applied", callback_data=f"o:a:{uid}"),
            InlineKeyboardButton(text="❌ Exclude", callback_data=f"o:x:{uid}"),
        ]

    rows: list[list[InlineKeyboardButton]] = [row1, row2]

    if back_query is not None and index is not None and total > 1:
        nav_row: list[InlineKeyboardButton] = []
        if index > 0:
            nav_row.append(
                InlineKeyboardButton(
                    text="◀️ Prev",
                    callback_data=encode_query_callback("oc", back_query, extra=str(index - 1)),
                )
            )
        nav_row.append(InlineKeyboardButton(text=f"{index + 1}/{total}", callback_data="noop"))
        if index + 1 < total:
            nav_row.append(
                InlineKeyboardButton(
                    text="Next ▶️",
                    callback_data=encode_query_callback("oc", back_query, extra=str(index + 1)),
                )
            )
        rows.append(nav_row)

    if back_query is not None:
        rows.append(
            [
                InlineKeyboardButton(
                    text="◀️ Back to list",
                    callback_data=encode_query_callback("pg", back_query),
                )
            ]
        )

    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_digest_list_keyboard(
    *,
    query: DigestQuery,
    page_items: list[EnrichedOffer],
    page_size: int,
    total_pages: int,
    seen_count: int = 0,
    profile_threshold: float | None = None,
) -> InlineKeyboardMarkup:
    """Build single-message list keyboard: numbered item selectors, pagination, mark-seen, filters."""
    rows: list[list[InlineKeyboardButton]] = []

    # Numbered drill-down buttons (max 5 per row for comfortable mobile tapping)
    base_idx = query.page * page_size
    num_buttons = [
        InlineKeyboardButton(
            text=str(offset + 1),
            callback_data=encode_query_callback("vw", query, extra=str(base_idx + offset)),
        )
        for offset in range(len(page_items))
    ]
    for i in range(0, len(num_buttons), 5):
        rows.append(num_buttons[i : i + 5])

    if total_pages > 0:
        nav: list[InlineKeyboardButton] = []
        if query.page > 0:
            nav.append(
                InlineKeyboardButton(
                    text="◀️ Prev",
                    callback_data=encode_query_callback("pg", replace(query, page=query.page - 1)),
                )
            )
        nav.append(
            InlineKeyboardButton(text=f"Page {query.page + 1}/{total_pages}", callback_data="noop")
        )
        if query.page + 1 < total_pages:
            nav.append(
                InlineKeyboardButton(
                    text="Next ▶️",
                    callback_data=encode_query_callback("pg", replace(query, page=query.page + 1)),
                )
            )
        rows.append(nav)

    new_on_page = sum(1 for item in page_items if item.status == OfferStatus.NEW)
    if new_on_page > 0:
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"👀 Mark page seen ({new_on_page})",
                    callback_data=encode_query_callback("ms", query),
                )
            ]
        )

    rows.extend(
        build_filter_rows(
            query=query,
            seen_count=seen_count,
            profile_threshold=profile_threshold,
        )
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def format_offer_description_page(
    item: EnrichedOffer,
    *,
    part: int = 0,
    max_chunk_len: int = 3300,
) -> tuple[str, int, int]:
    """
    Format an in-place full-description view for an offer.

    Returns (html_text, clamped_part_index, total_parts).
    """
    o = item.offer
    badge = _STATUS_BADGE.get(item.status, "")
    star = "⭐ " if item.is_starred else ""
    body_raw = o.description.strip()
    chunks = chunk_text(body_raw, max_len=max_chunk_len) if body_raw else []

    total_parts = max(1, len(chunks))
    clamped = max(0, min(part, total_parts - 1))
    part_suffix = f" · Part {clamped + 1}/{total_parts}" if total_parts > 1 else ""
    header = (
        f"📄 {badge} {star}<b>{esc(o.title)}</b> — {esc(o.company)} "
        f"(#{esc(o.external_id)}{part_suffix})\n\n"
    )
    body_html = (
        esc(chunks[clamped])
        if chunks
        else "<i>No description embedded in saved .htm file (check IS-Academia portal).</i>"
    )
    return header + body_html, clamped, total_parts


def build_description_keyboard(
    item: EnrichedOffer,
    *,
    part: int,
    total_parts: int,
    back_query: DigestQuery | None = None,
) -> InlineKeyboardMarkup:
    """Build inline controls for in-place description view (chunk paging + direct offer actions)."""
    uid = item.offer.uid
    rows: list[list[InlineKeyboardButton]] = []

    if total_parts > 1:
        pager: list[InlineKeyboardButton] = []
        if part > 0:
            pager.append(
                InlineKeyboardButton(text="◀️ Prev part", callback_data=f"od:{part - 1}:{uid}")
            )
        pager.append(InlineKeyboardButton(text=f"Part {part + 1}/{total_parts}", callback_data="noop"))
        if part + 1 < total_parts:
            pager.append(
                InlineKeyboardButton(text="Next part ▶️", callback_data=f"od:{part + 1}:{uid}")
            )
        rows.append(pager)

    star_label = "★ Unstar" if item.is_starred else "☆ Star"
    actions = [InlineKeyboardButton(text=star_label, callback_data=f"of:s:{part}:{uid}")]
    if item.status in {OfferStatus.EXCLUDED, OfferStatus.APPLIED}:
        actions.append(InlineKeyboardButton(text="↩️ Restore", callback_data=f"of:r:{part}:{uid}"))
        if item.status != OfferStatus.APPLIED:
            actions.append(InlineKeyboardButton(text="✅ Applied", callback_data=f"of:a:{part}:{uid}"))
    else:
        actions.append(InlineKeyboardButton(text="✅ Applied", callback_data=f"of:a:{part}:{uid}"))
        actions.append(InlineKeyboardButton(text="❌ Exclude", callback_data=f"of:x:{part}:{uid}"))
    rows.append(actions)

    back_row = [InlineKeyboardButton(text="◀️ Back to card", callback_data=f"ob:{uid}")]
    if back_query is not None:
        back_row.append(
            InlineKeyboardButton(
                text="📋 Back to list",
                callback_data=encode_query_callback("pg", back_query),
            )
        )
    rows.append(back_row)
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_filter_rows(
    *,
    query: DigestQuery,
    seen_count: int = 0,
    profile_threshold: float | None = None,
) -> list[list[InlineKeyboardButton]]:
    """Build reusable inline recency preset, basis toggle, threshold toggle, and view switch rows."""
    rows: list[list[InlineKeyboardButton]] = []

    recency_row: list[InlineKeyboardButton] = []
    for label, days in RECENCY_PRESETS:
        active = query.max_age_days == days
        btn_text = f"• {label} •" if active else label
        target = replace(query, page=0, max_age_days=days)
        recency_row.append(
            InlineKeyboardButton(text=btn_text, callback_data=encode_query_callback("pg", target))
        )
    rows.append(recency_row)

    toggles: list[InlineKeyboardButton] = []
    if query.max_age_days is not None:
        next_basis: RecencyBasis = "ingested" if query.recency_basis == "posted" else "posted"
        basis_label = "📅 Posted" if query.recency_basis == "posted" else "📥 Uploaded"
        toggles.append(
            InlineKeyboardButton(
                text=f"{basis_label} ⇄",
                callback_data=encode_query_callback(
                    "pg", replace(query, page=0, recency_basis=next_basis)
                ),
            )
        )

    if profile_threshold is not None and query.view in {"new", "all", "seen"}:
        if query.ignore_threshold:
            thresh_label = f"🎯 All (re-enable ≥{profile_threshold:g})"
        else:
            thresh_label = f"🎯 ≥{profile_threshold:g} (show all)"
        toggles.append(
            InlineKeyboardButton(
                text=thresh_label,
                callback_data=encode_query_callback(
                    "pg", replace(query, page=0, ignore_threshold=not query.ignore_threshold)
                ),
            )
        )

    if query.view == "new" and seen_count > 0:
        toggles.append(
            InlineKeyboardButton(
                text=f"👀 +Seen ({seen_count})",
                callback_data=encode_query_callback("pg", replace(query, view="all", page=0)),
            )
        )
    elif query.view == "all":
        toggles.append(
            InlineKeyboardButton(
                text="🆕 New Only",
                callback_data=encode_query_callback("pg", replace(query, view="new", page=0)),
            )
        )

    if toggles:
        rows.append(toggles)

    return rows


def build_scoring_cancel_keyboard() -> InlineKeyboardMarkup:
    """Single cancel button shown on the in-progress scoring status message."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🛑 Cancel scoring", callback_data="sc:cancel")]
        ]
    )


def build_pagination_keyboard(
    *,
    query: DigestQuery,
    total_pages: int,
    seen_count: int = 0,
    profile_threshold: float | None = None,
) -> InlineKeyboardMarkup:
    """Build navigation footer with Prev/Next, recency presets, and view/threshold toggles."""
    rows: list[list[InlineKeyboardButton]] = []
    if total_pages > 0:
        nav: list[InlineKeyboardButton] = []
        if query.page > 0:
            nav.append(
                InlineKeyboardButton(
                    text="◀️ Prev",
                    callback_data=encode_query_callback("pg", replace(query, page=query.page - 1)),
                )
            )
        nav.append(
            InlineKeyboardButton(text=f"Page {query.page + 1}/{total_pages}", callback_data="noop")
        )
        if query.page + 1 < total_pages:
            nav.append(
                InlineKeyboardButton(
                    text="Next ▶️",
                    callback_data=encode_query_callback("pg", replace(query, page=query.page + 1)),
                )
            )
        if nav:
            rows.append(nav)

    rows.extend(
        build_filter_rows(
            query=query,
            seen_count=seen_count,
            profile_threshold=profile_threshold,
        )
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def format_threshold_label(threshold: float | None) -> str:
    """Format a minimum score threshold for display."""
    if threshold is None or threshold <= 0.0:
        return "Off (show all)"
    return f"≥ {threshold:g}/10"


def format_profile_summary(profile: UserProfile) -> str:
    """Render the user's current profile settings and CV status."""
    cv_line = (
        f"✅ <code>{esc(profile.cv_filename)}</code> ({len(profile.cv_text)} chars)"
        if profile.cv_text
        else "<i>Not uploaded</i> (/cv)"
    )
    return (
        "🧭 <b>Your Profile Settings</b>\n\n"
        f"• <b>Target roles:</b> {esc(profile.target_roles) if profile.target_roles else '<i>not set</i>'}\n"
        f"• <b>Preferred locations:</b> {esc(profile.preferred_locations) if profile.preferred_locations else '<i>not set</i>'}\n"
        f"• <b>Accepted formats:</b> {esc(', '.join(profile.accepted_formats)) if profile.accepted_formats else '<i>Any</i>'}\n"
        f"• <b>Languages &amp; dealbreakers:</b> {esc(profile.dealbreakers) if profile.dealbreakers else '<i>not set</i>'}\n"
        f"• <b>Min score threshold:</b> <code>{esc(format_threshold_label(profile.min_score_threshold))}</code>\n"
        f"• <b>CV:</b> {cv_line}\n\n"
        "<i>Shortcuts:</i> <code>/profile edit</code> · <code>/profile threshold 7</code> · <code>/profile threshold off</code>"
    )


def build_profile_keyboard() -> InlineKeyboardMarkup:
    """Inline action buttons attached to the /profile settings view."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="✏️ Edit Full Profile", callback_data="prof:edit"),
                InlineKeyboardButton(text="🎯 Set Score Threshold", callback_data="prof:thresh"),
            ]
        ]
    )


def chunk_text(text: str, max_len: int = 3800) -> list[str]:
    """Split long descriptions safely under Telegram's 4,096-character limit."""
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_len:
            chunks.append(remaining)
            break
        split_at = remaining.rfind("\n", 0, max_len)
        if split_at < max_len // 2:
            split_at = remaining.rfind(" ", 0, max_len)
        if split_at <= 0:
            split_at = max_len
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    return chunks
