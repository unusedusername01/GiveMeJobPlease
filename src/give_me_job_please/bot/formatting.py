"""Safe HTML rendering and inline keyboards for Telegram (respecting 4,096-char & 64-byte limits)."""

from __future__ import annotations

import html

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from give_me_job_please.core.models import EnrichedOffer, OfferStatus, UserProfile

_STATUS_BADGE = {
    OfferStatus.NEW: "🆕",
    OfferStatus.SEEN: "👀",
    OfferStatus.APPLIED: "✅",
    OfferStatus.EXCLUDED: "🚫",
}


def esc(text: str) -> str:
    """Escape dynamic text for Telegram HTML parse mode."""
    return html.escape(text or "", quote=False)


def format_offer_card(item: EnrichedOffer) -> str:
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

    lines = [
        f"{badge} {star}<b>{esc(o.title)}</b>",
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


def build_offer_keyboard(item: EnrichedOffer) -> InlineKeyboardMarkup:
    """Build per-offer inline action buttons (all callback_data strictly < 64 bytes)."""
    uid = item.offer.uid
    star_label = "★ Unstar" if item.is_starred else "☆ Star"

    row1 = [
        InlineKeyboardButton(text=star_label, callback_data=f"o:s:{uid}"),
        InlineKeyboardButton(text="📄 Details", callback_data=f"o:d:{uid}"),
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

    return InlineKeyboardMarkup(inline_keyboard=[row1, row2])


def build_pagination_keyboard(
    *,
    view: str,
    page: int,
    total_pages: int,
    seen_count: int = 0,
) -> InlineKeyboardMarkup:
    """Build navigation footer with Prev/Next and optional 'Include Seen' switch."""
    rows: list[list[InlineKeyboardButton]] = []
    nav: list[InlineKeyboardButton] = []
    if page > 0:
        nav.append(InlineKeyboardButton(text="◀️ Prev", callback_data=f"pg:{view}:{page - 1}"))
    nav.append(InlineKeyboardButton(text=f"Page {page + 1}/{total_pages}", callback_data="noop"))
    if page + 1 < total_pages:
        nav.append(InlineKeyboardButton(text="Next ▶️", callback_data=f"pg:{view}:{page + 1}"))
    if nav:
        rows.append(nav)

    if view == "new" and seen_count > 0:
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"👀 Include Seen ({seen_count}) — Show All Eligible",
                    callback_data="pg:all:0",
                )
            ]
        )
    elif view == "all":
        rows.append(
            [
                InlineKeyboardButton(
                    text="🆕 Show New Only",
                    callback_data="pg:new:0",
                )
            ]
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
