"""All Telegram command handlers, FSM /profile wizard, file uploads, inline actions, and error handler."""

from __future__ import annotations

import io
import math
from pathlib import Path

import structlog
from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandObject
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import BufferedInputFile, CallbackQuery, ErrorEvent, Message

from give_me_job_please import __version__
from give_me_job_please.bot.formatting import (
    build_offer_keyboard,
    build_pagination_keyboard,
    chunk_text,
    esc,
    format_offer_card,
)
from give_me_job_please.config import Settings
from give_me_job_please.core.auth.crypto import SecretBox
from give_me_job_please.core.auth.openrouter_pkce import (
    OpenRouterPkceManager,
    build_key_deep_links,
    search_openrouter_models,
    verify_openrouter_model_exists,
)
from give_me_job_please.core.diff import sort_by_relevance
from give_me_job_please.core.exporter import export_csv_bytes, export_descriptions_json_bytes
from give_me_job_please.core.models import EnrichedOffer, OfferStatus
from give_me_job_please.core.parsers.cv_parser import parse_cv_bytes
from give_me_job_please.core.parsers.isacademia import (
    REGISTRATIONS_PARSER_VERSION,
    SHOWROOM_PARSER_VERSION,
    detect_isacademia_file_type,
    parse_registrations_html,
    parse_showroom_html,
)
from give_me_job_please.core.scoring.base import compute_scoring_fingerprint
from give_me_job_please.core.scoring.local_embedder import (
    LOCAL_PROMPT_VERSION,
    LocalEmbedderScorer,
)
from give_me_job_please.core.scoring.openrouter_scorer import (
    OPENROUTER_PROMPT_VERSION,
    OpenRouterScorer,
)
from give_me_job_please.errors import AppError, AuthError, ParseError, ValidationError
from give_me_job_please.logging_setup import read_recent_error_logs
from give_me_job_please.storage.repository import OfferRepository

logger = structlog.get_logger(__name__)


class ProfileWizard(StatesGroup):
    target_roles = State()
    preferred_locations = State()
    accepted_formats = State()
    dealbreakers = State()


class BotDependencies:
    """Container holding shared services injected into Telegram handlers."""

    def __init__(
        self,
        *,
        settings: Settings,
        repo: OfferRepository,
        secret_box: SecretBox,
        pkce_manager: OpenRouterPkceManager,
        local_scorer: LocalEmbedderScorer,
        error_log_path: Path,
    ) -> None:
        self.settings = settings
        self.repo = repo
        self.secret_box = secret_box
        self.pkce = pkce_manager
        self.local_scorer = local_scorer
        self.error_log_path = error_log_path


def build_router(deps: BotDependencies) -> Router:
    """Create and wire the main aiogram Router with all commands and centralized error handling."""
    router = Router(name="give_me_job_please")

    # ------------------------------------------------------------------
    # Scoring orchestration helper (OpenRouter overrides Local by default)
    # ------------------------------------------------------------------

    async def ensure_scored(
        user_id: int,
        items: list[EnrichedOffer],
        *,
        force_local: bool = False,
        status_msg: Message | None = None,
    ) -> tuple[list[EnrichedOffer], str, str]:
        """
        Score any offers whose cached `scoring_fingerprint` does not match the active setup.
        Returns (updated_items, active_engine_label, active_model_id).
        """
        profile = await deps.repo.get_or_create_profile(user_id)
        or_state = await deps.repo.get_openrouter_state(user_id)
        use_openrouter = (not force_local) and bool(or_state.key_enc)

        if use_openrouter and or_state.key_enc:
            engine = "openrouter"
            model_id = or_state.model_id or deps.settings.default_openrouter_model
            prompt_ver = OPENROUTER_PROMPT_VERSION
        else:
            engine = "local_fastembed"
            model_id = deps.local_scorer.model_id
            prompt_ver = LOCAL_PROMPT_VERSION

        fingerprint = compute_scoring_fingerprint(
            profile_hash=profile.profile_hash(),
            cv_hash=profile.cv_hash(),
            engine=engine,
            model_id=model_id,
            prompt_version=prompt_ver,
        )

        to_score = [item.offer for item in items if item.scoring_fingerprint != fingerprint]
        if to_score:
            if status_msg is not None:
                mode_str = (
                    f"OpenRouter (<code>{esc(model_id)}</code>)"
                    if engine == "openrouter"
                    else f"Local ONNX (<code>{esc(model_id)}</code>)"
                )
                await status_msg.edit_text(
                    f"⏳ Scoring <b>{len(to_score)}</b> offer(s) using {mode_str}..."
                )

            if engine == "openrouter" and or_state.key_enc:
                api_key = deps.secret_box.decrypt(
                    or_state.key_enc, associated_data=f"openrouter:{user_id}"
                )
                scorer = OpenRouterScorer(
                    api_key=api_key,
                    model_id=model_id,
                    batch_size=deps.settings.openrouter_batch_size,
                    app_title=deps.settings.openrouter_app_title,
                    app_url=deps.settings.openrouter_app_url,
                )
                new_scores = await scorer.score_offers(profile, to_score, fingerprint)
            else:
                prof_vec, offer_vecs = await deps.repo.load_embedding_cache(
                    user_id, [o.uid for o in to_score], deps.local_scorer.model_id
                )
                new_scores, updated_prof_vec, newly_embedded = (
                    await deps.local_scorer.score_with_cache(
                        profile,
                        to_score,
                        fingerprint=fingerprint,
                        cached_profile_vec=prof_vec,
                        cached_offer_vecs=offer_vecs,
                    )
                )
                await deps.repo.save_embedding_cache(
                    user_id,
                    model_name=deps.local_scorer.model_id,
                    profile_vec=updated_prof_vec,
                    new_offer_vecs=newly_embedded,
                )

            await deps.repo.save_scores(user_id, new_scores)
            score_map = {s.offer_uid: s for s in new_scores}
            for item in items:
                s = score_map.get(item.offer.uid)
                if s is not None:
                    item.score = s.score
                    item.score_engine = s.score_engine
                    item.scoring_fingerprint = s.scoring_fingerprint
                    item.location_match = s.location_match
                    item.match_summary = s.match_summary
                    item.tags = s.tags

        return items, engine, model_id

    async def render_view_page(
        target_msg: Message,
        user_id: int,
        view: str,
        page: int,
        *,
        force_local: bool = False,
        mark_new_as_seen: bool = False,
    ) -> None:
        """Load, score (if needed), sort, and send a paginated list of offer cards."""
        if view == "new":
            items = await deps.repo.list_enriched_offers(user_id, statuses={OfferStatus.NEW})
            title = "🆕 New Eligible Offers"
        elif view == "all":
            items = await deps.repo.list_enriched_offers(
                user_id, statuses={OfferStatus.NEW, OfferStatus.SEEN}
            )
            title = "📋 All Eligible Offers (New + Seen)"
        elif view == "seen":
            items = await deps.repo.list_enriched_offers(user_id, statuses={OfferStatus.SEEN})
            title = "👀 Seen Offers"
        elif view == "starred":
            items = await deps.repo.list_enriched_offers(
                user_id, only_latest_showroom=False, starred_only=True
            )
            title = "⭐ Starred Offers"
        elif view == "applied":
            items = await deps.repo.list_enriched_offers(
                user_id, only_latest_showroom=False, statuses={OfferStatus.APPLIED}
            )
            title = "✅ Applied Offers"
        elif view == "excluded":
            items = await deps.repo.list_enriched_offers(
                user_id, only_latest_showroom=False, statuses={OfferStatus.EXCLUDED}
            )
            title = "🚫 Excluded Offers"
        else:
            raise ValidationError(
                "Unknown view. Use: /browse [all|new|seen|starred|applied|excluded]"
            )

        stats = await deps.repo.get_user_stats(user_id)
        if not items:
            if view == "new" and stats.seen_count > 0:
                kb = build_pagination_keyboard(
                    view="new", page=0, total_pages=1, seen_count=stats.seen_count
                )
                await target_msg.answer(
                    f"<b>{title}</b>\nNo brand-new offers waiting. "
                    f"You have <b>{stats.seen_count}</b> seen eligible offer(s).",
                    reply_markup=kb,
                )
            else:
                await target_msg.answer(
                    f"<b>{title}</b>\nNothing here yet. Drop an IS-Academia <code>.htm</code> export anytime."
                )
            return

        status_notice = await target_msg.answer(f"⏳ Preparing <b>{len(items)}</b> offer(s)...")
        items, engine, model_id = await ensure_scored(
            user_id, items, force_local=force_local, status_msg=status_notice
        )
        ranked = sort_by_relevance(items)

        page_size = deps.settings.digest_page_size
        total_pages = max(1, math.ceil(len(ranked) / page_size))
        page = max(0, min(page, total_pages - 1))
        slice_items = ranked[page * page_size : (page + 1) * page_size]

        engine_badge = (
            f"🤖 OpenRouter (<code>{esc(model_id)}</code>)"
            if engine == "openrouter"
            else f"🧠 Local ONNX (<code>{esc(model_id)}</code>)"
        )
        await status_notice.edit_text(
            f"<b>{title}</b> — showing <b>{len(slice_items)}</b> of <b>{len(ranked)}</b> "
            f"(Page {page + 1}/{total_pages})\nEngine: {engine_badge}"
        )

        displayed_new_uids: list[str] = []
        for item in slice_items:
            await target_msg.answer(
                format_offer_card(item),
                reply_markup=build_offer_keyboard(item),
            )
            if item.status == OfferStatus.NEW:
                displayed_new_uids.append(item.offer.uid)

        if mark_new_as_seen and displayed_new_uids:
            await deps.repo.mark_offers_seen(user_id, displayed_new_uids)

        updated_stats = await deps.repo.get_user_stats(user_id)
        await target_msg.answer(
            f"Navigation ({title}):",
            reply_markup=build_pagination_keyboard(
                view=view,
                page=page,
                total_pages=total_pages,
                seen_count=updated_stats.seen_count,
            ),
        )

    # ------------------------------------------------------------------
    # Basic Commands: /start, /help, /status, /logs
    # ------------------------------------------------------------------

    @router.message(Command("start", "help"))
    async def cmd_start(message: Message) -> None:
        if message.from_user is None:
            return
        await deps.repo.get_or_create_profile(message.from_user.id)
        await message.answer(
            f"👋 <b>GiveMeJobPlease v{esc(__version__)}</b>\n\n"
            "<b>Quick workflow:</b>\n"
            "1. Set your preferences via /profile and optionally attach your CV via /cv (or drop a PDF/TXT).\n"
            "2. Optional: Connect OpenRouter via /connect_openrouter (overrides local embeddings).\n"
            "3. Save IS-Academia <code>Internships showroom (stu).htm</code> (or <code>Registrations (stu).htm</code>) "
            "and drop the file right into this chat.\n"
            "4. Run /digest to review ranked offers.\n\n"
            "<b>Commands:</b>\n"
            "/profile — Guided preference setup\n"
            "/cv — View/upload CV\n"
            "/digest [local] — Ranked new offers (or force offline ONNX)\n"
            "/browse [all|new|seen|starred|applied|excluded]\n"
            "/connect_openrouter [limit] [expiration] — OAuth PKCE\n"
            "/auth &lt;code&gt; — Complete PKCE (message auto-deleted)\n"
            "/model [query_or_id] — Pick OpenRouter model\n"
            "/disconnect_openrouter — Remove stored key\n"
            "/export — Download enriched CSV + JSON\n"
            "/status — System & DB overview\n"
            "/logs — Recent traced errors"
        )

    @router.message(Command("status"))
    async def cmd_status(message: Message) -> None:
        if message.from_user is None:
            return
        uid = message.from_user.id
        profile = await deps.repo.get_or_create_profile(uid)
        or_state = await deps.repo.get_openrouter_state(uid)
        stats = await deps.repo.get_user_stats(uid)

        if or_state.key_enc:
            model_id = or_state.model_id or deps.settings.default_openrouter_model
            engine_line = f"🤖 <b>OpenRouter (Active — overrides local)</b>\nModel: <code>{esc(model_id)}</code>"
            if or_state.key_hash:
                logs_url, settings_url = build_key_deep_links(or_state.key_hash)
                engine_line += (
                    f'\nKey links: <a href="{esc(logs_url)}">Activity</a> · '
                    f'<a href="{esc(settings_url)}">Settings/Revoke</a>'
                )
        else:
            engine_line = (
                f"🧠 <b>Local ONNX (Zero-Key Default)</b>\n"
                f"Model: <code>{esc(deps.local_scorer.model_id)}</code>"
            )

        cv_line = (
            f"✅ {esc(profile.cv_filename)} ({len(profile.cv_text)} chars)"
            if profile.cv_text
            else "Not uploaded (/cv)"
        )
        last_sync = (
            stats.last_ingest_at.strftime("%Y-%m-%d %H:%M UTC")
            if stats.last_ingest_at
            else "Never"
        )
        eligible = stats.new_count + stats.seen_count

        await message.answer(
            f"📊 <b>GiveMeJobPlease Status</b>\n"
            f"App version: <code>v{esc(__version__)}</code>\n"
            f"Parser: <code>{esc(stats.last_parser_version)}</code>\n"
            f"Last ingest: <code>{esc(last_sync)}</code>\n\n"
            f"<b>Scoring Engine:</b>\n{engine_line}\n\n"
            f"<b>Profile &amp; CV:</b>\n"
            f"Profile configured: {'✅' if profile.is_configured() else '⚠️ Not yet (/profile)'}\n"
            f"CV: {cv_line}\n\n"
            f"<b>Showroom Counts ({stats.total_in_showroom} active):</b>\n"
            f"• Eligible pool: <b>{eligible}</b> (🆕 {stats.new_count} new · 👀 {stats.seen_count} seen)\n"
            f"• ⭐ Starred (all time): <b>{stats.starred_count}</b>\n"
            f"• ✅ Applied: <b>{stats.applied_count}</b>\n"
            f"• 🚫 Excluded: <b>{stats.excluded_count}</b>",
            disable_web_page_preview=True,
        )

    @router.message(Command("logs"))
    async def cmd_logs(message: Message) -> None:
        entries = read_recent_error_logs(deps.error_log_path, limit=5)
        if not entries:
            await message.answer("✅ No recorded error entries in <code>errors.jsonl</code>.")
            return
        lines = ["🪵 <b>Last 5 Traced Errors:</b>"]
        for entry in entries:
            ts = str(entry.get("timestamp", ""))[:19]
            tid = str(entry.get("trace_id", "n/a"))
            code = str(entry.get("error_code", entry.get("level", "error")))
            ev = str(entry.get("event", ""))[:140]
            lines.append(f"• <code>{esc(ts)}</code> [<code>{esc(tid)}</code>] <b>{esc(code)}</b>: {esc(ev)}")
        await message.answer("\n".join(lines))

    # ------------------------------------------------------------------
    # Profile FSM Wizard (/profile, /cancel) & CV (/cv)
    # ------------------------------------------------------------------

    @router.message(Command("cancel"))
    async def cmd_cancel(message: Message, state: FSMContext) -> None:
        await state.clear()
        await message.answer("Cancelled current setup flow.")

    @router.message(Command("profile"))
    async def cmd_profile(message: Message, state: FSMContext) -> None:
        if message.from_user is None:
            return
        profile = await deps.repo.get_or_create_profile(message.from_user.id)
        await state.set_state(ProfileWizard.target_roles)
        current = esc(profile.target_roles) if profile.target_roles else "<i>empty</i>"
        await message.answer(
            "🧭 <b>Profile Setup (Step 1/4)</b> — send /cancel anytime or <code>-</code> to keep current.\n\n"
            f"Current target roles: {current}\n\n"
            "<b>What roles, domains, and tech stack are you targeting?</b>\n"
            "<i>Example: MSc Cybersecurity / Systems / Applied AI Security, Python, Rust, C++, distributed systems</i>"
        )

    @router.message(ProfileWizard.target_roles)
    async def profile_step_roles(message: Message, state: FSMContext) -> None:
        if message.from_user is None or not message.text:
            return
        profile = await deps.repo.get_or_create_profile(message.from_user.id)
        text = message.text.strip()
        await state.update_data(target_roles=profile.target_roles if text == "-" else text)
        await state.set_state(ProfileWizard.preferred_locations)
        current = esc(profile.preferred_locations) if profile.preferred_locations else "<i>empty</i>"
        await message.answer(
            "📍 <b>Profile Setup (Step 2/4)</b>\n"
            f"Current locations: {current}\n\n"
            "<b>Preferred locations &amp; remote/relocation tolerance?</b> (comma-separated)\n"
            "<i>Example: Lausanne, Geneva, Zurich, Remote (no relocation outside Switzerland)</i>"
        )

    @router.message(ProfileWizard.preferred_locations)
    async def profile_step_locations(message: Message, state: FSMContext) -> None:
        if message.from_user is None or not message.text:
            return
        profile = await deps.repo.get_or_create_profile(message.from_user.id)
        text = message.text.strip()
        await state.update_data(
            preferred_locations=profile.preferred_locations if text == "-" else text
        )
        await state.set_state(ProfileWizard.accepted_formats)
        current = esc(", ".join(profile.accepted_formats)) if profile.accepted_formats else "<i>any</i>"
        await message.answer(
            "🏷 <b>Profile Setup (Step 3/4)</b>\n"
            f"Current accepted formats: {current}\n\n"
            "<b>Which formats do you accept?</b> (comma-separated, or <code>any</code>)\n"
            "<i>Options: Internship, master project or Internship, PDM coordonné</i>"
        )

    @router.message(ProfileWizard.accepted_formats)
    async def profile_step_formats(message: Message, state: FSMContext) -> None:
        if message.from_user is None or not message.text:
            return
        profile = await deps.repo.get_or_create_profile(message.from_user.id)
        text = message.text.strip()
        if text == "-":
            formats = profile.accepted_formats
        elif text.lower() in {"any", "all"}:
            formats = []
        else:
            formats = [p.strip() for p in text.split(",") if p.strip()]
        await state.update_data(accepted_formats=formats)
        await state.set_state(ProfileWizard.dealbreakers)
        current = esc(profile.dealbreakers) if profile.dealbreakers else "<i>empty</i>"
        await message.answer(
            "🚫 <b>Profile Setup (Step 4/4)</b>\n"
            f"Current dealbreakers/languages: {current}\n\n"
            "<b>Languages spoken &amp; hard dealbreakers?</b>\n"
            "<i>Example: Fluent English &amp; French; skip unpaid internships, pure frontend/UI, or PowerBI-only roles.</i>"
        )

    @router.message(ProfileWizard.dealbreakers)
    async def profile_step_dealbreakers(message: Message, state: FSMContext) -> None:
        if message.from_user is None or not message.text:
            return
        profile = await deps.repo.get_or_create_profile(message.from_user.id)
        data = await state.get_data()
        text = message.text.strip()

        profile.target_roles = str(data.get("target_roles", profile.target_roles))
        profile.preferred_locations = str(
            data.get("preferred_locations", profile.preferred_locations)
        )
        profile.accepted_formats = list(data.get("accepted_formats", profile.accepted_formats))
        profile.dealbreakers = profile.dealbreakers if text == "-" else text

        saved = await deps.repo.save_profile(profile)
        await state.clear()
        await message.answer(
            "✅ <b>Profile saved!</b> (Stale offer scores will automatically re-compute on your next /digest.)\n\n"
            f"• <b>Roles:</b> {esc(saved.target_roles or '—')}\n"
            f"• <b>Locations:</b> {esc(saved.preferred_locations or '—')}\n"
            f"• <b>Formats:</b> {esc(', '.join(saved.accepted_formats) or 'Any')}\n"
            f"• <b>Dealbreakers:</b> {esc(saved.dealbreakers or '—')}\n\n"
            "Tip: You can also drop a <code>.pdf</code>, <code>.txt</code>, or <code>.md</code> CV anytime."
        )

    @router.message(Command("cv"))
    async def cmd_cv(message: Message) -> None:
        if message.from_user is None:
            return
        profile = await deps.repo.get_or_create_profile(message.from_user.id)
        if profile.cv_text:
            preview = esc(profile.cv_text[:500])
            await message.answer(
                f"📄 <b>Current CV:</b> <code>{esc(profile.cv_filename)}</code> ({len(profile.cv_text)} chars)\n\n"
                f"<i>Preview:</i>\n<code>{preview}...</code>\n\n"
                "Send a new <code>.pdf</code>, <code>.txt</code>, or <code>.md</code> file to replace it."
            )
        else:
            await message.answer(
                "📄 No CV uploaded yet.\n"
                "Simply attach and send your CV as a <code>.pdf</code>, <code>.txt</code>, or <code>.md</code> file right in this chat."
            )

    # ------------------------------------------------------------------
    # Document Ingestion (.htm Showroom/Registrations + .pdf/.txt/.md CV)
    # ------------------------------------------------------------------

    @router.message(F.document)
    async def handle_document(message: Message, bot: Bot) -> None:
        if message.from_user is None or message.document is None:
            return
        doc = message.document
        filename = doc.file_name or "upload.bin"
        lower = filename.lower()

        if doc.file_size and doc.file_size > 20 * 1024 * 1024:
            raise ValidationError("File is too large (>20 MB).")

        buf = io.BytesIO()
        await bot.download(doc, destination=buf)
        raw_bytes = buf.getvalue()
        user_id = message.from_user.id

        # 1. CV upload (.pdf, .txt, .md)
        if lower.endswith((".pdf", ".txt", ".md")):
            cv_text = parse_cv_bytes(raw_bytes, filename)
            profile = await deps.repo.get_or_create_profile(user_id)
            profile.cv_text = cv_text
            profile.cv_filename = filename
            await deps.repo.save_profile(profile)
            await message.answer(
                f"✅ <b>CV parsed &amp; saved!</b>\n"
                f"File: <code>{esc(filename)}</code> ({len(cv_text)} chars extracted)\n"
                "Your next /digest will incorporate your CV context."
            )
            return

        # 2. IS-Academia HTML upload (.htm, .html)
        if lower.endswith((".htm", ".html")):
            file_type = detect_isacademia_file_type(raw_bytes)
            if file_type == "showroom":
                offers = parse_showroom_html(raw_bytes)
                summary = await deps.repo.ingest_showroom_offers(
                    user_id=user_id,
                    offers=offers,
                    filename=filename,
                    parser_version=SHOWROOM_PARSER_VERSION,
                )
                await message.answer(
                    f"📥 <b>Showroom Export Ingested</b> (<code>{esc(summary.parser_version)}</code>)\n"
                    f"• Parsed offers: <b>{summary.total_parsed}</b>\n"
                    f"• Brand-new offers added: <b>+{summary.new_offers}</b>\n"
                    f"• Existing updated: <b>{summary.updated_offers}</b>\n"
                    f"• Current eligible pool: <b>{summary.eligible_new_count}</b> new · "
                    f"<b>{summary.eligible_seen_count}</b> seen\n\n"
                    "Run /digest to score &amp; review new offers, or /browse all to see everything eligible."
                )
                return

            records = parse_registrations_html(raw_bytes)
            summary = await deps.repo.ingest_registrations(
                user_id=user_id,
                records=records,
                filename=filename,
                parser_version=REGISTRATIONS_PARSER_VERSION,
            )
            await message.answer(
                f"📥 <b>Registrations Export Synced</b> (<code>{esc(summary.parser_version)}</code>)\n"
                f"• Parsed registration rows: <b>{summary.total_parsed}</b>\n"
                f"• Offers newly marked ✅ Applied: <b>{summary.auto_marked_applied}</b>\n"
                f"• Remaining eligible pool: <b>{summary.eligible_new_count}</b> new · "
                f"<b>{summary.eligible_seen_count}</b> seen"
            )
            return

        raise ParseError(
            "Unsupported file type. Send an IS-Academia <code>.htm</code> export, "
            "or a CV as <code>.pdf</code>, <code>.txt</code>, or <code>.md</code>."
        )

    # ------------------------------------------------------------------
    # Digest & Browse
    # ------------------------------------------------------------------

    @router.message(Command("digest"))
    async def cmd_digest(message: Message, command: CommandObject) -> None:
        if message.from_user is None:
            return
        args = (command.args or "").strip().lower()
        force_local = "local" in args
        await render_view_page(
            message,
            message.from_user.id,
            view="new",
            page=0,
            force_local=force_local,
            mark_new_as_seen=True,
        )

    @router.message(Command("browse", "starred", "seen", "applied", "excluded"))
    async def cmd_browse(message: Message, command: CommandObject) -> None:
        if message.from_user is None:
            return
        cmd_name = (command.command or "browse").lower()
        if cmd_name in {"starred", "seen", "applied", "excluded"}:
            view = cmd_name
        else:
            view = (command.args or "all").strip().lower().split()[0]
        await render_view_page(
            message,
            message.from_user.id,
            view=view,
            page=0,
            force_local=False,
            mark_new_as_seen=(view in {"new", "all"}),
        )

    # ------------------------------------------------------------------
    # OpenRouter OAuth PKCE & Model Management
    # ------------------------------------------------------------------

    @router.message(Command("connect_openrouter"))
    async def cmd_connect_openrouter(message: Message, command: CommandObject) -> None:
        if message.from_user is None:
            return
        limit_usd: float | None = None
        expiration: str | None = None

        parts = (command.args or "").split()
        for part in parts:
            try:
                limit_usd = float(part)
            except ValueError:
                expiration = part

        auth_url = deps.pkce.start_headless_session(
            message.from_user.id,
            limit_usd=limit_usd,
            expiration=expiration,
        )
        await message.answer(
            "🔐 <b>Connect OpenRouter (OAuth PKCE — Headless)</b>\n\n"
            f'1. Open this authorization link:\n<a href="{esc(auth_url)}">Authorize GiveMeJobPlease on OpenRouter</a>\n'
            "2. Choose your credit cap &amp; expiry on OpenRouter and approve.\n"
            "3. Copy the single-use code OpenRouter displays and send:\n"
            "<code>/auth YOUR_CODE</code>\n\n"
            "<i>Security notes: Your /auth message will be deleted immediately, the code expires in 10 minutes, "
            "and your resulting key is encrypted at rest with AES-256-GCM.</i>",
            disable_web_page_preview=True,
        )

    @router.message(Command("auth"))
    async def cmd_auth(message: Message, command: CommandObject) -> None:
        if message.from_user is None:
            return
        code = (command.args or "").strip()
        # Immediately remove the user's message containing the single-use code from chat history
        try:
            await message.delete()
        except Exception:
            logger.warning("could_not_delete_auth_message")

        if not code:
            raise ValidationError("Usage: /auth <code> (after running /connect_openrouter).")

        uid = message.from_user.id
        exchanged = await deps.pkce.exchange_code(uid, code)
        encrypted = deps.secret_box.encrypt(
            exchanged.api_key, associated_data=f"openrouter:{uid}"
        )
        active_model = await deps.repo.save_openrouter_credentials(
            uid,
            key_enc=encrypted,
            key_hash=exchanged.key_hash,
            default_model=deps.settings.default_openrouter_model,
        )
        await message.answer(
            "✅ <b>OpenRouter Connected!</b>\n"
            "OpenRouter is now your primary scoring engine (overriding local embeddings).\n"
            f"Active model: <code>{esc(active_model)}</code> (change anytime with /model)\n\n"
            f'Manage your key: <a href="{esc(exchanged.logs_url)}">Activity Logs</a> · '
            f'<a href="{esc(exchanged.settings_url)}">Settings / Revoke</a>',
            disable_web_page_preview=True,
        )

    @router.message(Command("disconnect_openrouter"))
    async def cmd_disconnect_openrouter(message: Message) -> None:
        if message.from_user is None:
            return
        await deps.repo.clear_openrouter_credentials(message.from_user.id)
        await message.answer(
            "🔌 <b>OpenRouter disconnected.</b> Stored key wiped.\n"
            "Reverted to Mode A (Local ONNX multilingual embeddings)."
        )

    @router.message(Command("model"))
    async def cmd_model(message: Message, command: CommandObject) -> None:
        if message.from_user is None:
            return
        uid = message.from_user.id
        arg = (command.args or "").strip()

        if not arg:
            or_state = await deps.repo.get_openrouter_state(uid)
            current = or_state.model_id or deps.settings.default_openrouter_model
            await message.answer(
                f"🤖 Current OpenRouter model: <code>{esc(current)}</code>\n\n"
                "• Search models: <code>/model qwen</code> or <code>/model deepseek</code>\n"
                "• Set exact model: <code>/model provider/model-slug</code>"
            )
            return

        # If argument contains '/', treat as exact model slug selection
        if "/" in arg:
            found = await verify_openrouter_model_exists(arg)
            if found is None:
                raise ValidationError(
                    f"Model <code>{esc(arg)}</code> was not found in OpenRouter's catalog. "
                    "Search first with e.g. <code>/model deepseek</code>."
                )
            await deps.repo.set_openrouter_model(uid, found.id)
            p_in = f"${found.prompt_price_per_m:.3f}/M" if found.prompt_price_per_m is not None else "n/a"
            p_out = f"${found.completion_price_per_m:.3f}/M" if found.completion_price_per_m is not None else "n/a"
            await message.answer(
                f"✅ Active OpenRouter model set to <code>{esc(found.id)}</code>\n"
                f"Pricing: {p_in} in · {p_out} out"
            )
            return

        # Otherwise search OpenRouter catalog
        matches = await search_openrouter_models(query=arg, limit=10)
        if not matches:
            await message.answer(f"No OpenRouter models found matching <code>{esc(arg)}</code>.")
            return

        lines = [f"🔎 <b>OpenRouter models matching</b> <code>{esc(arg)}</code>:"]
        for m in matches:
            p_in = f"${m.prompt_price_per_m:.3f}/M" if m.prompt_price_per_m is not None else "n/a"
            p_out = f"${m.completion_price_per_m:.3f}/M" if m.completion_price_per_m is not None else "n/a"
            ctx = f"{m.context_length // 1000}k ctx" if m.context_length else ""
            lines.append(f"• <code>{esc(m.id)}</code> ({p_in} / {p_out} {ctx})")
        lines.append("\nSelect one with: <code>/model &lt;model_id&gt;</code>")
        await message.answer("\n".join(lines))

    # ------------------------------------------------------------------
    # Export (/export)
    # ------------------------------------------------------------------

    @router.message(Command("export"))
    async def cmd_export(message: Message) -> None:
        if message.from_user is None:
            return
        items = sort_by_relevance(
            await deps.repo.list_enriched_offers(
                message.from_user.id, only_latest_showroom=False
            )
        )
        if not items:
            await message.answer("Nothing to export yet. Drop an IS-Academia <code>.htm</code> file first.")
            return

        csv_bytes = export_csv_bytes(items)
        json_bytes = export_descriptions_json_bytes(items)
        await message.answer_document(
            BufferedInputFile(csv_bytes, filename="internships_enriched.csv"),
            caption=f"📊 Enriched CSV ({len(items)} offers · v{__version__})",
        )
        await message.answer_document(
            BufferedInputFile(json_bytes, filename="internships_descriptions.json"),
            caption="📄 Full descriptions + scores JSON",
        )

    # ------------------------------------------------------------------
    # Inline Button Callbacks
    # ------------------------------------------------------------------

    @router.callback_query(F.data == "noop")
    async def cb_noop(callback: CallbackQuery) -> None:
        await callback.answer()

    @router.callback_query(F.data.startswith("pg:"))
    async def cb_paginate(callback: CallbackQuery) -> None:
        if callback.from_user is None or callback.message is None or not callback.data:
            return
        _, view, raw_page = callback.data.split(":", 2)
        await callback.answer()
        if isinstance(callback.message, Message):
            await render_view_page(
                callback.message,
                callback.from_user.id,
                view=view,
                page=int(raw_page),
                mark_new_as_seen=(view in {"new", "all"}),
            )

    @router.callback_query(F.data.startswith("o:"))
    async def cb_offer_action(callback: CallbackQuery) -> None:
        if callback.from_user is None or callback.message is None or not callback.data:
            return
        parts = callback.data.split(":", 2)
        if len(parts) != 3:
            await callback.answer("Invalid action", show_alert=False)
            return
        _, action, offer_uid = parts
        uid = callback.from_user.id

        if action == "d":
            item = await deps.repo.get_enriched_offer(uid, offer_uid)
            if item is None:
                await callback.answer("Offer not found.", show_alert=True)
                return
            await callback.answer()
            o = item.offer
            header = f"📄 <b>{esc(o.title)}</b> — {esc(o.company)} (#{esc(o.external_id)})\n\n"
            body = esc(o.description) if o.description else "<i>No description embedded in saved .htm file.</i>"
            if isinstance(callback.message, Message):
                for idx, chunk in enumerate(chunk_text(header + body)):
                    await callback.message.answer(chunk if idx == 0 else chunk)
            return

        updated: EnrichedOffer | None = None
        toast = ""
        if action == "s":
            updated = await deps.repo.toggle_star(uid, offer_uid)
            toast = "⭐ Starred" if (updated and updated.is_starred) else "☆ Unstarred"
        elif action == "a":
            updated = await deps.repo.set_offer_status(uid, offer_uid, OfferStatus.APPLIED)
            toast = "✅ Marked as Applied"
        elif action == "x":
            updated = await deps.repo.set_offer_status(uid, offer_uid, OfferStatus.EXCLUDED)
            toast = "🚫 Excluded"
        elif action == "r":
            updated = await deps.repo.set_offer_status(uid, offer_uid, OfferStatus.SEEN)
            toast = "↩️ Restored to Seen"

        if updated is None:
            await callback.answer("Offer not found in database.", show_alert=True)
            return

        await callback.answer(toast)
        if isinstance(callback.message, Message):
            await callback.message.edit_text(
                format_offer_card(updated),
                reply_markup=build_offer_keyboard(updated),
            )

    # ------------------------------------------------------------------
    # Centralized Global Error Handler (Structured Trace + Safe User Reply)
    # ------------------------------------------------------------------

    @router.error()
    async def global_error_handler(event: ErrorEvent) -> None:
        exc = event.exception
        ctx = structlog.contextvars.get_contextvars()
        trace_id = str(ctx.get("trace_id") or "unknown")

        if isinstance(exc, AppError):
            logger.error(
                "handled_app_error",
                error_code=exc.code,
                detail=exc.detail,
                exc_info=isinstance(exc, (AuthError,)) is False,
            )
            reply_text = f"⚠️ <b>{esc(exc.code)}</b>\n{esc(exc.user_message)}\n\n<i>[trace: <code>{esc(trace_id)}</code>]</i>"
        else:
            logger.error(
                "unhandled_exception",
                error_code="INTERNAL_ERROR",
                exc_info=exc,
            )
            reply_text = (
                "❌ <b>Unexpected internal error</b>\n"
                "Details were logged safely (secrets redacted). Run /logs or check Docker logs.\n\n"
                f"<i>[trace: <code>{esc(trace_id)}</code>]</i>"
            )

        update = event.update
        if update.callback_query is not None:
            try:
                await update.callback_query.answer("Error occurred — see message.", show_alert=False)
            except Exception:
                pass
            if isinstance(update.callback_query.message, Message):
                await update.callback_query.message.answer(reply_text)
        elif update.message is not None:
            await update.message.answer(reply_text)

    return router
