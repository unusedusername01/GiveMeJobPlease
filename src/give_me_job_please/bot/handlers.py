"""All Telegram command handlers, FSM /profile wizard, file uploads, inline actions, and error handler."""

from __future__ import annotations

import asyncio
import contextlib
import io
import math
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path

import structlog
from aiogram import Bot, F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandObject
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    ErrorEvent,
    InlineKeyboardMarkup,
    Message,
)

from give_me_job_please import __version__
from give_me_job_please.bot.formatting import (
    DigestQuery,
    build_description_keyboard,
    build_digest_list_keyboard,
    build_offer_keyboard,
    build_pagination_keyboard,
    build_profile_keyboard,
    build_scoring_cancel_keyboard,
    decode_query_callback,
    esc,
    format_digest_list,
    format_offer_card,
    format_offer_description_page,
    format_profile_summary,
    format_recency_label,
    format_threshold_label,
)
from give_me_job_please.config import Settings
from give_me_job_please.core.auth.crypto import SecretBox
from give_me_job_please.core.auth.openrouter_pkce import (
    OpenRouterPkceManager,
    build_key_deep_links,
    search_openrouter_models,
    verify_openrouter_model_exists,
)
from give_me_job_please.core.diff import (
    PreScoringFilter,
    apply_pre_scoring_filters,
    filter_by_min_score,
    parse_digest_args,
    parse_min_score_threshold,
    sort_by_relevance,
)
from give_me_job_please.core.exporter import export_csv_bytes, export_descriptions_json_bytes
from give_me_job_please.core.models import EnrichedOffer, OfferScore, OfferStatus, UserProfile
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
from give_me_job_please.storage.repository import OfferRepository, UserStats

logger = structlog.get_logger(__name__)

_MAX_MESSAGE_CONTEXTS = 200


@dataclass(slots=True)
class DigestMessageContext:
    """In-memory snapshot for smooth in-place Prev/Next card navigation (with stateless fallback)."""

    query: DigestQuery
    ordered_uids: list[str]
    page_size: int


@dataclass(slots=True)
class _ScoringProgress:
    """Mutable per-run counters shared with the checkpoint closure for cancel summaries."""

    total: int = 0
    scored: int = 0


@dataclass(slots=True)
class _PreparedDigest:
    title: str
    ranked: list[EnrichedOffer]
    total_before_pre_filter: int
    filtered_out_pre: int
    active_threshold: float | None
    hidden_below_threshold: int
    engine: str
    model_id: str
    profile: UserProfile
    stats: UserStats


async def _safe_edit_text(
    msg: Message,
    text: str,
    reply_markup: InlineKeyboardMarkup | None = None,
) -> None:
    """Edit a Telegram message in-place, ignoring harmless 'message is not modified' errors."""
    try:
        await msg.edit_text(text, reply_markup=reply_markup)
    except TelegramBadRequest as exc:
        if "message is not modified" in str(exc).lower():
            return
        raise


class ProfileWizard(StatesGroup):
    target_roles = State()
    preferred_locations = State()
    accepted_formats = State()
    dealbreakers = State()
    min_score_threshold = State()
    quick_threshold = State()


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
        # One lock per user: prevents duplicate concurrent scoring runs (and billing).
        self.scoring_locks: dict[int, asyncio.Lock] = {}
        # Active scoring task per user, so /cancel or the 🛑 button can abort a run.
        self.scoring_tasks: dict[int, asyncio.Task[None]] = {}
        # Bounded LRU mapping (chat_id, message_id) -> DigestMessageContext for in-place UX.
        self.digest_contexts: OrderedDict[tuple[int, int], DigestMessageContext] = OrderedDict()

    def set_digest_context(
        self, chat_id: int, message_id: int, ctx: DigestMessageContext
    ) -> None:
        key = (chat_id, message_id)
        self.digest_contexts[key] = ctx
        self.digest_contexts.move_to_end(key)
        while len(self.digest_contexts) > _MAX_MESSAGE_CONTEXTS:
            self.digest_contexts.popitem(last=False)

    def get_digest_context(
        self, chat_id: int, message_id: int
    ) -> DigestMessageContext | None:
        key = (chat_id, message_id)
        ctx = self.digest_contexts.get(key)
        if ctx is not None:
            self.digest_contexts.move_to_end(key)
        return ctx


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
        """Serialise scoring per user so double taps cannot run duplicate paid passes."""
        lock = deps.scoring_locks.setdefault(user_id, asyncio.Lock())
        if lock.locked():
            raise ValidationError(
                "A scoring run is already in progress for your account. "
                "Please wait for it to finish, or send /cancel to stop it."
            )
        progress = _ScoringProgress()
        task = asyncio.current_task()
        if task is not None:
            deps.scoring_tasks[user_id] = task
        try:
            async with lock:
                return await _ensure_scored_unlocked(
                    user_id,
                    items,
                    force_local=force_local,
                    status_msg=status_msg,
                    progress=progress,
                )
        except asyncio.CancelledError:
            if status_msg is not None:
                with contextlib.suppress(Exception):
                    await _safe_edit_text(
                        status_msg,
                        f"🛑 <b>Scoring cancelled</b> — {progress.scored} of {progress.total} "
                        "offer(s) scored and saved.\n"
                        "Run /digest again to resume: only unscored offers are re-billed.",
                    )
            raise
        finally:
            if deps.scoring_tasks.get(user_id) is task:
                deps.scoring_tasks.pop(user_id, None)

    async def _ensure_scored_unlocked(
        user_id: int,
        items: list[EnrichedOffer],
        *,
        force_local: bool = False,
        status_msg: Message | None = None,
        progress: _ScoringProgress | None = None,
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
            if progress is not None:
                progress.total = len(to_score)
            if status_msg is not None:
                mode_str = (
                    f"OpenRouter (<code>{esc(model_id)}</code>)"
                    if engine == "openrouter"
                    else f"Local ONNX (<code>{esc(model_id)}</code>)"
                )
                await _safe_edit_text(
                    status_msg,
                    f"⏳ Scoring <b>{len(to_score)}</b> offer(s) using {mode_str}...",
                    reply_markup=build_scoring_cancel_keyboard(),
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

                async def _checkpoint(batch_scores: list[OfferScore]) -> None:
                    # Persist each completed batch immediately: a later batch failure
                    # or cancellation must not discard (and re-bill) finished batches.
                    await deps.repo.save_scores(user_id, batch_scores)
                    if progress is not None:
                        progress.scored += len(batch_scores)

                new_scores = await scorer.score_offers(
                    profile, to_score, fingerprint, on_batch_complete=_checkpoint
                )
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
                if progress is not None:
                    progress.scored = len(new_scores)

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

    async def _prepare_digest_data(
        user_id: int,
        query: DigestQuery,
        *,
        status_msg: Message | None = None,
    ) -> _PreparedDigest:
        """Load view offers, apply pre-scoring recency filters, score, rank, and apply threshold."""
        view = query.view
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
        profile = await deps.repo.get_or_create_profile(user_id)
        total_before_pre_filter = len(items)

        # Pre-scoring filter pipeline: runs BEFORE OpenRouter/FastEmbed so excluded offers
        # (e.g. outside the recency window) never consume paid LLM tokens.
        pre_filter = PreScoringFilter(
            max_age_days=query.max_age_days,
            recency_basis=query.recency_basis,
        )
        filtered_items = apply_pre_scoring_filters(items, pre_filter)
        filtered_out_pre = total_before_pre_filter - len(filtered_items)

        if not filtered_items:
            return _PreparedDigest(
                title=title,
                ranked=[],
                total_before_pre_filter=total_before_pre_filter,
                filtered_out_pre=filtered_out_pre,
                active_threshold=None,
                hidden_below_threshold=0,
                engine="",
                model_id="",
                profile=profile,
                stats=stats,
            )

        scored_items, engine, model_id = await ensure_scored(
            user_id,
            filtered_items,
            force_local=query.force_local,
            status_msg=status_msg,
        )
        ranked = sort_by_relevance(scored_items)

        # Apply the user's configured score threshold on discovery views only
        # (never hide starred/applied/excluded items the user explicitly tracks).
        active_threshold = (
            profile.min_score_threshold
            if (not query.ignore_threshold and view in {"new", "all", "seen"})
            else None
        )
        total_before_threshold = len(ranked)
        if active_threshold is not None:
            ranked = filter_by_min_score(ranked, active_threshold)
        hidden_below_threshold = total_before_threshold - len(ranked)

        return _PreparedDigest(
            title=title,
            ranked=ranked,
            total_before_pre_filter=total_before_pre_filter,
            filtered_out_pre=filtered_out_pre,
            active_threshold=active_threshold,
            hidden_below_threshold=hidden_below_threshold,
            engine=engine,
            model_id=model_id,
            profile=profile,
            stats=stats,
        )

    async def render_view_page(
        target_msg: Message,
        user_id: int,
        query: DigestQuery,
        *,
        edit_in_place: bool = False,
    ) -> None:
        """Render the compact digest list into a single message (created once, then edited in-place)."""
        display_msg = (
            target_msg
            if edit_in_place
            else await target_msg.answer("⏳ Preparing offers...")
        )

        prepared = await _prepare_digest_data(
            user_id,
            query,
            status_msg=display_msg,
        )

        recency_label = format_recency_label(query.max_age_days, query.recency_basis)
        if prepared.total_before_pre_filter == 0:
            if query.view == "new" and prepared.stats.seen_count > 0:
                kb = build_pagination_keyboard(
                    query=replace(query, page=0),
                    total_pages=0,
                    seen_count=prepared.stats.seen_count,
                    profile_threshold=prepared.profile.min_score_threshold,
                )
                await _safe_edit_text(
                    display_msg,
                    f"<b>{prepared.title}</b>\nNo brand-new offers waiting. "
                    f"You have <b>{prepared.stats.seen_count}</b> seen eligible offer(s).",
                    reply_markup=kb,
                )
            else:
                await _safe_edit_text(
                    display_msg,
                    f"<b>{prepared.title}</b>\nNothing here yet. Drop an IS-Academia <code>.htm</code> export anytime.",
                )
            return

        if not prepared.ranked and prepared.filtered_out_pre == prepared.total_before_pre_filter:
            kb = build_pagination_keyboard(
                query=replace(query, page=0),
                total_pages=0,
                seen_count=prepared.stats.seen_count,
                profile_threshold=prepared.profile.min_score_threshold,
            )
            await _safe_edit_text(
                display_msg,
                f"<b>{prepared.title}</b> · 📅 {esc(recency_label)}\n"
                f"No offers matched the recency window "
                f"(<b>{prepared.filtered_out_pre}</b> older offer(s) skipped before scoring).\n"
                "Widen the range or switch Posted ⇄ Uploaded below.",
                reply_markup=kb,
            )
            return

        if not prepared.ranked and prepared.active_threshold is not None:
            engine_badge = (
                f"🤖 OpenRouter (<code>{esc(prepared.model_id)}</code>)"
                if prepared.engine == "openrouter"
                else f"🧠 Local ONNX (<code>{esc(prepared.model_id)}</code>)"
            )
            kb = build_pagination_keyboard(
                query=replace(query, page=0),
                total_pages=0,
                seen_count=prepared.stats.seen_count,
                profile_threshold=prepared.profile.min_score_threshold,
            )
            await _safe_edit_text(
                display_msg,
                f"<b>{prepared.title}</b> · 📅 {esc(recency_label)}\n"
                f"No offers scored <b>≥ {prepared.active_threshold:g}/10</b> "
                f"(<b>{prepared.hidden_below_threshold}</b> lower-scored offer(s) hidden).\n"
                f"Engine: {engine_badge}\n\n"
                "Tap below to show all scores, or update with <code>/profile threshold off</code>.",
                reply_markup=kb,
            )
            return

        page_size = deps.settings.digest_page_size
        total_pages = max(1, math.ceil(len(prepared.ranked) / page_size))
        page = max(0, min(query.page, total_pages - 1))
        active_query = replace(query, page=page)
        slice_items = prepared.ranked[page * page_size : (page + 1) * page_size]

        deps.set_digest_context(
            display_msg.chat.id,
            display_msg.message_id,
            DigestMessageContext(
                query=active_query,
                ordered_uids=[item.offer.uid for item in prepared.ranked],
                page_size=page_size,
            ),
        )

        text = format_digest_list(
            title=prepared.title,
            items=slice_items,
            total_count=len(prepared.ranked),
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            query=active_query,
            engine=prepared.engine,
            model_id=prepared.model_id,
            filtered_out_pre=prepared.filtered_out_pre,
            active_threshold=prepared.active_threshold,
            hidden_below_threshold=prepared.hidden_below_threshold,
        )
        kb = build_digest_list_keyboard(
            query=active_query,
            page_items=slice_items,
            page_size=page_size,
            total_pages=total_pages,
            seen_count=prepared.stats.seen_count,
            profile_threshold=prepared.profile.min_score_threshold,
        )
        await _safe_edit_text(display_msg, text, reply_markup=kb)

    async def _render_offer_card_in_place(
        target_msg: Message,
        user_id: int,
        offer_uid: str,
        *,
        ctx: DigestMessageContext | None = None,
        index: int | None = None,
        mark_seen: bool = True,
    ) -> bool:
        """Edit `target_msg` in-place to display a single offer card with contextual navigation."""
        item = await deps.repo.get_enriched_offer(user_id, offer_uid)
        if item is None:
            return False

        if mark_seen and item.status == OfferStatus.NEW:
            await deps.repo.mark_offers_seen(user_id, [offer_uid])
            item.status = OfferStatus.SEEN

        resolved_idx = index
        total = 0
        back_query: DigestQuery | None = None
        if ctx is not None:
            total = len(ctx.ordered_uids)
            if resolved_idx is None:
                try:
                    resolved_idx = ctx.ordered_uids.index(offer_uid)
                except ValueError:
                    resolved_idx = None
            if resolved_idx is not None and ctx.page_size > 0:
                back_query = replace(ctx.query, page=resolved_idx // ctx.page_size)
            else:
                back_query = ctx.query

        pos_label = (
            f"{resolved_idx + 1}/{total}"
            if (resolved_idx is not None and total > 0)
            else ""
        )
        await _safe_edit_text(
            target_msg,
            format_offer_card(item, position_label=pos_label),
            reply_markup=build_offer_keyboard(
                item,
                index=resolved_idx,
                total=total,
                back_query=back_query,
            ),
        )
        return True

    async def _apply_offer_action(
        user_id: int, action: str, offer_uid: str
    ) -> tuple[EnrichedOffer | None, str]:
        if action == "s":
            updated = await deps.repo.toggle_star(user_id, offer_uid)
            toast = "⭐ Starred" if (updated and updated.is_starred) else "☆ Unstarred"
            return updated, toast
        if action == "a":
            updated = await deps.repo.set_offer_status(user_id, offer_uid, OfferStatus.APPLIED)
            return updated, "✅ Marked as Applied"
        if action == "x":
            updated = await deps.repo.set_offer_status(user_id, offer_uid, OfferStatus.EXCLUDED)
            return updated, "🚫 Excluded"
        if action == "r":
            updated = await deps.repo.set_offer_status(user_id, offer_uid, OfferStatus.SEEN)
            return updated, "↩️ Restored to Seen"
        return None, "Invalid action"

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
            "   ⚠️ Upload an ANONYMISED CV only: replace name, email, phone and street address "
            "with mock data (e.g. John Doe, john.doe@example.com, Vaud). See the README.\n"
            "2. Optional: Connect OpenRouter via /connect_openrouter (overrides local embeddings).\n"
            "3. Save IS-Academia <code>Internships showroom (stu).htm</code> (or <code>Registrations (stu).htm</code>) "
            "and drop the file right into this chat.\n"
            "4. Run /digest to review ranked offers.\n\n"
            "<b>Commands:</b>\n"
            "/profile [edit|threshold &lt;val|off&gt;] — View or update preferences &amp; score threshold\n"
            "/cv — View/upload CV\n"
            "/digest [3d|7d|14d|30d|all] [posted|ingested] [local] — Ranked new offers (recency filters before LLM scoring)\n"
            "/browse [all|new|seen|starred|applied|excluded] [7d] [ingested]\n"
            "/connect_openrouter [limit] [expiration] — OAuth PKCE\n"
            "/auth &lt;code&gt; — Complete PKCE (message auto-deleted)\n"
            "/model [query_or_id] — Pick OpenRouter model\n"
            "/disconnect_openrouter — Remove stored key\n"
            "/export — Download enriched CSV + JSON\n"
            "/cancel — Stop an active scoring run (saved batches kept) or exit setup\n"
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
            f"Min score threshold: <code>{esc(format_threshold_label(profile.min_score_threshold))}</code>\n"
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
        if message.from_user is None:
            return
        await state.clear()
        task = deps.scoring_tasks.get(message.from_user.id)
        if task is not None and not task.done():
            task.cancel()
            # The cancelled run edits its own status message with the saved-progress summary.
            await message.answer(
                "🛑 Cancelling the active scoring run… "
                "Completed batches are already saved; /digest resumes where it stopped."
            )
            return
        await message.answer("Cancelled current setup flow.")

    @router.callback_query(F.data == "sc:cancel")
    async def cb_scoring_cancel(callback: CallbackQuery) -> None:
        if callback.from_user is None:
            return
        task = deps.scoring_tasks.get(callback.from_user.id)
        if task is None or task.done():
            await callback.answer("No scoring run in progress.", show_alert=False)
            return
        task.cancel()
        await callback.answer("Cancelling… completed batches are saved.", show_alert=False)

    async def _start_profile_wizard(target_msg: Message, user_id: int, state: FSMContext) -> None:
        profile = await deps.repo.get_or_create_profile(user_id)
        await state.set_state(ProfileWizard.target_roles)
        current = esc(profile.target_roles) if profile.target_roles else "<i>empty</i>"
        await target_msg.answer(
            "🧭 <b>Profile Setup (Step 1/5)</b> — send /cancel anytime or <code>-</code> to keep current.\n\n"
            f"Current target roles: {current}\n\n"
            "<b>What roles, domains, and tech stack are you targeting?</b>\n"
            "<i>Example: MSc Cybersecurity / Systems / Applied AI Security, Python, Rust, C++, distributed systems</i>"
        )

    async def _prompt_quick_threshold(target_msg: Message, user_id: int, state: FSMContext) -> None:
        profile = await deps.repo.get_or_create_profile(user_id)
        await state.set_state(ProfileWizard.quick_threshold)
        current = esc(format_threshold_label(profile.min_score_threshold))
        await target_msg.answer(
            "🎯 <b>Set Minimum Score Threshold</b> — send /cancel anytime.\n\n"
            f"Current threshold: <code>{current}</code>\n\n"
            "Enter a number from <b>0 to 10</b> (e.g. <code>6.5</code> or <code>7</code>) to hide "
            "lower-scoring offers in /digest and /browse, or send <code>off</code> to show all.\n"
            "<i>Note: Changing this threshold does not re-run paid LLM scoring.</i>"
        )

    @router.message(Command("profile"))
    async def cmd_profile(message: Message, command: CommandObject, state: FSMContext) -> None:
        if message.from_user is None:
            return
        uid = message.from_user.id
        raw_args = (command.args or "").strip()
        if not raw_args or raw_args.lower() in {"show", "view"}:
            profile = await deps.repo.get_or_create_profile(uid)
            await message.answer(
                format_profile_summary(profile),
                reply_markup=build_profile_keyboard(),
            )
            return

        parts = raw_args.split(None, 1)
        subcmd = parts[0].lower()
        if subcmd == "edit":
            await _start_profile_wizard(message, uid, state)
            return
        if subcmd in {"threshold", "min", "cutoff"}:
            if len(parts) < 2:
                await _prompt_quick_threshold(message, uid, state)
                return
            threshold = parse_min_score_threshold(parts[1])
            saved = await deps.repo.set_min_score_threshold(uid, threshold)
            await message.answer(
                f"✅ <b>Min score threshold updated to</b> "
                f"<code>{esc(format_threshold_label(saved.min_score_threshold))}</code>.\n"
                "<i>Cached offer scores are preserved (no re-scoring needed).</i>"
            )
            return

        raise ValidationError(
            "Usage: /profile (view settings), /profile edit (guided wizard), "
            "or /profile threshold <0-10|off>."
        )

    @router.callback_query(F.data == "prof:edit")
    async def cb_profile_edit(callback: CallbackQuery, state: FSMContext) -> None:
        if callback.from_user is None or not isinstance(callback.message, Message):
            return
        await callback.answer()
        await _start_profile_wizard(callback.message, callback.from_user.id, state)

    @router.callback_query(F.data == "prof:thresh")
    async def cb_profile_threshold(callback: CallbackQuery, state: FSMContext) -> None:
        if callback.from_user is None or not isinstance(callback.message, Message):
            return
        await callback.answer()
        await _prompt_quick_threshold(callback.message, callback.from_user.id, state)

    @router.message(ProfileWizard.quick_threshold)
    async def profile_step_quick_threshold(message: Message, state: FSMContext) -> None:
        if message.from_user is None or not message.text:
            return
        text = message.text.strip()
        uid = message.from_user.id
        if text == "-":
            profile = await deps.repo.get_or_create_profile(uid)
            saved = profile
        else:
            threshold = parse_min_score_threshold(text)
            saved = await deps.repo.set_min_score_threshold(uid, threshold)
        await state.clear()
        await message.answer(
            f"✅ <b>Min score threshold set to</b> "
            f"<code>{esc(format_threshold_label(saved.min_score_threshold))}</code>.\n"
            "<i>Cached offer scores are preserved (no re-scoring needed).</i>\n\n"
            f"{format_profile_summary(saved)}",
            reply_markup=build_profile_keyboard(),
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
            "📍 <b>Profile Setup (Step 2/5)</b>\n"
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
            "🏷 <b>Profile Setup (Step 3/5)</b>\n"
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
            "🚫 <b>Profile Setup (Step 4/5)</b>\n"
            f"Current dealbreakers/languages: {current}\n\n"
            "<b>Languages spoken &amp; hard dealbreakers?</b>\n"
            "<i>Example: Fluent English &amp; French; skip unpaid internships, pure frontend/UI, or PowerBI-only roles.</i>"
        )

    @router.message(ProfileWizard.dealbreakers)
    async def profile_step_dealbreakers(message: Message, state: FSMContext) -> None:
        if message.from_user is None or not message.text:
            return
        profile = await deps.repo.get_or_create_profile(message.from_user.id)
        text = message.text.strip()
        await state.update_data(dealbreakers=profile.dealbreakers if text == "-" else text)
        await state.set_state(ProfileWizard.min_score_threshold)
        current = esc(format_threshold_label(profile.min_score_threshold))
        await message.answer(
            "🎯 <b>Profile Setup (Step 5/5)</b>\n"
            f"Current min score threshold: <code>{current}</code>\n\n"
            "<b>Minimum score threshold (0–10) to display in /digest?</b>\n"
            "<i>Send a number (e.g. <code>6.5</code> or <code>7</code>), <code>off</code> to show all, "
            "or <code>-</code> to keep current. Does not trigger re-scoring.</i>"
        )

    @router.message(ProfileWizard.min_score_threshold)
    async def profile_step_threshold(message: Message, state: FSMContext) -> None:
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
        profile.dealbreakers = str(data.get("dealbreakers", profile.dealbreakers))
        if text != "-":
            profile.min_score_threshold = parse_min_score_threshold(text)

        saved = await deps.repo.save_profile(profile)
        await state.clear()
        await message.answer(
            "✅ <b>Profile saved!</b> (Stale offer scores will automatically re-compute if roles/locations/formats/dealbreakers changed.)\n\n"
            f"{format_profile_summary(saved)}\n\n"
            "Tip: You can also drop a <code>.pdf</code>, <code>.txt</code>, or <code>.md</code> CV anytime.",
            reply_markup=build_profile_keyboard(),
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
                "Simply attach and send your CV as a <code>.pdf</code>, <code>.txt</code>, or <code>.md</code> file right in this chat.\n\n"
                "⚠️ <b>Upload an anonymised CV only.</b> Replace your name, email, phone, and street "
                "address with mock data (e.g. <code>John Doe</code>, <code>john.doe@example.com</code>, "
                "region only like <code>Vaud</code>). The text is encrypted at rest and, while OpenRouter "
                "is connected, sent to the selected model provider. See the README for details."
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
            cv_text = await asyncio.to_thread(parse_cv_bytes, raw_bytes, filename)
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
            file_type = await asyncio.to_thread(detect_isacademia_file_type, raw_bytes)
            if file_type == "showroom":
                offers = await asyncio.to_thread(parse_showroom_html, raw_bytes)
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

            records = await asyncio.to_thread(parse_registrations_html, raw_bytes)
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
        parsed = parse_digest_args(command.args or "")
        await render_view_page(
            message,
            message.from_user.id,
            DigestQuery(
                view="new",
                page=0,
                max_age_days=parsed.max_age_days,
                recency_basis=parsed.recency_basis,
                force_local=parsed.force_local,
            ),
            edit_in_place=False,
        )

    @router.message(Command("browse", "starred", "seen", "applied", "excluded"))
    async def cmd_browse(message: Message, command: CommandObject) -> None:
        if message.from_user is None:
            return
        cmd_name = (command.command or "browse").lower()
        raw_args = (command.args or "").strip()
        if cmd_name in {"starred", "seen", "applied", "excluded"}:
            view = cmd_name
            extra_args = raw_args
        else:
            parts = raw_args.split(None, 1)
            if not parts:
                view = "all"
                extra_args = ""
            elif parts[0].lower() in {"all", "new", "seen", "starred", "applied", "excluded"}:
                view = parts[0].lower()
                extra_args = parts[1] if len(parts) > 1 else ""
            else:
                # Allow shorthand like `/browse 7d` (defaults view to 'all')
                view = "all"
                extra_args = raw_args
        parsed = parse_digest_args(extra_args)
        await render_view_page(
            message,
            message.from_user.id,
            DigestQuery(
                view=view,
                page=0,
                max_age_days=parsed.max_age_days,
                recency_basis=parsed.recency_basis,
                force_local=parsed.force_local,
            ),
            edit_in_place=False,
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
                    f"Model '{arg}' was not found in OpenRouter's catalog. "
                    "Search first with e.g. /model deepseek."
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
        if callback.from_user is None or not isinstance(callback.message, Message) or not callback.data:
            return
        _, query, _ = decode_query_callback(callback.data)
        await callback.answer()
        await render_view_page(
            callback.message,
            callback.from_user.id,
            query,
            edit_in_place=True,
        )

    @router.callback_query(F.data.startswith("ms:"))
    async def cb_mark_page_seen(callback: CallbackQuery) -> None:
        if callback.from_user is None or not isinstance(callback.message, Message) or not callback.data:
            return
        _, query, _ = decode_query_callback(callback.data)
        uid = callback.from_user.id
        msg = callback.message
        ctx = deps.get_digest_context(msg.chat.id, msg.message_id)
        page_size = ctx.page_size if ctx is not None else deps.settings.digest_page_size

        if ctx is not None and ctx.query == query:
            page_uids = ctx.ordered_uids[query.page * page_size : (query.page + 1) * page_size]
        else:
            prepared = await _prepare_digest_data(uid, query)
            page_uids = [
                item.offer.uid
                for item in prepared.ranked[query.page * page_size : (query.page + 1) * page_size]
            ]

        marked = await deps.repo.mark_offers_seen(uid, page_uids)
        await callback.answer(
            f"👀 Marked {marked} offer(s) as seen" if marked else "Page already seen"
        )
        await render_view_page(msg, uid, query, edit_in_place=True)

    @router.callback_query(F.data.startswith(("vw:", "oc:")))
    async def cb_open_card(callback: CallbackQuery) -> None:
        if callback.from_user is None or not isinstance(callback.message, Message) or not callback.data:
            return
        _, query, raw_idx = decode_query_callback(callback.data)
        try:
            idx = max(0, int(raw_idx))
        except ValueError as exc:
            raise ValidationError("Invalid offer index. Please refresh with /digest.") from exc

        uid = callback.from_user.id
        msg = callback.message
        ctx = deps.get_digest_context(msg.chat.id, msg.message_id)
        if ctx is None or idx >= len(ctx.ordered_uids):
            # Stateless restart-safe fallback: rebuild context from the encoded DigestQuery.
            prepared = await _prepare_digest_data(uid, query)
            if not prepared.ranked:
                await callback.answer("No offers remain in this view.", show_alert=False)
                await render_view_page(msg, uid, query, edit_in_place=True)
                return
            ctx = DigestMessageContext(
                query=query,
                ordered_uids=[item.offer.uid for item in prepared.ranked],
                page_size=deps.settings.digest_page_size,
            )
            deps.set_digest_context(msg.chat.id, msg.message_id, ctx)

        clamped_idx = max(0, min(idx, len(ctx.ordered_uids) - 1))
        offer_uid = ctx.ordered_uids[clamped_idx]
        await callback.answer()
        rendered = await _render_offer_card_in_place(
            msg,
            uid,
            offer_uid,
            ctx=ctx,
            index=clamped_idx,
            mark_seen=True,
        )
        if not rendered:
            await render_view_page(msg, uid, query, edit_in_place=True)

    @router.callback_query(F.data.startswith("od:"))
    async def cb_offer_description(callback: CallbackQuery) -> None:
        if callback.from_user is None or not isinstance(callback.message, Message) or not callback.data:
            return
        parts = callback.data.split(":", 2)
        if len(parts) != 3:
            await callback.answer("Invalid description request", show_alert=False)
            return
        _, raw_part, offer_uid = parts
        try:
            part = max(0, int(raw_part))
        except ValueError:
            part = 0

        uid = callback.from_user.id
        msg = callback.message
        item = await deps.repo.get_enriched_offer(uid, offer_uid)
        if item is None:
            await callback.answer("Offer not found.", show_alert=True)
            return

        await callback.answer()
        ctx = deps.get_digest_context(msg.chat.id, msg.message_id)
        back_query = ctx.query if ctx is not None else None
        text, clamped, total_parts = format_offer_description_page(item, part=part)
        await _safe_edit_text(
            msg,
            text,
            reply_markup=build_description_keyboard(
                item,
                part=clamped,
                total_parts=total_parts,
                back_query=back_query,
            ),
        )

    @router.callback_query(F.data.startswith("ob:"))
    async def cb_back_to_card(callback: CallbackQuery) -> None:
        if callback.from_user is None or not isinstance(callback.message, Message) or not callback.data:
            return
        offer_uid = callback.data.split(":", 1)[1]
        uid = callback.from_user.id
        msg = callback.message
        ctx = deps.get_digest_context(msg.chat.id, msg.message_id)
        await callback.answer()
        rendered = await _render_offer_card_in_place(
            msg,
            uid,
            offer_uid,
            ctx=ctx,
            mark_seen=False,
        )
        if not rendered:
            await callback.answer("Offer not found.", show_alert=True)

    @router.callback_query(F.data.startswith("of:"))
    async def cb_offer_action_in_description(callback: CallbackQuery) -> None:
        if callback.from_user is None or not isinstance(callback.message, Message) or not callback.data:
            return
        parts = callback.data.split(":", 3)
        if len(parts) != 4:
            await callback.answer("Invalid action", show_alert=False)
            return
        _, action, raw_part, offer_uid = parts
        try:
            part = max(0, int(raw_part))
        except ValueError:
            part = 0

        uid = callback.from_user.id
        msg = callback.message
        updated, toast = await _apply_offer_action(uid, action, offer_uid)
        if updated is None:
            await callback.answer("Offer not found in database.", show_alert=True)
            return

        await callback.answer(toast)
        ctx = deps.get_digest_context(msg.chat.id, msg.message_id)
        back_query = ctx.query if ctx is not None else None
        text, clamped, total_parts = format_offer_description_page(updated, part=part)
        await _safe_edit_text(
            msg,
            text,
            reply_markup=build_description_keyboard(
                updated,
                part=clamped,
                total_parts=total_parts,
                back_query=back_query,
            ),
        )

    @router.callback_query(F.data.startswith("o:"))
    async def cb_offer_action(callback: CallbackQuery) -> None:
        if callback.from_user is None or not isinstance(callback.message, Message) or not callback.data:
            return
        parts = callback.data.split(":", 2)
        if len(parts) != 3:
            await callback.answer("Invalid action", show_alert=False)
            return
        _, action, offer_uid = parts
        uid = callback.from_user.id
        msg = callback.message
        ctx = deps.get_digest_context(msg.chat.id, msg.message_id)

        # Backwards compatibility for older cards with `o:d:<uid>`
        if action == "d":
            item = await deps.repo.get_enriched_offer(uid, offer_uid)
            if item is None:
                await callback.answer("Offer not found.", show_alert=True)
                return
            await callback.answer()
            back_query = ctx.query if ctx is not None else None
            text, clamped, total_parts = format_offer_description_page(item, part=0)
            await _safe_edit_text(
                msg,
                text,
                reply_markup=build_description_keyboard(
                    item,
                    part=clamped,
                    total_parts=total_parts,
                    back_query=back_query,
                ),
            )
            return

        updated, toast = await _apply_offer_action(uid, action, offer_uid)
        if updated is None:
            await callback.answer("Offer not found in database.", show_alert=True)
            return

        await callback.answer(toast)
        await _render_offer_card_in_place(
            msg,
            uid,
            offer_uid,
            ctx=ctx,
            mark_seen=False,
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
            with contextlib.suppress(Exception):
                await update.callback_query.answer("Error occurred — see message.", show_alert=False)
            if isinstance(update.callback_query.message, Message):
                await update.callback_query.message.answer(reply_text)
        elif update.message is not None:
            await update.message.answer(reply_text)

    return router
