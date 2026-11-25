"""Wizard prompts are split: step details (HTML) + raw copyable current value (plain)."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from give_me_job_please.bot.formatting import format_threshold_value
from give_me_job_please.bot.handlers import BotDependencies, ProfileWizard, build_router
from give_me_job_please.config import Settings
from give_me_job_please.core.auth.crypto import SecretBox
from give_me_job_please.core.auth.openrouter_pkce import OpenRouterPkceManager
from give_me_job_please.core.models import UserProfile
from give_me_job_please.core.scoring.local_embedder import LocalEmbedderScorer
from give_me_job_please.storage.db import create_engine_and_sessionmaker
from give_me_job_please.storage.orm import Base
from give_me_job_please.storage.repository import OfferRepository


class _RecordingMessage:
    def __init__(self) -> None:
        self.answers: list[tuple[str, Any]] = []

    async def answer(self, text: str, parse_mode: Any = "HTML", **kwargs: Any) -> None:
        self.answers.append((text, parse_mode))


class _RecordingState:
    def __init__(self) -> None:
        self.state: Any = None
        self.data: dict[str, Any] = {}

    async def set_state(self, state: Any) -> None:
        self.state = state

    async def update_data(self, **kwargs: Any) -> None:
        self.data.update(kwargs)

    async def get_data(self) -> dict[str, Any]:
        return dict(self.data)

    async def clear(self) -> None:
        self.state = None
        self.data = {}


async def _make_deps() -> tuple[BotDependencies, Any]:
    engine, session_factory = create_engine_and_sessionmaker("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    box = SecretBox(os.urandom(32))
    settings = Settings(
        _env_file=None,
        telegram_bot_token="123456789:" + "A" * 35,
        telegram_allowed_user_ids=[1],
        encryption_key=base64.urlsafe_b64encode(os.urandom(32)).decode(),
    )
    deps = BotDependencies(
        settings=settings,
        repo=OfferRepository(session_factory, secret_box=box),
        secret_box=box,
        pkce_manager=cast(OpenRouterPkceManager, object()),
        local_scorer=cast(LocalEmbedderScorer, object()),
        error_log_path=Path("errors.jsonl"),
    )
    return deps, engine


def _find_handler(router: Any, name: str) -> Any:
    for observer in (router.message, router.callback_query):
        for h in observer.handlers:
            if getattr(h.callback, "__name__", "") == name:
                return h.callback
    raise AssertionError(f"handler {name} not found")


def _find_closure_fn(router: Any, name: str) -> Any:
    def walk(fn: Any, depth: int = 3) -> Any:
        if getattr(fn, "__name__", "") == name:
            return fn
        if depth <= 0:
            return None
        for cell in getattr(fn, "__closure__", None) or ():
            try:
                val = cell.cell_contents
            except ValueError:
                continue
            if callable(val):
                found = walk(val, depth - 1)
                if found is not None:
                    return found
        return None

    for observer in (router.message, router.callback_query):
        for h in observer.handlers:
            found = walk(h.callback)
            if found is not None:
                return found
    raise AssertionError(f"closure fn {name} not found")


def test_format_threshold_value_is_sendable_input() -> None:
    assert format_threshold_value(None) == "off"
    assert format_threshold_value(0.0) == "off"
    assert format_threshold_value(7.0) == "7"
    assert format_threshold_value(6.5) == "6.5"


async def test_wizard_step1_sends_prompt_then_plain_current_value() -> None:
    deps, engine = await _make_deps()
    try:
        uid = 1
        await deps.repo.save_profile(
            UserProfile(
                user_id=uid,
                target_roles="MSc Cybersecurity <AI> & Python",
                preferred_locations="Lausanne, Remote",
                accepted_formats=["Internship"],
                dealbreakers="No unpaid",
                min_score_threshold=6.5,
            )
        )
        router = build_router(deps)
        start = _find_closure_fn(router, "_start_profile_wizard")
        msg = _RecordingMessage()
        state = _RecordingState()
        await start(cast(Any, msg), uid, cast(Any, state))

        assert state.state == ProfileWizard.target_roles
        assert len(msg.answers) == 2
        prompt, current = msg.answers
        # Message 1: step details, no current value embedded.
        assert "Step 1/5" in prompt[0]
        assert "<AI>" not in prompt[0]
        # Message 2: raw value, plain text, unescaped.
        assert current == ("MSc Cybersecurity <AI> & Python", None)
    finally:
        await engine.dispose()


async def test_wizard_empty_values_use_sendable_fallbacks() -> None:
    deps, engine = await _make_deps()
    try:
        uid = 1
        await deps.repo.get_or_create_profile(uid)  # empty profile
        router = build_router(deps)
        start = _find_closure_fn(router, "_start_profile_wizard")
        msg = _RecordingMessage()
        await start(cast(Any, msg), uid, cast(Any, _RecordingState()))
        assert msg.answers[1] == ("-", None)

        # Step 3 (formats): empty list → 'any'; step 5 threshold: unset → 'off'.
        step_locations = _find_handler(router, "profile_step_locations")
        msg2 = _RecordingMessage()
        msg2.text = "-"  # type: ignore[attr-defined]
        msg2.from_user = SimpleNamespace(id=uid)  # type: ignore[attr-defined]
        await step_locations(cast(Any, msg2), cast(Any, _RecordingState()))
        assert msg2.answers[1] == ("any", None)

        quick = _find_closure_fn(router, "_prompt_quick_threshold")
        msg3 = _RecordingMessage()
        await quick(cast(Any, msg3), uid, cast(Any, _RecordingState()))
        assert msg3.answers[1] == ("off", None)
    finally:
        await engine.dispose()


async def test_wizard_threshold_current_value_roundtrips_through_parser() -> None:
    from give_me_job_please.core.diff import parse_min_score_threshold

    deps, engine = await _make_deps()
    try:
        uid = 1
        await deps.repo.set_min_score_threshold(uid, 6.5)
        router = build_router(deps)
        quick = _find_closure_fn(router, "_prompt_quick_threshold")
        msg = _RecordingMessage()
        await quick(cast(Any, msg), uid, cast(Any, _RecordingState()))
        value_text, parse_mode = msg.answers[1]
        assert parse_mode is None
        # The copied value must be accepted verbatim by the threshold parser.
        assert parse_min_score_threshold(value_text) == 6.5
    finally:
        await engine.dispose()