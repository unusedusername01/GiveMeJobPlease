"""/cv sends a plain-text, copyable preview message with truncation."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from give_me_job_please.bot.handlers import _CV_PREVIEW_CHARS, BotDependencies, build_router
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
        self.from_user = SimpleNamespace(id=1)

    async def answer(self, text: str, parse_mode: Any = "HTML", **kwargs: Any) -> None:
        self.answers.append((text, parse_mode))


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


def _cmd_cv_handler(router: Any) -> Any:
    for h in router.message.handlers:
        if getattr(h.callback, "__name__", "") == "cmd_cv":
            return h.callback
    raise AssertionError("cmd_cv handler not found")


async def test_cv_preview_is_plain_unescaped_and_truncated() -> None:
    deps, engine = await _make_deps()
    try:
        raw = "Skills: C++ <templates> & Rust\n" + ("line of CV text\n" * 400)
        await deps.repo.save_profile(
            UserProfile(user_id=1, cv_text=raw, cv_filename="cv.pdf")
        )
        handler = _cmd_cv_handler(build_router(deps))
        msg = _RecordingMessage()
        await handler(cast(Any, msg))

        assert len(msg.answers) == 2
        summary, preview = msg.answers
        assert "truncated" in summary[0]
        assert preview[1] is None  # plain text, no HTML parsing
        assert preview[0] == raw[:_CV_PREVIEW_CHARS]
        assert "<templates>" in preview[0]  # not HTML-escaped
        assert len(preview[0]) <= _CV_PREVIEW_CHARS < 4096
    finally:
        await engine.dispose()


async def test_cv_preview_short_text_not_marked_truncated() -> None:
    deps, engine = await _make_deps()
    try:
        await deps.repo.save_profile(
            UserProfile(user_id=1, cv_text="Short CV", cv_filename="cv.txt")
        )
        handler = _cmd_cv_handler(build_router(deps))
        msg = _RecordingMessage()
        await handler(cast(Any, msg))
        assert "Full extracted text" in msg.answers[0][0]
        assert msg.answers[1] == ("Short CV", None)
    finally:
        await engine.dispose()