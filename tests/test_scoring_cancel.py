"""Regression tests for cancellable scoring runs (cooperative asyncio cancellation)."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from give_me_job_please.bot.formatting import build_scoring_cancel_keyboard
from give_me_job_please.bot.handlers import BotDependencies, build_router
from give_me_job_please.config import Settings
from give_me_job_please.core.auth.crypto import SecretBox
from give_me_job_please.core.auth.openrouter_pkce import OpenRouterPkceManager
from give_me_job_please.core.models import EnrichedOffer, JobOffer, UserProfile
from give_me_job_please.core.scoring.local_embedder import LocalEmbedderScorer
from give_me_job_please.core.scoring.openrouter_scorer import OpenRouterScorer
from give_me_job_please.errors import ScorerError
from give_me_job_please.storage.db import create_engine_and_sessionmaker
from give_me_job_please.storage.orm import Base
from give_me_job_please.storage.repository import OfferRepository


def _offer(i: int) -> JobOffer:
    return JobOffer(
        uid=f"isacademia:{i}",
        external_id=str(i),
        title=f"Offer {i}",
        company="Lab",
        assignment_location="Lausanne",
        description="Research internship in applied security.",
    )


def _llm_response(uids: list[str], cost: float | None = 0.001) -> Any:
    payload = (
        '{"results": ['
        + ",".join(
            f'{{"uid": "{u}", "score": 8.0, "location_match": true, '
            '"one_liner": "Good fit", "tags": ["AI"]}'
            for u in uids
        )
        + "]}"
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=payload))],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20, cost=cost),
    )


class _BlockingChat:
    """Fake SDK chat namespace: batch 1 returns instantly, batch 2 blocks forever."""

    def __init__(self) -> None:
        self.calls = 0
        self.second_started = asyncio.Event()

    async def send_async(self, **kwargs: Any) -> Any:
        self.calls += 1
        m = re.search(
            r"<offers_json>\n(.*?)\n</offers_json>", kwargs["messages"][1]["content"], re.DOTALL
        )
        assert m is not None
        uids = [o["uid"] for o in json.loads(m.group(1))]
        if self.calls == 1:
            return _llm_response(uids)
        self.second_started.set()
        await asyncio.Event().wait()  # block until cancelled
        raise AssertionError("unreachable")


class _BlockingClient:
    def __init__(self, **kwargs: Any) -> None:
        self.chat = _BlockingChat()

    async def __aenter__(self) -> _BlockingClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


async def test_openrouter_cancel_preserves_checkpointed_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "give_me_job_please.core.scoring.openrouter_scorer.OpenRouter", _BlockingClient
    )
    scorer = OpenRouterScorer(api_key="k", model_id="m", batch_size=2)
    offers = [_offer(i) for i in range(6)]  # 3 batches of 2
    checkpointed: list[str] = []

    async def checkpoint(scores: list[Any]) -> None:
        checkpointed.extend(s.offer_uid for s in scores)

    task = asyncio.create_task(
        scorer.score_offers(UserProfile(user_id=1), offers, "fp", on_batch_complete=checkpoint)
    )
    # Wait until batch 2 is in flight, then cancel the run.
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Batch 1 was checkpointed before cancellation; nothing from batch 2/3 leaked.
    assert checkpointed == ["isacademia:0", "isacademia:1"]


async def test_openrouter_retry_does_not_swallow_cancellation() -> None:
    scorer = OpenRouterScorer(api_key="k", model_id="m")

    class _CancellingChat:
        async def send_async(self, **kwargs: Any) -> Any:
            raise asyncio.CancelledError

    client = SimpleNamespace(chat=_CancellingChat())
    with pytest.raises(asyncio.CancelledError):
        await scorer._call_with_retry(client, [])  # type: ignore[arg-type]


async def test_openrouter_generic_failure_still_wrapped_as_scorer_error() -> None:
    scorer = OpenRouterScorer(api_key="k", model_id="m")

    class _BrokenChat:
        async def send_async(self, **kwargs: Any) -> Any:
            raise RuntimeError("HTTP 418 teapot")

    client = SimpleNamespace(chat=_BrokenChat())
    with pytest.raises(ScorerError):
        await scorer._call_with_retry(client, [])  # type: ignore[arg-type]


class _SlowModel:
    """Fake fastembed model: each chunk takes a moment, records chunk sizes."""

    def __init__(self) -> None:
        self.chunks: list[int] = []

    def embed(self, texts: list[str]) -> Any:
        self.chunks.append(len(texts))
        time.sleep(0.05)
        for _ in texts:
            yield np.ones(4, dtype=np.float32)


async def test_local_embedder_cancellable_between_chunks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    scorer = LocalEmbedderScorer(model_name="fake", cache_dir=tmp_path)
    model = _SlowModel()
    scorer._model = model  # bypass real fastembed load

    task = asyncio.create_task(scorer.embed_texts([f"text {i}" for i in range(200)]))
    # Poll until the first chunk has been handed to the worker thread, then cancel
    # while it is still running (thread sleeps 50 ms per chunk).
    for _ in range(200):
        if model.chunks:
            break
        await asyncio.sleep(0.005)
    assert model.chunks == [64]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The in-flight chunk finished in its thread, but no further chunk was started.
    assert model.chunks == [64]


def test_scoring_cancel_keyboard_respects_64_byte_limit() -> None:
    kb = build_scoring_cancel_keyboard()
    cbs = [btn.callback_data or "" for row in kb.inline_keyboard for btn in row]
    assert cbs == ["sc:cancel"]
    assert all(len(cb.encode("utf-8")) < 64 for cb in cbs)


class _RecordingMessage:
    """Minimal aiogram Message stand-in capturing edit_text calls."""

    def __init__(self) -> None:
        self.edits: list[tuple[str, Any]] = []
        self.chat = SimpleNamespace(id=1)
        self.message_id = 1

    async def edit_text(self, text: str, reply_markup: Any = None) -> None:
        self.edits.append((text, reply_markup))


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
        openrouter_batch_size=2,
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


def _enriched(i: int) -> EnrichedOffer:
    return EnrichedOffer(offer=_offer(i))


async def test_bot_cancel_edits_status_and_resume_rescores_only_unscored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deps, engine = await _make_deps()
    try:
        uid = 1
        await deps.repo.ingest_showroom_offers(
            user_id=uid,
            offers=[_offer(i) for i in range(6)],
            filename="showroom.htm",
            parser_version="test",
        )
        await deps.repo.save_openrouter_credentials(
            uid,
            key_enc=deps.secret_box.encrypt("sk-test", associated_data=f"openrouter:{uid}"),
            key_hash="h",
            default_model="m",
        )
        monkeypatch.setattr(
            "give_me_job_please.core.scoring.openrouter_scorer.OpenRouter", _BlockingClient
        )
        # ensure_scored is a closure inside build_router; extract it from a handler cell.
        router = build_router(deps)
        def _find_ensure(fn: Any, depth: int = 3) -> Any:
            if getattr(fn, "__name__", "") == "ensure_scored":
                return fn
            if depth <= 0:
                return None
            for cell in getattr(fn, "__closure__", None) or ():
                try:
                    val = cell.cell_contents
                except ValueError:
                    continue
                if callable(val):
                    found = _find_ensure(val, depth - 1)
                    if found is not None:
                        return found
            return None

        ensure = None
        for observer in (router.message, router.callback_query):
            for h in observer.handlers:
                ensure = _find_ensure(h.callback)
                if ensure is not None:
                    break
            if ensure is not None:
                break
        assert ensure is not None

        items = [_enriched(i) for i in range(6)]
        status = _RecordingMessage()

        async def _run() -> None:
            await ensure(uid, items, status_msg=cast(Any, status))

        task = asyncio.create_task(_run())
        await asyncio.sleep(0.05)
        assert uid in deps.scoring_tasks
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Registry cleaned up; status message shows the cancelled summary.
        assert uid not in deps.scoring_tasks
        assert any("Scoring cancelled" in text for text, _ in status.edits)
        assert any("2 of 6" in text for text, _ in status.edits)
        # The scoring status edit carried the cancel button.
        assert any(markup is not None for _, markup in status.edits)

        # Resume: a fresh run must only re-score the 4 unscored offers.
        seen_batches: list[int] = []

        class _ResumeChat:
            async def send_async(self, **kwargs: Any) -> Any:
                m = re.search(
                    r"<offers_json>\n(.*?)\n</offers_json>",
                    kwargs["messages"][1]["content"],
                    re.DOTALL,
                )
                uids = [o["uid"] for o in json.loads(m.group(1))]
                seen_batches.append(len(uids))
                return _llm_response(uids)

        class _ResumeClient:
            def __init__(self, **kwargs: Any) -> None:
                self.chat = _ResumeChat()

            async def __aenter__(self) -> _ResumeClient:
                return self

            async def __aexit__(self, *exc: Any) -> None:
                return None

        monkeypatch.setattr(
            "give_me_job_please.core.scoring.openrouter_scorer.OpenRouter", _ResumeClient
        )
        # Reload from DB like the real /digest flow: checkpointed fingerprints must
        # survive so only the 4 unscored offers are re-sent (two batches of 2).
        items2 = await deps.repo.list_enriched_offers(uid)
        assert len(items2) == 6
        await ensure(uid, items2)
        assert seen_batches == [2, 2]
        assert all(item.score == 8.0 for item in items2)
    finally:
        await engine.dispose()


async def test_local_embedder_chunking_preserves_order_and_results(tmp_path: Any) -> None:
    scorer = LocalEmbedderScorer(model_name="fake", cache_dir=tmp_path)

    class _Model:
        def embed(self, texts: list[str]) -> Any:
            for t in texts:
                yield np.full(4, float(t.split()[-1]), dtype=np.float32)

    scorer._model = _Model()
    vecs = await scorer.embed_texts([f"t {i}" for i in range(130)])
    assert len(vecs) == 130
    assert float(vecs[129][0]) == 129.0