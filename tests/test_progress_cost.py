"""Per-batch progress + cost meter fed by OpenRouter's automatic usage accounting."""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from give_me_job_please.bot.handlers import BotDependencies, build_router
from give_me_job_please.config import Settings
from give_me_job_please.core.auth.crypto import SecretBox
from give_me_job_please.core.auth.openrouter_pkce import OpenRouterPkceManager
from give_me_job_please.core.models import JobOffer, UserProfile
from give_me_job_please.core.scoring.local_embedder import LocalEmbedderScorer
from give_me_job_please.core.scoring.openrouter_scorer import (
    BatchUsage,
    OpenRouterScorer,
    ProgressUpdate,
    _extract_usage,
)
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
    )


def test_extract_usage_handles_missing_and_null_cost() -> None:
    assert _extract_usage(SimpleNamespace()) == BatchUsage()
    resp = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, cost=None)
    )
    assert _extract_usage(resp) == BatchUsage(prompt_tokens=10, completion_tokens=5, cost=None)
    resp2 = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, cost=0.002)
    )
    assert _extract_usage(resp2).cost == 0.002
    # UNSET-like sentinel (non-numeric) must not crash or fabricate a cost.
    resp3 = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, cost="UNSET"))
    assert _extract_usage(resp3).cost is None


class _UsageClient:
    """Fake SDK client returning per-batch usage; cost omitted on the last batch."""

    def __init__(self, **kwargs: Any) -> None:
        self.chat = self
        self.calls = 0

    async def send_async(self, **kwargs: Any) -> Any:
        self.calls += 1
        m = re.search(
            r"<offers_json>\n(.*?)\n</offers_json>",
            kwargs["messages"][1]["content"],
            re.DOTALL,
        )
        uids = [o["uid"] for o in json.loads(m.group(1))]
        payload = (
            '{"results": ['
            + ",".join(
                f'{{"uid": "{u}", "score": 7.0, "location_match": true, '
                '"one_liner": "ok", "tags": []}'
                for u in uids
            )
            + "]}"
        )
        cost = 0.001 if self.calls < 3 else None
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=payload))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20, cost=cost),
        )

    async def __aenter__(self) -> _UsageClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


async def test_progress_updates_are_cumulative_and_monotonic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "give_me_job_please.core.scoring.openrouter_scorer.OpenRouter", _UsageClient
    )
    scorer = OpenRouterScorer(api_key="k", model_id="m", batch_size=2)
    updates: list[ProgressUpdate] = []

    async def on_progress(u: ProgressUpdate) -> None:
        updates.append(u)

    scores = await scorer.score_offers(
        UserProfile(user_id=1), [_offer(i) for i in range(5)], "fp", on_progress=on_progress
    )
    assert len(scores) == 5
    assert [u.batch_index for u in updates] == [1, 2, 3]
    assert all(u.total_batches == 3 and u.total_offers == 5 for u in updates)
    assert [u.scored_offers for u in updates] == [2, 4, 5]
    assert [u.prompt_tokens for u in updates] == [100, 200, 300]
    assert [u.completion_tokens for u in updates] == [20, 40, 60]
    # Third batch omits cost: cumulative stays at the last known value.
    assert updates[0].cost == pytest.approx(0.001)
    assert updates[1].cost == pytest.approx(0.002)
    assert updates[2].cost == pytest.approx(0.002)


async def test_progress_none_cost_when_no_batch_reports_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NoCostClient(_UsageClient):
        async def send_async(self, **kwargs: Any) -> Any:
            resp = await super().send_async(**kwargs)
            resp.usage.cost = None
            return resp

    monkeypatch.setattr(
        "give_me_job_please.core.scoring.openrouter_scorer.OpenRouter", _NoCostClient
    )
    scorer = OpenRouterScorer(api_key="k", model_id="m", batch_size=5)
    updates: list[ProgressUpdate] = []

    async def on_progress(u: ProgressUpdate) -> None:
        updates.append(u)

    await scorer.score_offers(
        UserProfile(user_id=1), [_offer(i) for i in range(3)], "fp", on_progress=on_progress
    )
    assert len(updates) == 1
    assert updates[0].cost is None


class _RecordingMessage:
    def __init__(self) -> None:
        self.edits: list[str] = []

    async def edit_text(self, text: str, reply_markup: Any = None) -> None:
        self.edits.append(text)


async def test_bot_progress_edits_are_throttled_and_last_run_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, session_factory = create_engine_and_sessionmaker("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        box = SecretBox(os.urandom(32))
        settings = Settings(
            _env_file=None,
            telegram_bot_token="123456789:" + "A" * 35,
            telegram_allowed_user_ids=[1],
            encryption_key=base64.urlsafe_b64encode(os.urandom(32)).decode(),
            openrouter_batch_size=1,
        )
        deps = BotDependencies(
            settings=settings,
            repo=OfferRepository(session_factory, secret_box=box),
            secret_box=box,
            pkce_manager=cast(OpenRouterPkceManager, object()),
            local_scorer=cast(LocalEmbedderScorer, object()),
            error_log_path=Path("errors.jsonl"),
        )
        uid = 1
        await deps.repo.ingest_showroom_offers(
            user_id=uid,
            offers=[_offer(i) for i in range(4)],
            filename="s.htm",
            parser_version="test",
        )
        await deps.repo.save_openrouter_credentials(
            uid,
            key_enc=box.encrypt("sk-test", associated_data=f"openrouter:{uid}"),
            key_hash="h",
            default_model="m",
        )
        monkeypatch.setattr(
            "give_me_job_please.core.scoring.openrouter_scorer.OpenRouter", _UsageClient
        )

        def walk(fn: Any, depth: int = 3) -> Any:
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
                    found = walk(val, depth - 1)
                    if found is not None:
                        return found
            return None

        router = build_router(deps)
        ensure = None
        for observer in (router.message, router.callback_query):
            for h in observer.handlers:
                ensure = walk(h.callback)
                if ensure is not None:
                    break
            if ensure is not None:
                break
        assert ensure is not None

        items = await deps.repo.list_enriched_offers(uid)
        status = _RecordingMessage()
        await ensure(uid, items, status_msg=cast(Any, status))

        # 4 batches complete near-instantly: initial edit + first progress + final
        # progress only (middle edits throttled by the 3 s interval).
        progress_edits = [t for t in status.edits if "batch" in t]
        assert len(progress_edits) == 2
        assert "4/4" in progress_edits[-1] and "batch 4/4" in progress_edits[-1]
        assert "$0.0020" in progress_edits[-1]  # batches 1-2 reported cost, 3-4 None

        last = deps.last_run_usage[uid]
        assert last.offers == 4 and not last.cancelled
        assert last.prompt_tokens == 400 and last.completion_tokens == 80
        assert last.cost == pytest.approx(0.002)
    finally:
        await engine.dispose()