"""Regression tests for cancellable scoring runs (cooperative asyncio cancellation)."""

from __future__ import annotations

import asyncio
import json
import re
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from give_me_job_please.core.models import JobOffer, UserProfile
from give_me_job_please.core.scoring.local_embedder import LocalEmbedderScorer
from give_me_job_please.core.scoring.openrouter_scorer import OpenRouterScorer
from give_me_job_please.errors import ScorerError


def _offer(i: int) -> JobOffer:
    return JobOffer(
        uid=f"isacademia:{i}",
        external_id=str(i),
        title=f"Offer {i}",
        company="Lab",
        assignment_location="Lausanne",
        description="Research internship in applied security.",
    )


def _llm_response(uids: list[str]) -> Any:
    payload = (
        '{"results": ['
        + ",".join(
            f'{{"uid": "{u}", "score": 8.0, "location_match": true, '
            '"one_liner": "Good fit", "tags": ["AI"]}'
            for u in uids
        )
        + "]}"
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=payload))])


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