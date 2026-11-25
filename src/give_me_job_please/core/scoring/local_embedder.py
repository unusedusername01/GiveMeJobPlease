"""
Mode A (Zero-Key Default): Local multilingual semantic similarity scorer.

Uses `fastembed` with ONNX Runtime (CPU-only, no PyTorch) and caches Float32
vectors in SQLite so weekly runs only embed newly added offers.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import structlog

from give_me_job_please.core.models import JobOffer, OfferScore, UserProfile
from give_me_job_please.errors import ScorerError

logger = structlog.get_logger(__name__)

LOCAL_PROMPT_VERSION = "fastembed_cosine:1.0"


def vector_to_bytes(vec: np.ndarray) -> bytes:
    """Serialize a 1D float32 numpy vector for SQLite BLOB storage."""
    return np.asarray(vec, dtype=np.float32).tobytes()


def bytes_to_vector(blob: bytes) -> np.ndarray:
    """Deserialize a SQLite BLOB back into a float32 numpy vector."""
    return np.frombuffer(blob, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity in [-1.0, 1.0]."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class LocalEmbedderScorer:
    """Thread-safe, lazily loaded FastEmbed ONNX multilingual scorer."""

    engine_name = "local_fastembed"
    prompt_version = LOCAL_PROMPT_VERSION

    def __init__(self, *, model_name: str, cache_dir: Path) -> None:
        self.model_id = model_name
        self._cache_dir = cache_dir
        self._model: object | None = None
        self._load_lock = asyncio.Lock()

    async def _get_model(self) -> object:
        if self._model is not None:
            return self._model
        async with self._load_lock:
            if self._model is not None:
                return self._model
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "loading_local_embedding_model",
                model_name=self.model_id,
                cache_dir=str(self._cache_dir),
            )
            try:
                from fastembed import TextEmbedding

                self._model = await asyncio.to_thread(
                    TextEmbedding,
                    model_name=self.model_id,
                    cache_dir=str(self._cache_dir),
                )
            except Exception as exc:
                raise ScorerError(
                    "Failed to initialize the local Hugging Face ONNX embedding model. "
                    "Check container disk/memory limits or logs.",
                    detail=f"fastembed init failed for {self.model_id}: {exc}",
                ) from exc
            return self._model

    async def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a batch of texts off the asyncio event loop."""
        if not texts:
            return []
        model = await self._get_model()

        def _run() -> list[np.ndarray]:
            # fastembed returns a generator of 1D float32 arrays
            return [np.asarray(v, dtype=np.float32) for v in model.embed(texts)]  # type: ignore[attr-defined]

        try:
            return await asyncio.to_thread(_run)
        except Exception as exc:
            raise ScorerError(
                "Local embedding computation failed. Please inspect /logs.",
                detail=f"fastembed.embed failed: {exc}",
            ) from exc

    async def score_with_cache(
        self,
        profile: UserProfile,
        offers: list[JobOffer],
        *,
        fingerprint: str,
        cached_profile_vec: np.ndarray | None,
        cached_offer_vecs: dict[str, np.ndarray],
    ) -> tuple[list[OfferScore], np.ndarray, dict[str, np.ndarray]]:
        """
        Score all offers using cached vectors where available.

        Returns:
            (scores, profile_vector, newly_computed_offer_vectors)
        """
        if cached_profile_vec is not None:
            profile_vec = cached_profile_vec
        else:
            profile_vec = (await self.embed_texts([profile.render_for_matching()]))[0]

        missing_offers = [o for o in offers if o.uid not in cached_offer_vecs]
        new_vecs: dict[str, np.ndarray] = {}
        if missing_offers:
            logger.info("embedding_new_offers_locally", count=len(missing_offers))
            computed = await self.embed_texts([o.embedding_text() for o in missing_offers])
            for offer, vec in zip(missing_offers, computed, strict=True):
                new_vecs[offer.uid] = vec

        all_vecs = {**cached_offer_vecs, **new_vecs}
        scores: list[OfferScore] = []

        preferred_locs = [
            tok.strip().lower()
            for tok in profile.preferred_locations.replace(";", ",").split(",")
            if tok.strip()
        ]

        for offer in offers:
            vec = all_vecs[offer.uid]
            sim = cosine_similarity(profile_vec, vec)
            # Map cosine similarity [0, 1] -> readable 0.0..10.0 scale
            clamped = max(0.0, min(1.0, sim))
            score_10 = round(clamped * 10.0, 2)

            loc_haystack = f"{offer.assignment_location} {offer.company_hq}".lower()
            loc_match: bool | None = None
            if preferred_locs:
                loc_match = any(loc in loc_haystack for loc in preferred_locs)

            summary = (
                f"Semantic match: {int(round(clamped * 100))}% (local ONNX)"
                if offer.has_description
                else f"Title/company match: {int(round(clamped * 100))}% (no description in export)"
            )
            scores.append(
                OfferScore(
                    offer_uid=offer.uid,
                    score=score_10,
                    score_engine=self.engine_name,
                    scoring_fingerprint=fingerprint,
                    location_match=loc_match,
                    match_summary=summary,
                    tags=offer.label_durable[:3],
                )
            )

        return scores, profile_vec, new_vecs

    async def score_offers(
        self,
        profile: UserProfile,
        offers: list[JobOffer],
        fingerprint: str,
    ) -> list[OfferScore]:
        """Protocol-compatible uncached entry point."""
        scores, _, _ = await self.score_with_cache(
            profile,
            offers,
            fingerprint=fingerprint,
            cached_profile_vec=None,
            cached_offer_vecs={},
        )
        return scores
