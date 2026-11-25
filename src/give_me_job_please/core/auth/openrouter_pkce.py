"""
Headless OpenRouter OAuth PKCE (S256) flow + live OpenRouter model catalog helper.

Follows OpenRouter's official PKCE spec (zero inbound ports required):
  1. Generate cryptographic code_verifier (in-memory only, 10-minute TTL).
  2. Build https://openrouter.ai/auth?code_challenge=...&code_challenge_method=S256...
     (omitting callback_url so OpenRouter displays the single-use code on screen).
  3. Exchange code + code_verifier via POST https://openrouter.ai/api/v1/auth/keys.
  4. Provide safe SHA-256 deep links for activity logs & key revocation.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import time
from dataclasses import dataclass
from urllib.parse import urlencode

import httpx

from give_me_job_please.core.auth.crypto import sha256_hex
from give_me_job_please.errors import AuthError, ValidationError

OPENROUTER_AUTH_URL = "https://openrouter.ai/auth"
OPENROUTER_EXCHANGE_URL = "https://openrouter.ai/api/v1/auth/keys"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

VALID_EXPIRATIONS = {
    "1-hour",
    "1-day",
    "7-days",
    "30-days",
    "90-days",
    "180-days",
    "1-year",
    "never",
}


@dataclass(slots=True)
class PendingPkceSession:
    code_verifier: str
    code_challenge: str
    created_at: float
    ttl_seconds: float = 600.0  # Matches OpenRouter's 10-minute code expiry

    @property
    def is_expired(self) -> bool:
        return (time.monotonic() - self.created_at) > self.ttl_seconds


@dataclass(slots=True)
class ExchangedKeyResult:
    api_key: str
    key_hash: str
    logs_url: str
    settings_url: str


@dataclass(slots=True)
class OpenRouterModelInfo:
    id: str
    name: str
    context_length: int | None
    prompt_price_per_m: float | None
    completion_price_per_m: float | None


def create_sha256_code_challenge(code_verifier: str) -> str:
    """Compute base64url(SHA256(verifier)) without '=' padding per RFC 7636."""
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def build_key_deep_links(key_hash: str) -> tuple[str, str]:
    """Return (activity_logs_url, key_settings_url) for the signed-in key owner."""
    return (
        f"https://openrouter.ai/logs?api_key_hash={key_hash}",
        f"https://openrouter.ai/keys/{key_hash}",
    )


class OpenRouterPkceManager:
    """Manages ephemeral in-memory PKCE sessions per Telegram user."""

    def __init__(self, *, app_label: str = "GiveMeJobPlease", ttl_seconds: float = 600.0) -> None:
        self._app_label = app_label
        self._ttl_seconds = ttl_seconds
        self._sessions: dict[int, PendingPkceSession] = {}

    def start_headless_session(
        self,
        user_id: int,
        *,
        limit_usd: float | None = None,
        expiration: str | None = None,
    ) -> str:
        """Create a fresh PKCE verifier/challenge and return the OpenRouter authorize URL."""
        self._purge_expired()
        verifier = secrets.token_urlsafe(32)
        challenge = create_sha256_code_challenge(verifier)
        self._sessions[user_id] = PendingPkceSession(
            code_verifier=verifier,
            code_challenge=challenge,
            created_at=time.monotonic(),
            ttl_seconds=self._ttl_seconds,
        )

        params: dict[str, str] = {
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "key_label": self._app_label[:100],
        }
        if limit_usd is not None:
            if limit_usd <= 0 or limit_usd >= 1_000_000:
                raise ValidationError("Credit limit must be between 0.01 and 999999 USD.")
            params["limit"] = f"{limit_usd:.2f}"
        if expiration is not None:
            if expiration not in VALID_EXPIRATIONS:
                raise ValidationError(
                    f"Invalid expiration '{expiration}'. Valid values: {', '.join(sorted(VALID_EXPIRATIONS))}."
                )
            params["expiration"] = expiration

        return f"{OPENROUTER_AUTH_URL}?{urlencode(params)}"

    async def exchange_code(self, user_id: int, code: str) -> ExchangedKeyResult:
        """Exchange the single-use OpenRouter code + stored verifier for a user API key."""
        self._purge_expired()
        session = self._sessions.pop(user_id, None)
        if session is None:
            raise AuthError(
                "No active OpenRouter authorization session found (or it expired after 10 minutes). "
                "Please run /connect_openrouter again."
            )
        if session.is_expired:
            raise AuthError(
                "Your OpenRouter authorization session expired (>10 minutes). "
                "Please run /connect_openrouter again."
            )

        cleaned_code = code.strip()
        if not cleaned_code:
            raise ValidationError("Please provide the authorization code: /auth <code>")

        payload = {
            "code": cleaned_code,
            "code_verifier": session.code_verifier,
            "code_challenge_method": "S256",
        }

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(
                    OPENROUTER_EXCHANGE_URL,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
        except httpx.HTTPError as exc:
            raise AuthError(
                "Network error while contacting OpenRouter. Please try /connect_openrouter again.",
                detail=f"HTTP error during PKCE exchange: {exc}",
            ) from exc

        if resp.status_code == 403:
            raise AuthError(
                "OpenRouter rejected the code (invalid, already used, or expired after 10 minutes). "
                "Run /connect_openrouter to generate a fresh link.",
                detail=f"OpenRouter 403 on PKCE exchange: {resp.text[:200]}",
            )
        if resp.status_code >= 400:
            raise AuthError(
                f"OpenRouter returned HTTP {resp.status_code} during key exchange. "
                "Please restart with /connect_openrouter.",
                detail=f"OpenRouter {resp.status_code}: {resp.text[:200]}",
            )

        data = resp.json()
        api_key = str(data.get("key") or "").strip()
        if not api_key:
            raise AuthError(
                "OpenRouter response did not contain an API key. Please try /connect_openrouter again.",
                detail="Missing 'key' field in OpenRouter exchange response.",
            )

        key_hash = sha256_hex(api_key)
        logs_url, settings_url = build_key_deep_links(key_hash)
        return ExchangedKeyResult(
            api_key=api_key,
            key_hash=key_hash,
            logs_url=logs_url,
            settings_url=settings_url,
        )

    def _purge_expired(self) -> None:
        expired_ids = [uid for uid, s in self._sessions.items() if s.is_expired]
        for uid in expired_ids:
            self._sessions.pop(uid, None)


async def search_openrouter_models(query: str = "", limit: int = 10) -> list[OpenRouterModelInfo]:
    """Query OpenRouter's public /api/v1/models catalog to help the user pick a model ID."""
    params: dict[str, str | int] = {"limit": 100}
    if query.strip():
        params["q"] = query.strip()

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(OPENROUTER_MODELS_URL, params=params)
            resp.raise_for_status()
            payload = resp.json()
    except httpx.HTTPError as exc:
        raise AuthError(
            "Could not fetch the OpenRouter model catalog right now.",
            detail=f"OpenRouter models catalog error: {exc}",
        ) from exc

    raw_models = payload.get("data", [])
    q_lower = query.strip().lower()
    results: list[OpenRouterModelInfo] = []

    for item in raw_models:
        model_id = str(item.get("id", ""))
        name = str(item.get("name", model_id))
        if q_lower and q_lower not in model_id.lower() and q_lower not in name.lower():
            continue
        pricing = item.get("pricing") or {}
        prompt_p = _to_per_million(pricing.get("prompt"))
        comp_p = _to_per_million(pricing.get("completion"))
        ctx = item.get("context_length")
        results.append(
            OpenRouterModelInfo(
                id=model_id,
                name=name,
                context_length=int(ctx) if isinstance(ctx, int | float) else None,
                prompt_price_per_m=prompt_p,
                completion_price_per_m=comp_p,
            )
        )
        if len(results) >= limit:
            break

    return results


async def verify_openrouter_model_exists(model_id: str) -> OpenRouterModelInfo | None:
    """Verify whether an exact model slug exists in OpenRouter's catalog."""
    matches = await search_openrouter_models(query=model_id, limit=25)
    for item in matches:
        if item.id == model_id:
            return item
    return None


def _to_per_million(raw_price: object) -> float | None:
    if raw_price is None:
        return None
    try:
        val = float(str(raw_price))
        if val < 0:
            return None
        return round(val * 1_000_000, 4)
    except ValueError:
        return None
