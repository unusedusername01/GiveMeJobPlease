"""Typed application error hierarchy with safe, actionable user messages."""

from __future__ import annotations


class AppError(Exception):
    """Base exception carrying a machine-readable code and user-safe Telegram hint."""

    code: str = "APP_ERROR"

    def __init__(self, user_message: str, *, detail: str | None = None, code: str | None = None) -> None:
        super().__init__(detail or user_message)
        self.user_message = user_message
        self.detail = detail or user_message
        if code:
            self.code = code


class ParseError(AppError):
    """Raised when an uploaded .htm or CV document cannot be parsed deterministically."""

    code = "PARSE_ERROR"


class AuthError(AppError):
    """Raised on PKCE expiration, invalid verifier, decryption failure, or revoked key."""

    code = "AUTH_ERROR"


class ScorerError(AppError):
    """Raised when local embedding or OpenRouter scoring fails."""

    code = "SCORER_ERROR"


class StorageError(AppError):
    """Raised on unexpected SQLite / repository failures."""

    code = "STORAGE_ERROR"


class ValidationError(AppError):
    """Raised on invalid user command arguments or invalid model selections."""

    code = "VALIDATION_ERROR"
