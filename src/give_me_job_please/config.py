"""Validated environment configuration via Pydantic Settings."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and optional .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    telegram_bot_token: SecretStr
    telegram_allowed_user_ids: Annotated[list[int], NoDecode] = Field(min_length=1)
    encryption_key: SecretStr

    database_url: str = "sqlite+aiosqlite:///data/give_me_job_please.db"
    log_dir: Path = Path("data/logs")
    log_level: str = "INFO"

    # Mode A (Zero-Key Local Fallback)
    local_embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    fastembed_cache_path: Path = Path("data/model_cache")

    # Mode B (OpenRouter OAuth PKCE — primary when connected)
    default_openrouter_model: str = "deepseek/deepseek-chat"
    openrouter_batch_size: int = Field(default=15, ge=1, le=40)
    openrouter_app_title: str = "GiveMeJobPlease"
    openrouter_app_url: str = "https://github.com/unusedusername01/GiveMeJobPlease"

    # Telegram UI pagination
    digest_page_size: int = Field(default=5, ge=1, le=15)

    @field_validator("telegram_allowed_user_ids", mode="before")
    @classmethod
    def parse_allowed_ids(cls, value: object) -> list[int]:
        """Accept comma-separated integers ('123,456'), a single int, or a JSON list."""
        if isinstance(value, list):
            return [int(v) for v in value]
        if isinstance(value, int):
            return [value]
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return []
            if raw.startswith("["):
                parsed = json.loads(raw)
                return [int(v) for v in parsed]
            return [int(part.strip()) for part in raw.split(",") if part.strip()]
        raise ValueError("TELEGRAM_ALLOWED_USER_IDS must be a comma-separated list of integers.")

    @field_validator("encryption_key")
    @classmethod
    def validate_encryption_key(cls, value: SecretStr) -> SecretStr:
        """Verify ENCRYPTION_KEY decodes to exactly 32 bytes for AES-256-GCM."""
        secret = value.get_secret_value().strip()
        try:
            decoded = base64.urlsafe_b64decode(secret.encode("ascii"))
        except Exception as exc:
            raise ValueError(
                "ENCRYPTION_KEY must be urlsafe-base64 encoded 32 random bytes. "
                "Generate one with: python -c \"import os, base64; "
                "print(base64.urlsafe_b64encode(os.urandom(32)).decode())\""
            ) from exc
        if len(decoded) != 32:
            raise ValueError(
                f"ENCRYPTION_KEY decoded to {len(decoded)} bytes; expected exactly 32 bytes."
            )
        return value

    def encryption_key_bytes(self) -> bytes:
        """Return the raw 32-byte AES-256 master key."""
        return base64.urlsafe_b64decode(self.encryption_key.get_secret_value().strip().encode("ascii"))
