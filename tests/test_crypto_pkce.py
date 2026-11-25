"""Unit tests for AES-256-GCM encryption, PKCE S256 generation, and log secret redaction."""

from __future__ import annotations

import os
from urllib.parse import parse_qs, urlparse

import pytest

from give_me_job_please.core.auth.crypto import SecretBox
from give_me_job_please.core.auth.openrouter_pkce import (
    OpenRouterPkceManager,
    create_sha256_code_challenge,
)
from give_me_job_please.errors import AuthError, ValidationError
from give_me_job_please.logging_setup import redact_secrets_processor, redact_string


def test_aes_gcm_roundtrip_and_aad_binding() -> None:
    box = SecretBox(os.urandom(32))
    token = box.encrypt("sk-or-v1-supersecretkey1234567890", associated_data="openrouter:42")
    assert "supersecret" not in token
    assert box.decrypt(token, associated_data="openrouter:42") == "sk-or-v1-supersecretkey1234567890"

    # Wrong user AAD must fail authentication tag verification
    with pytest.raises(AuthError):
        box.decrypt(token, associated_data="openrouter:999")


def test_pkce_headless_url_and_s256() -> None:
    mgr = OpenRouterPkceManager(app_label="Test App", ttl_seconds=60.0)
    url = mgr.start_headless_session(user_id=7, limit_usd=5.0, expiration="30-days")
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)

    assert parsed.netloc == "openrouter.ai"
    assert "callback_url" not in qs  # Headless mode omits callback_url
    assert qs["code_challenge_method"] == ["S256"]
    assert qs["limit"] == ["5.00"]
    assert qs["expiration"] == ["30-days"]

    session = mgr._sessions[7]
    assert create_sha256_code_challenge(session.code_verifier) == qs["code_challenge"][0]

    with pytest.raises(ValidationError):
        mgr.start_headless_session(user_id=7, expiration="invalid-ttl")


def test_secret_redaction_in_logs() -> None:
    leaked = "Error calling API with sk-or-v1-abcdef1234567890xyz and bot 123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567"
    cleaned = redact_string(leaked)
    assert "sk-or-v1-" not in cleaned
    assert "ABCDEFGHIJKLMNOPQRSTUVWXYZ" not in cleaned

    event = redact_secrets_processor(
        None,
        "error",
        {"event": leaked, "api_key": "sk-or-v1-12345678901234567890", "code_verifier": "secret"},
    )
    assert event["api_key"] == "[REDACTED]"
    assert event["code_verifier"] == "[REDACTED]"
