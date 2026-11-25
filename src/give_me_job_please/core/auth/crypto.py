"""Authenticated symmetric encryption (AES-256-GCM) for storing OpenRouter keys at rest."""

from __future__ import annotations

import base64
import hashlib
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from give_me_job_please.errors import AuthError

_NONCE_LEN = 12


class SecretBox:
    """Encrypts and decrypts user secrets using AES-256-GCM with per-value random nonces."""

    def __init__(self, master_key: bytes) -> None:
        if len(master_key) != 32:
            raise ValueError("AES-256-GCM master key must be exactly 32 bytes.")
        self._aead = AESGCM(master_key)

    def encrypt(self, plaintext: str, *, associated_data: str = "") -> str:
        """Return urlsafe base64(nonce || ciphertext || tag)."""
        nonce = os.urandom(_NONCE_LEN)
        aad = associated_data.encode("utf-8") if associated_data else None
        ct = self._aead.encrypt(nonce, plaintext.encode("utf-8"), aad)
        return base64.urlsafe_b64encode(nonce + ct).decode("ascii")

    def decrypt(self, token: str, *, associated_data: str = "") -> str:
        """Verify tag and decrypt a previously encrypted secret."""
        try:
            raw = base64.urlsafe_b64decode(token.encode("ascii"))
            if len(raw) <= _NONCE_LEN:
                raise AuthError("Stored API key ciphertext is malformed.")
            nonce, ct = raw[:_NONCE_LEN], raw[_NONCE_LEN:]
            aad = associated_data.encode("utf-8") if associated_data else None
            plaintext = self._aead.decrypt(nonce, ct, aad)
            return plaintext.decode("utf-8")
        except InvalidTag as exc:
            raise AuthError(
                "Could not decrypt your stored OpenRouter key (master ENCRYPTION_KEY may have changed). "
                "Please reconnect using /connect_openrouter.",
                detail="AES-GCM authentication tag mismatch.",
            ) from exc
        except AuthError:
            raise
        except Exception as exc:
            raise AuthError(
                "Stored OpenRouter key could not be decoded. Please reconnect using /connect_openrouter.",
                detail=f"Decrypt error: {exc}",
            ) from exc


def sha256_hex(value: str) -> str:
    """Return lowercase SHA-256 hex digest (used for OpenRouter deep links)."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
