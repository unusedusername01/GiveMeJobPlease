"""Aiogram outer middlewares for trace_id correlation and strict user whitelisting."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from aiogram import BaseMiddleware
from aiogram.types import TelegramObject, Update

logger = structlog.get_logger(__name__)


def _extract_user_id(update: Update) -> int | None:
    if update.message and update.message.from_user:
        return update.message.from_user.id
    if update.callback_query and update.callback_query.from_user:
        return update.callback_query.from_user.id
    if update.edited_message and update.edited_message.from_user:
        return update.edited_message.from_user.id
    return None


class TraceContextMiddleware(BaseMiddleware):
    """Assign a unique 8-char trace_id to every incoming Telegram update."""

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        trace_id = uuid.uuid4().hex[:8]
        user_id = _extract_user_id(event) if isinstance(event, Update) else None
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(trace_id=trace_id, user_id=user_id)
        data["trace_id"] = trace_id
        try:
            return await handler(event, data)
        finally:
            structlog.contextvars.clear_contextvars()


class WhitelistMiddleware(BaseMiddleware):
    """Silently drop any update from Telegram users not in TELEGRAM_ALLOWED_USER_IDS."""

    def __init__(self, allowed_user_ids: set[int]) -> None:
        self._allowed = allowed_user_ids

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        if isinstance(event, Update):
            user_id = _extract_user_id(event)
            if user_id is None or user_id not in self._allowed:
                logger.warning("unauthorized_telegram_access_blocked", blocked_user_id=user_id)
                return None
        return await handler(event, data)
