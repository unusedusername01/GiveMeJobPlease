"""Application entrypoint wiring configuration, logging, DB, scorers, and aiogram polling."""

from __future__ import annotations

import asyncio

import structlog
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage

from give_me_job_please import __version__
from give_me_job_please.bot.handlers import BotDependencies, build_router
from give_me_job_please.bot.middlewares import TraceContextMiddleware, WhitelistMiddleware
from give_me_job_please.config import Settings
from give_me_job_please.core.auth.crypto import SecretBox
from give_me_job_please.core.auth.openrouter_pkce import OpenRouterPkceManager
from give_me_job_please.core.scoring.local_embedder import LocalEmbedderScorer
from give_me_job_please.logging_setup import configure_logging
from give_me_job_please.storage.db import create_engine_and_sessionmaker
from give_me_job_please.storage.orm import Base
from give_me_job_please.storage.repository import OfferRepository


async def run_bot() -> None:
    settings = Settings()  # type: ignore[call-arg]
    error_log_path = configure_logging(settings.log_dir, settings.log_level)
    logger = structlog.get_logger(__name__)

    logger.info(
        "starting_give_me_job_please",
        version=__version__,
        allowed_users_count=len(settings.telegram_allowed_user_ids),
        local_model=settings.local_embedding_model,
        default_openrouter_model=settings.default_openrouter_model,
    )

    engine, session_factory = create_engine_and_sessionmaker(settings.database_url)
    # Defensive ensure tables exist when launched outside entrypoint.sh
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    deps = BotDependencies(
        settings=settings,
        repo=OfferRepository(session_factory),
        secret_box=SecretBox(settings.encryption_key_bytes()),
        pkce_manager=OpenRouterPkceManager(app_label=settings.openrouter_app_title),
        local_scorer=LocalEmbedderScorer(
            model_name=settings.local_embedding_model,
            cache_dir=settings.fastembed_cache_path,
        ),
        error_log_path=error_log_path,
    )

    bot = Bot(
        token=settings.telegram_bot_token.get_secret_value(),
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher(storage=MemoryStorage())

    # Outer middlewares: trace_id first, then strict whitelist enforcement
    dp.update.outer_middleware(TraceContextMiddleware())
    dp.update.outer_middleware(WhitelistMiddleware(set(settings.telegram_allowed_user_ids)))
    dp.include_router(build_router(deps))

    try:
        await bot.delete_webhook(drop_pending_updates=False)
        await dp.start_polling(bot)
    finally:
        await bot.session.close()
        await engine.dispose()
        logger.info("give_me_job_please_stopped")


def main() -> None:
    asyncio.run(run_bot())


if __name__ == "__main__":
    main()
