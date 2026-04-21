"""Entry point for the nan.discord.bot."""

import asyncio
import signal
from pathlib import Path

from bot.base import NanBot
from bot.config import DEFAULT_DOCS_DIR, logger, settings
from bot.knowledge import load_documentation, SimpleVectorStore
from bot.llm import LLMClient


async def init_knowledge_base(store: SimpleVectorStore) -> None:
    """Load docs and create embeddings. Non-fatal on failure."""
    llm = LLMClient()
    await load_documentation(store, DEFAULT_DOCS_DIR)
    try:
        embedded = await llm.embed_chunks(store)
        if embedded:
            store.save()
            logger.info("Created embeddings for %d chunks", embedded)
        else:
            logger.info("No new chunks to embed")
    except Exception as e:
        logger.error("Embedding init failed (bot will start without embeddings): %s", type(e).__name__)


async def main() -> None:
    store = SimpleVectorStore(Path("vector_db"))
    await init_knowledge_base(store)
    bot = NanBot()
    bot.store = store
    await bot.setup_commands()

    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_event.set)

    shutdown_task = asyncio.create_task(shutdown_event.wait())

    try:
        await asyncio.gather(
            bot.start(settings.discord_token),
            shutdown_task,
        )
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("Shutting down...")
        # Cancel background tasks
        for task in asyncio.all_tasks():
            task.cancel()
        try:
            await asyncio.gather(*asyncio.all_tasks(), return_exceptions=True)
        except asyncio.CancelledError:
            pass
        # Save and close store
        if store:
            try:
                store.save()
            except Exception:
                pass
            try:
                store.close()
            except Exception:
                pass
        # Close LLM clients
        if bot.llm:
            try:
                await bot.llm._client.close()
                await bot.llm._embed_client.close()
            except Exception:
                pass
        # Stop health check server
        if hasattr(bot, '_health_server') and bot._health_server:
            try:
                bot._health_server.shutdown()
            except Exception:
                pass
        # Stop the bot
        try:
            await bot.close()
        except Exception:
            pass
        logger.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
