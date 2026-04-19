"""Entry point for the nan.discord.bot."""

import asyncio
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
        logger.error("Embedding init failed (bot will start without embeddings): %s", e)


async def main() -> None:
    store = SimpleVectorStore(Path("vector_db"))
    await init_knowledge_base(store)
    bot = NanBot()
    bot.store = store
    await bot.start(settings.discord_token)


if __name__ == "__main__":
    asyncio.run(main())
