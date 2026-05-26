"""Entry point for the nan.discord.bot."""

import asyncio
import hashlib
import signal
from pathlib import Path

from bot.base import NanBot
from bot.config import DEFAULT_DOCS_DIR, logger, settings
from bot.knowledge import SimpleVectorStore, canonicalize_doc_text, load_documentation
from bot.llm import LLMClient


async def init_knowledge_base(store: SimpleVectorStore) -> None:
    """Load docs and create embeddings. Non-fatal on failure."""
    from bot.docs_client import DocsClient
    from bot.knowledge import load_documentation_from_remote

    llm = LLMClient()
    mode = settings.docs_use_remote

    if mode == "remote":
        async with DocsClient() as client:
            result = await load_documentation_from_remote(
                store,
                client,
                fallback_docs_dir=DEFAULT_DOCS_DIR,
            )
    elif mode == "shadow":
        result = await load_documentation(store, DEFAULT_DOCS_DIR)

        try:
            async with DocsClient() as client:
                manifest = await client.fetch_manifest()

                # Compare on the same canonical text the chunker would see, so
                # the diff is signal (real content divergence) not noise from
                # frontmatter or line-ending differences.
                local_hashes: dict[str, str] = {}
                for md_file in sorted(DEFAULT_DOCS_DIR.glob("*.md")):
                    local_text = canonicalize_doc_text(
                        md_file.read_text(encoding="utf-8"),
                        strip_frontmatter=True,
                    )
                    local_hashes[md_file.stem] = hashlib.sha256(local_text.encode("utf-8")).hexdigest()

                remote_hashes: dict[str, str] = {}
                for entry in manifest.entries:
                    try:
                        doc_body = await client.fetch_body(entry)
                    except Exception as e:
                        logger.warning("Shadow fetch failed for %s: %s", entry.slug, type(e).__name__)
                        continue
                    remote_text = canonicalize_doc_text(doc_body.body, strip_frontmatter=False)
                    remote_hashes[entry.slug] = hashlib.sha256(remote_text.encode("utf-8")).hexdigest()

                only_local = sorted(set(local_hashes) - set(remote_hashes))
                only_remote = sorted(set(remote_hashes) - set(local_hashes))
                changed = sorted(
                    slug
                    for slug in (set(local_hashes) & set(remote_hashes))
                    if local_hashes[slug] != remote_hashes[slug]
                )

                logger.info(
                    "Shadow diff: local_only=%s remote_only=%s changed=%s",
                    only_local or "-",
                    only_remote or "-",
                    changed or "-",
                )
        except Exception as e:
            logger.warning("Shadow mode remote comparison failed: %s", type(e).__name__)
    else:
        result = await load_documentation(store, DEFAULT_DOCS_DIR)

    try:
        if result.new_chunks:
            embedded = await llm.embed_chunks(store)
            store.save()
            logger.info("Created embeddings for %d new chunks", embedded)
        elif result.stale_removed:
            store.save()
            logger.info("Persisted stale source cleanup")
        else:
            logger.info("No doc changes detected, skipping embedding API calls")
    except Exception as e:
        logger.error("Embedding init failed (bot will start without embeddings): %s", type(e).__name__)


async def main() -> None:
    store = SimpleVectorStore(Path("vector_db"))
    await init_knowledge_base(store)
    bot = NanBot()
    bot.store = store

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
        if hasattr(bot, "_health_server") and bot._health_server:
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
