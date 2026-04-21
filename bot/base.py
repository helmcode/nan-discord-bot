"""Main Discord bot implementation."""

import asyncio
import json
import re
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import discord
from discord.ext import commands

from bot.config import settings, logger
from bot.knowledge import SimpleVectorStore
from bot.llm import LLMClient
from bot.news import fetch_rss_feeds, deduplicate, llm_select_top5, send_news_to_discord


# Rate limiting: max 3 mentions per user per 60-second window
_RATE_LIMIT = 3
_RATE_WINDOW = 60
_user_rate_limits: dict[tuple[int, int], list[float]] = defaultdict(list)


def _check_rate_limit(author_id: int, channel_id: int) -> bool:
    """Check if a user has exceeded the rate limit. Returns True if allowed."""
    now = time.time()
    window = _user_rate_limits[(author_id, channel_id)]
    window[:] = [t for t in window if now - t < _RATE_WINDOW]
    if len(window) >= _RATE_LIMIT:
        return False
    window.append(now)
    return True


# Sanitize usernames to prevent prompt injection
_SAFE_USERNAME_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\u200c\u200d\u2060\ufeff\u202a\u202b\u202c\u202d\u202e\u202f]')
_MAX_USERNAME_LEN = 100


def _sanitize_username(name: str) -> str:
    """Sanitize a Discord username for safe inclusion in LLM prompts."""
    if not name:
        return ""
    safe = _SAFE_USERNAME_RE.sub("", name)
    # Strip bidirectional override, zero-width, and other problematic Unicode
    safe = re.sub(r'[\u00ad\u0300-\u036f\u1ab0-\u1aff\u1dc0-\u1dff]', '', safe)
    return safe[:_MAX_USERNAME_LEN]


class NanBot(commands.Bot):
    """Main bot class."""

    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True

        super().__init__(
            command_prefix="/",
            intents=intents,
        )

        self.llm = LLMClient()
        self.store: SimpleVectorStore | None = None
        self._initialized = False
        self._ready = False
        # Limit concurrent LLM calls to prevent overwhelming the API
        self._llm_semaphore = asyncio.Semaphore(5)
        self._health_port = 9100
        self._health_server: HTTPServer | None = None
        self._health_thread: Thread | None = None

    def _start_health_server(self) -> None:
        """Start a lightweight HTTP health check server in a background thread."""

        class HealthHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/health":
                    health_data = {
                        "status": "healthy" if self.bot._ready else "starting",
                        "initialized": self.bot._initialized,
                        "store_chunks": len(self.bot.store.chunks) if self.bot.store else 0,
                    }
                    body = json.dumps(health_data).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format: str, *args: object) -> None:
                logger.debug("Health server: %s", format % args)

        HealthHandler.bot = self

        try:
            self._health_server = HTTPServer(("0.0.0.0", self._health_port), HealthHandler)
            self._health_thread = Thread(target=self._health_server.serve_forever, daemon=True)
            self._health_thread.start()
            logger.info("Health check server started on port %d", self._health_port)
        except OSError as e:
            logger.warning("Could not start health check server on port %d: %s", self._health_port, e)

    async def setup_hook(self) -> None:
        self._initialized = True
        self._start_health_server()

    async def on_ready(self) -> None:
        self._ready = True
        logger.info("Bot ready: %s (ID: %s)", self.user, self.user.id)
        logger.info("News channel ID configured: %s", settings.news_channel_id_value)
        logger.info("News send hour: %s", settings.news_send_hour)
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="/help | nan.builders",
            )
        )
        await self.start_daily_news()

    async def _fetch_and_send_news(self) -> None:
        """Fetch AI news, select top 5, and send to Discord channel."""
        logger.info("Fetching daily AI news...")
        try:
            articles = await fetch_rss_feeds()
            logger.info("Fetched %d raw articles from RSS feeds", len(articles))

            if not articles:
                logger.warning("No articles fetched from RSS feeds")
                return

            unique_articles = deduplicate(articles)
            logger.info("Deduplicated to %d unique articles", len(unique_articles))

            top5_data = await llm_select_top5(unique_articles[:30], self.llm)
            logger.info("LLM selected %d top articles", len(top5_data))

            if settings.news_channel_id_value:
                await send_news_to_discord(settings.news_channel_id_value, top5_data, self)
                logger.info("Successfully sent daily news")
        except Exception as e:
            logger.error("Daily news task failed: %s", type(e).__name__, exc_info=True)

    @commands.command(name="news", description="Manually trigger daily AI news fetch")
    @commands.cooldown(1, 3600, commands.BucketType.default)
    async def trigger_news(self, ctx: commands.Context) -> None:
        """Manually trigger the news fetch (rate limited: 1 per hour)."""
        await self._fetch_and_send_news()
        await ctx.send("Fetching AI news... This may take a moment.")

    @commands.command(name="news-now", description="Immediately fetch and send AI news")
    async def news_now(self, ctx: commands.Context) -> None:
        """Immediately fetch and send AI news without waiting."""
        if settings.news_channel_id_value is None:
            await ctx.send("News channel not configured.")
            return

        await ctx.send("Fetching AI news NOW... This may take a moment.")
        await self._fetch_and_send_news()

    @trigger_news.error
    async def trigger_news_error(self, ctx: commands.Context, error) -> None:
        if isinstance(error, commands.CommandOnCooldown):
            await ctx.send(f"News fetch is on cooldown. Try again in {int(error.retry_after)} seconds.")

    async def _schedule_daily_news(self) -> None:
        """Schedule daily news to run at the configured hour."""
        logger.info("_schedule_daily_news entered, channel=%s", settings.news_channel_id_value)
        if settings.news_channel_id_value is None:
            logger.info("News channel not configured, skipping daily news task")
            return

        from datetime import datetime, timedelta, timezone

        target_hour = settings.news_send_hour
        now = datetime.now(timezone.utc)
        logger.info("Target hour=%d, now=%s", target_hour, now)
        next_run = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)
        logger.info("next_run=%s, now=%s, next_run<=now=%s", next_run, now, next_run <= now)
        if next_run <= now:
            next_run += timedelta(days=1)
            logger.info("Scheduled for tomorrow: %s", next_run)

        delay = (next_run - now).total_seconds()
        logger.info("Daily news scheduled for %s (in %.1f hours)", next_run.strftime("%H:%M"), delay / 3600)

        await asyncio.sleep(delay)
        await self._fetch_and_send_news()

        # Schedule next run (24 hours from now)
        while True:
            await asyncio.sleep(86400)
            await self._fetch_and_send_news()

    async def start_daily_news(self) -> None:
        """Start the daily news scheduler as a background task."""
        logger.info("start_daily_news called, news_channel_id_value=%s", settings.news_channel_id_value)
        if settings.news_channel_id_value is not None:
            asyncio.create_task(self._schedule_daily_news())
            logger.info("Daily news scheduler started (first run at %d:00, channel %s)", settings.news_send_hour, settings.news_channel_id_value)
        else:
            logger.info("News channel not configured, skipping daily news task")

    async def on_message(self, message: discord.Message) -> None:
        """Process incoming messages for auto-responses."""
        if message.author == self.user:
            return

        # Delegate to slash commands before processing auto-responses
        await self.process_commands(message)

        channel_id = message.channel.id
        allowed = settings.allowed_channel_ids

        is_in_channel = not allowed or channel_id in allowed
        is_mentioned = self.user.mentioned_in(message)

        logger.debug(
            "Message from %s in #%s (id=%s): allowed=%s mentioned=%s",
            message.author, message.channel.name, channel_id, is_in_channel, is_mentioned,
        )

        if not is_in_channel or not is_mentioned:
            return

        # Rate limiting
        if not _check_rate_limit(message.author.id, channel_id):
            logger.warning("Rate limit exceeded for user %s in channel %s", message.author, channel_id)
            try:
                await message.reply("Demasiadas peticiones. Espera un momento e intenta de nuevo.", allowed_mentions=discord.AllowedMentions.none())
            except discord.Forbidden:
                pass
            return

        content = re.sub(rf"<@!?{self.user.id}>\s*", "", message.content).strip()
        if not content:
            return

        # Truncate to prevent overly expensive embedding/LLM calls
        content = content[:1500]

        try:
            await message.channel.send(
                embed=discord.Embed(
                    title="Thinking...",
                    description="Searching documentation...",
                    color=discord.Color.blue(),
                ),
                mention_author=False,
            )
        except discord.Forbidden:
            pass

        try:
            query_vector = await asyncio.wait_for(
                self.llm.embed(content), timeout=15.0
            )
            results = self.store.search(query_vector, top_k=settings.top_k) if self.store else []
        except asyncio.TimeoutError:
            logger.error("Embedding timed out")
            results = []
        except Exception as e:
            logger.error("Embedding failed: %s", type(e).__name__)
            results = []

        try:
            async with self._llm_semaphore:
                answer = await asyncio.wait_for(
                    self.llm.answer_with_context(
                        question=content,
                        context_chunks=results,
                        user_name=_sanitize_username(message.author.display_name),
                    ),
                    timeout=60.0,
                )
        except asyncio.TimeoutError:
            logger.error("LLM response timed out")
            answer = "La respuesta tardó demasiado. Intenta de nuevo."
        except Exception as e:
            logger.error("LLM response failed: %s", type(e).__name__)
            answer = "Lo siento, hubo un error generando la respuesta. Intenta de nuevo o contacta a un admin."

        if len(answer) > 2000:
            answer = answer[:1997] + "..."

        try:
            await asyncio.wait_for(
                message.reply(answer, allowed_mentions=discord.AllowedMentions.none()),
                timeout=10.0,
            )
        except (discord.Forbidden, asyncio.TimeoutError):
            pass

    @commands.command(name="health", description="Check bot health and knowledge base")
    @commands.guild_only()
    async def health(self, ctx: commands.Context) -> None:
        chunk_count = len(self.store.chunks) if self.store else 0
        embed = discord.Embed(title="Bot Health", color=discord.Color.green())
        embed.add_field(name="Status", value="Online", inline=True)
        embed.add_field(name="Knowledge Base", value=f"{chunk_count} chunks", inline=True)
        await ctx.send(embed=embed)

    @commands.command(name="docs", description="List available documentation files")
    async def docs(self, ctx: commands.Context) -> None:
        from bot.config import DOCS_DIR
        docs = list(DOCS_DIR.glob("**/*.md"))
        if not docs:
            await ctx.send("No documentation files loaded yet.")
            return
        doc_list = "\n".join(f"- {d.name}" for d in docs)
        embed = discord.Embed(title="Documentation", description=doc_list, color=discord.Color.blue())
        await ctx.send(embed=embed)

    @commands.command(name="search", description="Search the knowledge base")
    async def search(self, ctx: commands.Context, *, query: str) -> None:
        if not self.store:
            await ctx.send("Knowledge base not initialized.")
            return

        try:
            query_vector = await asyncio.wait_for(
                self.llm.embed(query), timeout=15.0
            )
            results = self.store.search(query_vector, top_k=3)
        except asyncio.TimeoutError:
            logger.error("Search timed out")
            await ctx.send("La búsqueda tardó demasiado. Intenta de nuevo.")
            return
        except Exception as e:
            logger.error("Search failed: %s", type(e).__name__)
            await ctx.send("Error performing search.")
            return

        if not results:
            await ctx.send("No results found for that query.")
            return

        parts = []
        for i, result in enumerate(results, 1):
            preview = result.chunk.text[:200] + "..."
            parts.append(f"**[{result.chunk.source}]** (score: {result.score:.3f})\n{preview}")

        await ctx.send("\n\n".join(parts))
