"""Main Discord bot implementation."""

import asyncio
import json
import re
import time
from collections import defaultdict
from datetime import UTC
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import discord
from discord.ext import commands

from bot.config import logger, settings
from bot.docs_client import DocsClient
from bot.knowledge import SimpleVectorStore, load_documentation_from_remote
from bot.llm import LLMClient
from bot.metrics import send_metrics_report, send_user_metrics_report
from bot.slack import SlackNotifier, SupportThreadEvent

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
_SAFE_USERNAME_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f"
    r"\u200b\u200c\u200d\u2060\ufeff"
    r"\u202a\u202b\u202c\u202d\u202e\u202f]"
)
_MAX_USERNAME_LEN = 100


def _sanitize_username(name: str) -> str:
    """Sanitize a Discord username for safe inclusion in LLM prompts."""
    if not name:
        return ""
    safe = _SAFE_USERNAME_RE.sub("", name)
    # Strip bidirectional override, zero-width, and other problematic Unicode
    safe = re.sub(r"[\u00ad\u0300-\u036f\u1ab0-\u1aff\u1dc0-\u1dff]", "", safe)
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
        self.slack = SlackNotifier()
        self.store: SimpleVectorStore | None = None
        self._initialized = False
        self._ready = False
        # Limit concurrent LLM calls to prevent overwhelming the API
        self._llm_semaphore = asyncio.Semaphore(5)
        self._health_port = 9101
        self._health_server: HTTPServer | None = None
        self._health_thread: Thread | None = None
        self._docs_last_sync: str | None = None
        self._docs_last_sync_ok: bool = False
        self._docs_refresh_task: asyncio.Task[None] | None = None
        self._docs_refresh_lock = asyncio.Lock()

    def _start_health_server(self) -> None:
        """Start a lightweight HTTP health check server in a background thread."""

        class HealthHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/health":
                    health_data = {
                        "status": "healthy" if self.bot._ready else "starting",
                        "initialized": self.bot._initialized,
                        "store_chunks": len(self.bot.store.chunks) if self.bot.store else 0,
                        "docs_last_sync": self.bot._docs_last_sync,
                        "docs_last_sync_ok": self.bot._docs_last_sync_ok,
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
        @self.tree.command(name="metrics", description="Manually trigger token usage metrics report")
        @discord.app_commands.checks.cooldown(1, 3600)
        async def metrics_command(interaction: discord.Interaction) -> None:
            if settings.status_channel_id_value is None or settings.litellm_admin_key is None:
                await interaction.response.send_message("Metrics not configured.")
                return
            await interaction.response.send_message("Fetching LiteLLM token metrics... This may take a moment.")
            await send_metrics_report(self)

        @self.tree.command(
            name="my-metrics",
            description="View your personal token usage metrics for the last 24 hours",
        )
        @discord.app_commands.checks.cooldown(1, 300)
        async def my_metrics_command(interaction: discord.Interaction) -> None:
            if settings.litellm_admin_key is None:
                await interaction.response.send_message("Metrics not configured.")
                return
            alias = interaction.user.display_name
            msg = f"Fetching your token metrics for `{alias}`... This may take a moment."
            await interaction.response.send_message(msg)
            await send_user_metrics_report(self, alias, interaction.channel)

        await self.tree.sync()
        logger.info("Synced %d commands", len(self.tree.get_commands()))
        self._initialized = True
        self._start_health_server()

    async def on_ready(self) -> None:
        self._ready = True
        logger.info("Bot ready: %s (ID: %s)", self.user, self.user.id)
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="/help | nan.builders",
            )
        )
        await self.start_daily_metrics()

        if self._docs_refresh_task is None or self._docs_refresh_task.done():
            self._docs_refresh_task = asyncio.create_task(self._schedule_docs_refresh())
        else:
            logger.info("Docs refresh scheduler already running")

    async def _refresh_docs_once(self) -> None:
        from datetime import UTC, datetime

        if self.store is None:
            return

        async with self._docs_refresh_lock:
            try:
                async with DocsClient() as client:
                    result = await load_documentation_from_remote(self.store, client)

                if result.new_chunks:
                    embedded = await self.llm.embed_chunks(self.store)
                    self.store.save()
                    logger.info("Refresh: embedded %d new chunks", embedded)
                elif result.stale_removed:
                    self.store.save()

                self._docs_last_sync_ok = True
            except Exception as e:
                logger.error("Docs refresh failed: %s", type(e).__name__)
                self._docs_last_sync_ok = False
            finally:
                self._docs_last_sync = datetime.now(UTC).isoformat()

    async def _schedule_docs_refresh(self) -> None:
        if settings.docs_use_remote == "local":
            logger.info("DOCS_USE_REMOTE=local, skipping remote docs refresh")
            return

        interval = max(60, settings.docs_refresh_interval)
        logger.info(
            "Docs refresh scheduler started (mode=%s, interval=%ds)",
            settings.docs_use_remote,
            interval,
        )

        try:
            while True:
                await self._refresh_docs_once()
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("Docs refresh scheduler cancelled")

    async def _schedule_daily_metrics(self) -> None:
        """Schedule daily metrics to run at the configured hour."""
        if settings.status_channel_id_value is None or settings.litellm_admin_key is None:
            logger.info("Metrics channel or LiteLLM admin key not configured, skipping daily metrics")
            return

        from datetime import datetime, timedelta

        target_hour = settings.metrics_send_hour
        now = datetime.now(UTC)
        next_run = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)
        if next_run <= now:
            next_run += timedelta(days=1)

        delay = (next_run - now).total_seconds()
        logger.info("Daily metrics scheduled for %s (in %.1f hours)", next_run.strftime("%H:%M"), delay / 3600)

        try:
            await asyncio.sleep(delay)
            await send_metrics_report(self)

            # Schedule next run (24 hours from now)
            while True:
                await asyncio.sleep(86400)
                await send_metrics_report(self)
        except asyncio.CancelledError:
            logger.info("Daily metrics scheduler cancelled")

    async def start_daily_metrics(self) -> None:
        """Start the daily metrics scheduler as a background task."""
        if settings.status_channel_id_value is not None and settings.litellm_admin_key:
            asyncio.create_task(self._schedule_daily_metrics())
            logger.info(
                "Daily metrics scheduler started (first run at %d:00, channel %s)",
                settings.metrics_send_hour,
                settings.status_channel_id_value,
            )
        else:
            logger.info("Metrics channel or LiteLLM admin key not configured, skipping daily metrics")

    async def _fetch_starter_text(self, thread: discord.Thread) -> str:
        """Best-effort text of the message that opened the thread.

        Forum posts carry their opening message inside the thread under the
        thread's own ID. Threads started from a message in a text channel keep
        that message in the parent channel, so there is nothing to fetch and the
        preview is simply omitted.
        """
        starter = thread.starter_message
        if starter is not None:
            return starter.content or ""

        # THREAD_CREATE can outrun the opening message, so a NotFound on the
        # first try is worth one retry; Forbidden never is.
        for delay in (0, 1.0):
            if delay:
                await asyncio.sleep(delay)
            try:
                message = await thread.fetch_message(thread.id)
            except discord.Forbidden:
                logger.debug("No permission to read the starter message of thread %s", thread.id)
                return ""
            except discord.HTTPException as e:
                logger.debug("Could not fetch starter message for thread %s: %s", thread.id, type(e).__name__)
                continue
            return message.content or ""
        return ""

    async def on_thread_create(self, thread: discord.Thread) -> None:
        """Announce new threads in the configured support channels on Slack."""
        support_channels = settings.support_channel_id_set
        if not support_channels or thread.parent_id not in support_channels:
            return

        if not self.slack.enabled:
            logger.warning("New thread in support channel but SLACK_WEBHOOK_URL is not configured")
            return

        preview = await self._fetch_starter_text(thread)

        owner = thread.owner
        author = owner.display_name if owner is not None else f"user {thread.owner_id}"
        parent = thread.parent
        channel_name = parent.name if parent is not None else str(thread.parent_id)

        event = SupportThreadEvent(
            thread_name=thread.name,
            thread_url=thread.jump_url,
            channel_name=channel_name,
            author=_sanitize_username(author),
            preview=preview,
        )

        try:
            sent = await asyncio.wait_for(self.slack.notify_support_thread(event), timeout=30.0)
        except TimeoutError:
            logger.error("Slack notification timed out for thread %s", thread.id)
            return
        except Exception as e:
            logger.error("Slack notification failed for thread %s: %s", thread.id, type(e).__name__)
            return

        if sent:
            logger.info("Notified Slack about thread %s in #%s", thread.id, channel_name)

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

        channel_name = message.channel.name if hasattr(message.channel, "name") else str(channel_id)
        logger.debug(
            "Message from %s in #%s (id=%s): allowed=%s mentioned=%s",
            message.author,
            channel_name,
            channel_id,
            is_in_channel,
            is_mentioned,
        )

        if not is_in_channel or not is_mentioned:
            return

        # Rate limiting
        if not _check_rate_limit(message.author.id, channel_id):
            logger.warning("Rate limit exceeded for user %s in channel %s", message.author, channel_id)
            try:
                await message.reply(
                    "Demasiadas peticiones. Espera un momento e intenta de nuevo.",
                    allowed_mentions=discord.AllowedMentions.none(),
                )
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
            query_vector = await asyncio.wait_for(self.llm.embed(content), timeout=15.0)
            results = self.store.search(query_vector, top_k=settings.top_k) if self.store else []
        except TimeoutError:
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
        except TimeoutError:
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
        except (TimeoutError, discord.Forbidden):
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
        if not self.store:
            await ctx.send("Knowledge base not initialized.")
            return

        docs = sorted(self.store.get_tracked_sources())
        if not docs:
            await ctx.send("No documentation files loaded yet.")
            return

        doc_list = "\n".join(f"- {doc}" for doc in docs)
        embed = discord.Embed(title="Documentation", description=doc_list, color=discord.Color.blue())
        await ctx.send(embed=embed)

    @commands.command(name="search", description="Search the knowledge base")
    async def search(self, ctx: commands.Context, *, query: str) -> None:
        if not self.store:
            await ctx.send("Knowledge base not initialized.")
            return

        try:
            query_vector = await asyncio.wait_for(self.llm.embed(query), timeout=15.0)
            results = self.store.search(query_vector, top_k=3)
        except TimeoutError:
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
