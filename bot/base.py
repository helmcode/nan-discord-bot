"""Main Discord bot implementation."""

import asyncio
import re
import time
from collections import defaultdict

import discord
from discord.ext import commands

from bot.config import settings, logger
from bot.knowledge import SimpleVectorStore
from bot.llm import LLMClient


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
_SAFE_USERNAME_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_MAX_USERNAME_LEN = 100


def _sanitize_username(name: str) -> str:
    """Sanitize a Discord username for safe inclusion in LLM prompts."""
    if not name:
        return ""
    safe = _SAFE_USERNAME_RE.sub("", name)
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
        # Limit concurrent LLM calls to prevent overwhelming the API
        self._llm_semaphore = asyncio.Semaphore(5)

    async def setup_hook(self) -> None:
        await self.tree.sync()
        logger.info("Synced %d commands", len(self.tree.get_commands()))
        self._initialized = True

    async def on_ready(self) -> None:
        logger.info("Bot ready: %s (ID: %s)", self.user, self.user.id)
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="/help | nan.builders",
            )
        )

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
    async def health(self, ctx: commands.Context) -> None:
        chunk_count = len(self.store._chunks) if self.store else 0
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
