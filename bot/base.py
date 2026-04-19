"""Main Discord bot implementation."""

import re

import discord
from discord.ext import commands

from bot.config import settings, logger
from bot.knowledge import SimpleVectorStore
from bot.llm import LLMClient


class NanBot(commands.Bot):
    """Main bot class."""

    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True

        super().__init__(
            command_prefix="/",
            intents=intents,
            tree_options=commands.BotTreeOptions(description="nan.builders support bot"),
        )

        self.llm = LLMClient()
        self.store: SimpleVectorStore | None = None
        self._ready = False

    async def setup_hook(self) -> None:
        await self.tree.sync()
        logger.info("Synced %d commands", len(self.tree.get_commands()))
        self._ready = True

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

        channel_id = message.channel.id
        allowed = settings.allowed_channel_ids
        support = settings.support_channel_ids

        is_in_channel = not allowed or channel_id in allowed
        is_mentioned = self.user.mentioned_in(message)

        if not is_in_channel or not (is_mentioned or channel_id in support):
            return

        content = re.sub(rf"<@!?{self.user.id}>\s*", "", message.content).strip()
        if not content:
            return

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
            query_vector = await self.llm.embed(content)
            results = self.store.search(query_vector, top_k=settings.top_k) if self.store else []
        except Exception as e:
            logger.error("Embedding failed: %s", e)
            results = []

        try:
            answer = await self.llm.answer_with_context(
                question=content,
                context_chunks=results,
                user_name=message.author.display_name,
            )
        except Exception as e:
            logger.error("LLM response failed: %s", e)
            answer = "Lo siento, hubo un error generando la respuesta. Intenta de nuevo o contacta a un admin."

        if len(answer) > 2000:
            answer = answer[:1997] + "..."

        try:
            await message.reply(answer, allowed_mentions=discord.AllowedMentions.none())
        except discord.Forbidden:
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
            query_vector = await self.llm.embed(query)
            results = self.store.search(query_vector, top_k=3)
        except Exception as e:
            logger.error("Search failed: %s", e)
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
