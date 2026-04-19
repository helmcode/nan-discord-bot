"""LiteLLM API integration for chat completions and embeddings."""

from openai import AsyncOpenAI

from bot.config import settings, logger
from bot.knowledge import DocumentChunk, SearchResult, SimpleVectorStore


class LLMClient:
    """Client for LiteLLM API (chat + embeddings)."""

    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.litellm_api_key,
            base_url=settings.litellm_base_url,
        )

    async def chat(self, messages: list[dict], model: str = "qwen3.6") -> str:
        """Send a chat completion request."""
        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
        )
        return response.choices[0].message.content or ""

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding for a text."""
        response = await self._client.embeddings.create(
            model=settings.embedding_model,
            input=text,
        )
        return response.data[0].embedding

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        results = await self._client.embeddings.create(
            model=settings.embedding_model,
            input=texts,
        )
        return [r.embedding for r in results.data]

    async def embed_chunks(self, store: SimpleVectorStore) -> int:
        """Embed all chunks in the store that don't have embeddings yet."""
        chunks_without_embedding = [
            chunk for chunk in store._chunks if not chunk.embedding
        ]
        if not chunks_without_embedding:
            return 0

        batch_size = 100
        embedded = 0

        for i in range(0, len(chunks_without_embedding), batch_size):
            batch = chunks_without_embedding[i:i + batch_size]
            texts = [c.text for c in batch]
            embeddings = await self.embed_many(texts)

            for chunk, embedding in zip(batch, embeddings):
                chunk.embedding = embedding
                embedded += 1

            logger.info("Embedded %d/%d chunks", embedded, len(chunks_without_embedding))

        return embedded

    async def answer_with_context(
        self,
        question: str,
        context_chunks: list[SearchResult],
        user_name: str = "",
    ) -> str:
        """Answer a question using retrieved context from the knowledge base."""
        if not context_chunks:
            system_prompt = (
                "You are a helpful AI assistant for the nan.builders community. "
                "The community provides access to AI models (like qwen3.6, qwen3-embedding, kokoro, whisper) "
                "for developers and AI enthusiasts. "
                "Answer questions about AI, machine learning, and the nan.builders project. "
                "If you don't know something specific about community setup, suggest they check the documentation or ask in the support channel."
            )
        else:
            context_parts = []
            for i, result in enumerate(context_chunks, 1):
                context_parts.append(
                    f"[Document {i} - {result.chunk.source}]\n{result.chunk.text}\n"
                )
            context = "\n---\n".join(context_parts)

            system_prompt = (
                "You are a helpful support assistant for the nan.builders community. "
                "Your job is to help community members connect and configure AI client tools "
                "(OpenCode, OpenClaw, Cline, etc.) to use nan.builders API.\n\n"
                "Use the following documentation to answer the user's question. "
                "Be concise and practical. If the documentation doesn't cover the question, "
                "say so clearly and suggest alternatives.\n\n"
                "Documentation:\n\n" + context
            )

        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
        ]

        if user_name:
            messages.append({
                "role": "user",
                "content": f"{user_name} asks: {question}",
            })
        else:
            messages.append({
                "role": "user",
                "content": question,
            })

        return await self.chat(messages)
