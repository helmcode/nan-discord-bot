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
                "Eres un asistente de IA para la comunidad de NaN. La comunidad proporciona acceso a modelos de IA "
                "(como qwen3.6, qwen3-embedding, kokoro, whisper) para desarrolladores y entusiastas de la IA. "
                "Responde preguntas sobre IA, machine learning, agentes de IA, harness de agentes de IA y NaN. "
                "Si no sabes algo específico sobre la configuración de la comunidad, sugiere que revisen la documentación "
                "(https://nan.builders/docs) o pregunten en el canal de #support. "
                "NO respondas a preguntas fuera de estos temas (IA, machine learning, agentes de IA, harness de agentes de IA, "
                "o la comunidad NaN). Si una pregunta no está relacionada con estos temas, indica claramente que no puedes responderla."
            )
        else:
            context_parts = []
            for i, result in enumerate(context_chunks, 1):
                context_parts.append(
                    f"[Document {i} - {result.chunk.source}]\n{result.chunk.text}\n"
                )
            context = "\n---\n".join(context_parts)

            system_prompt = (
                "Eres un asistente de soporte para la comunidad de NaN. Tu trabajo es ayudar a miembros de la comunidad "
                "a conectar y configurar herramientas cliente de IA (OpenCode, OpenClaw, Cline, etc.) "
                "para usar la API de nan.builders.\n\n"
                "Usa la siguiente documentación para responder la pregunta del usuario. "
                "Sé conciso y práctico. Si la documentación no cubre la pregunta, "
                "dilo claramente y sugiere alternativas. "
                "o indicales que revisen la documentación oficial de la comunidad: https://nan.builders/docs "
                "o que avisen al Staff.\n\n"
                "NO respondas a preguntas fuera de estos temas: IA, machine learning, agentes de IA, "
                "harness de agentes de IA, o la comunidad NaN. "
                "Si una pregunta no está relacionada con estos temas, indica claramente que no puedes responderla.\n\n"
                "Documentación:\n\n" + context
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
