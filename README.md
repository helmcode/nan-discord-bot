# NaN Discord Bot

Support bot for the nan.builders Discord community. Helps members connect their tools (OpenCode, Cursor, Cline, etc.) to NaN's AI models and answers AI-related questions.

## Features

- **Knowledge base** — Semantic search over community documentation using `qwen3-embedding` (4096-dim vectors)
- **Auto-responses** — Bot answers questions in configured support channels when mentioned
- **Slash commands** — `/health`, `/docs`, `/search`
- **Fallback** — Uses general LLM knowledge when no docs match

## Setup

### 1. Clone and install

```bash
cd discord-bot
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
DISCORD_TOKEN=your-bot-token
DISCORD_GUILD_ID=your-guild-id
LITELLM_BASE_URL=https://api.nan.builders/v1
LITELLM_API_KEY=your-api-key
```

### 3. Add documentation

Place markdown files in `bot/docs/knowledge/`. The bot will chunk and embed them on startup.

### 4. Run

```bash
python main.py
```

## Production (Docker)

```bash
docker compose up -d
```

The bot runs on the inference server (`46.225.241.235`) alongside LiteLLM.

## Architecture

```
discord-bot/
├── main.py                 # Entry point
├── bot/
│   ├── config.py           # Settings & paths
│   ├── base.py             # Bot class, events, slash commands
│   ├── knowledge.py        # Vector store + doc chunking
│   ├── llm.py              # LiteLLM client (chat + embeddings)
│   └── docs/knowledge/     # Markdown documentation files
└── vector_db/              # SQLite embeddings storage (auto-created)
```

## Knowledge Base

The bot loads markdown files from `bot/docs/knowledge/`, chunks them, and generates embeddings using the NaN server's `qwen3-embedding` model. When a user asks a question:

1. The question is embedded
2. Top-K similar chunks are found via cosine similarity
3. The LLM (`qwen3.6`) answers using the retrieved context

Vectors are persisted in `vector_db/vectors.db` (SQLite) and reloaded on startup.

## Slash Commands

- `/health` — Check bot health and knowledge base status
- `/docs` — List loaded documentation files
- `/search <query>` — Search knowledge base and show results
