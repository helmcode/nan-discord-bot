# nan-discord-bot

Community Discord bot for [nan.builders](https://nan.builders). It answers member questions in configured support channels using semantic search over a curated markdown knowledge base (a `SimpleVectorStore` backed by SQLite, `qwen3-embedding` 4096-dim vectors, cosine similarity computed in Python) and the `qwen3.6` chat model exposed through the NaN LiteLLM gateway. The bot also reports daily and per-user token usage pulled from the LiteLLM admin API.

## Features

- Mention-triggered auto-response in channels listed in `ALLOWED_CHANNELS`, with retrieval-augmented answers from the local vector store and a `qwen3.6` fallback.
- Per-user rate limiting (3 mentions per 60-second window per `(user, channel)` pair).
- Username sanitization to mitigate prompt injection via Discord display names.
- Text commands (prefix `/`): `/health`, `/docs`, `/search <query>`.
- Slash commands (Discord interactions): `/metrics`, `/my-metrics`.
- Daily token usage report posted to `STATUS_CHANNEL_ID` at `METRICS_SEND_HOUR` (UTC), pulled from the LiteLLM proxy.
- HTTP health endpoint on port `9101` (`GET /health`) consumed by the Docker `HEALTHCHECK`.
- Slack notification on every new thread opened in the channels listed in `SUPPORT_CHANNEL_IDS` (e.g. `#support`), posted through an Incoming Webhook.
- Mirroring of support threads into Slack: with a bot token, every human message posted in a tracked thread is echoed as a reply under the Slack announcement.
- Doc-hash optimization: unchanged markdown files are skipped on startup, so embeddings are only recomputed when content actually changes.

## Tech stack

- Python 3.11 (Dockerfile `python:3.11-slim`, `pyproject.toml` `requires-python = ">=3.11"`).
- [discord.py](https://discordpy.readthedocs.io/) >= 2.3.2.
- [openai](https://github.com/openai/openai-python) async SDK, pointed at the NaN LiteLLM gateway (`AsyncOpenAI`).
- `aiohttp` for the LiteLLM admin metrics calls.
- [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) for `.env` parsing.
- SQLite (stdlib `sqlite3`) as the vector store backend.
- Hatchling build backend.
- Ruff for linting and formatting.

## Architecture

`main.py` boots a `SimpleVectorStore` against `vector_db/vectors.db`, loads markdown files from `bot/docs/knowledge/` through `load_documentation`, embeds any new or changed chunks via the LiteLLM embeddings endpoint, and starts the `NanBot` discord.py client. Graceful shutdown is wired through `SIGINT`/`SIGTERM` handlers that cancel pending tasks, persist the vector store, close the OpenAI clients, stop the health HTTP server, and close the bot connection.

On message events, `NanBot.on_message` filters by `ALLOWED_CHANNELS` and mention, applies the rate limiter, embeds the question, runs a cosine-similarity search over the in-memory chunks (top-K configurable via `TOP_K`), and calls `LLMClient.answer_with_context` against the `qwen3.6` chat model. A `CircuitBreaker` (5 failures, 60-second cool-off) protects the chat endpoint from cascading failures, and an `asyncio.Semaphore(5)` caps concurrent LLM calls.

Metrics live in `bot/metrics.py`. They hit the LiteLLM proxy `/spend/logs/ui` endpoint (configured via `LITELLM_PROXY_URL` and `LITELLM_ADMIN_KEY`) and aggregate token usage per `user_api_key_alias`. The daily scheduler sleeps until `METRICS_SEND_HOUR` UTC, posts the top-10 report to `STATUS_CHANNEL_ID`, and then loops every 24 hours.

Slack notifications live in `bot/slack.py`. `NanBot.on_thread_create` fires on Discord's `THREAD_CREATE` gateway event (only for newly created threads, so re-joins do not re-notify), filters by `SUPPORT_CHANNEL_IDS`, resolves the opening message — `Thread.starter_message` when cached, otherwise `fetch_message(thread.id)` with one retry for the forum race where the event outruns the message — and POSTs a Block Kit payload to `SLACK_WEBHOOK_URL`. The `SlackNotifier` retries 5xx and 429 with a 0.5/1/2 s backoff, never retries other 4xx, and escapes `&`, `<`, `>` in every user-controlled field so thread titles cannot forge Slack links. With `SLACK_WEBHOOK_URL` empty the notifier is a no-op.

When `SLACK_BOT_TOKEN` and `SLACK_CHANNEL_ID` are set, `SlackApiClient` takes over and posts through `chat.postMessage` instead. That is what makes mirroring possible: the Web API returns the message `ts`, which an Incoming Webhook never does, and a threaded reply needs it as `thread_ts`. The `ts` is stored in `vector_db/slack_threads.db` by `bot/thread_map.py` — inside the Docker volume, so a redeploy does not orphan live threads — and purged after 30 days. `NanBot._mirror_to_slack` then echoes each human message of a tracked thread, skipping bots and the forum opening message (already the body of the announcement). Posts are serialised behind a lock with a ~1 s spacing to respect Slack's per-channel rate limit.

## Project structure

```
discord-bot/
├── main.py                          # Entry point and shutdown wiring
├── bot/
│   ├── __init__.py
│   ├── base.py                      # NanBot, commands, message handler, health HTTP server
│   ├── config.py                    # pydantic-settings, paths, logger
│   ├── knowledge.py                 # SimpleVectorStore, chunking, doc loader
│   ├── llm.py                       # LLMClient, CircuitBreaker, RAG prompt
│   ├── metrics.py                   # LiteLLM spend log aggregation and reports
│   ├── slack.py                     # Slack notifier (webhook + Web API) and payload builders
│   ├── thread_map.py                # Discord thread -> Slack ts mapping (SQLite)
│   └── docs/
│       └── knowledge/               # Embedded markdown corpus
│           ├── intro.md
│           ├── getting-started.md
│           └── models.md
├── Dockerfile
├── entrypoint.sh
├── docker-compose.yml               # Local build
├── docker-compose.prod.yml          # Production override (pulled image)
├── pyproject.toml
├── .env.example
└── .github/
    └── workflows/
        └── deploy.yml               # GHCR build + SSH deploy
```

## Getting started

### Prerequisites

- Python 3.11.
- A Discord application with a bot user, a token, and the following privileged intents enabled in the [Discord Developer Portal](https://discord.com/developers/applications): **MESSAGE CONTENT INTENT** and **SERVER MEMBERS INTENT**. Without them the bot fails to connect with `PrivilegedIntentsRequired`.
- The bot invited to your guild with permissions to read messages, send messages, embed links, and use slash commands.
- A LiteLLM API key. The bot defaults to `https://api.nan.builders/v1`; override with `LITELLM_BASE_URL` if you run your own gateway.
- For the Slack support-thread notifications: a Slack app with **Incoming Webhooks** enabled and a webhook added to the destination channel, plus the Discord bot having *View Channel* and *Read Message History* on the support channel so it can read the thread's opening message for the preview.
- For the metrics features: network reachability to the LiteLLM proxy URL (defaults to `http://localhost:4000`, i.e. the bot is expected to run on the same host) and an admin key with read access to `/spend/logs/ui`.

### Local setup (without Docker)

```bash
git clone https://github.com/helmcode/nan-discord-bot.git
cd nan-discord-bot
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Fill in DISCORD_TOKEN, DISCORD_GUILD_ID, LITELLM_API_KEY, ALLOWED_CHANNELS, etc.
python main.py
```

### Local setup (Docker)

```bash
cp .env.example .env
# Fill in the required variables.
docker compose up --build
```

The container exposes the health endpoint on port `9101` (consumed by the `HEALTHCHECK` defined in the `Dockerfile`).

## Commands

`NanBot` registers two flavors of commands. Text commands use the `commands.Bot` prefix (currently `/`) and slash commands are real Discord interactions registered via `bot.tree`.

| Command       | Type   | Description                                                                       | Cooldown        |
|---------------|--------|-----------------------------------------------------------------------------------|-----------------|
| `/health`     | text   | Bot status and the number of chunks currently loaded in the vector store.         | none            |
| `/docs`       | text   | List of markdown files loaded from `bot/docs/knowledge/`.                         | none            |
| `/search <q>` | text   | Top-3 chunks from the knowledge base for the given query, with cosine score.      | none            |
| `/metrics`    | slash  | Manually trigger the global LiteLLM top-10 token usage report (last 24 hours).    | 1 per 3600 s    |
| `/my-metrics` | slash  | The caller's personal token usage and per-model breakdown (last 24 hours).        | 1 per 300 s     |

Auto-response is triggered when the bot is **mentioned** inside a channel listed in `ALLOWED_CHANNELS`. Rate limiting allows at most 3 mentions per user per channel per 60-second window; excess messages get a Spanish "demasiadas peticiones" reply.

## Environment variables

| Name                 | Required | Default                          | Description                                                                                              |
|----------------------|----------|----------------------------------|----------------------------------------------------------------------------------------------------------|
| `DISCORD_TOKEN`      | yes      | —                                | Bot token from the Discord Developer Portal.                                                             |
| `DISCORD_GUILD_ID`   | yes      | —                                | Guild (server) ID the bot is associated with.                                                            |
| `LITELLM_BASE_URL`   | no       | `https://api.nan.builders/v1`    | OpenAI-compatible base URL used for chat completions and embeddings.                                     |
| `LITELLM_API_KEY`    | yes      | —                                | LiteLLM key used by both the chat and embeddings clients.                                                |
| `LITELLM_PROXY_URL`  | no       | `http://localhost:4000`          | LiteLLM proxy base URL used by the metrics module to call `/spend/logs/ui`.                              |
| `LITELLM_ADMIN_KEY`  | no       | `""` (disables metrics)          | Admin key for the LiteLLM proxy. When empty, metrics commands and the daily report are skipped.          |
| `EMBEDDING_MODEL`    | no       | `qwen3-embedding`                | Embedding model identifier sent to the LiteLLM gateway.                                                  |
| `EMBEDDING_DIM`      | no       | `4096`                           | Expected embedding dimensionality. Informational; not enforced at write time.                            |
| `TOP_K`              | no       | `5`                              | Number of chunks returned by the vector search used to build the RAG context.                            |
| `ALLOWED_CHANNELS`   | no       | `""` (all channels)              | Comma-separated Discord channel IDs the bot will respond in. Empty means every channel is allowed.       |
| `STATUS_CHANNEL_ID`     | no       | `""` (disables daily report)     | Channel ID where the daily metrics report is posted. Required for the scheduler to run.                  |
| `METRICS_SEND_HOUR`     | no       | `9`                              | UTC hour (0–23) at which the daily metrics report is posted.                                             |
| `SUPPORT_CHANNEL_IDS`   | no       | `""` (disables notifications)   | Comma-separated Discord channel IDs (text or forum) whose new threads are announced in Slack.            |
| `SLACK_WEBHOOK_URL`     | no       | `""` (disables notifications)   | Slack Incoming Webhook URL. The destination channel is fixed in the Slack app, not here.                 |
| `SLACK_BOT_TOKEN`       | no       | `""` (webhook mode)             | Slack bot token (`xoxb-…`) with `chat:write`. Enables thread mirroring; takes precedence over the webhook. |
| `SLACK_CHANNEL_ID`      | no       | `""`                             | Destination Slack channel ID. Required with `SLACK_BOT_TOKEN`.                                           |
| `SLACK_HTTP_TIMEOUT`    | no       | `10`                             | Per-request HTTP timeout (seconds) for the webhook POST.                                                 |
| `DOCS_USE_REMOTE`       | no       | `local`                          | Source for docs: `local` (`bot/docs/knowledge/`), `remote` (web docs API), or `shadow` (local + warn on remote drift). |
| `DOCS_BASE_URL`         | no       | `https://nan.builders`           | Base URL of the web that serves `/api/docs/manifest.json` and `/api/docs/{slug}.md`.                     |
| `DOCS_REFRESH_INTERVAL` | no       | `900`                            | Seconds between docs syncs when `DOCS_USE_REMOTE` is not `local`. Aligned with the web `Cache-Control`.   |
| `DOCS_CACHE_DIR`        | no       | `vector_db/docs_cache`           | Directory for the cached `manifest.json`, `etags.json`, and per-slug body files used for conditional GETs. |
| `DOCS_HTTP_TIMEOUT`     | no       | `10`                             | Per-request HTTP timeout (seconds) for the docs client.                                                  |

## Knowledge base

When `DOCS_USE_REMOTE=local` (the default), markdown files in `bot/docs/knowledge/` are loaded at startup by `SimpleVectorStore`, chunked on paragraph boundaries (target ~2000 chars per chunk with overlap), embedded via the LiteLLM embeddings endpoint, and persisted to `vector_db/vectors.db`. A `doc_hashes` table stores a SHA-256 of each canonical source so unchanged docs are skipped on subsequent boots; sources that disappear have their chunks evicted from the database.

To update the corpus in `local` mode, edit or add `.md` files under `bot/docs/knowledge/` and restart the bot. Only sources whose content hash changed will trigger new embedding API calls.

When `DOCS_USE_REMOTE=remote`, the corpus is pulled from `<DOCS_BASE_URL>/api/docs/manifest.json` instead and refreshed every `DOCS_REFRESH_INTERVAL` seconds (post-ready, in the background). The client honours `If-None-Match`/304 against `DOCS_CACHE_DIR` and short-circuits the entire sync when the manifest `version` (a hash over `[(slug, contentHash)]`) matches the value persisted in the SQLite `meta` table. `shadow` mode runs the local indexer but also fetches the remote bodies and logs any divergence between local and remote canonical hashes — useful while migrating to `remote`.

## Development

- Lint: `ruff check .`
- Format: `ruff format .`
- Ruff is configured in `pyproject.toml` (`line-length = 120`, `target-version = "py311"`, rules `E, F, I, N, W, UP`).
- Tests: `pytest` (full suite). The `dev` extra installs `pytest`, `pytest-asyncio`, and `pytest-httpx`. Tests live under `tests/` and mirror the `bot/` package layout. Mock `httpx` responses with `pytest-httpx`; do not hit live services. See `CONTRIBUTING.md` for the full testing policy.

## Deployment

Production runs as a Docker container on the inference server.

- On every push to `main`, `.github/workflows/deploy.yml` builds the image and pushes it to GHCR (`ghcr.io/helmcode/nan-discord-bot`) tagged with both `latest` and the commit SHA.
- The same workflow then SSHes into the deploy target, pulls the new image (by SHA, with `latest` as fallback), retags it as `latest`, and runs `docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --remove-orphans`.
- The production compose override uses `network_mode: host`, mounts the read-only knowledge directory and a named volume for `vector_db`, and caps the container at 512 MiB / 1 vCPU.
- The Dockerfile starts as root, runs `entrypoint.sh` which `chown`s the `vector_db` volume to the unprivileged `bot` user, then `exec gosu bot python main.py`.

### Required GitHub repository secrets

| Secret             | Purpose                                                                                                              |
|--------------------|----------------------------------------------------------------------------------------------------------------------|
| `SERVER_HOST`      | Hostname or IP of the deploy target.                                                                                 |
| `SERVER_USER`      | SSH username on the deploy target.                                                                                   |
| `SSH_PRIVATE_KEY`  | SSH private key authorized on the deploy target.                                                                     |
| `DEPLOY_DIR`       | Optional. Working directory on the deploy target. Falls back to `$HOME/nan-discord-bot` when empty or unset.         |

## Contributing

- Branch from `main` and open a pull request.
- Run `ruff check .` and `ruff format .` before pushing.
- Never commit `.env` files or any token, API key, or secret.
