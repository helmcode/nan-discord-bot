# Contributing to nan-discord-bot

Thanks for your interest in contributing! We favor small, focused PRs and clear
intent over big bangs. This guide explains how to get set up and the workflow
we use.

## Quick Start

Prerequisites

- **Python 3.11+**
- Git
- A LiteLLM API key for local testing (talk to a maintainer or point at your
  own gateway via `LITELLM_BASE_URL`)

Setup

```bash
git clone https://github.com/<you>/nan-discord-bot.git
cd nan-discord-bot

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# fill in your tokens
python main.py
```

See `README.md` for the full list of environment variables and the
Docker-based local setup.

## Development Workflow

1. **Create a feature branch**

   ```
   git checkout -b feat/<short-slug>
   ```

2. **Make changes and keep PRs small and focused**

   - Prefer a series of small PRs over one large one.
   - Update `README.md` when behavior or configuration changes.

3. **Run checks locally before opening a PR**

   ```bash
   ruff check .              # Lint (must pass)
   ruff format --check .     # Formatting (must pass)
   pytest                    # Tests (must pass)
   ```

   `ruff format --check .` reports issues without rewriting; run
   `ruff format .` to apply fixes.

4. **Commit using Conventional Commits**

   - `feat:` / `fix:` / `chore:` / `refactor:` / `docs:` / `perf:` / `test:` ...

   Example: `feat(docs): sync knowledge base from remote docs API`

5. **Open a Pull Request**

   - Describe the change, rationale, and testing steps.
   - Link related Issues.
   - Keep the PR title in Conventional Commit format.

## Testing

**Every new feature, fix, or refactor must ship with tests.** PRs that add
functionality without tests will not be merged.

- The project uses [pytest](https://docs.pytest.org/) with
  [pytest-asyncio](https://pytest-asyncio.readthedocs.io/) (`asyncio_mode =
  "auto"` is already configured in `pyproject.toml`).
- The test suite is being grown from scratch. Place new tests under `tests/`,
  mirroring the `bot/` package layout (`tests/test_knowledge.py`,
  `tests/test_docs_client.py`, etc.).
- For HTTP clients (`docs_client.py`, `llm.py`, `metrics.py`): mock `httpx` /
  `aiohttp` responses; do not hit live services in tests.
- For loaders and parsers (`knowledge.py`, `_strip_frontmatter`,
  `_SAFE_SLUG_RE`): cover positive cases, negative cases, and edge cases
  (empty input, malformed input, unicode) with explicit fixtures.
- For Discord-side handlers: stub `discord.Message` / `discord.Channel` and
  assert on the bot's reactions to specific inputs (rate limiting, mention
  parsing, sanitization).

```bash
pytest          # full suite
pytest -k name  # filter by test name
pytest -x       # stop at first failure
```

## Code Style

- Follow the existing style in the codebase.
- `ruff` is the linter and formatter (config in `pyproject.toml`,
  `line-length = 120`, `target-version = "py311"`).
- Use type hints. Target Python 3.11+ syntax (`list[str]`, `str | None`,
  ...).
- Never commit secrets, API keys, Discord tokens, or LiteLLM keys. Use `.env`
  locally (gitignored) and the production secrets store for deploys. Keep
  `.env.example` in sync when adding new settings.

## Issue Reports and Feature Requests

Use GitHub Issues. Include Python version and OS, steps to reproduce,
relevant logs (with secrets redacted), and the bot commit SHA if running a
build.
