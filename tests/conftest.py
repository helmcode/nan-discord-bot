import os

# Required Settings fields must be defined before bot.config is imported.
os.environ.setdefault("DISCORD_TOKEN", "test-token")
os.environ.setdefault("DISCORD_GUILD_ID", "1")
os.environ.setdefault("LITELLM_API_KEY", "test-litellm-key")
os.environ.setdefault("DOCS_USE_REMOTE", "local")
