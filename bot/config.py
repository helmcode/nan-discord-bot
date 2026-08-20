import logging
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


def _parse_channel_ids(raw: str) -> set[int]:
    """Parse a comma-separated list of Discord snowflake IDs, ignoring junk."""
    if not raw:
        return set()
    ids = set()
    for x in raw.split(","):
        x = x.strip()
        if x and x.isdigit() and len(x) < 22:
            ids.add(int(x))
    return ids


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    discord_token: str
    discord_guild_id: int

    litellm_base_url: str = "https://api.nan.builders/v1"
    litellm_api_key: str
    litellm_proxy_url: str = "http://localhost:4000"
    litellm_admin_key: str = ""

    embedding_model: str = "qwen3-embedding"
    embedding_dim: int = 4096
    top_k: int = 5

    allowed_channels: str = ""
    status_channel_id: str = ""
    metrics_send_hour: int = 9

    docs_base_url: str = "https://nan.builders"
    docs_refresh_interval: int = 900
    docs_use_remote: Literal["local", "remote", "shadow"] = "local"
    docs_cache_dir: str = "vector_db/docs_cache"
    docs_http_timeout: int = 10

    slack_webhook_url: str = ""
    slack_http_timeout: int = 10
    support_channel_ids: str = ""

    @property
    def allowed_channel_ids(self) -> set[int]:
        return _parse_channel_ids(self.allowed_channels)

    @property
    def support_channel_id_set(self) -> set[int]:
        """Channel IDs whose new threads are announced in Slack."""
        return _parse_channel_ids(self.support_channel_ids)

    @property
    def status_channel_id_value(self) -> int | None:
        if not self.status_channel_id:
            return None
        x = self.status_channel_id.strip()
        if x and x.isdigit() and len(x) < 22:
            return int(x)
        logger.warning("Invalid STATUS_CHANNEL_ID: %r", self.status_channel_id)
        return None


BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
DB_DIR = BASE_DIR / "vector_db"
DEFAULT_DOCS_DIR = DOCS_DIR / "knowledge"

settings = Settings()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# httpx logs every request at INFO with the full URL. SLACK_WEBHOOK_URL carries
# its secret in the path, so inheriting INFO from the root would write that
# secret to the container logs on every notification.
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger("nan-bot")
