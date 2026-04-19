import logging
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    discord_token: str
    discord_guild_id: int

    litellm_base_url: str = "https://api.nan.builders/v1"
    litellm_api_key: str

    embedding_model: str = "qwen3-embedding"
    embedding_dim: int = 4096
    top_k: int = 5

    support_channels: str = ""
    allowed_channels: str = ""

    @property
    def support_channel_ids(self) -> set[int]:
        if not self.support_channels:
            return set()
        return {int(x.strip()) for x in self.support_channels.split(",") if x.strip()}

    @property
    def allowed_channel_ids(self) -> set[int]:
        if not self.allowed_channels:
            return set()
        return {int(x.strip()) for x in self.allowed_channels.split(",") if x.strip()}


BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
DB_DIR = BASE_DIR / "vector_db"
DEFAULT_DOCS_DIR = DOCS_DIR / "knowledge"

settings = Settings()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nan-bot")
