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
    news_channel_id: str = ""
    news_send_hour: int = 9
    news_feeds: str = (
        "https://news.ycombinator.com/front,"
        "https://techcrunch.com/category/artificial-intelligence/feed/,"
        "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml,"
        "https://feeds.arstechnica.com/arstechnica/technology-lab,"
        "https://www.technologyreview.com/feed/,"
        "https://rss.art19.com/the-latest-artificial-intelligence-news-updates-from-mit-technology-review.xml"
    )

    @property
    def support_channel_ids(self) -> set[int]:
        if not self.support_channels:
            return set()
        ids = set()
        for x in self.support_channels.split(","):
            x = x.strip()
            if x and x.isdigit() and len(x) < 22:
                ids.add(int(x))
        return ids

    @property
    def allowed_channel_ids(self) -> set[int]:
        if not self.allowed_channels:
            return set()
        ids = set()
        for x in self.allowed_channels.split(","):
            x = x.strip()
            if x and x.isdigit() and len(x) < 22:
                ids.add(int(x))
        return ids

    @property
    def news_channel_id_value(self) -> int | None:
        if not self.news_channel_id:
            return None
        x = self.news_channel_id.strip()
        if x and x.isdigit() and len(x) < 22:
            return int(x)
        logger.warning("Invalid NEWS_CHANNEL_ID: %r", self.news_channel_id)
        return None

    @property
    def news_feed_urls(self) -> list[str]:
        if not self.news_feeds:
            return []
        urls = []
        for url in self.news_feeds.split(","):
            url = url.strip()
            if url:
                urls.append(url)
        return urls


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
