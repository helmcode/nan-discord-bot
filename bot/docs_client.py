from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

import httpx

from bot.config import logger, settings


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_SAFE_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


@dataclass
class ManifestEntry:
    slug: str
    title: str
    description: str
    order: int
    content_hash: str
    content_url: str


@dataclass
class Manifest:
    version: str
    entries: list[ManifestEntry]


@dataclass
class DocBody:
    slug: str
    raw: str
    body: str
    content_hash: str


def _strip_frontmatter(raw: str) -> str:
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return raw
    return raw[match.end():]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DocsClient:
    def __init__(
        self,
        base_url: str | None = None,
        cache_dir: Path | None = None,
        timeout: float | None = None,
    ) -> None:
        self._base_url = (base_url or settings.docs_base_url).rstrip("/")
        self._cache_dir = Path(cache_dir or settings.docs_cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._timeout = timeout if timeout is not None else settings.docs_http_timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def manifest_url(self) -> str:
        return f"{self._base_url}/api/docs/manifest.json"

    def resolve_content_url(self, content_url: str) -> str:
        return urljoin(f"{self._base_url}/", content_url.lstrip("/"))

    async def __aenter__(self) -> "DocsClient":
        self._client = httpx.AsyncClient(
            timeout=self._timeout,
            headers={"User-Agent": "nan-discord-bot/0.1 (+docs-sync)"},
            follow_redirects=False,
        )
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def fetch_manifest(self) -> Manifest:
        assert self._client is not None, "DocsClient not entered"
        resp = await self._client.get(self.manifest_url)
        resp.raise_for_status()
        data = resp.json()

        entries = [
            ManifestEntry(
                slug=e["slug"],
                title=e["title"],
                description=e["description"],
                order=int(e["order"]),
                content_hash=e["contentHash"],
                content_url=e["contentUrl"],
            )
            for e in data["entries"]
            if _SAFE_SLUG_RE.match(e["slug"])
        ]
        manifest = Manifest(version=data["version"], entries=entries)

        tmp = self._cache_dir / "manifest.json.tmp"
        dst = self._cache_dir / "manifest.json"
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(dst)
        return manifest

    async def fetch_body(self, entry: ManifestEntry) -> DocBody:
        assert self._client is not None, "DocsClient not entered"
        if not _SAFE_SLUG_RE.match(entry.slug):
            raise ValueError(f"Unsafe slug: {entry.slug!r}")

        resp = await self._client.get(self.resolve_content_url(entry.content_url))
        resp.raise_for_status()
        raw = resp.text
        body = _strip_frontmatter(raw)
        computed = f"sha256:{_sha256(body)}"

        if computed != entry.content_hash:
            logger.warning(
                "Body hash mismatch for %s: manifest=%s computed=%s",
                entry.slug, entry.content_hash, computed,
            )

        tmp = self._cache_dir / f"{entry.slug}.md.tmp"
        dst = self._cache_dir / f"{entry.slug}.md"
        tmp.write_text(raw, encoding="utf-8")
        tmp.replace(dst)

        return DocBody(slug=entry.slug, raw=raw, body=body, content_hash=computed)

    def load_cached_manifest(self) -> Manifest | None:
        p = self._cache_dir / "manifest.json"
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return Manifest(
                version=data["version"],
                entries=[
                    ManifestEntry(
                        slug=e["slug"],
                        title=e["title"],
                        description=e["description"],
                        order=int(e["order"]),
                        content_hash=e["contentHash"],
                        content_url=e["contentUrl"],
                    )
                    for e in data["entries"]
                    if _SAFE_SLUG_RE.match(e["slug"])
                ],
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Failed to load cached manifest: %s", type(e).__name__)
            return None

    def load_cached_body(self, slug: str) -> DocBody | None:
        if not _SAFE_SLUG_RE.match(slug):
            return None
        p = self._cache_dir / f"{slug}.md"
        if not p.exists():
            return None
        raw = p.read_text(encoding="utf-8")
        body = _strip_frontmatter(raw)
        return DocBody(
            slug=slug,
            raw=raw,
            body=body,
            content_hash=f"sha256:{_sha256(body)}",
        )
