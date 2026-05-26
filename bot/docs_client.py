from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

import httpx

from bot.config import logger, settings
from bot.knowledge import canonicalize_doc_text


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_SAFE_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_BACKOFF_SECONDS = (0.5, 1.0, 2.0)


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


class _EtagStore:
    """Persists per-resource ETags in a JSON file alongside the docs cache.

    Layout::

        {
          "manifest": "\"sha256:...\"",
          "bodies": {"<slug>": "\"sha256:...\""}
        }

    Tolerant to a missing or corrupt file (treats it as empty).
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: dict[str, object] = {"manifest": None, "bodies": {}}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to load etags.json (%s); starting empty", type(e).__name__)
            return
        if not isinstance(raw, dict):
            return
        manifest = raw.get("manifest")
        bodies = raw.get("bodies") or {}
        self._data["manifest"] = manifest if isinstance(manifest, str) else None
        self._data["bodies"] = {
            k: v for k, v in bodies.items() if isinstance(k, str) and isinstance(v, str)
        }

    def _save(self) -> None:
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._path)

    def get_manifest(self) -> str | None:
        v = self._data.get("manifest")
        return v if isinstance(v, str) else None

    def set_manifest(self, etag: str | None) -> None:
        self._data["manifest"] = etag
        self._save()

    def get_body(self, slug: str) -> str | None:
        bodies = self._data.get("bodies")
        if not isinstance(bodies, dict):
            return None
        v = bodies.get(slug)
        return v if isinstance(v, str) else None

    def set_body(self, slug: str, etag: str | None) -> None:
        bodies = self._data.setdefault("bodies", {})
        if not isinstance(bodies, dict):
            bodies = {}
            self._data["bodies"] = bodies
        if etag is None:
            bodies.pop(slug, None)
        else:
            bodies[slug] = etag
        self._save()


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
        self._etags = _EtagStore(self._cache_dir / "etags.json")

    @property
    def manifest_url(self) -> str:
        return f"{self._base_url}/api/docs/manifest.json"

    def resolve_content_url(self, content_url: str) -> str:
        return urljoin(f"{self._base_url}/", content_url.lstrip("/"))

    async def __aenter__(self) -> "DocsClient":
        transport = httpx.AsyncHTTPTransport(retries=3)
        self._client = httpx.AsyncClient(
            timeout=self._timeout,
            headers={"User-Agent": "nan-discord-bot/0.1 (+docs-sync)"},
            follow_redirects=False,
            transport=transport,
        )
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _get_with_backoff(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """GET with bounded retries on 5xx and transient connection errors.

        Returns the last response if every attempt returned a 5xx; raises the
        last httpx.HTTPError when every attempt raised. Non-5xx responses are
        returned immediately.
        """
        assert self._client is not None, "DocsClient not entered"
        last_exc: BaseException | None = None
        last_resp: httpx.Response | None = None
        attempts = len(_BACKOFF_SECONDS)
        for i in range(attempts):
            try:
                resp = await self._client.get(url, headers=headers)
            except httpx.HTTPError as e:
                last_exc = e
                if i < attempts - 1:
                    await asyncio.sleep(_BACKOFF_SECONDS[i])
                    continue
                raise
            last_resp = resp
            if resp.status_code < 500:
                return resp
            if i < attempts - 1:
                await asyncio.sleep(_BACKOFF_SECONDS[i])
                continue
        assert last_resp is not None or last_exc is not None
        if last_resp is not None:
            return last_resp
        raise last_exc  # type: ignore[misc]

    async def fetch_manifest(self) -> Manifest:
        assert self._client is not None, "DocsClient not entered"

        headers: dict[str, str] = {}
        prev_etag = self._etags.get_manifest()
        if prev_etag:
            headers["If-None-Match"] = prev_etag

        resp = await self._get_with_backoff(self.manifest_url, headers=headers)

        if resp.status_code == 304:
            cached = self.load_cached_manifest()
            if cached is not None:
                return cached
            logger.warning("Manifest 304 but local cache missing; refetching unconditionally")
            resp = await self._get_with_backoff(self.manifest_url)

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

        new_etag = resp.headers.get("etag")
        if new_etag:
            self._etags.set_manifest(new_etag)
        else:
            # The server forgot to send one — drop our stale token so we stop
            # claiming we have a validator we can't actually prove.
            self._etags.set_manifest(None)
        return manifest

    async def fetch_body(self, entry: ManifestEntry) -> DocBody:
        assert self._client is not None, "DocsClient not entered"
        if not _SAFE_SLUG_RE.match(entry.slug):
            raise ValueError(f"Unsafe slug: {entry.slug!r}")

        headers: dict[str, str] = {}
        prev_etag = self._etags.get_body(entry.slug)
        if prev_etag:
            headers["If-None-Match"] = prev_etag

        url = self.resolve_content_url(entry.content_url)
        resp = await self._get_with_backoff(url, headers=headers)

        if resp.status_code == 304:
            cached = self.load_cached_body(entry.slug)
            if cached is not None:
                return cached
            logger.warning("Body 304 for %s but cache missing; refetching", entry.slug)
            resp = await self._get_with_backoff(url)

        resp.raise_for_status()
        raw = resp.text
        # strip_frontmatter=True is intentional for forward-compat: the new
        # endpoint never sends frontmatter, but old caches / mixed deploys may.
        body = canonicalize_doc_text(raw, strip_frontmatter=True)
        computed = f"sha256:{_sha256(body)}"

        if computed != entry.content_hash:
            logger.warning(
                "Body hash mismatch for %s: manifest=%s computed=%s",
                entry.slug, entry.content_hash, computed,
            )

        tmp = self._cache_dir / f"{entry.slug}.md.tmp"
        dst = self._cache_dir / f"{entry.slug}.md"
        tmp.write_text(body, encoding="utf-8")
        tmp.replace(dst)

        new_etag = resp.headers.get("etag")
        if new_etag:
            self._etags.set_body(entry.slug, new_etag)
        else:
            self._etags.set_body(entry.slug, None)

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
        body = canonicalize_doc_text(raw, strip_frontmatter=True)
        return DocBody(
            slug=slug,
            raw=raw,
            body=body,
            content_hash=f"sha256:{_sha256(body)}",
        )
