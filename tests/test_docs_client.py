from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bot.docs_client import (
    _SAFE_SLUG_RE,
    DocsClient,
    ManifestEntry,
    _strip_frontmatter,
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_manifest_json(version: str, entries: list[dict]) -> str:
    return json.dumps({"version": version, "entries": entries}, ensure_ascii=False)


def _entry(slug: str, content_hash: str, *, order: int = 1) -> dict:
    return {
        "slug": slug,
        "title": slug.title(),
        "description": f"{slug} desc",
        "order": order,
        "contentHash": content_hash,
        "contentUrl": f"/api/docs/{slug}.md",
    }


class TestStripFrontmatter:
    def test_with_frontmatter(self) -> None:
        raw = "---\ntitle: foo\n---\nbody"
        assert _strip_frontmatter(raw) == "body"

    def test_without_frontmatter(self) -> None:
        raw = "no frontmatter here\nbody"
        assert _strip_frontmatter(raw) == raw

    def test_malformed(self) -> None:
        raw = "---\nincomplete\nbody"
        assert _strip_frontmatter(raw) == raw


class TestSafeSlugRe:
    @pytest.mark.parametrize(
        "slug,expected",
        [
            ("api", True),
            ("getting-started", True),
            ("a", True),
            ("../etc/passwd", False),
            ("Foo", False),
            ("", False),
            ("a" * 65, False),
            ("-bad", False),
            ("api/", False),
            ("a" * 64, True),
        ],
    )
    def test_match(self, slug: str, expected: bool) -> None:
        assert bool(_SAFE_SLUG_RE.match(slug)) is expected


@pytest.mark.asyncio
async def test_fetch_manifest_valid(httpx_mock, tmp_path: Path) -> None:
    body = "intro body"
    entry = _entry("intro", f"sha256:{_sha256(body)}")
    httpx_mock.add_response(
        url="https://example.test/api/docs/manifest.json",
        method="GET",
        text=_make_manifest_json("sha256:" + "a" * 64, [entry]),
        headers={"ETag": '"sha256:' + "a" * 64 + '"'},
    )

    async with DocsClient(base_url="https://example.test", cache_dir=tmp_path) as client:
        manifest = await client.fetch_manifest()

    assert manifest.version == "sha256:" + "a" * 64
    assert [e.slug for e in manifest.entries] == ["intro"]
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "etags.json").exists()
    etag_data = json.loads((tmp_path / "etags.json").read_text())
    assert etag_data["manifest"] == '"sha256:' + "a" * 64 + '"'


@pytest.mark.asyncio
async def test_fetch_manifest_filters_unsafe_slug(httpx_mock, tmp_path: Path) -> None:
    safe = _entry("intro", f"sha256:{_sha256('a')}")
    unsafe = _entry("../etc", f"sha256:{_sha256('b')}")
    httpx_mock.add_response(
        url="https://example.test/api/docs/manifest.json",
        method="GET",
        text=_make_manifest_json("sha256:" + "b" * 64, [safe, unsafe]),
    )

    async with DocsClient(base_url="https://example.test", cache_dir=tmp_path) as client:
        manifest = await client.fetch_manifest()

    assert [e.slug for e in manifest.entries] == ["intro"]


@pytest.mark.asyncio
async def test_fetch_manifest_sends_if_none_match(httpx_mock, tmp_path: Path) -> None:
    """Second call must reuse the persisted ETag."""
    text = _make_manifest_json("sha256:" + "c" * 64, [_entry("intro", f"sha256:{_sha256('x')}")])
    etag = '"sha256:' + "c" * 64 + '"'
    httpx_mock.add_response(
        url="https://example.test/api/docs/manifest.json",
        method="GET",
        text=text,
        headers={"ETag": etag},
    )

    async with DocsClient(base_url="https://example.test", cache_dir=tmp_path) as client:
        await client.fetch_manifest()

    # Second client uses persisted ETag from disk.
    httpx_mock.add_response(
        url="https://example.test/api/docs/manifest.json",
        method="GET",
        status_code=304,
        match_headers={"If-None-Match": etag},
    )
    async with DocsClient(base_url="https://example.test", cache_dir=tmp_path) as client:
        manifest = await client.fetch_manifest()
    assert [e.slug for e in manifest.entries] == ["intro"]


@pytest.mark.asyncio
async def test_fetch_body_hash_mismatch_warns_but_returns(
    httpx_mock, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    body_text = "real body"
    declared = "sha256:" + "0" * 64  # wrong on purpose
    entry = ManifestEntry(
        slug="intro",
        title="Intro",
        description="d",
        order=1,
        content_hash=declared,
        content_url="/api/docs/intro.md",
    )
    httpx_mock.add_response(
        url="https://example.test/api/docs/intro.md",
        method="GET",
        text=body_text,
        headers={"ETag": '"sha256:wrong"'},
    )

    async with DocsClient(base_url="https://example.test", cache_dir=tmp_path) as client:
        with caplog.at_level("WARNING"):
            doc = await client.fetch_body(entry)

    assert doc.body == "real body"
    assert any("hash mismatch" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_fetch_body_writes_canonical_to_cache(httpx_mock, tmp_path: Path) -> None:
    body_text = "---\ntitle: t\n---\nhello\r\n\r\nworld"
    canonical = "hello\n\nworld"
    entry = ManifestEntry(
        slug="intro",
        title="Intro",
        description="d",
        order=1,
        content_hash=f"sha256:{_sha256(canonical)}",
        content_url="/api/docs/intro.md",
    )
    httpx_mock.add_response(
        url="https://example.test/api/docs/intro.md",
        method="GET",
        text=body_text,
    )

    async with DocsClient(base_url="https://example.test", cache_dir=tmp_path) as client:
        await client.fetch_body(entry)

    cached = (tmp_path / "intro.md").read_text(encoding="utf-8")
    assert cached == canonical


@pytest.mark.asyncio
async def test_fetch_body_304_uses_cache(httpx_mock, tmp_path: Path) -> None:
    body_canonical = "cached body"
    (tmp_path / "intro.md").write_text(body_canonical, encoding="utf-8")
    # Pre-seed an ETag so the next request sends If-None-Match
    (tmp_path / "etags.json").write_text(
        json.dumps({"manifest": None, "bodies": {"intro": '"sha256:abc"'}}),
        encoding="utf-8",
    )
    entry = ManifestEntry(
        slug="intro",
        title="Intro",
        description="d",
        order=1,
        content_hash=f"sha256:{_sha256(body_canonical)}",
        content_url="/api/docs/intro.md",
    )
    httpx_mock.add_response(
        url="https://example.test/api/docs/intro.md",
        method="GET",
        status_code=304,
        match_headers={"If-None-Match": '"sha256:abc"'},
    )

    async with DocsClient(base_url="https://example.test", cache_dir=tmp_path) as client:
        doc = await client.fetch_body(entry)

    assert doc.body == body_canonical


@pytest.mark.asyncio
async def test_load_cached_body_normalises_legacy_frontmatter(tmp_path: Path) -> None:
    legacy = "---\ntitle: t\n---\nhello\r\n\r\nworld\n"
    (tmp_path / "intro.md").write_text(legacy, encoding="utf-8")

    client = DocsClient(base_url="https://example.test", cache_dir=tmp_path)
    doc = client.load_cached_body("intro")
    assert doc is not None
    assert doc.body == "hello\n\nworld"
