from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from bot.docs_client import DocBody, Manifest, ManifestEntry
from bot.knowledge import SimpleVectorStore, load_documentation_from_remote


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _entry(slug: str, body: str, *, order: int = 1) -> ManifestEntry:
    return ManifestEntry(
        slug=slug,
        title=slug.title(),
        description=f"{slug} desc",
        order=order,
        content_hash=f"sha256:{_sha256(body)}",
        content_url=f"/api/docs/{slug}.md",
    )


@dataclass
class FakeClient:
    """Stub satisfying the surface of DocsClient used by load_documentation_from_remote."""

    manifest: Manifest | None
    bodies: dict[str, str] = field(default_factory=dict)
    fail_manifest: bool = False
    cached_manifest: Manifest | None = None
    fail_bodies: set[str] = field(default_factory=set)
    cached_bodies: dict[str, str] = field(default_factory=dict)

    async def fetch_manifest(self) -> Manifest:
        if self.fail_manifest:
            raise RuntimeError("boom")
        assert self.manifest is not None
        return self.manifest

    async def fetch_body(self, entry: ManifestEntry) -> DocBody:
        if entry.slug in self.fail_bodies:
            raise RuntimeError("body boom")
        body = self.bodies[entry.slug]
        return DocBody(
            slug=entry.slug,
            raw=body,
            body=body,
            content_hash=f"sha256:{_sha256(body)}",
        )

    def load_cached_manifest(self) -> Manifest | None:
        return self.cached_manifest

    def load_cached_body(self, slug: str) -> DocBody | None:
        if slug not in self.cached_bodies:
            return None
        body = self.cached_bodies[slug]
        return DocBody(
            slug=slug,
            raw=body,
            body=body,
            content_hash=f"sha256:{_sha256(body)}",
        )


@pytest.mark.asyncio
async def test_unchanged_skip(tmp_path: Path) -> None:
    body = "hello"
    entry = _entry("intro", body)
    manifest = Manifest(version="v1", entries=[entry])
    store = SimpleVectorStore(tmp_path / "db")
    client = FakeClient(manifest=manifest, bodies={"intro": body})

    first = await load_documentation_from_remote(store, client)
    assert first.new_chunks > 0
    store.save()

    # Reload with a fresh store on the same DB — version short-circuit must
    # treat this as unchanged on the second remote sync.
    store2 = SimpleVectorStore(tmp_path / "db")
    second = await load_documentation_from_remote(store2, client)
    assert second.new_chunks == 0
    assert second.stale_removed == 0


@pytest.mark.asyncio
async def test_changed_reindex(tmp_path: Path) -> None:
    body1 = "hello v1"
    body2 = "hello v2"
    store = SimpleVectorStore(tmp_path / "db")

    manifest1 = Manifest(version="v1", entries=[_entry("intro", body1)])
    client1 = FakeClient(manifest=manifest1, bodies={"intro": body1})
    await load_documentation_from_remote(store, client1)
    store.save()
    first_hash = store.get_doc_hash("intro.md")

    manifest2 = Manifest(version="v2", entries=[_entry("intro", body2)])
    client2 = FakeClient(manifest=manifest2, bodies={"intro": body2})
    result = await load_documentation_from_remote(store, client2)
    assert result.new_chunks > 0
    assert store.get_doc_hash("intro.md") != first_hash


@pytest.mark.asyncio
async def test_removed_cleanup(tmp_path: Path) -> None:
    store = SimpleVectorStore(tmp_path / "db")
    body_a = "a body"
    body_b = "b body"
    manifest1 = Manifest(
        version="v1",
        entries=[_entry("a", body_a), _entry("b", body_b)],
    )
    client1 = FakeClient(manifest=manifest1, bodies={"a": body_a, "b": body_b})
    await load_documentation_from_remote(store, client1)
    store.save()
    assert {"a.md", "b.md"}.issubset(store.get_tracked_sources())

    manifest2 = Manifest(version="v2", entries=[_entry("a", body_a)])
    client2 = FakeClient(manifest=manifest2, bodies={"a": body_a})
    result = await load_documentation_from_remote(store, client2)
    assert result.stale_removed == 1
    assert "b.md" not in store.get_tracked_sources()


@pytest.mark.asyncio
async def test_fallback_remote_to_cache(tmp_path: Path) -> None:
    body = "from cache"
    cached_manifest = Manifest(version="v-cache", entries=[_entry("intro", body)])
    store = SimpleVectorStore(tmp_path / "db")
    client = FakeClient(
        manifest=None,
        bodies={"intro": body},
        fail_manifest=True,
        cached_manifest=cached_manifest,
    )

    result = await load_documentation_from_remote(store, client)
    assert result.new_chunks > 0
    # source_of_truth=="cache" means the short-circuit must NOT be triggered.
    # But after a successful cache-based sync we still persist the version,
    # so a subsequent cache sync with the same version would re-walk entries.


@pytest.mark.asyncio
async def test_fallback_remote_to_local(tmp_path: Path) -> None:
    local_docs = tmp_path / "local"
    local_docs.mkdir()
    (local_docs / "intro.md").write_text("local content", encoding="utf-8")

    store = SimpleVectorStore(tmp_path / "db")
    client = FakeClient(manifest=None, fail_manifest=True, cached_manifest=None)

    result = await load_documentation_from_remote(store, client, fallback_docs_dir=local_docs)
    assert result.new_chunks > 0
    assert "intro.md" in store.get_tracked_sources()


@pytest.mark.asyncio
async def test_manifest_version_unchanged_short_circuits(tmp_path: Path) -> None:
    body = "hello"
    manifest = Manifest(version="v1", entries=[_entry("intro", body)])
    store = SimpleVectorStore(tmp_path / "db")
    client = FakeClient(manifest=manifest, bodies={"intro": body})

    first = await load_documentation_from_remote(store, client)
    assert first.new_chunks > 0

    # Mutate bodies dict so any fetch_body call would change the chunks.
    client.bodies["intro"] = "different"
    result = await load_documentation_from_remote(store, client)
    # Short-circuit must hit before per-entry walk.
    assert result.new_chunks == 0
    assert result.stale_removed == 0


@pytest.mark.asyncio
async def test_cache_source_does_not_short_circuit(tmp_path: Path) -> None:
    body = "hello"
    cached_manifest = Manifest(version="v1", entries=[_entry("intro", body)])
    store = SimpleVectorStore(tmp_path / "db")
    # Pretend a prior remote sync already recorded version "v1".
    store.set_meta("docs_manifest_version", "v1")

    client = FakeClient(
        manifest=None,
        bodies={"intro": body},
        fail_manifest=True,
        cached_manifest=cached_manifest,
    )
    result = await load_documentation_from_remote(store, client)
    # We came from cache, so the short-circuit must not fire; the walk
    # actually inspects the entry and indexes it (no stored hash yet).
    assert result.new_chunks > 0
