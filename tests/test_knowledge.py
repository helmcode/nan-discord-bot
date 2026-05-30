from __future__ import annotations

from pathlib import Path

import pytest

from bot.knowledge import SimpleVectorStore, canonicalize_doc_text, load_documentation


class TestCanonicalizeDocText:
    def test_strip_frontmatter(self) -> None:
        raw = "---\ntitle: foo\nauthor: bar\n---\nhello world"
        assert canonicalize_doc_text(raw, strip_frontmatter=True) == "hello world"

    def test_strip_frontmatter_false_preserves_body(self) -> None:
        raw = "---\ntitle: foo\n---\nhello"
        # Without the strip flag, the leading dashes stay verbatim.
        assert canonicalize_doc_text(raw, strip_frontmatter=False) == raw

    def test_normalises_crlf(self) -> None:
        raw = "hello\r\nworld\r\n"
        assert canonicalize_doc_text(raw, strip_frontmatter=False) == "hello\nworld"

    def test_collapses_blank_lines(self) -> None:
        raw = "a\n\n\n\nb"
        assert canonicalize_doc_text(raw, strip_frontmatter=False) == "a\n\nb"

    def test_trims_final(self) -> None:
        raw = "\n\n  hello\n\n"
        assert canonicalize_doc_text(raw, strip_frontmatter=False) == "hello"


@pytest.mark.asyncio
async def test_load_documentation_ignores_frontmatter_and_crlf_differences(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    db = tmp_path / "db"

    # Two equivalent payloads modulo frontmatter and line endings.
    a = "---\ntitle: t\n---\nhello\n\nworld"
    (docs / "a.md").write_text(a, encoding="utf-8")
    store = SimpleVectorStore(db)
    result = await load_documentation(store, docs)
    assert result.new_chunks > 0
    store.save()
    first_hash = store.get_doc_hash("a.md")
    store.close()

    # Same logical content, different frontmatter + CRLF — should not re-index.
    b = "---\ntitle: other\n---\nhello\r\n\r\nworld\r\n"
    (docs / "a.md").write_text(b, encoding="utf-8")
    store2 = SimpleVectorStore(db)
    result2 = await load_documentation(store2, docs)
    assert result2.new_chunks == 0
    assert store2.get_doc_hash("a.md") == first_hash
    store2.close()


@pytest.mark.asyncio
async def test_load_documentation_second_run_no_changes_is_noop(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "x.md").write_text("# Hello\n\nbody", encoding="utf-8")
    store = SimpleVectorStore(tmp_path / "db")

    first = await load_documentation(store, docs)
    assert first.new_chunks > 0
    store.save()

    second = await load_documentation(store, docs)
    assert second.new_chunks == 0
    assert second.stale_removed == 0
    store.close()
