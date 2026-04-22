"""
Knowledge base system: load documentation, create embeddings, semantic search.

Uses LiteLLM API for embeddings and stores vectors in SQLite with
cosine similarity search in Python.
"""

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from bot.config import DB_DIR, logger


@dataclass
class DocumentChunk:
    """A chunk of text with its embedding."""
    id: str
    text: str
    source: str  # original file name
    embedding: list[float] | None = None


@dataclass
class SearchResult:
    """Result from a similarity search."""
    chunk: DocumentChunk
    score: float


class SimpleVectorStore:
    """
    Persistent vector store using SQLite.
    Stores embeddings as JSON arrays in a single table.
    Cosine similarity computed in Python.
    """

    def __init__(self, db_dir: Path) -> None:
        self._db_dir = db_dir
        self._db_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = db_dir / "vectors.db"
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._chunks: list[DocumentChunk] = []
        self._init_schema()
        self._load()

    @property
    def chunks(self) -> list[DocumentChunk]:
        """Return a copy of all chunks for external access."""
        return list(self._chunks)

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                source TEXT NOT NULL,
                embedding TEXT  -- JSON array of floats
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
            CREATE TABLE IF NOT EXISTS doc_hashes (
                source TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL
            );
        """)

    def _load_all(self) -> None:
        """Load all chunks from SQLite into memory for search."""
        cursor = self._conn.execute("SELECT id, text, source, embedding FROM chunks")
        self._chunks: list[DocumentChunk] = []
        for row in cursor:
            embedding: list[float] | None = None
            if row["embedding"]:
                embedding = json.loads(row["embedding"])
            self._chunks.append(DocumentChunk(
                id=row["id"],
                text=row["text"],
                source=row["source"],
                embedding=embedding,
            ))

    def _load(self) -> None:
        """Check if DB has data and load it."""
        cursor = self._conn.execute("SELECT COUNT(*) as cnt FROM chunks")
        count = cursor.fetchone()["cnt"]
        if count > 0:
            self._load_all()
            logger.info("Loaded %d chunks from SQLite", count)

    def get_doc_hash(self, source: str) -> str | None:
        """Get the stored content hash for a document source."""
        cursor = self._conn.execute(
            "SELECT content_hash FROM doc_hashes WHERE source = ?", (source,)
        )
        row = cursor.fetchone()
        return row["content_hash"] if row else None

    def set_doc_hash(self, source: str, content_hash: str) -> None:
        """Store a content hash for a document source."""
        self._conn.execute(
            "INSERT OR REPLACE INTO doc_hashes (source, content_hash) VALUES (?, ?)",
            (source, content_hash),
        )

    def remove_source(self, source: str) -> None:
        """Remove all chunks and metadata for a source."""
        self._chunks = [c for c in self._chunks if c.source != source]
        self._conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
        self._conn.execute("DELETE FROM doc_hashes WHERE source = ?", (source,))

    def get_tracked_sources(self) -> set[str]:
        """Return all sources that have stored hashes."""
        cursor = self._conn.execute("SELECT source FROM doc_hashes")
        return {row["source"] for row in cursor}

    def save(self) -> None:
        """Persist chunks to SQLite (upsert)."""
        self._conn.executemany(
            """INSERT OR REPLACE INTO chunks (id, text, source, embedding)
               VALUES (?, ?, ?, ?)""",
            [
                (
                    c.id,
                    c.text,
                    c.source,
                    json.dumps(c.embedding) if c.embedding else None,
                )
                for c in self._chunks
            ],
        )
        self._conn.commit()
        logger.info("Saved %d chunks to SQLite", len(self._chunks))

    def add(self, chunk: DocumentChunk) -> None:
        """Add a chunk (in-memory, call save() to persist)."""
        self._chunks.append(chunk)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[SearchResult]:
        """Find top-K most similar chunks using cosine similarity."""
        if not self._chunks:
            return []

        # Normalize query
        q_norm = sum(x * x for x in query_vector) ** 0.5
        if q_norm == 0:
            return []
        q_norm_vec = [x / q_norm for x in query_vector]

        # Compute cosine similarity with all chunks that have embeddings
        results: list[tuple[float, DocumentChunk]] = []
        for chunk in self._chunks:
            if not chunk.embedding:
                continue
            # Dot product (vectors already normalized → cosine similarity)
            score = sum(a * b for a, b in zip(q_norm_vec, chunk.embedding))
            if score > 0.1:
                results.append((score, chunk))

        results.sort(key=lambda x: x[0], reverse=True)
        return [SearchResult(chunk=c, score=s) for s, c in results[:top_k]]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def chunk_text(text: str, source: str, max_chars: int = 2000, overlap: int = 100) -> list[DocumentChunk]:
    """
    Split text into overlapping chunks suitable for embedding.
    Tries to split on paragraph boundaries first, then on sentences.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[DocumentChunk] = []
    current_text = ""
    current_id = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_text) + len(para) > max_chars:
            if current_text:
                chunks.append(DocumentChunk(
                    id=f"{source}#{current_id}",
                    text=current_text,
                    source=source,
                    embedding=None,
                ))
                current_id += 1
                # Overlap: keep last sentence
                last_sentence_end = current_text.rfind(".")
                if last_sentence_end > len(current_text) * 0.3:
                    current_text = current_text[last_sentence_end + 1:].strip()[:overlap]
                else:
                    current_text = ""

        current_text = f"{current_text}\n\n{para}" if current_text else para

    if current_text:
        chunks.append(DocumentChunk(
            id=f"{source}#{current_id}",
            text=current_text,
            source=source,
            embedding=None,
        ))

    return chunks


@dataclass
class LoadResult:
    """Result of loading documentation into the store."""
    new_chunks: int
    stale_removed: int


async def load_documentation(store: SimpleVectorStore, docs_dir: Path) -> LoadResult:
    """
    Load markdown files from docs_dir into the vector store.
    Uses content hashing to skip unchanged files and remove deleted ones.
    Returns a LoadResult with counts of new chunks and removed stale sources.
    """
    if not docs_dir.exists():
        logger.warning("Docs directory not found: %s", docs_dir)
        return LoadResult(new_chunks=0, stale_removed=0)

    docs_dir_resolved = docs_dir.resolve()
    current_sources: set[str] = set()
    new_chunks = 0

    for md_file in sorted(docs_dir.glob("**/*.md")):
        resolved = md_file.resolve()
        if not str(resolved).startswith(str(docs_dir_resolved / "")):
            logger.warning("Skipping potentially malicious file: %s", md_file)
            continue

        source = md_file.name
        current_sources.add(source)
        text = md_file.read_text(encoding="utf-8")
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        stored_hash = store.get_doc_hash(source)
        if stored_hash == content_hash:
            logger.info("Unchanged, skipping: %s", source)
            continue

        logger.info("Changed (or new), re-indexing: %s", source)
        store.remove_source(source)
        chunks = chunk_text(text, source=source)
        for chunk in chunks:
            store.add(chunk)
            new_chunks += 1
        store.set_doc_hash(source, content_hash)

    # Remove chunks for deleted files
    stale_sources = store.get_tracked_sources() - current_sources
    for source in stale_sources:
        logger.info("File removed, cleaning up: %s", source)
        store.remove_source(source)

    if new_chunks:
        logger.info("Re-indexed %d chunks from changed files", new_chunks)
    elif stale_sources:
        logger.info("Removed %d stale sources", len(stale_sources))
    else:
        logger.info("All %d docs unchanged, no re-indexing needed", len(current_sources))

    return LoadResult(new_chunks=new_chunks, stale_removed=len(stale_sources))
