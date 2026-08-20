"""Persistent mapping from a Discord thread to its Slack message.

Slack replies are threaded by passing the parent message's ``ts``. That value
only exists after the parent has been posted, so it has to outlive the process:
without it, a restart orphans every live support thread.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from bot.config import logger

# Discord archives support threads long before this, and a Slack thread that old
# is not worth replying into. Keeps the table from growing without bound.
_RETENTION_SECONDS = 30 * 24 * 3600


class SlackThreadMap:
    """Maps Discord thread IDs to the Slack message ts that announced them."""

    def __init__(self, db_dir: Path) -> None:
        db_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = db_dir / "slack_threads.db"
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS slack_threads (
                discord_thread_id INTEGER PRIMARY KEY,
                slack_ts TEXT NOT NULL,
                slack_channel TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
        """)
        self._conn.commit()

    def remember(self, discord_thread_id: int, slack_ts: str, slack_channel: str) -> None:
        """Record the Slack message that announced a Discord thread."""
        self._conn.execute(
            "INSERT OR REPLACE INTO slack_threads "
            "(discord_thread_id, slack_ts, slack_channel, created_at) VALUES (?, ?, ?, ?)",
            (discord_thread_id, slack_ts, slack_channel, int(time.time())),
        )
        self._conn.commit()

    def lookup(self, discord_thread_id: int) -> tuple[str, str] | None:
        """Return (slack_ts, slack_channel) for a thread, or None if unknown."""
        row = self._conn.execute(
            "SELECT slack_ts, slack_channel FROM slack_threads WHERE discord_thread_id = ?",
            (discord_thread_id,),
        ).fetchone()
        if row is None:
            return None
        return row["slack_ts"], row["slack_channel"]

    def purge_older_than(self, seconds: int = _RETENTION_SECONDS) -> int:
        """Drop mappings older than `seconds`. Returns how many were removed."""
        cutoff = int(time.time()) - seconds
        cursor = self._conn.execute("DELETE FROM slack_threads WHERE created_at < ?", (cutoff,))
        self._conn.commit()
        removed = cursor.rowcount
        if removed:
            logger.info("Purged %d stale Slack thread mappings", removed)
        return removed

    def close(self) -> None:
        self._conn.close()
