"""Tests for the Discord thread -> Slack message mapping."""

from __future__ import annotations

import time

from bot.thread_map import SlackThreadMap


def test_remembers_and_looks_up_a_mapping(tmp_path):
    tm = SlackThreadMap(tmp_path)
    tm.remember(555, "1610144875.000600", "C0123")

    assert tm.lookup(555) == ("1610144875.000600", "C0123")
    tm.close()


def test_unknown_thread_returns_none(tmp_path):
    tm = SlackThreadMap(tmp_path)
    assert tm.lookup(999) is None
    tm.close()


def test_mapping_survives_reopening_the_database(tmp_path):
    """A redeploy restarts the process; the mapping has to outlive it."""
    tm = SlackThreadMap(tmp_path)
    tm.remember(555, "1610144875.000600", "C0123")
    tm.close()

    reopened = SlackThreadMap(tmp_path)
    assert reopened.lookup(555) == ("1610144875.000600", "C0123")
    reopened.close()


def test_remembering_the_same_thread_twice_overwrites(tmp_path):
    tm = SlackThreadMap(tmp_path)
    tm.remember(555, "111.000", "C0123")
    tm.remember(555, "222.000", "C0123")

    assert tm.lookup(555) == ("222.000", "C0123")
    tm.close()


def test_purge_drops_only_old_mappings(tmp_path):
    tm = SlackThreadMap(tmp_path)
    tm.remember(1, "111.000", "C0123")
    tm.remember(2, "222.000", "C0123")
    # Backdate one row by 40 days.
    tm._conn.execute(
        "UPDATE slack_threads SET created_at = ? WHERE discord_thread_id = 1",
        (int(time.time()) - 40 * 24 * 3600,),
    )
    tm._conn.commit()

    assert tm.purge_older_than() == 1
    assert tm.lookup(1) is None
    assert tm.lookup(2) == ("222.000", "C0123")
    tm.close()
