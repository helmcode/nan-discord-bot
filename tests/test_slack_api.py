"""Tests for the Slack Web API client used to thread replies."""

from __future__ import annotations

import asyncio
import json
import logging

import httpx
import pytest

from bot.slack import SlackApiClient, SupportThreadEvent, build_reply_payload

EVENT = SupportThreadEvent(
    thread_name="El bot no responde",
    thread_url="https://discord.com/channels/1/2",
    channel_name="support",
    author="crstian",
)


def _client(handler, **kwargs) -> SlackApiClient:
    api = SlackApiClient(bot_token="xoxb-test", channel="C0123", **kwargs)
    api._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    api._MIN_INTERVAL = 0  # no throttling in tests
    return api


async def test_disabled_without_a_token_or_channel():
    assert SlackApiClient(bot_token="", channel="C0123").enabled is False
    assert SlackApiClient(bot_token="xoxb-test", channel="").enabled is False
    assert SlackApiClient(bot_token="xoxb-test", channel="C0123").enabled is True


async def test_announcing_a_thread_returns_the_ts():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer xoxb-test"
        body = json.loads(request.content)
        assert body["channel"] == "C0123"
        assert "thread_ts" not in body
        return httpx.Response(200, json={"ok": True, "ts": "1610144875.000600"})

    api = _client(handler)
    assert await api.notify_support_thread(EVENT) == "1610144875.000600"
    await api.close()


async def test_a_reply_carries_thread_ts():
    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"ok": True, "ts": "1610144999.000100"})

    api = _client(handler)
    assert await api.reply_in_thread("1610144875.000600", "crstian", "no arranca") is True
    assert seen["thread_ts"] == "1610144875.000600"
    assert "crstian" in json.dumps(seen)
    await api.close()


async def test_an_error_inside_a_200_is_not_a_delivery():
    """The Web API reports failures in the body, with HTTP 200."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": False, "error": "channel_not_found"})

    api = _client(handler)
    assert await api.notify_support_thread(EVENT) is None
    await api.close()


async def test_permanent_errors_are_not_retried():
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"ok": False, "error": "not_in_channel"})

    api = _client(handler)
    assert await api.notify_support_thread(EVENT) is None
    assert calls == 1
    await api.close()


async def test_ratelimited_is_retried(monkeypatch):
    monkeypatch.setattr("bot.slack._BACKOFF_SECONDS", (0, 0, 0))
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(200, json={"ok": False, "error": "ratelimited"})
        return httpx.Response(200, json={"ok": True, "ts": "1.2"})

    api = _client(handler)
    assert await api.notify_support_thread(EVENT) == "1.2"
    assert calls == 3
    await api.close()


@pytest.mark.parametrize(
    ("author", "text", "expected"),
    [
        ("crstian", "hola", "*crstian*\nhola"),
        ("<@here>", "a > b", "*&lt;@here&gt;*\na &gt; b"),
    ],
)
def test_reply_payload_escapes_user_content(author, text, expected):
    payload = build_reply_payload(author, text)
    assert payload["blocks"][0]["text"]["text"] == expected
    # The fallback text is what Slack puts in notifications.
    author_line, body = expected.split("\n", 1)
    assert payload["text"] == f"{author_line.strip('*')}: {body}"


def test_reply_fallback_text_cannot_ping_the_channel_or_forge_links():
    payload = build_reply_payload("crstian", "<!channel> look <https://evil.test|here>")
    assert "<" not in payload["text"]
    assert ">" not in payload["text"]
    assert "&lt;!channel&gt;" in payload["text"]


def test_reply_payload_truncates_long_messages():
    payload = build_reply_payload("crstian", "x" * 5000)
    assert len(payload["blocks"][0]["text"]["text"]) <= 3000
    assert "…" in payload["blocks"][0]["text"]["text"]


@pytest.mark.parametrize("char", ["<", ">", "&"])
def test_escaping_cannot_push_a_reply_past_the_section_limit(char):
    """One "<" becomes "&lt;": truncating before escaping lets 2000 chars grow to 8000."""
    payload = build_reply_payload("a", char * 2000)
    text = payload["blocks"][0]["text"]["text"]

    assert len(text) <= 3000
    body = text.split("\n", 1)[1].removesuffix("…")
    entity = {"<": "&lt;", ">": "&gt;", "&": "&amp;"}[char]
    # Only whole entities: nothing is left of a half-cut "&lt;".
    assert body.replace(entity, "") == ""


def test_a_long_author_still_fits_the_section_limit():
    payload = build_reply_payload("&" * 500, "<" * 2000)
    assert len(payload["blocks"][0]["text"]["text"]) <= 3000


# --- configuration --------------------------------------------------------


@pytest.mark.parametrize(
    ("token", "channel", "message"),
    [
        ("xoxb-test", "", "SLACK_CHANNEL_ID is empty"),
        ("", "C0123", "SLACK_BOT_TOKEN is empty"),
        ("xoxp-user-token", "C0123", "not a bot token"),
        ("https://hooks.slack.com/services/T/B/X", "C0123", "not a bot token"),
    ],
)
def test_a_half_configured_token_mode_is_one_error_at_startup(token, channel, message, caplog):
    with caplog.at_level(logging.ERROR, logger="nan-bot"):
        api = SlackApiClient(bot_token=token, channel=channel)

    assert api.enabled is False
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert message in errors[0].getMessage()
    # The token itself must never reach the logs.
    if token:
        assert token not in caplog.text


def test_no_token_and_no_channel_is_silent(caplog):
    with caplog.at_level(logging.DEBUG, logger="nan-bot"):
        assert SlackApiClient(bot_token="", channel="").enabled is False
    assert caplog.records == []


async def test_a_reply_goes_to_the_requested_channel():
    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"ok": True, "ts": "2.2"})

    api = _client(handler)
    assert await api.reply_in_thread("1.1", "crstian", "hola", channel="C_OLD") is True
    assert seen["channel"] == "C_OLD"
    await api.close()


# --- retries ---------------------------------------------------------------


@pytest.mark.parametrize(
    "failure",
    [
        httpx.Response(200, json={"ok": False, "error": "internal_error"}),
        httpx.Response(200, json={"ok": False, "error": "service_unavailable"}),
        httpx.ConnectError("connection refused"),
        httpx.Response(502, text="<html><body>502 Bad Gateway</body></html>"),
    ],
    ids=["internal_error", "service_unavailable", "connect_error", "502_html"],
)
async def test_transient_failures_are_retried(failure, monkeypatch):
    monkeypatch.setattr("bot.slack._BACKOFF_SECONDS", (0, 0, 0))
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            if isinstance(failure, Exception):
                raise failure
            return failure
        return httpx.Response(200, json={"ok": True, "ts": "1.2"})

    api = _client(handler)
    assert await api.notify_support_thread(EVENT) == "1.2"
    assert calls == 2
    await api.close()


async def test_gives_up_after_four_attempts(monkeypatch):
    monkeypatch.setattr("bot.slack._BACKOFF_SECONDS", (0, 0, 0))
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("connection refused")

    api = _client(handler)
    assert await api.notify_support_thread(EVENT) is None
    assert calls == 4
    await api.close()


# --- throttling ------------------------------------------------------------


@pytest.fixture
def recorded_sleeps(monkeypatch):
    """Record asyncio.sleep calls without waiting, but still yield to the loop."""
    real_sleep = asyncio.sleep
    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr("bot.slack.asyncio.sleep", fake_sleep)
    return waits, real_sleep


async def test_consecutive_posts_are_spaced_by_the_channel_rate_limit(recorded_sleeps):
    waits, _ = recorded_sleeps

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True, "ts": "1.2"})

    api = _client(handler)
    api._MIN_INTERVAL = 1.05

    await api.notify_support_thread(EVENT)
    assert waits == []  # the first post goes out immediately
    await api.notify_support_thread(EVENT)

    assert len(waits) == 1
    assert waits[0] == pytest.approx(1.05, abs=0.05)
    await api.close()


async def test_concurrent_posts_are_serialised(recorded_sleeps):
    waits, real_sleep = recorded_sleeps
    in_flight = 0
    peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        for _ in range(5):
            await real_sleep(0)
        in_flight -= 1
        return httpx.Response(200, json={"ok": True, "ts": "1.2"})

    api = _client(handler)
    api._MIN_INTERVAL = 1.05

    results = await asyncio.gather(api.notify_support_thread(EVENT), api.notify_support_thread(EVENT))

    assert results == ["1.2", "1.2"]
    assert peak == 1
    assert len(waits) == 1
    assert waits[0] == pytest.approx(1.05, abs=0.05)
    await api.close()
