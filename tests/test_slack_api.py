"""Tests for the Slack Web API client used to thread replies."""

from __future__ import annotations

import json

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


def test_reply_payload_truncates_long_messages():
    payload = build_reply_payload("crstian", "x" * 5000)
    assert len(payload["blocks"][0]["text"]["text"]) < 2100
    assert "…" in payload["blocks"][0]["text"]["text"]
