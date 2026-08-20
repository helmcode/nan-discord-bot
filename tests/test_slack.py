from __future__ import annotations

import httpx
import pytest

from bot.config import _parse_channel_ids
from bot.slack import SlackNotifier, SupportThreadEvent, build_support_thread_payload, escape_mrkdwn

EVENT = SupportThreadEvent(
    thread_name="Bot no responde en prod",
    thread_url="https://discord.com/channels/1/2",
    channel_name="support",
    author="crstian",
    preview="Levanté el contenedor y el health devuelve starting.",
)


def _block_text(payload: dict) -> str:
    parts = []
    for block in payload["blocks"]:
        if "text" in block:
            parts.append(block["text"]["text"])
        for element in block.get("elements", []):
            parts.append(element["text"])
    return "\n".join(parts)


def test_payload_includes_thread_link_author_and_preview():
    payload = build_support_thread_payload(EVENT)
    text = _block_text(payload)

    assert f"<{EVENT.thread_url}|{EVENT.thread_name}>" in text
    assert "crstian" in text
    assert "support" in text
    assert "Levanté el contenedor" in text
    # Fallback text is what Slack shows in notifications.
    assert EVENT.thread_name in payload["text"]


def test_payload_omits_preview_section_when_there_is_no_starter_text():
    payload = build_support_thread_payload(
        SupportThreadEvent(
            thread_name="Sin cuerpo",
            thread_url="https://discord.com/channels/1/2",
            channel_name="support",
            author="crstian",
        )
    )
    assert len(payload["blocks"]) == 2


def test_payload_escapes_mrkdwn_in_user_controlled_fields():
    payload = build_support_thread_payload(
        SupportThreadEvent(
            thread_name="<https://evil.test|click me> & co",
            thread_url="https://discord.com/channels/1/2",
            channel_name="support",
            author="<@here>",
            preview="a > b & c < d",
        )
    )
    text = _block_text(payload)

    assert "<https://evil.test" not in text
    assert "&lt;https://evil.test|click me&gt; &amp; co" in text
    assert "&lt;@here&gt;" in text
    assert "a &gt; b &amp; c &lt; d" in text
    # The link the bot builds itself must survive escaping.
    assert f"<{EVENT.thread_url}|" in text


def test_payload_truncates_long_title_and_preview():
    payload = build_support_thread_payload(
        SupportThreadEvent(
            thread_name="t" * 500,
            thread_url="https://discord.com/channels/1/2",
            channel_name="support",
            author="crstian",
            preview="p" * 2000,
        )
    )
    text = _block_text(payload)

    assert "t" * 200 not in text
    assert "p" * 600 not in text
    assert "…" in text


def test_escape_mrkdwn_escapes_ampersand_before_angle_brackets():
    assert escape_mrkdwn("<a>") == "&lt;a&gt;"


async def test_notifier_is_disabled_without_a_webhook_url():
    notifier = SlackNotifier(webhook_url="")
    assert notifier.enabled is False
    assert await notifier.notify_support_thread(EVENT) is False


async def test_notifier_posts_the_payload_to_the_webhook(monkeypatch):
    sent: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        sent.append(__import__("json").loads(request.content))
        return httpx.Response(200, text="ok")

    notifier = SlackNotifier(webhook_url="https://hooks.slack.test/services/T/B/X")
    notifier._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    assert await notifier.notify_support_thread(EVENT) is True
    assert len(sent) == 1
    assert sent[0]["blocks"]
    await notifier.close()


async def test_notifier_does_not_retry_on_client_error():
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404, text="no_service")

    notifier = SlackNotifier(webhook_url="https://hooks.slack.test/services/T/B/X")
    notifier._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    assert await notifier.notify_support_thread(EVENT) is False
    assert calls == 1
    await notifier.close()


async def test_notifier_retries_on_server_error_then_succeeds(monkeypatch):
    monkeypatch.setattr("bot.slack._BACKOFF_SECONDS", (0, 0, 0))
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(503)
        return httpx.Response(200, text="ok")

    notifier = SlackNotifier(webhook_url="https://hooks.slack.test/services/T/B/X")
    notifier._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    assert await notifier.notify_support_thread(EVENT) is True
    assert calls == 3
    await notifier.close()


async def test_notifier_gives_up_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr("bot.slack._BACKOFF_SECONDS", (0, 0, 0))
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("boom", request=request)

    notifier = SlackNotifier(webhook_url="https://hooks.slack.test/services/T/B/X")
    notifier._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    assert await notifier.notify_support_thread(EVENT) is False
    assert calls == 4
    await notifier.close()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", set()),
        ("123", {123}),
        (" 123 , 456 ", {123, 456}),
        ("123,,456", {123, 456}),
        ("123,abc,456", {123, 456}),
        ("1" * 22, set()),
    ],
)
def test_parse_channel_ids(raw, expected):
    assert _parse_channel_ids(raw) == expected


def test_httpx_request_logging_cannot_leak_the_webhook_secret(caplog):
    """The webhook secret lives in the URL path, so httpx must not log requests."""
    import logging

    import bot.config  # noqa: F401  (importing configures logging)

    assert logging.getLogger("httpx").level >= logging.WARNING
