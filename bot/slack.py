"""Slack notifications via Incoming Webhook."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx

from bot.config import logger, settings

_BACKOFF_SECONDS = (0.5, 1.0, 2.0)

# Only this prefix is an Incoming Webhook endpoint. A Slack client URL
# (https://app.slack.com/client/...) answers a POST with the web app and HTTP
# 200, which would otherwise read as a delivered message.
_WEBHOOK_PREFIX = "https://hooks.slack.com/services/"
_MAX_TITLE_LEN = 200
_MAX_PREVIEW_LEN = 600

# Slack mrkdwn requires these three characters to be HTML-escaped so user text
# cannot forge links or entities inside a block.
_ESCAPES = (("&", "&amp;"), ("<", "&lt;"), (">", "&gt;"))


def escape_mrkdwn(text: str) -> str:
    """Escape user-controlled text for safe inclusion in Slack mrkdwn."""
    for char, replacement in _ESCAPES:
        text = text.replace(char, replacement)
    return text


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


@dataclass
class SupportThreadEvent:
    """The data needed to announce a new Discord support thread in Slack."""

    thread_name: str
    thread_url: str
    channel_name: str
    author: str
    preview: str = ""


def build_support_thread_payload(event: SupportThreadEvent) -> dict:
    """Build the Slack Block Kit payload for a new support thread."""
    title = escape_mrkdwn(_truncate(event.thread_name, _MAX_TITLE_LEN)) or "(untitled)"
    author = escape_mrkdwn(_truncate(event.author, _MAX_TITLE_LEN)) or "unknown"
    channel = escape_mrkdwn(_truncate(event.channel_name, _MAX_TITLE_LEN))

    blocks: list[dict] = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*<{event.thread_url}|{title}>*",
            },
        },
        {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": f"Discord #{channel} · by *{author}*"},
            ],
        },
    ]

    preview = _truncate(event.preview, _MAX_PREVIEW_LEN)
    if preview:
        blocks.insert(
            1,
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f">{escape_mrkdwn(preview)}"},
            },
        )

    return {
        # Fallback text for notifications and clients that cannot render blocks.
        "text": f"New support thread in Discord #{channel}: {title}",
        "blocks": blocks,
    }


class SlackNotifier:
    """Posts messages to a Slack Incoming Webhook.

    Disabled (a no-op) when ``SLACK_WEBHOOK_URL`` is unset, so the bot runs
    unchanged in environments without Slack configured.
    """

    def __init__(self, webhook_url: str | None = None, timeout: float | None = None) -> None:
        self._webhook_url = (webhook_url if webhook_url is not None else settings.slack_webhook_url).strip()
        self._timeout = timeout if timeout is not None else float(settings.slack_http_timeout)
        self._client: httpx.AsyncClient | None = None
        # Validated once, at startup, so a misconfigured URL is one loud error
        # in the logs instead of one per notification.
        self._enabled = self._validate_url()

    def _validate_url(self) -> bool:
        if not self._webhook_url:
            return False
        if not self._webhook_url.startswith(_WEBHOOK_PREFIX):
            logger.error(
                "SLACK_WEBHOOK_URL is not an Incoming Webhook (must start with %s); notifications are disabled",
                _WEBHOOK_PREFIX,
            )
            return False
        return True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def post(self, payload: dict) -> bool:
        """POST a payload to the webhook. Returns True when Slack accepted it."""
        if not self.enabled:
            logger.debug("Slack webhook not configured, skipping notification")
            return False

        client = self._get_client()
        last_error: str | None = None

        for attempt, backoff in enumerate((*_BACKOFF_SECONDS, None)):
            try:
                resp = await client.post(self._webhook_url, json=payload)
            except httpx.HTTPError as e:
                last_error = type(e).__name__
            else:
                if resp.status_code < 400:
                    # An Incoming Webhook answers "ok". Anything else behind a
                    # 2xx means the URL is not a webhook, so the message was
                    # never delivered however healthy the status looks.
                    body = resp.text.strip()
                    if body.lower() == "ok":
                        return True
                    logger.error(
                        "Slack returned HTTP %d but not an Incoming Webhook response (body starts with %r); "
                        "check that SLACK_WEBHOOK_URL is a %s… URL",
                        resp.status_code,
                        body[:40],
                        _WEBHOOK_PREFIX,
                    )
                    return False
                # 4xx means a bad payload or a revoked webhook: retrying cannot help.
                if resp.status_code < 500 and resp.status_code != 429:
                    logger.error("Slack webhook rejected the message (HTTP %d)", resp.status_code)
                    return False
                last_error = f"HTTP {resp.status_code}"

            if backoff is None:
                break
            logger.warning("Slack webhook attempt %d failed (%s), retrying", attempt + 1, last_error)
            await asyncio.sleep(backoff)

        logger.error("Slack webhook failed after %d attempts: %s", len(_BACKOFF_SECONDS) + 1, last_error)
        return False

    async def notify_support_thread(self, event: SupportThreadEvent) -> bool:
        """Announce a new Discord support thread in Slack."""
        return await self.post(build_support_thread_payload(event))
