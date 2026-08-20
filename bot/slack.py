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
_MAX_REPLY_LEN = 2000

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


def build_reply_payload(author: str, text: str) -> dict:
    """Build the Slack payload mirroring one Discord message into a thread."""
    safe_author = escape_mrkdwn(_truncate(author, _MAX_TITLE_LEN)) or "unknown"
    body = _truncate(text, _MAX_REPLY_LEN)
    rendered = escape_mrkdwn(body) if body else "_(sin texto)_"

    return {
        "text": f"{safe_author}: {body or '(sin texto)'}",
        "blocks": [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*{safe_author}*\n{rendered}"},
            }
        ],
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


class SlackApiClient:
    """Posts through chat.postMessage with a bot token.

    Unlike an Incoming Webhook, the Web API returns the message ``ts``, which is
    what a threaded reply needs as ``thread_ts``. Slack allows roughly one
    message per second per channel, so posts are serialised behind a lock.
    """

    _URL = "https://slack.com/api/chat.postMessage"
    _MIN_INTERVAL = 1.05

    def __init__(
        self,
        bot_token: str | None = None,
        channel: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._token = (bot_token if bot_token is not None else settings.slack_bot_token).strip()
        self._channel = (channel if channel is not None else settings.slack_channel_id).strip()
        self._timeout = timeout if timeout is not None else float(settings.slack_http_timeout)
        self._client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()
        self._last_post = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self._token and self._channel)

    @property
    def channel(self) -> str:
        return self._channel

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def _throttle(self) -> None:
        """Space out posts to stay under Slack's per-channel rate limit."""
        loop = asyncio.get_running_loop()
        elapsed = loop.time() - self._last_post
        if elapsed < self._MIN_INTERVAL:
            await asyncio.sleep(self._MIN_INTERVAL - elapsed)
        self._last_post = loop.time()

    async def post(self, payload: dict, thread_ts: str | None = None) -> str | None:
        """Post a message. Returns its ``ts``, or None when it was not delivered."""
        if not self.enabled:
            logger.debug("Slack bot token or channel not configured, skipping")
            return None

        body = {**payload, "channel": self._channel}
        if thread_ts:
            body["thread_ts"] = thread_ts

        headers = {"Authorization": f"Bearer {self._token}"}
        client = self._get_client()
        last_error: str | None = None

        for attempt, backoff in enumerate((*_BACKOFF_SECONDS, None)):
            async with self._lock:
                await self._throttle()
                try:
                    resp = await client.post(self._URL, json=body, headers=headers)
                except httpx.HTTPError as e:
                    last_error = type(e).__name__
                    resp = None

            if resp is not None:
                # The Web API reports failures inside a 200 body, so the status
                # code alone never tells you whether the message landed.
                try:
                    data = resp.json()
                except ValueError:
                    last_error = f"HTTP {resp.status_code} with a non-JSON body"
                    data = {}
                else:
                    if data.get("ok"):
                        return data.get("ts")
                    error = data.get("error", "unknown_error")
                    if error not in ("ratelimited", "internal_error", "service_unavailable"):
                        logger.error("Slack rejected the message: %s", error)
                        return None
                    last_error = error

            if backoff is None:
                break
            logger.warning("Slack API attempt %d failed (%s), retrying", attempt + 1, last_error)
            await asyncio.sleep(backoff)

        logger.error("Slack API failed after %d attempts: %s", len(_BACKOFF_SECONDS) + 1, last_error)
        return None

    async def notify_support_thread(self, event: SupportThreadEvent) -> str | None:
        """Announce a new Discord support thread. Returns the Slack ``ts``."""
        return await self.post(build_support_thread_payload(event))

    async def reply_in_thread(self, thread_ts: str, author: str, text: str) -> bool:
        """Mirror one Discord message as a reply under the announcement."""
        ts = await self.post(build_reply_payload(author, text), thread_ts=thread_ts)
        return ts is not None
