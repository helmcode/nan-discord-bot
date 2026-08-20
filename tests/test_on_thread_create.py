"""Tests for the Slack notification triggered by new support threads."""

from __future__ import annotations

from dataclasses import dataclass, field

import discord
import pytest

from bot.base import NanBot
from bot.config import settings

SUPPORT_CHANNEL_ID = 111222333
OTHER_CHANNEL_ID = 999888777


@dataclass
class FakeOwner:
    display_name: str = "crstian"


@dataclass
class FakeParent:
    name: str = "support"


@dataclass
class FakeMessage:
    content: str = ""


@dataclass
class FakeThread:
    id: int = 555
    name: str = "El bot no responde"
    parent_id: int = SUPPORT_CHANNEL_ID
    owner_id: int = 42
    owner: FakeOwner | None = field(default_factory=FakeOwner)
    parent: FakeParent | None = field(default_factory=FakeParent)
    jump_url: str = "https://discord.com/channels/1/555"
    starter_message: FakeMessage | None = None
    # One entry per fetch_message call: a FakeMessage to return or an exception
    # to raise. Exhausting the list keeps raising the last entry.
    fetch_results: list[object] = field(default_factory=list)
    fetch_calls: int = 0

    async def fetch_message(self, message_id: int) -> FakeMessage:
        assert message_id == self.id
        self.fetch_calls += 1
        if not self.fetch_results:
            raise discord.NotFound(_FakeResponse(), "unknown message")
        index = min(self.fetch_calls - 1, len(self.fetch_results) - 1)
        result = self.fetch_results[index]
        if isinstance(result, Exception):
            raise result
        return result


class _FakeResponse:
    status = 404
    reason = "Not Found"


class RecordingSlack:
    def __init__(self, enabled: bool = True, result: bool = True) -> None:
        self.enabled = enabled
        self._result = result
        self.events: list[object] = []

    async def notify_support_thread(self, event) -> bool:
        self.events.append(event)
        return self._result


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path, monkeypatch):
    """NanBot opens the thread-map DB relative to the cwd; keep it out of the repo."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Keep the starter-message retry backoff from slowing the suite down."""

    async def instant(_seconds: float) -> None:
        return None

    monkeypatch.setattr("bot.base.asyncio.sleep", instant)


@pytest.fixture
def bot(monkeypatch):
    monkeypatch.setattr(settings, "support_channel_ids", str(SUPPORT_CHANNEL_ID))
    instance = NanBot()
    instance.slack = RecordingSlack()
    return instance


async def test_notifies_slack_for_a_thread_in_a_support_channel(bot):
    thread = FakeThread(starter_message=FakeMessage("El health devuelve starting"))

    await bot.on_thread_create(thread)

    assert len(bot.slack.events) == 1
    event = bot.slack.events[0]
    assert event.thread_name == "El bot no responde"
    assert event.thread_url == thread.jump_url
    assert event.channel_name == "support"
    assert event.author == "crstian"
    assert event.preview == "El health devuelve starting"


async def test_ignores_threads_from_other_channels(bot):
    await bot.on_thread_create(FakeThread(parent_id=OTHER_CHANNEL_ID))
    assert bot.slack.events == []


async def test_ignores_every_thread_when_no_support_channel_is_configured(monkeypatch):
    monkeypatch.setattr(settings, "support_channel_ids", "")
    instance = NanBot()
    instance.slack = RecordingSlack()

    await instance.on_thread_create(FakeThread())

    assert instance.slack.events == []


async def test_does_nothing_when_slack_is_not_configured(bot):
    bot.slack = RecordingSlack(enabled=False)
    await bot.on_thread_create(FakeThread())
    assert bot.slack.events == []


async def test_falls_back_to_fetching_the_forum_starter_message(bot):
    thread = FakeThread(fetch_results=[FakeMessage("Cuerpo del post de foro")])

    await bot.on_thread_create(thread)

    assert thread.fetch_calls == 1
    assert bot.slack.events[0].preview == "Cuerpo del post de foro"


async def test_retries_once_when_the_event_outruns_the_starter_message(bot):
    thread = FakeThread(
        fetch_results=[
            discord.NotFound(_FakeResponse(), "unknown message"),
            FakeMessage("Llegó tarde pero llegó"),
        ]
    )

    await bot.on_thread_create(thread)

    assert thread.fetch_calls == 2
    assert bot.slack.events[0].preview == "Llegó tarde pero llegó"


async def test_notifies_without_a_preview_when_the_message_never_appears(bot):
    thread = FakeThread(fetch_results=[])

    await bot.on_thread_create(thread)

    assert thread.fetch_calls == 2
    assert bot.slack.events[0].preview == ""


async def test_does_not_retry_when_reading_the_starter_message_is_forbidden(bot):
    thread = FakeThread(fetch_results=[discord.Forbidden(_FakeResponse(), "missing access")])

    await bot.on_thread_create(thread)

    assert thread.fetch_calls == 1
    assert bot.slack.events[0].preview == ""


async def test_uses_owner_id_and_parent_id_when_the_cache_is_empty(bot):
    thread = FakeThread(owner=None, parent=None, starter_message=FakeMessage("hola"))

    await bot.on_thread_create(thread)

    event = bot.slack.events[0]
    assert event.author == "user 42"
    assert event.channel_name == str(SUPPORT_CHANNEL_ID)


async def test_a_slack_failure_does_not_propagate(bot):
    class ExplodingSlack(RecordingSlack):
        async def notify_support_thread(self, event) -> bool:
            raise RuntimeError("slack down")

    bot.slack = ExplodingSlack()

    await bot.on_thread_create(FakeThread(starter_message=FakeMessage("hola")))


# --- mirroring later messages into the Slack thread -----------------------


@dataclass
class FakeAuthor:
    display_name: str = "crstian"
    bot: bool = False


@dataclass
class FakeAttachment:
    filename: str


@dataclass
class FakeDiscordMessage:
    id: int = 777
    content: str = "sigue sin arrancar"
    author: FakeAuthor = field(default_factory=FakeAuthor)
    channel: object = None
    attachments: list = field(default_factory=list)


class RecordingApi:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.channel = "C0123"
        self.replies: list[tuple[str, str, str]] = []
        self.announced: list[object] = []

    async def notify_support_thread(self, event):
        self.announced.append(event)
        return "1610144875.000600"

    async def reply_in_thread(self, thread_ts: str, author: str, text: str) -> bool:
        self.replies.append((thread_ts, author, text))
        return True


@pytest.fixture
def mirroring_bot(bot):
    bot.slack_api = RecordingApi()
    return bot


def _thread_channel(bot, thread_id: int = 555, parent_id: int = SUPPORT_CHANNEL_ID):
    """A discord.Thread stand-in that passes the isinstance check."""
    channel = discord.Thread.__new__(discord.Thread)
    object.__setattr__(channel, "id", thread_id)
    object.__setattr__(channel, "parent_id", parent_id)
    return channel


async def test_announcing_a_thread_stores_the_slack_ts(mirroring_bot):
    thread = FakeThread(starter_message=FakeMessage("hola"))

    await mirroring_bot.on_thread_create(thread)

    assert mirroring_bot.thread_map.lookup(thread.id) == ("1610144875.000600", "C0123")


async def test_a_message_in_a_tracked_thread_is_mirrored(mirroring_bot):
    mirroring_bot.thread_map.remember(555, "1610144875.000600", "C0123")
    msg = FakeDiscordMessage(channel=_thread_channel(mirroring_bot))

    await mirroring_bot._mirror_to_slack(msg)

    assert mirroring_bot.slack_api.replies == [("1610144875.000600", "crstian", "sigue sin arrancar")]


async def test_the_forum_opening_message_is_not_mirrored_twice(mirroring_bot):
    """It is already the body of the announcement."""
    mirroring_bot.thread_map.remember(555, "1610144875.000600", "C0123")
    msg = FakeDiscordMessage(id=555, channel=_thread_channel(mirroring_bot, thread_id=555))

    await mirroring_bot._mirror_to_slack(msg)

    assert mirroring_bot.slack_api.replies == []


async def test_a_thread_with_no_mapping_is_skipped(mirroring_bot):
    msg = FakeDiscordMessage(channel=_thread_channel(mirroring_bot))
    await mirroring_bot._mirror_to_slack(msg)
    assert mirroring_bot.slack_api.replies == []


async def test_messages_outside_support_channels_are_skipped(mirroring_bot):
    mirroring_bot.thread_map.remember(555, "1610144875.000600", "C0123")
    msg = FakeDiscordMessage(channel=_thread_channel(mirroring_bot, parent_id=OTHER_CHANNEL_ID))

    await mirroring_bot._mirror_to_slack(msg)

    assert mirroring_bot.slack_api.replies == []


async def test_attachments_are_named_in_the_mirrored_message(mirroring_bot):
    mirroring_bot.thread_map.remember(555, "1610144875.000600", "C0123")
    msg = FakeDiscordMessage(
        content="mira esto",
        channel=_thread_channel(mirroring_bot),
        attachments=[FakeAttachment("error.png")],
    )

    await mirroring_bot._mirror_to_slack(msg)

    assert "error.png" in mirroring_bot.slack_api.replies[0][2]


async def test_an_empty_message_is_not_mirrored(mirroring_bot):
    mirroring_bot.thread_map.remember(555, "1610144875.000600", "C0123")
    msg = FakeDiscordMessage(content="", channel=_thread_channel(mirroring_bot))

    await mirroring_bot._mirror_to_slack(msg)

    assert mirroring_bot.slack_api.replies == []


async def test_a_mirror_failure_does_not_propagate(mirroring_bot):
    class ExplodingApi(RecordingApi):
        async def reply_in_thread(self, thread_ts, author, text):
            raise RuntimeError("slack down")

    mirroring_bot.slack_api = ExplodingApi()
    mirroring_bot.thread_map.remember(555, "1610144875.000600", "C0123")

    await mirroring_bot._mirror_to_slack(FakeDiscordMessage(channel=_thread_channel(mirroring_bot)))


async def test_bot_messages_are_not_mirrored(mirroring_bot):
    """Slack mirrors the human conversation, not the bot's LLM answers."""
    mirroring_bot.thread_map.remember(555, "1610144875.000600", "C0123")
    msg = FakeDiscordMessage(
        author=FakeAuthor(display_name="NaN Builders", bot=True),
        channel=_thread_channel(mirroring_bot),
    )

    await mirroring_bot._mirror_to_slack(msg)

    assert mirroring_bot.slack_api.replies == []
