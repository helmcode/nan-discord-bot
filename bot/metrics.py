"""Daily LiteLLM token usage metrics report."""

import asyncio
import urllib.parse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp

from bot.config import settings, logger


async def fetch_token_metrics(session: aiohttp.ClientSession) -> list[dict[str, Any]]:
    """Fetch token usage data from LiteLLM proxy for the last 24 hours.

    Returns a list of dicts with keys: alias, total_tokens, requests.
    """
    proxy_url = settings.litellm_proxy_url
    admin_key = settings.litellm_admin_key

    if not proxy_url or not admin_key:
        logger.warning("LiteLLM proxy URL or admin key not configured, skipping metrics")
        return []

    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=24)

    base_url = proxy_url.rstrip("/") + "/spend/logs/ui"
    params = {
        "start_date": start.strftime("%Y-%m-%d %H:%M:%S"),
        "end_date": now.strftime("%Y-%m-%d %H:%M:%S"),
        "limit": "50",
    }

    headers = {"Authorization": "Bearer " + admin_key}

    alias_totals = defaultdict(lambda: {"total_tokens": 0, "requests": 0})
    page = 1
    total_pages = 1
    total_logs = 0

    while page <= total_pages:
        url = base_url + "?" + urllib.parse.urlencode(params) + "&page=" + str(page)
        try:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status != 200:
                    logger.error("LiteLLM API returned status %d on page %d", resp.status, page)
                    break
                data = await resp.json()
        except asyncio.TimeoutError:
            logger.error("Timeout fetching LiteLLM spend logs page %d", page)
            break
        except Exception as e:
            logger.error("Error fetching LiteLLM spend logs page %d: %s", page, type(e).__name__)
            break

        total_pages = data.get("total_pages", 1)
        total_logs += len(data.get("data", []))

        for log in data.get("data", []):
            alias = log.get("metadata", {}).get("user_api_key_alias", "unknown")
            alias_totals[alias]["total_tokens"] += log.get("total_tokens", 0)
            alias_totals[alias]["requests"] += 1

        page += 1

        if page % 100 == 0:
            logger.info("LiteLLM metrics: processed page %d/%d", page, total_pages)

    logger.info("LiteLLM metrics: fetched %d logs, %d unique aliases", total_logs, len(alias_totals))

    result = []
    for alias, stats in alias_totals.items():
        result.append({
            "alias": alias,
            "total_tokens": stats["total_tokens"],
            "requests": stats["requests"],
        })

    result.sort(key=lambda x: x["total_tokens"], reverse=True)
    return result


def format_metrics_report(top10: list[dict[str, Any]], total_aliases: int) -> str:
    """Format the top 10 metrics into a Discord-friendly report string."""
    lines = ["**📊 Token Usage — Last 24 Hours**\n"]

    for rank, entry in enumerate(top10, 1):
        tokens = "{:,}".format(entry["total_tokens"])
        requests = entry["requests"]
        emoji = ["🥇", "🥈", "🥉"][rank - 1] if rank <= 3 else str(rank)
        lines.append(f"{emoji} `{entry['alias']}` — {tokens} tokens ({requests} requests)")

    lines.append("")
    lines.append(f"Total aliases: {total_aliases}")
    return "\n".join(lines)


async def send_metrics_report(bot: Any) -> None:
    """Fetch metrics and send the report to the status channel."""
    proxy_url = settings.litellm_proxy_url
    status_channel_id = settings.status_channel_id

    if not proxy_url or not status_channel_id:
        logger.warning("Metrics or status channel not configured, skipping")
        return

    logger.info("Fetching LiteLLM token metrics...")

    try:
        async with aiohttp.ClientSession() as session:
            top10 = await fetch_token_metrics(session)
    except Exception as e:
        logger.error("Failed to fetch metrics: %s", type(e).__name__, exc_info=True)
        return

    if not top10:
        logger.warning("No metrics data returned from LiteLLM")
        return

    report = format_metrics_report(top10[:10], len(top10))

    try:
        channel = await bot.fetch_channel(status_channel_id)
        await channel.send(report)
        logger.info("Metrics report sent to channel %s", status_channel_id)
    except Exception as e:
        logger.error("Failed to send metrics report: %s", type(e).__name__, exc_info=True)
