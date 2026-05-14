"""Daily LiteLLM token usage metrics report."""

import urllib.parse
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

import aiohttp

from bot.config import logger, settings


async def fetch_token_metrics(session: aiohttp.ClientSession, alias: str | None = None) -> list[dict[str, Any]]:
    """Fetch token usage data from LiteLLM proxy for the last 24 hours.

    If alias is provided, filter to that specific user's data.
    Returns a list of dicts with keys: alias, total_tokens, requests.
    """
    proxy_url = settings.litellm_proxy_url
    admin_key = settings.litellm_admin_key

    if not proxy_url or not admin_key:
        logger.warning("LiteLLM proxy URL or admin key not configured, skipping metrics")
        return []

    now = datetime.now(UTC)
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
    filtered_logs = 0

    while page <= total_pages:
        url = base_url + "?" + urllib.parse.urlencode(params) + "&page=" + str(page)
        try:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status != 200:
                    logger.error("LiteLLM API returned status %d on page %d", resp.status, page)
                    break
                data = await resp.json()
        except TimeoutError:
            logger.error("Timeout fetching LiteLLM spend logs page %d", page)
            break
        except Exception as e:
            logger.error("Error fetching LiteLLM spend logs page %d: %s", page, type(e).__name__)
            break

        total_pages = data.get("total_pages", 1)
        raw_logs = data.get("data", [])
        total_logs += len(raw_logs)

        for log in raw_logs:
            log_alias = log.get("metadata", {}).get("user_api_key_alias", "unknown")
            if alias is not None and log_alias != alias:
                continue
            filtered_logs += 1
            alias_totals[log_alias]["total_tokens"] += log.get("total_tokens", 0)
            alias_totals[log_alias]["requests"] += 1

        page += 1

        if page % 100 == 0:
            logger.info("LiteLLM metrics: processed page %d/%d", page, total_pages)

    if alias is not None:
        logger.info("LiteLLM metrics: fetched %d logs, %d matching alias '%s'", filtered_logs, len(alias_totals), alias)
    else:
        logger.info("LiteLLM metrics: fetched %d logs, %d unique aliases", total_logs, len(alias_totals))

    result = []
    for log_alias, stats in alias_totals.items():
        result.append({
            "alias": log_alias,
            "total_tokens": stats["total_tokens"],
            "requests": stats["requests"],
        })

    result.sort(key=lambda x: x["total_tokens"], reverse=True)
    return result


async def fetch_user_model_breakdown(session: aiohttp.ClientSession, alias: str) -> list[dict[str, Any]]:
    """Fetch per-model token breakdown for a specific user alias.

    Returns a list of dicts with keys: model, total_tokens, requests.
    """
    proxy_url = settings.litellm_proxy_url
    admin_key = settings.litellm_admin_key

    if not proxy_url or not admin_key:
        return []

    now = datetime.now(UTC)
    start = now - timedelta(hours=24)

    base_url = proxy_url.rstrip("/") + "/spend/logs/ui"
    params = {
        "start_date": start.strftime("%Y-%m-%d %H:%M:%S"),
        "end_date": now.strftime("%Y-%m-%d %H:%M:%S"),
        "limit": "50",
    }

    headers = {"Authorization": "Bearer " + admin_key}

    model_totals = defaultdict(lambda: {"total_tokens": 0, "requests": 0})
    page = 1
    total_pages = 1

    while page <= total_pages:
        url = base_url + "?" + urllib.parse.urlencode(params) + "&page=" + str(page)
        try:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status != 200:
                    break
                data = await resp.json()
        except (TimeoutError, Exception):
            break

        total_pages = data.get("total_pages", 1)

        for log in data.get("data", []):
            log_alias = log.get("metadata", {}).get("user_api_key_alias", "")
            if log_alias != alias:
                continue
            model = log.get("model", "unknown")
            model_totals[model]["total_tokens"] += log.get("total_tokens", 0)
            model_totals[model]["requests"] += 1

        page += 1

    result = []
    for model, stats in model_totals.items():
        result.append({
            "model": model,
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


def format_user_metrics_report(
    alias: str,
    total_tokens: int,
    total_requests: int,
    model_breakdown: list[dict[str, Any]],
) -> str:
    """Format a user's personal metrics report."""
    lines = [f"**📊 {alias} — Token Usage (Last 24 Hours)**\n"]
    lines.append(f"Total tokens: `{total_tokens:,}`")
    lines.append(f"Total requests: `{total_requests}`")

    if model_breakdown:
        lines.append("")
        lines.append("**By Model:**")
        for entry in model_breakdown:
            tokens = "{:,}".format(entry["total_tokens"])
            requests = entry["requests"]
            pct = (entry["total_tokens"] / total_tokens * 100) if total_tokens > 0 else 0
            lines.append(f"- `{entry['model']}` — {tokens} tokens ({requests} requests, {pct:.1f}%)")

    if not model_breakdown:
        lines.append("")
        lines.append("No usage data found for the last 24 hours.")

    return "\n".join(lines)


async def send_user_metrics_report(bot: Any, alias: str, channel: Any) -> None:
    """Fetch user metrics and send the report to a channel."""
    proxy_url = settings.litellm_proxy_url

    if not proxy_url or not settings.litellm_admin_key:
        logger.warning("Metrics or admin key not configured, skipping")
        return

    logger.info("Fetching LiteLLM token metrics for user '%s'...", alias)

    try:
        async with aiohttp.ClientSession() as session:
            user_metrics = await fetch_token_metrics(session, alias=alias)
            model_breakdown = await fetch_user_model_breakdown(session, alias)
    except Exception as e:
        logger.error("Failed to fetch user metrics: %s", type(e).__name__, exc_info=True)
        await channel.send("Error fetching your metrics. Please try again later.")
        return

    if not user_metrics:
        await channel.send(f"No usage data found for `{alias}` in the last 24 hours.")
        return

    entry = user_metrics[0]
    report = format_user_metrics_report(alias, entry["total_tokens"], entry["requests"], model_breakdown)

    try:
        await channel.send(report)
        logger.info("User metrics report sent for '%s'", alias)
    except Exception as e:
        logger.error("Failed to send user metrics report: %s", type(e).__name__, exc_info=True)


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
        channel_id = int(status_channel_id)
        channel = bot.get_channel(channel_id)
        if channel is None:
            logger.info("Channel %s not in cache, fetching from API...", status_channel_id)
            channel = await bot.fetch_channel(channel_id)
        if channel is None:
            logger.error("Could not resolve channel %s", status_channel_id)
            return
        await channel.send(report)
        logger.info("Metrics report sent to channel %s", status_channel_id)
    except Exception as e:
        logger.error("Failed to send metrics report: %s", type(e).__name__, exc_info=True)
