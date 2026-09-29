"""RSS/news fetching helpers for the scheduler.

Fetches and parses RSS/Atom feeds (via feedparser), optionally extracts the
full article body (via BeautifulSoup) so the brain learns real content, not
just the headline.
"""

from __future__ import annotations

import html as html_lib
import re
from dataclasses import dataclass
from typing import Any

import feedparser

from core.logging import get_logger
from core.net import fetch_text_bounded
from core.readability import extract_article

logger = get_logger(__name__)

# Some "RSS" feeds pretend to be FeedsBurner etc. — map to a real format
FEED_ALIASES: dict[str, str] = {}


@dataclass
class FeedItem:
    """A normalized news item."""

    title: str
    url: str
    content: str
    summary: str
    published: str | None
    categories: list[str]


def _clean_html(raw: str) -> str:
    """Strip HTML tags to clean text."""
    if not raw:
        return ""
    text = re.sub(r"<script.*?</script>", " ", raw, flags=re.S | re.I)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_lib.unescape(text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


def _published(entry: dict[str, Any]) -> str | None:
    for key in ("published", "updated", "created"):
        if entry.get(key):
            val = entry[key]
            if hasattr(val, "isoformat"):
                try:
                    return val.isoformat()
                except Exception:
                    return str(val)
            return str(val)
    return None


def _extract_summary(entry: dict[str, Any]) -> str:
    for key in ("summary", "description", "content"):
        if entry.get(key):
            val = entry[key]
            if isinstance(val, list):
                val = val[0].get("value", "") if val else ""
            return _clean_html(str(val))
    return ""


async def fetch_feed(feed_url: str, limit: int = 10) -> list[FeedItem]:
    """Fetch and parse an RSS/Atom feed, returning up to `limit` items.

    Raises ``ValueError`` for non-public targets (SSRF guard).
    """
    raw, _ = await fetch_text_bounded(feed_url)

    feed: dict[str, Any] = feedparser.parse(raw)
    if feed.get("bozo") and not feed.get("entries"):
        raise RuntimeError(f"Feed parse error: {feed.get('bozo_exception')}")

    items: list[FeedItem] = []
    for entry in feed.get("entries", [])[:limit]:
        title = _clean_html(str(entry.get("title", ""))).strip()
        link = str(entry.get("link", "")).strip()
        if not title or not link:
            continue
        cats = [c.get("term", "") if isinstance(c, dict) else str(c) for c in entry.get("tags", [])]
        cats = [c for c in cats if c]
        content = _extract_summary(entry)
        items.append(
            FeedItem(
                title=title,
                url=link,
                content=content or title,
                summary=content,
                published=_published(entry),
                categories=cats[:5],
            )
        )
    return items


async def scrape_url(url: str, limit_chars: int = 8000) -> str:
    """Fetch a URL and extract readable main text (SSRF-safe, bounded).

    Uses the readability-grade extractor (trafilatura with bs4 fallback).
    Raises ``ValueError`` for non-public targets and ``RuntimeError`` when the
    response is non-textual or exceeds the outbound size cap.
    """
    html, _ = await fetch_text_bounded(url)
    text = re.sub(r"\n{3,}", "\n\n", extract_article(html, url=url, max_chars=limit_chars))
    return text
