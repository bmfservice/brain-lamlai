"""Braind Tool - RSS fetch + URL scrape.

Lets the agent pull live content from RSS feeds or scrape a URL and store
it in the brain. This keeps Braind continuously learning from external
sources on demand.

"""

from __future__ import annotations

from typing import Any

from core.agent.tool import BaseTool
from core.logging import get_logger
from services.ingest.service import IngestService
from services.scheduler.rss import fetch_feed, scrape_url

logger = get_logger(__name__)


class FetchRSSTool(BaseTool):
    """Fetch the latest items from an RSS/Atom feed into the brain."""

    name = "fetch_rss"
    description = (
        "Lấy các tin tức mới nhất từ một RSS feed và lưu vào bộ nhớ Braind. "
        "Dùng khi cần cập nhật tin tức / kiến thức từ nguồn cấp."
    )
    parameters = {
        "type": "object",
        "properties": {
            "feed_url": {"type": "string", "description": "URL của RSS/Atom feed"},
            "limit": {"type": "integer", "description": "Số bài tối đa (mặc định 5)"},
        },
        "required": ["feed_url"],
    }

    def __init__(self, ingest: IngestService) -> None:
        self.ingest = ingest

    async def execute(self, feed_url: str, limit: int = 5) -> Any:
        try:
            items = await fetch_feed(feed_url, limit=limit)
        except Exception as exc:
            logger.error("tool.rss_error", url=feed_url, error=str(exc))
            return {"error": f"Fetch RSS failed: {exc}", "items": []}

        ingested = []
        for item in items:
            try:
                res = await self.ingest.ingest_text(
                    item.content,
                    title=item.title,
                    source_type="rss",
                    url=item.url,
                    tags=["rss"],
                    meta={"published": item.published},
                )
                ingested.append({"title": item.title, "saved": res.added})
            except Exception as exc:
                logger.warning("tool.rss_ingest_error", title=item.title, error=str(exc))

        return {
            "feed": feed_url,
            "fetched": len(items),
            "ingested": ingested,
        }


class FetchURLTool(BaseTool):
    """Fetch + scrape a URL's readable content into the brain."""

    name = "fetch_url"
    description = (
        "Lấy và trích xuất nội dung chính từ một URL web, lưu vào bộ nhớ Braind. "
        "Dùng khi cần đọc một bài viết / trang web cụ thể."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL cần đọc"},
            "limit_chars": {"type": "integer", "description": "Số ký tự tối đa"},
        },
        "required": ["url"],
    }

    def __init__(self, ingest: IngestService) -> None:
        self.ingest = ingest

    async def execute(self, url: str, limit_chars: int = 8000) -> Any:
        try:
            content = await scrape_url(url, limit_chars=limit_chars)
        except Exception as exc:
            logger.error("tool.url_error", url=url, error=str(exc))
            return {"error": f"Fetch URL failed: {exc}"}

        res = await self.ingest.ingest_text(
            content,
            title=f"Web: {url[:80]}",
            source_type="web",
            url=url,
            tags=["web"],
        )
        return {"url": url, "chars": len(content), "saved": res.added, "preview": content[:300]}
