"""Braind Tool - External web search.

Uses DuckDuckGo HTML search (no API key needed), then optionally ingests the
top results into the knowledge base so the brain "learns" new information.

"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote_plus, unquote, urljoin, urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup

from core.agent.tool import BaseTool
from core.config import settings
from core.logging import get_logger
from core.net import fetch_text_bounded, is_safe_url
from core.readability import extract_article, extract_title
from core.retry import fetch_with_retry
from services.ingest.service import IngestService

logger = get_logger(__name__)


class WebSearchTool(BaseTool):
    """Search the web via DuckDuckGo and optionally save results to brain."""

    name = "web_search"
    description = (
        "Tìm kiếm thông tin trên web (DuckDuckGo). Dùng khi kiến thức trong bộ nhớ "
        "không đủ, cần thông tin mới nhất từ bên ngoài."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Từ khóa tìm kiếm"},
            "max_results": {
                "type": "integer",
                "description": "Số kết quả trả về (mặc định 10)",
            },
            "save_to_brain": {
                "type": "boolean",
                "description": "True nếu muốn lưu kết quả vào bộ nhớ Braind",
            },
        },
        "required": ["query"],
    }

    def __init__(self, ingest: IngestService) -> None:
        self.ingest = ingest

    async def execute(self, query: str, max_results: int = 10, save_to_brain: bool = False) -> Any:
        # Cap + floor so the agent always gets a useful research set.
        max_results = max(1, min(int(max_results or 10), 15))
        try:
            raw = await self._duckduckgo_search(query, max(max_results * 2, 15))
            results = self._diversify(raw, max_results)
        except Exception as exc:
            logger.error("tool.websearch_error", error=str(exc), exc_info=True)
            return {"error": f"Web search failed: {exc}", "results": []}

        saved = 0
        if save_to_brain and results:
            for r in results:
                try:
                    res = await self.ingest.ingest_text(
                        f"{r['title']}\n\n{r['snippet']}",
                        title=f"Web: {r['title'][:80]}",
                        source_type="web",
                        url=r["url"],
                        tags=["web", "search"],
                        meta={"search_query": query},
                    )
                    saved += res.added
                except Exception as exc:
                    logger.warning("tool.websearch_ingest_error", error=str(exc))

        return {
            "query": query,
            "results": results,
            "saved_to_brain": saved,
        }

    @staticmethod
    def _is_search_result_url(url: str) -> bool:
        """Return whether a result is a direct, public page URL, not an ad redirect."""
        if not is_safe_url(url):
            return False
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        path = parsed.path.lower()
        return not (host.endswith("duckduckgo.com") and path.startswith("/y.js"))

    @staticmethod
    def _diversify(results: list[dict], max_results: int) -> list[dict]:
        """Dedupe by canonical URL, prioritizing different hosts."""
        by_url: dict[str, dict] = {}
        for r in results:
            url = (r.get("url") or "").strip()
            if not url:
                continue
            try:
                p = urlparse(url)
                scheme = (p.scheme or "https").lower()
                netloc = (p.hostname or "").lower()
                path_part = (p.path or "/").rstrip("/") or "/"
                key = f"{scheme}://" + netloc + path_part
            except Exception:
                key = url.lower()
            by_url.setdefault(key, r)

        unique = list(by_url.values())
        # First pass: one result per host.
        seen_host: set[str] = set()
        primary: list[dict] = []
        overflow: list[dict] = []
        for r in unique:
            try:
                host = (urlparse(r.get("url") or "").hostname or "").lower()
            except Exception:
                host = ""
            if host and host in seen_host:
                overflow.append(r)
                continue
            if host:
                seen_host.add(host)
            primary.append(r)
            if len(primary) >= max_results:
                return primary[:max_results]
        # Fill remaining slots from same-domain overflow.
        for r in overflow:
            primary.append(r)
            if len(primary) >= max_results:
                break
        return primary[:max_results]

    @staticmethod
    def _extract_ddg_url(link: str) -> str:
        """Extract the real URL from a DuckDuckGo redirect link."""
        if not link:
            return ""
        # Direct URL
        if link.startswith("http") and "duckduckgo.com" not in link:
            return link
        # DDG redirect with uddg param
        m = re.search(r"uddg=([^&]+)", link)
        if m:
            return unquote(m.group(1))
        # DDG redirect with u parameter
        m = re.search(r"[?&]u=([^&]+)", link)
        if m:
            try:
                return unquote(unquote(m.group(1)))
            except Exception:
                return unquote(m.group(1))
        # Try to find a real URL in the href
        m = re.search(r"(https?%3A[^&]+)", link)
        if m:
            try:
                return unquote(unquote(m.group(1)))
            except Exception:
                return unquote(m.group(1))
        return ""

    async def _duckduckgo_search(self, query: str, max_results: int) -> list[dict]:
        """Try DuckDuckGo HTML search first; fall back to Bing RSS on CAPTCHA.

        Goes through the stealth engine (real Chrome TLS fingerprint) with
        transient-failure retry — the html endpoint CAPTCHAs plain httpx clients.
        """
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        html, _ = await fetch_with_retry(
            lambda: fetch_text_bounded(url, timeout_seconds=30.0, stealth=True)
        )

        soup = BeautifulSoup(html, "lxml")
        results_list: list[dict] = []
        for block in soup.select(".result")[: max(max_results, 20)]:
            a = block.select_one("a.result__a")
            if not a:
                continue
            title = a.get_text(" ", strip=True)
            link = a.get("href", "")
            link = self._extract_ddg_url(link)
            snippet_el = block.select_one(".result__snippet")
            snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
            if title and link:
                results_list.append({"title": title, "url": link, "snippet": snippet})

        # DuckDuckGo returned a CAPTCHA challenge — fall back to Bing RSS.
        if not results_list:
            logger.info("tool.ddg_fallback_bing", query=query)
            results_list = await self._bing_search(query, max_results)

        return results_list

    async def _bing_search(self, query: str, max_results: int) -> list[dict]:
        """Search via Bing RSS feed (no API key needed, no CAPTCHA).

        Plain engine + retry: the RSS endpoint does not TLS-fingerprint.
        """
        url = f"https://www.bing.com/search?q={quote_plus(query)}&format=rss"
        try:
            xml_text, _ = await fetch_with_retry(
                lambda: fetch_text_bounded(url, timeout_seconds=30.0, stealth=False)
            )

            soup = BeautifulSoup(xml_text, "xml")
            items = soup.find_all("item")
            results_list: list[dict] = []
            for item in items[:max_results]:
                title = item.find("title")
                link = item.find("link")
                desc = item.find("description")
                if title and link:
                    results_list.append(
                        {
                            "title": title.get_text(" ", strip=True),
                            "url": link.get_text(" ", strip=True),
                            "snippet": desc.get_text(" ", strip=True) if desc else "",
                        }
                    )
            return results_list
        except Exception as exc:
            logger.warning("tool.bing_search_error", error=str(exc))
            return []

    async def fetch_page_text(
        self,
        url: str,
        *,
        max_chars: int | None = None,
        max_bytes: int | None = None,
        timeout_seconds: float | None = None,
        include_links: bool = False,
        link_limit: int = 12,
        link_min_len: int = 30,
    ) -> dict:
        """Fetch a web page and extract clean article text (SSRF-safe).

        Returns a dict with keys ``title``, ``text``, ``url``, ``status``.
        Raises ``ValueError`` for unsafe URLs and ``httpx.HTTPError`` /
        ``RuntimeError`` for transport or size violations so callers can degrade
        gracefully. Does not execute JS, download assets, or send credentials.
        """
        max_chars = max_chars or settings.writer_web_max_page_chars
        max_bytes = max_bytes or settings.writer_web_max_response_bytes
        timeout_seconds = timeout_seconds or settings.writer_web_fetch_timeout_seconds

        html, final_url = await fetch_text_bounded(
            url, max_bytes=max_bytes, timeout_seconds=timeout_seconds
        )

        soup = BeautifulSoup(html, "lxml")

        # --- Optional: candidate same-topic links for the deep crawler ------
        links: list[str] = []
        if include_links:
            links = self._extract_links(soup, final_url, link_limit, link_min_len)

        title = extract_title(html)
        # Readability-grade extraction (trafilatura) with the bs4 fallback;
        # core.quality.assess remains the downstream gate, unchanged.
        text = re.sub(r"\n{3,}", "\n\n", extract_article(html, url=final_url, max_chars=max_chars))

        # Extract a compact SEO profile for the brain's "learn SEO" use-case.
        meta_desc = ""
        meta = soup.find("meta", attrs={"name": "description"})
        if meta:
            meta_desc = (meta.get("content") or "").strip()
        headings: list[str] = []
        for h in soup.find_all(["h1", "h2", "h3"]):
            txt = h.get_text(" ", strip=True)
            if txt and txt not in headings:
                headings.append(txt)

        return {
            "title": title[:300],
            "text": text[:max_chars],
            "url": final_url,
            "status": "ok",
            "seo": {
                "title": title[:300],
                "meta_description": meta_desc[:500],
                "headings": headings[:30],
                "heading_count": len(headings),
            },
            "links": links,
        }

    @staticmethod
    def _extract_links(
        soup: BeautifulSoup,
        page_url: str,
        limit: int,
        min_len: int,
    ) -> list[str]:
        """Extract capped, normalized, public links from an already parsed page."""
        nav_segments = {
            "login",
            "signin",
            "signup",
            "register",
            "account",
            "auth",
            "privacy",
            "terms",
            "legal",
            "about",
            "contact",
            "careers",
            "search",
            "tags",
            "tag",
            "category",
            "cart",
            "checkout",
            "wp-json",
            "feed",
            "rss",
            "xmlrpc",
            "logout",
            "password",
        }
        seen: set[str] = set()
        links: list[str] = []
        for anchor in soup.find_all("a", href=True):
            href = (anchor.get("href") or "").strip()
            if not href or href.startswith(("#", "mailto:", "tel:", "javascript:", "data:")):
                continue
            try:
                absolute = urljoin(page_url, href)
                parsed = urlparse(absolute)
            except Exception:
                continue
            if parsed.scheme not in ("http", "https") or not is_safe_url(absolute):
                continue
            segments = {segment.lower() for segment in parsed.path.split("/") if segment}
            if segments & nav_segments:
                continue
            normalized = urlunparse(
                (
                    parsed.scheme.lower(),
                    parsed.netloc.lower(),
                    parsed.path,
                    parsed.params,
                    parsed.query,
                    "",
                )
            ).rstrip("/")
            if len(normalized) < min_len or normalized in seen:
                continue
            seen.add(normalized)
            links.append(normalized)
            if len(links) >= limit:
                break
        return links

    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        """Read-only DuckDuckGo search (no brain ingestion).

        Returns a diversified list of ``{title, url, snippet}``. Safe to call
        repeatedly by the learning crawler without writing to the knowledge base.
        Only returns direct, public, non-ad page URLs.
        """
        try:
            raw = await self._duckduckgo_search(query, max(max_results * 2, 15))
            # Filter out ad redirects and non-public URLs
            raw = [r for r in raw if self._is_search_result_url(r.get("url") or "")]
            return self._diversify(raw, max_results)
        except Exception as exc:
            logger.warning("tool.websearch_readonly_error", error=str(exc))
            return []

    async def extract_candidate_links(
        self,
        page_url: str,
        *,
        limit: int = 12,
        min_len: int = 30,
    ) -> list[str]:
        """Return candidate links found in a page (SSRF-safe).

        Fetches ``page_url`` HTML (bounded, SSRF-safe), parses ``<a href>``
        anchors, resolves relatives, drops fragments and non-http/private
        targets, and filters out navigation/utility paths. The caller applies
        topic-relevance + history checks before fetching anything further.
        """
        try:
            html, final_url = await fetch_text_bounded(
                page_url,
                max_bytes=settings.writer_web_max_response_bytes,
                timeout_seconds=settings.writer_web_fetch_timeout_seconds,
            )
        except (ValueError, RuntimeError, httpx.HTTPError) as exc:
            logger.warning("tool.link_fetch_failed", url=page_url, error=str(exc))
            return []
        soup = BeautifulSoup(html, "lxml")
        return self._extract_links(soup, final_url, limit, min_len)
