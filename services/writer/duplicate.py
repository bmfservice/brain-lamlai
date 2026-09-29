"""External global web similarity checks for generated Writer content.

The checker asks the local model to summarize a completed article and propose
short search queries. It then uses the shared SSRF-safe web tool to fetch public
pages and records lexical and semantic scores separately. The audit is intended
for editorial review, not as a legal plagiarism verdict.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from urllib.parse import urlparse

from core.config import settings
from core.logging import get_logger
from core.vllm import VLLMClient
from services.tools.web import WebSearchTool
from services.writer.similarity import (
    combined_similarity,
    cosine_similarity,
    longest_shared_fragment,
    shingle_containment,
    tokenize,
)

logger = get_logger(__name__)

MAX_FETCH = 5
_MIN_CONTENT_CHARS = 60
_MAX_QUERY_CHARS = 180
_MAX_PAGE_TEXT_CHARS = 30_000


@dataclass
class WebDuplicateMatch:
    """One external page scored against generated content."""

    url: str
    canonical_url: str
    title: str
    similarity_percent: float
    semantic_percent: float | None
    match_type: str
    fetched: bool
    queries: list[str] = field(default_factory=list)
    matched_excerpt: str = ""
    source_excerpt: str = ""
    snippet: str = ""
    page_text: str = ""
    seo: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return bounded JSON-safe audit data."""
        return {
            "url": self.url,
            "canonical_url": self.canonical_url,
            "title": self.title[:300],
            "similarity_percent": round(self.similarity_percent, 1),
            "semantic_percent": self.semantic_percent,
            "match_type": self.match_type,
            "fetched": self.fetched,
            "queries": self.queries,
            "matched_excerpt": self.matched_excerpt[:300],
            "source_excerpt": self.source_excerpt[:300],
            "snippet": self.snippet[:300],
            "page_text": self.page_text[:_MAX_PAGE_TEXT_CHARS],
            "seo": self.seo or {},
        }


def _canonical(url: str) -> str:
    """Normalize an external URL for deduplication and page history."""
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        path = parsed.path.rstrip("/") or "/"
        query = f"?{parsed.query}" if parsed.query else ""
        return f"{parsed.scheme.lower()}://{host}{path}{query}"
    except ValueError:
        return url.strip().lower()


def _page_text_for_score(page: dict) -> str:
    """Use fetched article text where available, otherwise search metadata."""
    body = str(page.get("text") or "").strip()
    if len(body) >= _MIN_CONTENT_CHARS:
        return body
    return f"{page.get('title') or ''}\n\n{page.get('snippet') or ''}".strip()


class ExternalDuplicateChecker:
    """Perform model-planned global web similarity analysis."""

    def __init__(
        self,
        web_tool: WebSearchTool,
        vllm: VLLMClient | None = None,
        threshold: float | None = None,
    ) -> None:
        self.web_tool = web_tool
        self.vllm = vllm
        self.threshold = (
            threshold if threshold is not None else settings.writer_web_similarity_threshold
        )

    async def check(self, title: str, content: str) -> dict:
        """Create a structured, full-page global similarity audit."""
        audit: dict = {
            "status": "completed",
            "checked_at": datetime.now(UTC).isoformat(),
            "article_summary": "",
            "queries": [],
            "query_fallback": False,
            "candidates_checked": 0,
            "highest_similarity_percent": 0.0,
            "highest_semantic_percent": None,
            "threshold_percent": round(self.threshold * 100, 1),
            "pages": [],
            "matches": [],
            "errors": [],
        }
        if not settings.writer_web_check_enabled or not content.strip():
            audit["status"] = "skipped"
            return audit

        article_plan = await self._build_article_plan(title, content)
        audit["article_summary"] = article_plan["summary"]
        audit["queries"] = article_plan["queries"]
        audit["query_fallback"] = article_plan["fallback"]

        discovered: dict[str, dict] = {}
        for query in audit["queries"]:
            try:
                results = await self.web_tool.search(query, settings.writer_web_max_results)
            except Exception:
                logger.warning("writer.webcheck.search_failed", query=query, exc_info=True)
                audit["errors"].append("search_failed")
                continue
            for result in results:
                url = str(result.get("url") or "").strip()
                if not url:
                    continue
                key = _canonical(url)
                item = discovered.setdefault(
                    key,
                    {
                        "title": str(result.get("title") or ""),
                        "url": url,
                        "snippet": str(result.get("snippet") or ""),
                        "queries": [],
                    },
                )
                if query not in item["queries"]:
                    item["queries"].append(query)

        candidates = list(discovered.values())[:MAX_FETCH]
        fetched = await self._fetch_candidates(candidates)
        if any(page.get("fetch_error") for page in fetched):
            audit["errors"].append("page_fetch_failed")
        content_embedding = await self._embed_content(content, audit)
        pages: list[WebDuplicateMatch] = []
        for candidate, page in zip(candidates, fetched, strict=True):
            scored = await self._score(content, candidate, page, content_embedding, audit)
            if scored is not None:
                pages.append(scored)

        pages.sort(key=lambda item: item.similarity_percent, reverse=True)
        audit["candidates_checked"] = len(pages)
        audit["pages"] = [page.to_dict() for page in pages]
        audit["matches"] = [
            page.to_dict() for page in pages if page.similarity_percent >= self.threshold * 100
        ]
        if pages:
            audit["highest_similarity_percent"] = round(pages[0].similarity_percent, 1)
            semantic_scores = [
                page.semantic_percent for page in pages if page.semantic_percent is not None
            ]
            if semantic_scores:
                audit["highest_semantic_percent"] = round(max(semantic_scores), 1)
        if not candidates:
            audit["status"] = "unavailable" if audit["errors"] else "completed"
        elif audit["errors"]:
            audit["status"] = "partial"
        logger.info(
            "writer.webcheck.done",
            queries=len(audit["queries"]),
            checked=len(pages),
            matches=len(audit["matches"]),
            errors=len(audit["errors"]),
        )
        return audit

    async def _build_article_plan(self, title: str, content: str) -> dict:
        """Ask the local model for a summary and two or three global queries."""
        fallback_query = " ".join(title.split())[:_MAX_QUERY_CHARS]
        fallback = {
            "summary": "",
            "queries": [fallback_query] if fallback_query else [],
            "fallback": True,
        }
        if self.vllm is None or not fallback_query:
            return fallback
        prompt = (
            "Generated article title:\n"
            f"{title[:500]}\n\n"
            "Generated article:\n"
            f"{content[:6000]}\n\n"
            'Return JSON only: {"summary": "...", "queries": ["..."]}. '
            "Summarize the article in 2-4 sentences and generate 2 or 3 concise "
            "web search queries that find globally relevant public sources. Do not "
            "apply country, city, language, or Vietnam-specific restrictions."
        )
        try:
            response = await self.vllm.chat_json(
                [{"role": "user", "content": prompt}],
                system=(
                    "Return valid JSON only. Search queries must be factual and globally scoped."
                ),
                temperature=0.1,
                max_tokens=500,
            )
        except Exception:
            logger.warning("writer.webcheck.plan_failed", exc_info=True)
            return fallback
        if not isinstance(response, dict):
            return fallback
        summary = str(response.get("summary") or "").strip()[:1200]
        queries: list[str] = []
        raw_queries = response.get("queries")
        if isinstance(raw_queries, list):
            seen: set[str] = set()
            for raw_query in raw_queries:
                query = " ".join(str(raw_query).split())[:_MAX_QUERY_CHARS]
                key = query.casefold()
                if query and key not in seen:
                    seen.add(key)
                    queries.append(query)
                if len(queries) == 3:
                    break
        if len(queries) < 2:
            return fallback | {"summary": summary}
        return {"summary": summary, "queries": queries, "fallback": False}

    async def _fetch_candidates(self, candidates: list[dict]) -> list[dict]:
        """Fetch a bounded number of pages through the shared SSRF-safe tool."""
        semaphore = asyncio.Semaphore(2)

        async def fetch(candidate: dict) -> dict:
            async with semaphore:
                try:
                    page = await self.web_tool.fetch_page_text(candidate["url"])
                except Exception:
                    logger.info(
                        "writer.webcheck.fetch_failed", url=candidate["url"][:80], exc_info=True
                    )
                    return {"fetch_error": True}
                return page if isinstance(page, dict) else {"fetch_error": True}

        return await asyncio.gather(*(fetch(candidate) for candidate in candidates))

    async def _embed_content(self, content: str, audit: dict) -> list[float] | None:
        """Embed generated content once for the separate semantic metric."""
        if self.vllm is None:
            return None
        try:
            return await self.vllm.embed(content[:2000])
        except Exception:
            logger.info("writer.webcheck.embed_content_failed", exc_info=True)
            audit["errors"].append("semantic_embedding_failed")
            return None

    async def _score(
        self,
        content: str,
        candidate: dict,
        fetched_page: dict,
        content_embedding: list[float] | None,
        audit: dict,
    ) -> WebDuplicateMatch | None:
        """Score one discovered page while retaining fetched page provenance."""
        fetched = not fetched_page.get("fetch_error")
        page = {**candidate, **fetched_page} if fetched else candidate
        candidate_text = _page_text_for_score(page)
        content_tokens = tokenize(content)
        candidate_tokens = tokenize(candidate_text)
        if not content_tokens or not candidate_tokens:
            return None
        similarity_percent = combined_similarity(content_tokens, candidate_tokens) * 100
        verbatim = shingle_containment(content_tokens, candidate_tokens, 5) >= 0.5
        semantic_percent: float | None = None
        if content_embedding is not None and fetched:
            try:
                page_embedding = await self.vllm.embed(candidate_text[:2000])
                semantic_percent = round(
                    cosine_similarity(content_embedding, page_embedding) * 100, 1
                )
            except Exception:
                logger.info(
                    "writer.webcheck.embed_page_failed", url=candidate["url"][:80], exc_info=True
                )
                if "semantic_embedding_failed" not in audit["errors"]:
                    audit["errors"].append("semantic_embedding_failed")
        if verbatim:
            match_type = "verbatim"
        elif similarity_percent >= self.threshold * 100:
            match_type = "similar"
        else:
            match_type = "none"
        return WebDuplicateMatch(
            url=str(page.get("url") or candidate["url"]),
            canonical_url=_canonical(str(page.get("url") or candidate["url"])),
            title=str(page.get("title") or candidate.get("title") or "")[:300],
            similarity_percent=round(similarity_percent, 1),
            semantic_percent=semantic_percent,
            match_type=match_type,
            fetched=fetched,
            queries=list(candidate.get("queries") or []),
            matched_excerpt=longest_shared_fragment(content_tokens, candidate_tokens),
            source_excerpt=candidate_text[:300],
            snippet=str(candidate.get("snippet") or "")[:300],
            page_text=str(page.get("text") or "")[:_MAX_PAGE_TEXT_CHARS] if fetched else "",
            seo=page.get("seo") if isinstance(page.get("seo"), dict) else {},
        )
