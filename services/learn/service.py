"""Braind Learning Service — bounded, SEO-aware BFS web crawler."""

from __future__ import annotations

import asyncio
import datetime
import json
import re
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse, urlunparse

from sqlalchemy import text as sqltext

from core.config import settings
from core.logging import get_logger
from core.net import is_safe_url
from core.textanalysis import contains_chinese
from services.ingest.service import IngestService
from services.learn.live import LearnEventHub
from services.rag.pipeline import UnifiedRetrievalService
from services.tools.web import WebSearchTool

logger = get_logger(__name__)
EmitFn = Callable[[str, dict[str, Any]], Awaitable[None]]


@dataclass
class LearnTopic:
    """A learning topic definition."""

    name: str
    keywords: list[str]
    max_items: int = 20
    sources: list[str] = field(default_factory=list)


@dataclass
class LearnResult:
    """Result of the legacy topic learning endpoint."""

    topic: str
    searched: int = 0
    fetched: int = 0
    ingested: int = 0
    duplicates: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class PageOutcome:
    """Result emitted after one URL is processed."""

    url: str
    depth: int
    status: str
    title: str = ""
    category: str = ""
    relevance: float = 0.0
    ingested: int = 0
    links: list[str] = field(default_factory=list)
    error: str | None = None


class LearningService:
    """Autonomous learning engine with safe bounded spider crawling."""

    def __init__(
        self,
        session_factory: Any,
        vllm: Any,
        retrieval: UnifiedRetrievalService,
        ingest: IngestService,
        web_tool: WebSearchTool | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.vllm = vllm
        self.retrieval = retrieval
        self.ingest = ingest
        self.web_tool = web_tool or WebSearchTool(ingest)

    async def generate_queries(self, topic: str, keywords: list[str] | None = None) -> list[str]:
        """Generate focused research queries with a deterministic fallback.

        Bing RSS (the only available search backend) works best with short,
        topic-anchored queries that avoid generic or ambiguous words.
        """
        base = topic.strip()
        if not base:
            return []
        kw = [k.strip() for k in (keywords or []) if k.strip()]
        keyword_text = ", ".join(kw)

        # --- 1) Deterministic base set — always works if the local model is down.
        ordered: list[str] = []
        # Best-practices query is the most reliable for Bing.
        ordered.append(f"{base} best practices")
        if kw:
            ordered.append(f"{base} {' '.join(kw)}")
        # Additional narrow suffixes that Bing handles well.
        for suffix in ["examples", "tutorial", "implementation", "guide", "documentation"]:
            q = f"{base} {suffix}"
            if q not in ordered:
                ordered.append(q)
        base_queries = ordered[:6]

        # --- 2) LLM refinement (optional, constrained to the topic).
        llm_queries: list[str] = []
        try:
            llm_prompt = (
                f"Topic: {topic}\nKeywords: {keyword_text}\n"
                'Return JSON only: {"queries":["..."]}. '
                "Generate 3 to 5 concise web-search queries. Rules:\n"
                "- Every query MUST start with the exact topic name.\n"
                "- End with ONE narrow word: 'best practices', 'examples', "
                "'tutorial', 'documentation', 'guide', 'vs', 'tools'.\n"
                "- Avoid: common, pitfalls, solutions, tips, optimizing, "
                "using, for, getting started, what is, how to.\n"
                f"- Target: {keyword_text or topic}"
            )
            result = await self.vllm.chat_json(
                [{"role": "user", "content": llm_prompt}],
                system="Every query must start with the exact topic name and end "
                "with a narrow non-generic word. Never use 'common', 'pitfalls', "
                "'solutions', 'tips', 'using', 'for'.",
                temperature=0.1,
                max_tokens=500,
            )
            raw = result.get("queries", []) if isinstance(result, dict) else []
            llm_queries = [
                str(q).strip() for q in raw if str(q).strip() and topic.lower() in q.lower()
            ]
        except Exception as exc:
            logger.warning("learn.query_generation_failed", error=str(exc))

        # --- 3) Merge: deterministic anchors first, LLM refinement after.
        merged: list[str] = []
        for q in base_queries + llm_queries:
            if q not in merged:
                merged.append(q)
        merged = merged[:6]
        logger.debug("learn.generated_queries", queries=merged)
        return merged

    @staticmethod
    def normalize_url(url: str) -> str:
        """Canonicalize a URL for cross-run visit history."""
        parsed = urlparse(url.strip())
        return urlunparse(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                parsed.path.rstrip("/") or "/",
                parsed.params,
                parsed.query,
                "",
            )
        )

    @staticmethod
    def same_site(url: str, root_host: str) -> bool:
        """Whether ``url`` belongs to the same site as ``root_host``.

        A leading ``www.`` is ignored, and a subdomain of the root counts as the
        same site (``docs.example.com`` for a crawl rooted at ``example.com``),
        so a site split across subdomains is still crawled as one.
        """
        try:
            host = (urlparse(url).hostname or "").lower()
        except ValueError:
            return False
        if not host:
            return False
        host = host.removeprefix("www.")
        root = root_host.lower().removeprefix("www.")
        return host == root or host.endswith(f".{root}")

    async def get_seen_urls(self, urls: list[str]) -> set[str]:
        """Return normalized URLs already recorded by any learning run."""
        normalized = list(dict.fromkeys(self.normalize_url(url) for url in urls if url))
        if not normalized:
            return set()
        try:
            async with self.session_factory() as session:
                rows = (
                    (
                        await session.execute(
                            sqltext("SELECT url_norm FROM learn_urls WHERE url_norm = ANY(:urls)"),
                            {"urls": normalized},
                        )
                    )
                    .scalars()
                    .all()
                )
            return set(str(row) for row in rows)
        except Exception as exc:
            logger.warning("learn.history_lookup_failed", error=str(exc))
            return set()

    async def _create_run(self, topic: str, params: dict[str, Any]) -> uuid.UUID:
        """Create and return an audit run ID."""
        async with self.session_factory() as session:
            run_id = (
                await session.execute(
                    sqltext(
                        "INSERT INTO learn_runs(topic, status, params) "
                        "VALUES (:topic, 'running', CAST(:params AS jsonb)) RETURNING id"
                    ),
                    {"topic": topic, "params": json.dumps(params)},
                )
            ).scalar_one()
            await session.commit()
            return uuid.UUID(str(run_id))

    async def record_learn_url(
        self,
        run_id: uuid.UUID,
        url: str,
        depth: int,
        status: str,
        *,
        title: str = "",
        source_id: str | None = None,
        links_out: list[str] | None = None,
    ) -> None:
        """Insert or update one URL audit record."""
        normalized = self.normalize_url(url)
        async with self.session_factory() as session:
            await session.execute(
                sqltext(
                    "INSERT INTO learn_urls(run_id, url, url_norm, depth, status, title, "
                    "source_id, links_out, fetched_at) VALUES "
                    "(:run_id, :url, :url_norm, :depth, :status, :title, :source_id, "
                    "CAST(:links AS jsonb), CASE WHEN :fetched THEN now() ELSE NULL END) "
                    "ON CONFLICT (run_id, url_norm) DO UPDATE SET status=EXCLUDED.status, "
                    "title=EXCLUDED.title, source_id=EXCLUDED.source_id, "
                    "links_out=EXCLUDED.links_out, fetched_at=EXCLUDED.fetched_at, updated_at=now()"
                ),
                {
                    "run_id": run_id,
                    "url": url,
                    "url_norm": normalized,
                    "depth": depth,
                    "status": status,
                    "title": title[:500],
                    "source_id": source_id,
                    "links": json.dumps(links_out or []),
                    "fetched": status in {"success", "duplicate", "irrelevant", "error"},
                },
            )
            await session.commit()

    async def _finish_run(
        self, run_id: uuid.UUID, status: str, summary: dict[str, Any], log: list[dict[str, Any]]
    ) -> None:
        """Persist final run status, summary, and bounded logs."""
        async with self.session_factory() as session:
            await session.execute(
                sqltext(
                    "UPDATE learn_runs SET status=:status, summary=CAST(:summary AS jsonb), "
                    "log=CAST(:log AS jsonb), updated_at=now() WHERE id=:run_id"
                ),
                {
                    "status": status,
                    "summary": json.dumps(summary, ensure_ascii=False, default=str),
                    "log": json.dumps(
                        log[-settings.learn_max_log :], ensure_ascii=False, default=str
                    ),
                    "run_id": run_id,
                },
            )
            await session.commit()

    async def categorize_page(
        self, topic: str, title: str, content: str, seo: dict[str, Any]
    ) -> dict[str, Any]:
        """Classify a page and estimate relevance to the requested topic."""
        excerpt = content[:3500]
        try:
            result = await self.vllm.chat_json(
                [
                    {
                        "role": "user",
                        "content": (
                            f"Topic: {topic}\nTitle: {title}\nSEO: {seo}\nContent:\n{excerpt}\n"
                            "Return JSON only with category, subcategory, relevance (0..1), reason."
                        ),
                    }
                ],
                system="Classify web pages for a focused research crawler.",
                temperature=0.1,
                max_tokens=350,
            )
            relevance = max(0.0, min(1.0, float(result.get("relevance", 0.0))))
            return {
                "category": str(result.get("category") or "other")[:100],
                "subcategory": str(result.get("subcategory") or "")[:100],
                "relevance": relevance,
                "reason": str(result.get("reason") or "")[:500],
            }
        except Exception as exc:
            topic_words = {w.lower() for w in re.findall(r"\w+", topic) if len(w) > 2}
            haystack = f"{title} {content[:7000]}".lower()
            hits = sum(1 for word in topic_words if word in haystack)
            relevance = min(1.0, hits / max(len(topic_words), 1))
            logger.debug("learn.categorize_fallback", error=str(exc))
            return {
                "category": "related" if relevance >= 0.4 else "other",
                "subcategory": "",
                "relevance": relevance,
                "reason": "heuristic",
            }

    async def _process_page(
        self,
        run_id: uuid.UUID,
        topic: str,
        keywords: list[str],
        url: str,
        depth: int,
        emit: EmitFn,
        *,
        source_type: str = "web_learn",
        min_relevance: float = 0.35,
        link_limit: int | None = None,
        min_chars: int = 120,
        discovered_from: str | None = None,
    ) -> PageOutcome:
        """Fetch, classify, ingest, and discover links for one page.

        ``min_relevance`` of 0 skips the LLM relevance gate entirely, which is
        what a whole-site crawl wants when no topic was given.
        """
        link_limit = link_limit or settings.learn_links_per_page
        await emit("stage", {"stage": "fetch", "label": "TẢI TRANG", "depth": depth})
        try:
            page = await self.web_tool.fetch_page_text(
                url,
                include_links=True,
                link_limit=link_limit,
                link_min_len=settings.learn_min_link_len,
            )
            title = str(page.get("title") or url)
            text = str(page.get("text") or "")
            seo = page.get("seo") or {}
            links = [str(link) for link in (page.get("links") or [])[:link_limit]]
            if len(text) < min_chars:
                # Record the links: a short hub page is often the only route to
                # the content the caller actually wants.
                await self.record_learn_url(
                    run_id, url, depth, "short", title=title, links_out=links
                )
                return PageOutcome(url, depth, "short", title=title, links=links)
            # Language filter: absolutely prohibit Chinese ideographs.
            # Early-out before the LLM categorize call to avoid wasted work.
            if contains_chinese(text):
                await self.record_learn_url(run_id, url, depth, "chinese", title=title)
                return PageOutcome(url, depth, "chinese", title=title)
            category = {"category": "", "subcategory": "", "relevance": 1.0, "reason": ""}
            if min_relevance > 0:
                await emit("stage", {"stage": "analyze", "label": "PHÂN TÍCH SEO", "url": url})
                category = await self.categorize_page(topic, title, text, seo)
                if category["relevance"] < min_relevance:
                    await self.record_learn_url(
                        run_id, url, depth, "irrelevant", title=title, links_out=links
                    )
                    return PageOutcome(
                        url,
                        depth,
                        "irrelevant",
                        title=title,
                        category=category["category"],
                        relevance=category["relevance"],
                        links=links,
                    )
            await emit("stage", {"stage": "ingest", "label": "NẠP TRI THỨC", "url": url})
            result = await self.ingest.ingest_text(
                text,
                title=title,
                source_type=source_type,
                url=str(page.get("url") or url),
                tags=[tag for tag in (topic, *keywords) if tag],
                meta={
                    "learn_run_id": str(run_id),
                    "crawl_depth": depth,
                    "seo": seo,
                    "category": category["category"],
                    "subcategory": category["subcategory"],
                    "relevance": category["relevance"],
                    "discovered_from": discovered_from or url,
                },
            )
            status = "duplicate" if result.skipped or result.duplicate_of else "success"
            await self.record_learn_url(run_id, url, depth, status, title=title, links_out=links)
            return PageOutcome(
                url,
                depth,
                status,
                title,
                category["category"],
                category["relevance"],
                result.added,
                links,
            )
        except Exception as exc:
            logger.warning("learn.page_failed", url=url, error=str(exc))
            await self.record_learn_url(run_id, url, depth, "error", title=url)
            return PageOutcome(url, depth, "error", title=url, error=str(exc)[:500])

    async def run_deep_learn(
        self,
        topic: str,
        keywords: list[str] | None = None,
        *,
        depth: int | None = None,
        max_items: int | None = None,
        hub: LearnEventHub | None = None,
    ) -> dict[str, Any]:
        """Run a bounded BFS crawl and stream progress to ``hub``."""
        keywords = [k.strip() for k in (keywords or []) if k.strip()]
        max_depth = min(depth or settings.learn_max_depth, settings.learn_max_depth)
        per_level = min(max_items or settings.learn_per_level, settings.learn_per_level)
        total_cap = settings.learn_max_total_pages
        run_id = await self._create_run(
            topic, {"keywords": keywords, "depth": max_depth, "max_items": per_level}
        )
        event_hub = hub or LearnEventHub()
        logs: list[dict[str, Any]] = []
        summary: dict[str, Any] = {
            "run_id": str(run_id),
            "topic": topic,
            "pages": 0,
            "ingested": 0,
            "duplicates": 0,
            "errors": 0,
            "skipped": 0,
        }

        async def emit(event: str, data: dict[str, Any]) -> None:
            if event == "log":
                logs.append(data)
            await event_hub.emit(event, data)

        try:
            await emit("stage", {"stage": "queries", "label": "TẠO TRUY VẤN"})
            queries = await self.generate_queries(topic, keywords)
            await emit("log", {"level": "ok", "text": f"Đã tạo {len(queries)} truy vấn nghiên cứu"})
            await emit("stage", {"stage": "search", "label": "TÌM KIẾM"})
            discovered: list[tuple[str, int, str]] = []
            for query in queries:
                await emit("log", {"level": "info", "text": f"Đang tìm: {query}"})
                for result in await self.web_tool.search(query, max_results=5):
                    url = str(result.get("url") or "")
                    if url:
                        discovered.append((url, 0, query))
            unique: dict[str, tuple[str, int, str]] = {}
            for url, item_depth, query in discovered:
                unique.setdefault(self.normalize_url(url), (url, item_depth, query))
            # Keep ALL unique search results (up to total_cap) so every query's
            # top-5 contributes to the initial pool — the per_level cap still
            # governs how many are processed concurrently per BFS tier.
            queue = list(unique.values())[:total_cap]
            current_seen = await self.get_seen_urls([item[0] for item in queue])
            await emit("stage", {"stage": "history", "label": "KIỂM TRA LỊCH SỬ"})
            queue_new: list[tuple[str, int, str]] = []
            for url, item_depth, query in queue:
                if self.normalize_url(url) in current_seen:
                    summary["skipped"] += 1
                    await self.record_learn_url(run_id, url, item_depth, "seen")
                    await emit(
                        "log", {"level": "warn", "text": f"Bỏ qua URL đã học: {url}", "url": url}
                    )
                else:
                    await self.record_learn_url(run_id, url, item_depth, "queued")
                    queue_new.append((url, item_depth, query))
            queue = queue_new
            semaphore = asyncio.Semaphore(settings.learn_concurrency)
            started = datetime.datetime.now(datetime.UTC)

            # Inner crawl — extracted so we can enforce a hard wall-clock timeout.
            async def _crawl() -> None:
                for _current_depth in range(max_depth):
                    if not queue or summary["pages"] >= total_cap:
                        break
                    batch = queue[:per_level]
                    # Mutate in place so the closure sees the same list object
                    del queue[:per_level]

                    async def bounded(item: tuple[str, int, str]) -> PageOutcome:
                        async with semaphore:
                            return await self._process_page(
                                run_id, topic, keywords, item[0], item[1], emit
                            )

                    outcomes = await asyncio.gather(
                        *(bounded(item) for item in batch), return_exceptions=False
                    )
                    next_candidates: list[tuple[str, int, str]] = []
                    for outcome in outcomes:
                        summary["pages"] += 1
                        summary["ingested"] += outcome.ingested
                        summary["duplicates"] += int(outcome.status == "duplicate")
                        summary["errors"] += int(outcome.status == "error")
                        payload = {
                            "url": outcome.url,
                            "depth": outcome.depth,
                            "status": outcome.status,
                            "title": outcome.title,
                            "category": outcome.category,
                            "relevance": outcome.relevance,
                            "ingested": outcome.ingested,
                            "links": outcome.links,
                            "error": outcome.error,
                        }
                        await emit("page", payload)
                        await emit(
                            "log",
                            {
                                "level": "ok" if outcome.status == "success" else "warn",
                                "text": f"{outcome.status}: {outcome.title[:90]}",
                                "url": outcome.url,
                            },
                        )
                        if (
                            outcome.status in {"success", "duplicate"}
                            and outcome.depth + 1 < max_depth
                        ):
                            next_candidates.extend(
                                (link, outcome.depth + 1, outcome.url) for link in outcome.links
                            )
                    if not next_candidates:
                        break
                    link_urls = [item[0] for item in next_candidates]
                    seen_links = await self.get_seen_urls(link_urls)
                    local: set[str] = set()
                    for url, item_depth, source in next_candidates:
                        normalized = self.normalize_url(url)
                        if normalized in seen_links or normalized in local:
                            continue
                        local.add(normalized)
                        await self.record_learn_url(run_id, url, item_depth, "queued")
                        queue.append((url, item_depth, source))
                    await emit(
                        "stage", {"stage": "links", "label": "CHỌN LIÊN KẾT", "count": len(local)}
                    )

            await asyncio.wait_for(
                _crawl(),
                timeout=settings.learn_hard_timeout_sec,
            )
            summary["elapsed_seconds"] = (
                datetime.datetime.now(datetime.UTC) - started
            ).total_seconds()
            await self._finish_run(run_id, "done", summary, logs)
            await emit("summary", summary)
            await emit("done", {"run_id": str(run_id), "summary": summary})
            return {**summary, "queries": queries}
        except TimeoutError:
            safe_message = "The learning crawl timed out."
            summary["error"] = safe_message
            summary["elapsed_seconds"] = (
                datetime.datetime.now(datetime.UTC) - started
            ).total_seconds()
            await self._finish_run(run_id, "error", summary, logs)
            await emit(
                "error",
                {
                    "code": "upstream_error",
                    "message": safe_message,
                    "run_id": str(run_id),
                },
            )
            return {**summary, "queries": queries, "timed_out": True}
        except Exception as exc:
            safe_message = "The learning crawl could not complete."
            logger.error("learn.deep_error", error=str(exc), exc_info=True)
            summary["error"] = safe_message
            await self._finish_run(run_id, "error", summary, logs)
            await emit(
                "error",
                {
                    "code": "internal_error",
                    "message": safe_message,
                    "run_id": str(run_id),
                },
            )
            return {**summary, "queries": queries}
        finally:
            if hub is not None:
                hub.close()

    async def crawl_site(
        self,
        start_url: str,
        *,
        topic: str = "",
        keywords: list[str] | None = None,
        depth: int | None = None,
        max_pages: int | None = None,
        hub: LearnEventHub | None = None,
    ) -> dict[str, Any]:
        """Crawl one website from ``start_url`` and ingest every readable page.

        Unlike ``run_deep_learn`` this does not search the web: it follows the
        links found on each page, staying on the starting site. Every page that
        yields enough text is ingested. ``topic`` is optional — when given, each
        page is scored for relevance and off-topic pages are skipped; when
        empty, the only filters are the shared ingest ones (too short, Han
        ideographs, duplicate).

        Depth 10 on a real site is effectively unbounded, so ``max_pages`` and
        ``crawl_hard_timeout_sec`` are the limits that actually stop the run.
        """
        keywords = [k.strip() for k in (keywords or []) if k.strip()]
        max_depth = min(depth or settings.crawl_max_depth, settings.crawl_max_depth)
        page_cap = min(max_pages or settings.crawl_max_pages, settings.crawl_max_pages)
        root_host = (urlparse(start_url).hostname or "").lower()

        run_id = await self._create_run(
            topic or start_url,
            {
                "mode": "site_crawl",
                "start_url": start_url,
                "keywords": keywords,
                "depth": max_depth,
                "max_pages": page_cap,
            },
        )
        event_hub = hub or LearnEventHub()
        logs: list[dict[str, Any]] = []
        summary: dict[str, Any] = {
            "run_id": str(run_id),
            "mode": "site_crawl",
            "start_url": start_url,
            "site": root_host,
            "topic": topic,
            "pages": 0,
            "ingested": 0,
            "duplicates": 0,
            "errors": 0,
            "skipped": 0,
            "max_depth_reached": 0,
        }

        async def emit(event: str, data: dict[str, Any]) -> None:
            if event == "log":
                logs.append(data)
            await event_hub.emit(event, data)

        started = datetime.datetime.now(datetime.UTC)
        # Visited within this run. Cross-run history is deliberately NOT used:
        # crawling a site means reading that site now, not skipping whatever a
        # previous run happened to touch. Ingest-level dedup still prevents
        # duplicate rows.
        visited: set[str] = {self.normalize_url(start_url)}
        queue: list[tuple[str, int, str]] = [(start_url, 0, start_url)]
        semaphore = asyncio.Semaphore(settings.crawl_concurrency)

        async def _crawl() -> None:
            nonlocal queue
            for current_depth in range(max_depth):
                if not queue or summary["pages"] >= page_cap:
                    break
                budget = page_cap - summary["pages"]
                batch, queue = queue[:budget], queue[budget:]
                await emit(
                    "stage",
                    {
                        "stage": "level",
                        "label": f"MỨC {current_depth}",
                        "depth": current_depth,
                        "count": len(batch),
                    },
                )

                async def bounded(item: tuple[str, int, str]) -> PageOutcome:
                    async with semaphore:
                        return await self._process_page(
                            run_id,
                            topic,
                            keywords,
                            item[0],
                            item[1],
                            emit,
                            source_type="web_crawl",
                            min_relevance=0.35 if topic else 0.0,
                            link_limit=settings.crawl_links_per_page,
                            min_chars=settings.crawl_min_content_chars,
                            discovered_from=item[2],
                        )

                outcomes = await asyncio.gather(*(bounded(item) for item in batch))
                next_urls: list[tuple[str, int, str]] = []
                for outcome in outcomes:
                    summary["pages"] += 1
                    summary["ingested"] += outcome.ingested
                    summary["duplicates"] += int(outcome.status == "duplicate")
                    summary["errors"] += int(outcome.status == "error")
                    summary["skipped"] += int(
                        outcome.status in {"short", "chinese", "irrelevant"}
                    )
                    summary["max_depth_reached"] = max(
                        summary["max_depth_reached"], outcome.depth
                    )
                    await emit(
                        "page",
                        {
                            "url": outcome.url,
                            "depth": outcome.depth,
                            "status": outcome.status,
                            "title": outcome.title,
                            "category": outcome.category,
                            "relevance": outcome.relevance,
                            "ingested": outcome.ingested,
                            "links": outcome.links,
                            "error": outcome.error,
                        },
                    )
                    await emit(
                        "log",
                        {
                            "level": "ok" if outcome.status == "success" else "warn",
                            "text": f"[d{outcome.depth}] {outcome.status}: "
                            f"{(outcome.title or outcome.url)[:90]}",
                            "url": outcome.url,
                        },
                    )
                    # Follow links from any page that loaded, including ones
                    # skipped as too short, duplicate, or off-topic: a nav-only
                    # index page carries no prose of its own but is often the
                    # only route to the content. Chinese pages are not followed
                    # — the content policy rejects them, and their links lead
                    # deeper into the same rejected material.
                    if outcome.status in {"error", "chinese"}:
                        continue
                    for link in outcome.links:
                        normalized = self.normalize_url(link)
                        if normalized in visited or not self.same_site(link, root_host):
                            continue
                        visited.add(normalized)
                        next_urls.append((link, outcome.depth + 1, outcome.url))

                if not next_urls:
                    break
                remaining = page_cap - summary["pages"] - len(queue)
                if remaining > 0:
                    queue.extend(next_urls[:remaining])
                await emit(
                    "stage",
                    {
                        "stage": "links",
                        "label": "XẾP HÀNG LIÊN KẾT",
                        "count": len(queue),
                    },
                )

        try:
            if not is_safe_url(start_url):
                # Raised inside the try so the client receives an SSE error
                # event; raising before the stream opens would leave the browser
                # waiting on a socket that only ever carries keep-alive pings.
                raise ValueError("The start URL must be a public http(s) address.")
            await emit(
                "log",
                {"level": "info", "text": f"Bắt đầu duyệt {root_host} từ {start_url}"},
            )
            await asyncio.wait_for(_crawl(), timeout=settings.crawl_hard_timeout_sec)
            summary["elapsed_seconds"] = (
                datetime.datetime.now(datetime.UTC) - started
            ).total_seconds()
            summary["queued_unvisited"] = len(queue)
            await self._finish_run(run_id, "done", summary, logs)
            await emit("summary", summary)
            await emit("done", {"run_id": str(run_id), "summary": summary})
            return summary
        except TimeoutError:
            summary["error"] = "The site crawl timed out."
            summary["timed_out"] = True
            summary["elapsed_seconds"] = (
                datetime.datetime.now(datetime.UTC) - started
            ).total_seconds()
            # A timeout still leaves everything already ingested in place.
            await self._finish_run(run_id, "done", summary, logs)
            await emit("summary", summary)
            await emit("done", {"run_id": str(run_id), "summary": summary})
            return summary
        except ValueError as exc:
            # A rejected start URL: report the reason, since it is the caller's
            # input rather than an internal failure.
            summary["error"] = str(exc)
            await self._finish_run(run_id, "error", summary, logs)
            await emit(
                "error",
                {
                    "code": "validation_error",
                    "message": summary["error"],
                    "run_id": str(run_id),
                },
            )
            return summary
        except Exception as exc:
            logger.error("crawl.site_error", url=start_url, error=str(exc), exc_info=True)
            summary["error"] = "The site crawl could not complete."
            await self._finish_run(run_id, "error", summary, logs)
            await emit(
                "error",
                {
                    "code": "internal_error",
                    "message": summary["error"],
                    "run_id": str(run_id),
                },
            )
            return summary
        finally:
            if hub is not None:
                hub.close()

    async def learn_topic(
        self, topic: str, keywords: list[str] | None = None, max_items: int = 15
    ) -> LearnResult:
        """Legacy wrapper that now uses the deep crawler."""
        result = await self.run_deep_learn(topic, keywords, depth=1, max_items=min(max_items, 5))
        return LearnResult(
            topic=topic,
            searched=result.get("pages", 0),
            fetched=result.get("pages", 0),
            ingested=result.get("ingested", 0),
            duplicates=result.get("duplicates", 0),
            errors=[result["error"]] if result.get("error") else [],
        )

    async def spider_crawl(
        self, topic: str, max_depth: int = 2, per_level: int = 8
    ) -> dict[str, Any]:
        """Legacy wrapper for bounded deep crawling."""
        return await self.run_deep_learn(topic, depth=max_depth, max_items=min(per_level, 5))

    async def list_runs(self, limit: int = 30) -> list[dict[str, Any]]:
        """Return recent learning runs for the history sidebar."""
        async with self.session_factory() as session:
            rows = (
                (
                    await session.execute(
                        sqltext(
                            "SELECT id, topic, status, summary, created_at, updated_at "
                            "FROM learn_runs ORDER BY created_at DESC LIMIT :limit"
                        ),
                        {"limit": max(1, min(limit, 100))},
                    )
                )
                .mappings()
                .all()
            )
            return [dict(row) for row in rows]

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return one run and its URL audit records."""
        try:
            parsed_id = uuid.UUID(run_id)
        except (ValueError, AttributeError, TypeError):
            return None
        async with self.session_factory() as session:
            run = (
                (
                    await session.execute(
                        sqltext(
                            "SELECT id, topic, status, params, summary, log, created_at, "
                            "updated_at FROM learn_runs WHERE id=:id"
                        ),
                        {"id": parsed_id},
                    )
                )
                .mappings()
                .first()
            )
            if not run:
                return None
            urls = (
                (
                    await session.execute(
                        sqltext(
                            "SELECT url, url_norm, depth, status, title, links_out, fetched_at "
                            "FROM learn_urls WHERE run_id=:id ORDER BY depth, created_at"
                        ),
                        {"id": parsed_id},
                    )
                )
                .mappings()
                .all()
            )
            return {"run": dict(run), "urls": [dict(row) for row in urls]}

    async def dedup_check(self, text: str, threshold: float = 0.92) -> dict[str, Any]:
        """Check if similar content already exists (embedding cosine)."""
        try:
            vec = await self.vllm.embed(text[:4000])
            vector_literal = "[" + ",".join(str(value) for value in vec) + "]"
            async with self.session_factory() as sess:
                q = """
                    SELECT id, title, 1 - (embedding_vllm_1024 <=> CAST(:vec AS vector)) AS sim
                    FROM documents WHERE embedding_vllm_1024 IS NOT NULL
                    ORDER BY embedding_vllm_1024 <=> CAST(:vec AS vector) LIMIT 5
                """
                rows = (await sess.execute(sqltext(q), {"vec": vector_literal})).all()
            dups = [
                {"id": str(r[0]), "title": r[1], "similarity": float(r[2])}
                for r in rows
                if float(r[2]) >= threshold
            ]
            return {"duplicates": dups, "count": len(dups)}
        except Exception as exc:
            logger.warning("dedup.error", error=str(exc), exc_info=True)
            return {
                "duplicates": [],
                "count": 0,
                "error": "The duplicate check could not complete.",
            }
