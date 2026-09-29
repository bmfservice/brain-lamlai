"""Braind API - learning routes."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from api.deps import get_learn
from api.errors import safe_error_payload
from api.schemas import DeepLearnRequest
from core.errors import NotFoundError
from core.logging import get_logger
from services.learn.live import LearnEventHub
from services.learn.service import LearningService

logger = get_logger(__name__)
router = APIRouter(tags=["learn"])


class LearnTopicRequest(BaseModel):
    """Request to auto-learn a topic (web search → ingest)."""

    topic: str = Field(..., min_length=2, max_length=200)
    keywords: list[str] = Field(default_factory=list)
    max_items: int = Field(12, ge=1, le=30)
    depth: int = Field(2, ge=1, le=3)


@router.post("/api/learn/topic")
async def learn_topic(
    req: LearnTopicRequest,
    learn: Annotated[LearningService, Depends(get_learn)],
) -> dict:
    """Auto-learn a topic through web search and ingestion."""
    result = await learn.learn_topic(req.topic, req.keywords or None, max_items=req.max_items)
    return {
        "topic": result.topic,
        "searched": result.searched,
        "fetched": result.fetched,
        "ingested": result.ingested,
        "duplicates": result.duplicates,
        "errors": result.errors,
    }


@router.post("/api/learn/spider")
async def spider_crawl(
    req: LearnTopicRequest,
    learn: Annotated[LearningService, Depends(get_learn)],
) -> dict:
    """Spider crawl a topic with bounded expansion."""
    result = await learn.run_deep_learn(
        req.topic,
        req.keywords or None,
        depth=req.depth,
        max_items=req.max_items,
    )
    return {
        key: result.get(key)
        for key in (
            "run_id",
            "pages",
            "ingested",
            "duplicates",
            "errors",
        )
    }


class SiteCrawlRequest(BaseModel):
    """Request to crawl one website and learn every readable page."""

    url: str = Field(..., min_length=8, max_length=2000)
    topic: str = Field("", max_length=200)
    keywords: str = Field("", max_length=500)
    depth: int = Field(10, ge=1, le=10)
    max_pages: int = Field(300, ge=1, le=300)


def _sse_stream(
    coro_factory, hub: LearnEventHub
) -> EventSourceResponse:
    """Run a learning coroutine and forward its events as SSE.

    A disconnected client cancels the task; a task that dies before emitting a
    terminal event still yields one, so the browser never hangs on an open
    stream.
    """
    task = asyncio.create_task(coro_factory())

    async def event_source() -> AsyncIterator[dict[str, str]]:
        task_error: Exception | None = None
        try:
            async for event in hub.stream():
                yield event
        finally:
            if not task.done():
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                task_error = exc
                logger.error("api.learn_stream_task_failed", error=str(exc), exc_info=True)
            if task_error is not None and not hub.terminal_event_emitted:
                yield {
                    "event": "error",
                    "data": json.dumps(
                        safe_error_payload(task_error)["error"],
                        ensure_ascii=False,
                    ),
                }

    return EventSourceResponse(event_source())


@router.post("/api/learn/site")
async def learn_site(
    req: SiteCrawlRequest,
    learn: Annotated[LearningService, Depends(get_learn)],
) -> EventSourceResponse:
    """Crawl one website from a start URL, learning every page it links to.

    Stays on the starting site, follows links breadth-first, and ingests each
    page that yields enough text. ``topic`` is optional: supply it to skip
    off-topic pages, omit it to learn the whole site.
    """
    keywords = [k.strip() for k in req.keywords.split(",") if k.strip()]
    hub = LearnEventHub()
    return _sse_stream(
        lambda: learn.crawl_site(
            req.url.strip(),
            topic=req.topic.strip(),
            keywords=keywords,
            depth=req.depth,
            max_pages=req.max_pages,
            hub=hub,
        ),
        hub,
    )


@router.post("/api/learn/deep")
async def learn_deep(
    req: DeepLearnRequest,
    learn: Annotated[LearningService, Depends(get_learn)],
) -> EventSourceResponse:
    """Run the deep BFS learning crawl with live SSE progress."""
    keywords = [str(keyword).strip() for keyword in req.keywords.split(",") if str(keyword).strip()]
    hub = LearnEventHub()
    return _sse_stream(
        lambda: learn.run_deep_learn(
            req.topic,
            keywords,
            depth=req.depth,
            max_items=req.max_items,
            hub=hub,
        ),
        hub,
    )


@router.get("/api/learn/runs")
async def list_runs(learn: Annotated[LearningService, Depends(get_learn)], limit: int = 30) -> dict:
    """List recent learning runs for history cards."""
    runs = await learn.list_runs(limit=limit)
    return {"items": runs, "count": len(runs)}


@router.get("/api/learn/runs/{run_id}")
async def get_run(run_id: str, learn: Annotated[LearningService, Depends(get_learn)]) -> dict:
    """Return full detail for one learning run."""
    detail = await learn.get_run(run_id)
    if detail is None:
        raise NotFoundError("Learning run was not found")
    return detail


class DedupRequest(BaseModel):
    """Request to compare content with stored documents."""

    content: str = Field(..., min_length=10)
    threshold: float = Field(0.92, ge=0.5, le=1.0)


@router.post("/api/learn/dedup_check")
async def dedup_check(
    req: DedupRequest,
    learn: Annotated[LearningService, Depends(get_learn)],
) -> dict:
    """Check whether content similar to the text already exists."""
    return await learn.dedup_check(req.content, threshold=req.threshold)
