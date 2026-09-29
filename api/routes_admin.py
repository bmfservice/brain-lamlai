"""Braind API - writer generation and scheduler routes."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from api.deps import get_scheduler, get_writer
from api.schemas import (
    ContentListItem,
    RefineRequest,
    RunJobRequest,
    SchedulerJob,
    WriterBatchResponse,
    WriterRequest,
    WriterResponse,
)
from core.errors import NotFoundError, ValidationError
from services.scheduler.service import BraindScheduler
from services.writer.service import WriterService, writer_options

router = APIRouter(tags=["writer", "scheduler"])

WRITER_PREFIX = "/api/writer"


@router.get(WRITER_PREFIX + "/options")
async def options() -> dict[str, Any]:
    """Return controlled option lists for the writer UI."""
    return writer_options()


@router.post(WRITER_PREFIX + "/generate", response_model=WriterBatchResponse)
async def generate(
    req: WriterRequest,
    writer_service: Annotated[WriterService, Depends(get_writer)],
) -> WriterBatchResponse:
    """Generate one or more marketing or blog content variants."""
    try:
        result = await writer_service.generate(req.to_writer())
    except ValueError as exc:
        raise ValidationError("The writer request is not valid.") from exc
    return WriterBatchResponse(**result)


@router.post(WRITER_PREFIX + "/refine", response_model=WriterResponse)
async def refine(
    req: RefineRequest,
    writer_service: Annotated[WriterService, Depends(get_writer)],
) -> WriterResponse:
    """Refine a previously generated content item."""
    try:
        result = await writer_service.refine(req.content_id, req.instruction)
    except FileNotFoundError as exc:
        raise NotFoundError("Generated content was not found") from exc
    return WriterResponse(
        content_id=result["content_id"],
        content_type="",
        topic="",
        content=result["content"],
        meta=result["meta"],
    )


@router.get(WRITER_PREFIX + "/content", response_model=list[ContentListItem])
async def list_content(
    writer_service: Annotated[WriterService, Depends(get_writer)],
    content_type: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[ContentListItem]:
    """List generated content newest first."""
    items = await writer_service.list_content(
        content_type=content_type or None,
        limit=limit,
        offset=offset,
    )
    return [ContentListItem(**item) for item in items]


@router.get("/api/scheduler/jobs", response_model=list[SchedulerJob])
async def list_jobs(
    scheduler: Annotated[BraindScheduler, Depends(get_scheduler)],
) -> list[SchedulerJob]:
    """List registered background jobs."""
    return [SchedulerJob(**job) for job in scheduler.get_jobs()]


@router.post("/api/scheduler/run")
async def run_job_now(
    req: RunJobRequest,
    scheduler: Annotated[BraindScheduler, Depends(get_scheduler)],
) -> dict[str, str | bool]:
    """Manually trigger one registered background job."""
    try:
        await scheduler.run_job(req.name)
    except KeyError as exc:
        raise NotFoundError("The requested scheduler job was not found") from exc
    return {"ok": True, "job": req.name}
