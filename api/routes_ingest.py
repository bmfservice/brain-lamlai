"""Braind API - ingest routes."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Response, UploadFile

from api.deps import (
    get_file_policy,
    get_ingest,
    get_job_tracker,
    get_obsidian,
    get_session_context,
    get_session_factory,
)
from api.schemas import (
    IngestQualityReport,
    IngestResponse,
    IngestTextRequest,
    ObsidianExportResponse,
)
from core.config import settings
from core.documents import UnsupportedDocument, extract_text
from core.errors import ValidationError
from core.session_context import SESSION_COOKIE
from services.ingest.service import IngestService
from services.observability import jobs as job_obs
from services.obsidian import ObsidianExporter

router = APIRouter(prefix="/api/ingest", tags=["ingest"])


def _ingest_response(result) -> IngestResponse:
    """Convert an ingest result without changing the existing success schema."""
    return IngestResponse(
        source=result.source,
        added=result.added,
        updated=result.updated,
        deleted=result.deleted,
        skipped=result.skipped,
        characters_read=getattr(result, "characters_read", 0),
        errors=result.errors,
        job_id=getattr(result, "job_id", None),
    )


@router.post("/text", response_model=IngestResponse)
async def ingest_text(
    req: IngestTextRequest,
    ingest_service: Annotated[IngestService, Depends(get_ingest)],
    session_ctx: Annotated[Any, Depends(get_session_context)],
    response: Response,
) -> IngestResponse:
    """Ingest a raw text blob into the knowledge base (session-scoped)."""
    if session_ctx.minted:
        response.set_cookie(SESSION_COOKIE, session_ctx.session_id, samesite="lax", httponly=True)
    result = await ingest_service.ingest_text(
        req.content,
        title=req.title,
        source_type=req.source_type,
        path=req.path,
        url=req.url,
        tags=req.tags or None,
        session_id=session_ctx.session_id,
        fingerprint_hash=session_ctx.fingerprint_hash,
        meta=req.meta or None,
    )
    return _ingest_response(result)


@router.post("/file", response_model=IngestResponse)
async def ingest_file(
    file: Annotated[UploadFile, File(description="Document to ingest")],
    ingest_service: Annotated[IngestService, Depends(get_ingest)],
    session_ctx: Annotated[Any, Depends(get_session_context)],
    file_policy: Annotated[Any, Depends(get_file_policy)],
    response: Response,
) -> IngestResponse:
    """Upload and ingest a text or document file.

    Supports plain text and code, plus .pdf, .docx, .pptx, .xlsx, .odt and
    .epub. See ``core.documents`` for the extraction rules.
    """
    raw = await file.read()
    if len(raw) > settings.upload_max_bytes:
        raise ValidationError(
            f"The file exceeds the {settings.upload_max_bytes // 1_048_576}MB upload limit."
        )
    filename = file.filename or "upload"
    # Giai đoạn 5, Layer 6: governance file-import policy — real mime sniffing
    # plus DB rules. Code/config files are rejected before structural ingest.
    policy = await file_policy.check(
        filename=filename, content_type=file.content_type, head_bytes=raw[:4096]
    )
    if not policy.allowed:
        from core.errors import PolicyDeniedError

        raise PolicyDeniedError(f"File rejected by governance policy: {policy.reason}")
    try:
        content, source_format = extract_text(raw, filename, file.content_type)
    except UnsupportedDocument as exc:
        raise ValidationError(str(exc)) from exc
    if session_ctx.minted:
        response.set_cookie(SESSION_COOKIE, session_ctx.session_id, samesite="lax", httponly=True)
    result = await ingest_service.ingest_text(
        content,
        title=Path(filename).stem,
        source_type="file",
        path=filename,
        session_id=session_ctx.session_id,
        fingerprint_hash=session_ctx.fingerprint_hash,
        meta={"uploaded_filename": filename, "source_format": source_format},
    )
    return _ingest_response(result)


@router.post("/vault", response_model=IngestResponse)
async def sync_vault(
    ingest_service: Annotated[IngestService, Depends(get_ingest)],
) -> IngestResponse:
    """Force a full resync of the Obsidian vault."""
    return _ingest_response(await ingest_service.sync_vault())


@router.get("/stats")
async def get_stats(ingest_service: Annotated[IngestService, Depends(get_ingest)]) -> dict:
    """Return knowledge base statistics."""
    return await ingest_service.get_stats()


@router.get("/report", response_model=IngestQualityReport)
async def ingest_report(
    session_factory: Annotated[object, Depends(get_session_factory)],
) -> IngestQualityReport:
    """Return a data-quality report for the ingestion pipeline (Phase 7)."""
    return IngestQualityReport(**await job_obs.build_quality_report(session_factory))


@router.post("/obsidian-export", response_model=ObsidianExportResponse)
async def obsidian_export(
    exporter: Annotated[ObsidianExporter, Depends(get_obsidian)],
    session_factory: Annotated[object, Depends(get_session_factory)],
) -> ObsidianExportResponse:
    """Rebuild the whole Obsidian display layer from the semantic tables (§3).

    Every ingest exports its own source automatically; this endpoint is for the
    full rebuild after a schema change, a renderer change, or a restore. Pruning
    only ever removes generated files inside the export directory.
    """
    async with session_factory() as session:
        result = await exporter.export_all(session)
    return ObsidianExportResponse(
        documents=result.documents,
        entities=result.entities,
        canvases=result.canvases,
        pruned=result.pruned,
        errors=result.errors,
    )


@router.post("/{job_id}/retry")
async def retry_stage(
    job_id: uuid.UUID,
    tracker: Annotated[object, Depends(get_job_tracker)],
) -> dict:
    """Re-mark a failed ingestion stage as queued for replay (idempotent).

    Never deletes or overwrites prior work — it only flags the stage row so a
    consumer can re-run it against the already-committed source.
    """
    return await job_obs.retry_failed_stage(tracker.session_factory, job_id)
