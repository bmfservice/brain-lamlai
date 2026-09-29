"""Braind API - writer similarity checker + external duplicate check + content management."""

from __future__ import annotations

import re
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select

from api.deps import get_writer
from core.database import GeneratedContent
from core.errors import NotFoundError, ServiceUnavailableError
from services.writer.service import WriterService

router = APIRouter(tags=["writer", "similarity"])


class SimilarityCheckRequest(BaseModel):
    """Request to check similarity of new content against existing."""

    topic: str = Field(..., min_length=1, max_length=500)
    content_type: str | None = None
    keywords: list[str] = Field(default_factory=list)
    threshold: float = Field(0.75, ge=0.5, le=0.99)


class SimilarItem(BaseModel):
    """A similar existing content item."""

    content_id: str
    topic: str
    content_type: str
    similarity: float
    created_at: str | None = None


class SimilarityCheckResponse(BaseModel):
    """Response with similar items found."""

    query: str
    threshold: float
    matches: list[SimilarItem]
    count: int


@router.post("/api/writer/check_similarity", response_model=SimilarityCheckResponse)
async def check_similarity(
    req: SimilarityCheckRequest, writer: Annotated[WriterService, Depends(get_writer)]
) -> SimilarityCheckResponse:
    """Check if similar content already exists in generated content."""
    async with writer.session_factory() as session:
        rows = (
            (
                await session.execute(
                    select(GeneratedContent).order_by(GeneratedContent.created_at.desc()).limit(200)
                )
            )
            .scalars()
            .all()
        )
    query_words = set(word.lower() for word in re.findall(r"\w+", req.topic))
    query_words.update(word.lower() for word in req.keywords)
    matches: list[SimilarItem] = []
    for generated in rows:
        title_words = set(word.lower() for word in re.findall(r"\w+", generated.title or ""))
        meta_words = {str(word).lower() for word in (generated.meta or {}).get("keywords", [])}
        overlap = len(query_words & (title_words | meta_words))
        similarity = overlap / max(len(query_words), 1)
        if similarity >= req.threshold:
            matches.append(
                SimilarItem(
                    content_id=str(generated.id),
                    topic=generated.title or "",
                    content_type=generated.content_type,
                    similarity=round(similarity, 3),
                    created_at=generated.created_at.isoformat() if generated.created_at else None,
                )
            )
    matches.sort(key=lambda match: match.similarity, reverse=True)
    return SimilarityCheckResponse(
        query=req.topic, threshold=req.threshold, matches=matches[:10], count=len(matches)
    )


class ExternalCheckRequest(BaseModel):
    """Request to re-check a saved article against public web results."""

    content_id: str = Field(..., min_length=1)


@router.post("/api/writer/check_external")
async def check_external(
    req: ExternalCheckRequest, writer: Annotated[WriterService, Depends(get_writer)]
) -> dict:
    """Re-run the external duplicate check for a saved generated article.

    Clicking "Kiểm tra trùng web" on a finished article also learns the fetched
    external pages into the brain so Braind can study how other sites write about
    the topic. This is the only path that ingests external web references.
    """
    if writer.dupe_checker is None:
        raise ServiceUnavailableError("External duplicate checker has not been initialized")
    try:
        return await writer.check_external(req.content_id)
    except FileNotFoundError as exc:
        raise NotFoundError("Generated content was not found") from exc


@router.get("/api/writer/content/{content_id}")
async def get_content(
    content_id: str, writer: Annotated[WriterService, Depends(get_writer)]
) -> dict:
    """Return a generated article and its metadata."""
    try:
        return await writer.get_content(content_id)
    except FileNotFoundError as exc:
        raise NotFoundError("Generated content was not found") from exc


@router.delete("/api/writer/content/{content_id}")
async def delete_content(
    content_id: str, writer: Annotated[WriterService, Depends(get_writer)]
) -> dict:
    """Delete a generated content item for duplicate cleanup."""
    async with writer.session_factory() as session:
        generated = await session.get(GeneratedContent, content_id)
        if generated is None:
            raise NotFoundError("Generated content was not found")
        await session.delete(generated)
        await session.commit()
    return {"ok": True, "deleted": content_id}
