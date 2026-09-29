"""Braind API - real knowledge graph, memory statistics, and tools."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from sqlalchemy import func as safunc
from sqlalchemy import select

from api.deps import get_ingest, get_scheduler, get_session_factory
from core.database import Chunk, Document, GeneratedContent, KnowledgeLink, Source
from core.errors import ValidationError
from services.ingest.service import IngestService
from services.tools.web import WebSearchTool

router = APIRouter(tags=["memory"])


@router.get("/api/memory/graph")
async def memory_graph(
    session_factory: Annotated[Any, Depends(get_session_factory)],
    include_generated: bool = False,
) -> dict[str, Any]:
    """Return the persisted knowledge graph for the Memory Canvas."""
    excluded_types = ["writer_generated", "writer_duplicate_reference"]
    async with session_factory() as session:
        source_query = select(Source).order_by(Source.created_at.desc()).limit(250)
        if not include_generated:
            source_query = source_query.where(Source.source_type.not_in(excluded_types))
        sources = (await session.execute(source_query)).scalars().all()
        source_ids = {source.id for source in sources}
        if source_ids:
            degree_rows = (
                await session.execute(
                    select(
                        KnowledgeLink.source_id.label("source_id"),
                        safunc.count(KnowledgeLink.id).label("degree"),
                    )
                    .where(KnowledgeLink.source_id.in_(source_ids))
                    .group_by(KnowledgeLink.source_id)
                    .union_all(
                        select(
                            KnowledgeLink.target_id.label("source_id"),
                            safunc.count(KnowledgeLink.id).label("degree"),
                        )
                        .where(KnowledgeLink.target_id.in_(source_ids))
                        .group_by(KnowledgeLink.target_id)
                    )
                )
            ).all()
            degrees: dict[str, int] = {}
            for source_id, degree in degree_rows:
                key = str(source_id)
                degrees[key] = degrees.get(key, 0) + int(degree)
            edge_rows = (
                (
                    await session.execute(
                        select(KnowledgeLink)
                        .where(
                            KnowledgeLink.source_id.in_(source_ids),
                            KnowledgeLink.target_id.in_(source_ids),
                        )
                        .order_by(KnowledgeLink.weight.desc(), KnowledgeLink.created_at.desc())
                        .limit(800)
                    )
                )
                .scalars()
                .all()
            )
        else:
            degrees, edge_rows = {}, []
    nodes = [
        {
            "id": str(source.id),
            "label": source.name[:70],
            "type": source.source_type or "manual",
            "tags": source.tags or [],
            "summary": ((source.meta or {}).get("summary") or "")[:180],
            "parent_id": str(source.parent_id) if source.parent_id else None,
            "degree": degrees.get(str(source.id), 0),
            "created_at": source.created_at.isoformat() if source.created_at else None,
        }
        for source in sources
    ]
    edges = [
        {
            "source": str(link.source_id),
            "target": str(link.target_id),
            "weight": round(float(link.weight), 4),
            "relation_type": link.relation_type,
            "shared": (link.meta or {}).get("shared", []),
        }
        for link in edge_rows
    ]
    by_type: dict[str, int] = {}
    for node in nodes:
        by_type[node["type"]] = by_type.get(node["type"], 0) + 1
    return {
        "nodes": nodes,
        "edges": edges,
        "total": len(nodes),
        "stats": {"sources": len(nodes), "edges": len(edges), "by_type": by_type},
    }


@router.get("/api/memory/stats")
async def memory_stats(
    session_factory: Annotated[Any, Depends(get_session_factory)],
) -> dict[str, Any]:
    """Return aggregated memory statistics for the UI dashboard."""
    async with session_factory() as session:
        source_count = (await session.execute(select(safunc.count(Source.id)))).scalar() or 0
        document_count = (await session.execute(select(safunc.count(Document.id)))).scalar() or 0
        generated_count = (
            await session.execute(select(safunc.count(GeneratedContent.id)))
        ).scalar() or 0
        link_total = (await session.execute(select(safunc.count(KnowledgeLink.id)))).scalar() or 0
        type_rows = (
            await session.execute(
                select(Source.source_type, safunc.count(Source.id)).group_by(Source.source_type)
            )
        ).all()
        link_type_rows = (
            await session.execute(
                select(KnowledgeLink.relation_type, safunc.count(KnowledgeLink.id)).group_by(
                    KnowledgeLink.relation_type
                )
            )
        ).all()
        chunk_count = (await session.execute(select(safunc.count(Chunk.id)))).scalar() or 0
    return {
        "sources": source_count,
        "documents": document_count,
        "chunks": chunk_count,
        "generated": generated_count,
        "links": link_total,
        "by_type": {str(kind or "manual"): count for kind, count in type_rows},
        "links_by_type": {str(kind or "related"): count for kind, count in link_type_rows},
    }


@router.post("/api/tools/search_external")
async def search_external(
    payload: dict[str, Any], ingest: Annotated[IngestService, Depends(get_ingest)]
) -> dict[str, Any]:
    """Search the web and ingest selected results into the knowledge base."""
    query = str(payload.get("query", "")).strip()
    if not query:
        raise ValidationError("A search query is required")
    result = await WebSearchTool(ingest).execute(
        query=query, max_results=payload.get("max_results", 10), save_to_brain=True
    )
    return {"ok": True, "query": query, **result}


@router.post("/api/tools/fetch_rss")
async def trigger_rss(
    scheduler: Annotated[Any, Depends(get_scheduler)],
) -> dict[str, str | bool]:
    """Manually trigger the RSS fetch job."""
    await scheduler.run_job("rss-fetch")
    return {"ok": True, "job": "rss-fetch"}
