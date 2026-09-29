"""Unified search routes.

Exposes the analyzer-aware retrieval pipeline plus debug routes for the
per-stream diagnostics and the observability metrics snapshot.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query

from api.deps import get_session_context, get_unified_retrieval
from api.schemas import SearchRequest
from services.rag.pipeline import RetrievalResult, UnifiedRetrievalService

router = APIRouter(prefix="/api", tags=["search"])


class UnifiedSearchRequest(SearchRequest):
    """Request body for the unified analyzer-driven search endpoint."""

    tags: list[str] = []
    include_source_types: list[str] = []


def _profile_payload(result: RetrievalResult) -> dict:
    return {
        "query": result.query_profile.query,
        "query_type": result.query_profile.query_type,
        "entity_candidates": result.query_profile.entity_candidates,
        "negated": result.query_profile.negated,
        "temporal": {
            "valid_from": result.query_profile.temporal.valid_from,
            "valid_to": result.query_profile.temporal.valid_to,
            "is_current": result.query_profile.temporal.is_current,
        },
    }


@router.post("/search/unified", response_model=dict)
async def unified_search(
    req: UnifiedSearchRequest,
    retrieval: Annotated[UnifiedRetrievalService, Depends(get_unified_retrieval)],
    session_ctx: Annotated[Any, Depends(get_session_context)],
) -> dict:
    """Hybrid retrieval with query-analysis, entity/graph streams, and RRF."""
    result = await retrieval.search(
        req.query,
        top_k=req.top_k,
        tags=req.tags or None,
        excluded_types=list(req.include_source_types or []),
        session_id=session_ctx.session_id,
    )
    return {
        "query_profile": _profile_payload(result),
        "results": result.chunks,
        "diagnostics": result.diagnostics,
    }


@router.get("/search/debug", response_model=dict)
async def debug_retrieval(
    retrieval: Annotated[UnifiedRetrievalService, Depends(get_unified_retrieval)],
    q: str = Query(..., min_length=1),
    top_k: int = Query(8, ge=1, le=50),
) -> dict:
    """Return the full retrieval result including graph nodes and diagnostics."""
    result = await retrieval.search(q, top_k=top_k)
    return {
        "query_profile": _profile_payload(result),
        "results": result.chunks,
        "graph_nodes": [
            {"node_id": n.node_id, "relation_type": n.relation_type}
            for n in result.graph_nodes
        ],
        "diagnostics": result.diagnostics,
    }


@router.get("/debug/metrics", response_model=dict)
async def debug_metrics(
    retrieval: Annotated[UnifiedRetrievalService, Depends(get_unified_retrieval)],
) -> dict:
    """Return the current observability metrics snapshot."""
    return retrieval.metrics.snapshot()
