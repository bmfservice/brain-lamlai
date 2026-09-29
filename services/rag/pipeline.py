"""Unified retrieval pipeline (Brain Second Phase 8).

Builds on the three-stream ``RAGService`` with three extra capabilities the
master prompt requires:

1. **Query analyzer** — classifies the query into §65 families so downstream
   weights and filters adapt (temporal current/historical, multi-hop,
   contradiction, entity alias, negative, etc.).
2. **Entity + graph candidate streams** — resolved entities from the query
   trigger a graph-traversal hop, and neighbour chunks are merged into the
   candidate pool (bounded by ``graph_max_*`` config, never unbounded).
3. **Stream-specific RRF weights** — the query type boosts the streams most
   likely to answer that question.

The pipeline is still fully mocked in tests (no live PostgreSQL or vLLM).
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy import or_ as sql_or_
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import ARRAY

from core import database as db
from core.logging import get_logger
from services.observability.metrics import MetricsRegistry
from services.rag.analyzer import QueryAnalyzer, QueryProfile, QueryType
from services.rag.context import AssembledContext, ContextAssembler
from services.rag.preferences import RetrievalPreferences
from services.rag.preferences import for_query as preferences_for_query
from services.rag.reranker import FeatureReranker
from services.rag.service import RAGService, RetrievedChunk

if TYPE_CHECKING:
    from services.graph import GraphService

logger = get_logger(__name__)


@dataclass(frozen=True)
class RetrievalResult:
    """Unified retrieval output (Phase 8)."""

    query_profile: QueryProfile
    chunks: list[dict] = field(default_factory=list)
    graph_nodes: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


class UnifiedRetrievalService:
    """Hybrid retrieval orchestrator: analyzer → candidates → graph expansion."""

    # Per-query-type RRF weights.  Heavily based on which streams answer the
    # question best; all weights are config-driven and sum to 1.0 per family.
    _DEFAULT_STREAM_WEIGHTS: dict[str, dict[str, float]] = {
        "default": {"vector": 0.30, "fts": 0.30, "trgm": 0.25, "entity": 0.15},
        QueryType.EXACT: {"vector": 0.20, "fts": 0.45, "trgm": 0.25, "entity": 0.10},
        QueryType.SEMANTIC: {"vector": 0.50, "fts": 0.20, "trgm": 0.20, "entity": 0.10},
        QueryType.MULTI_HOP: {"vector": 0.25, "fts": 0.20, "trgm": 0.20, "entity": 0.35},
        QueryType.CROSS_DOCUMENT: {"vector": 0.20, "fts": 0.20, "trgm": 0.20, "entity": 0.40},
        QueryType.CONTRADICTION: {"vector": 0.20, "fts": 0.25, "trgm": 0.20, "entity": 0.35},
        QueryType.ENTITY_ALIAS: {"vector": 0.20, "fts": 0.30, "trgm": 0.20, "entity": 0.30},
        QueryType.TEMPORAL_CURRENT: {"vector": 0.30, "fts": 0.30, "trgm": 0.20, "entity": 0.20},
        QueryType.TEMPORAL_HISTORICAL: {"vector": 0.30, "fts": 0.30, "trgm": 0.20, "entity": 0.20},
        QueryType.NEGATIVE: {"vector": 0.30, "fts": 0.30, "trgm": 0.25, "entity": 0.15},
        QueryType.AMBIGUOUS: {"vector": 0.30, "fts": 0.30, "trgm": 0.25, "entity": 0.15},
        QueryType.GLOBAL: {"vector": 0.30, "fts": 0.30, "trgm": 0.25, "entity": 0.15},
    }

    def __init__(
        self,
        *,
        rag: RAGService,
        graph: GraphService | None = None,
        session_factory=None,
        analyzer: QueryAnalyzer | None = None,
        reranker: FeatureReranker | None = None,
        assembler: ContextAssembler | None = None,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self.rag = rag
        self.graph = graph
        self.session_factory = session_factory
        self.analyzer = analyzer or QueryAnalyzer()
        self.reranker = reranker or FeatureReranker()
        self.assembler = assembler or ContextAssembler()
        self.metrics = metrics or MetricsRegistry()

    async def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        tags: list[str] | None = None,
        excluded_types: list[str] | None = None,
        preferences: RetrievalPreferences | None = None,
        session_id: str | None = None,
    ) -> RetrievalResult:
        """Run the full Phase 8 retrieval pipeline for ``query``.

        ``preferences`` overrides what the query itself implies — the writer
        passes its own, because grounding material is judged by what it can be
        stated as fact, not by the shape of a question.
        """
        self.metrics.counter("retrieval.queries")
        profile = self.analyzer.analyze(query)
        prefs = preferences or preferences_for_query(profile)
        weights = self._stream_weights(profile.query_type)

        # 1. Base three-stream scores from RAGService (session-scoped).
        base_scores = await self.rag.component_scores(
            query,
            top_k=top_k * 2,
            tags=tags,
            excluded_types=excluded_types,
            session_id=session_id,
        )

        # 2. Entity-based candidate stream: direct text matches for named
        #    entities extracted from the query (CASE E alias expansion).
        entity_scores: dict[str, float] = {}
        if profile.entity_candidates:
            entity_scores = await self._entity_search(
                profile.entity_candidates,
                top_k * 2,
                tags,
                excluded_types,
                session_id=session_id,
            )

        # 3. Graph expansion: one bounded hop from the resolved entity ids.
        graph_nodes: list = []
        graph_chunks: dict[str, float] = {}
        if self.graph is not None and self.session_factory is not None:
            graph_chunks, graph_nodes = await self._graph_expansion(
                profile, top_k * 2, session_id=session_id
            )
            # Fold graph evidence into the entity stream (both identity-driven).
            for cid, score in graph_chunks.items():
                entity_scores[cid] = max(entity_scores.get(cid, 0.0), score)

        all_scores = {
            "vector": base_scores["vector"],
            "fts": base_scores["fts"],
            "trgm": base_scores["trgm"],
            "entity": entity_scores,
        }
        merged: dict[str, float] = {}
        for stream_name, score_dict in all_scores.items():
            weight = weights.get(stream_name, 0.0)
            if weight <= 0:
                continue
            for rank, chunk_id in enumerate(
                sorted(score_dict, key=score_dict.get, reverse=True)[:top_k]
            ):
                merged[chunk_id] = merged.get(chunk_id, 0.0) + weight / (60 + rank)

        # Deduplication is implicit in the dict merge above.
        top_ids = [
            cid
            for cid, _ in sorted(merged.items(), key=lambda x: x[1], reverse=True)[:top_k]
        ]
        candidates = await self._load_chunks(
            top_ids, query, merged, session_id=session_id
        )

        # 4. Feature-based re-ranking (Phase 9).
        query_terms = [
            t.lower().strip(string.punctuation)
            for t in query.split()
            if len(t.strip(string.punctuation)) > 2
        ]
        ranked = self.reranker.rank(
            candidates,
            query_terms=query_terms,
            entity_candidates=profile.entity_candidates,
            graph_nodes=graph_nodes,
            chunk_distances={cid: 1 for cid in graph_chunks},
            preferences=prefs,
        )
        by_id = {c["chunk_id"]: c for c in candidates}
        chunks = [
            {
                **by_id.get(r.chunk_id, {}),
                "chunk_id": r.chunk_id,
                "content": r.content,
                "score": r.score,
                "features": r.features,
            }
            for r in ranked
        ]

        diagnostics = {
            "streams_used": list(all_scores.keys()),
            "weights": {k: v for k, v in weights.items() if v > 0},
            "graph_hops": len(graph_nodes),
            "raw_merged_count": len(merged),
            "preferred_roles": list(prefs.preferred_roles),
            "preferred_doc_types": list(prefs.preferred_doc_types),
        }
        return RetrievalResult(
            query_profile=profile,
            chunks=chunks,
            graph_nodes=graph_nodes,
            diagnostics=diagnostics,
        )

    def assemble_context(
        self,
        chunks: list[dict],
        profile: QueryProfile,
    ) -> AssembledContext:
        """Turn ranked chunks into a token-bounded, cited context string."""
        return self.assembler.assemble(chunks, profile=profile)

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        tags: list[str] | None = None,
        exclude_source_types: list[str] | None = None,
        preferences: RetrievalPreferences | None = None,
        session_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """Drop-in replacement for ``RAGService.retrieve`` using the full pipeline.

        Lets chat, the agent, the knowledge tool and the writer share the
        analyzer/entity/graph/rerank path instead of the bare three-stream one,
        while still returning ``RetrievedChunk`` so callers stay unchanged.
        """
        top_k = top_k or self.rag.top_k
        result = await self.search(
            query,
            top_k=top_k,
            tags=tags,
            excluded_types=self.rag.default_excludes(exclude_source_types),
            preferences=preferences,
            session_id=session_id,
        )
        return [
            RetrievedChunk(
                chunk_id=c["chunk_id"],
                title=c.get("title") or "",
                content=c.get("content") or "",
                score=c.get("score", 0.0),
                source_id=c.get("source_id"),
                tags=c.get("tags") or [],
                meta=c.get("meta") or {},
                excerpt=self.rag.focused_excerpt(query, c.get("content") or ""),
            )
            for c in result.chunks
        ]

    async def recent_knowledge(
        self, limit: int = 20, *, session_id: str | None = None
    ) -> list[RetrievedChunk]:
        """Delegate the recency feed to the base service (no ranking involved)."""
        return await self.rag.recent_knowledge(limit=limit, session_id=session_id)

    def _stream_weights(self, query_type: QueryType) -> dict[str, float]:
        return self._DEFAULT_STREAM_WEIGHTS.get(
            str(query_type), self._DEFAULT_STREAM_WEIGHTS["default"]
        )

    async def _entity_search(
        self,
        entity_candidates: tuple[str, ...],
        top_k: int,
        tags: list[str] | None,
        excluded_types: list[str] | None,
        *,
        session_id: str | None = None,
    ) -> dict[str, float]:
        """Lightweight entity-name match stream (case-insensitive, OR)."""
        if not entity_candidates or self.session_factory is None:
            return {}
        try:
            async with self.session_factory() as session:
                # Find chunks whose content contains any candidate entity name.
                # Use OR-ed ILIKE filters; the FTS stream already covers most of
                # this, but this gives a deterministic boost for exact aliases.
                clauses = [
                    db.Document.content.ilike(f"%{name}%")
                    for name in entity_candidates
                ]
                if not clauses:
                    return {}
                filter_ = clauses[0] if len(clauses) == 1 else sql_or_(*clauses)
                stmt = (
                    select(db.Document.id, db.Document.content)
                    .where(filter_)
                    .limit(top_k)
                )
                if session_id:
                    # Giai đoạn 4: this stream does not join sources, so the
                    # documents-side session column is the scope.
                    stmt = stmt.where(
                        sql_or_(
                            db.Document.session_id.is_(None),
                            db.Document.session_id == session_id,
                        )
                    )
                rows = (await session.execute(stmt)).all()
        except Exception:
            logger.warning("retrieval.entity_stream_failed", exc_info=True)
            return {}
        scores: dict[str, float] = {}
        for row in rows:
            chunk_id = str(row[0])
            content = (row[1] or "").lower()
            hits = sum(1 for name in entity_candidates if name.lower() in content)
            if hits:
                scores[chunk_id] = min(1.0, 0.5 + hits * 0.1)
        return scores

    async def _graph_expansion(
        self,
        profile: QueryProfile,
        top_k: int,
        *,
        session_id: str | None = None,
    ) -> tuple[dict[str, float], list]:
        """Bounded graph expansion around entities mentioned in the query.

        Looks up canonical entities in the DB, walks one bounded hop via the
        graph service, then returns any chunks referencing the neighbours.
        """
        if self.graph is None or self.session_factory is None:
            return {}, []
        graph_nodes: list = []
        try:
            async with self.session_factory() as session:
                # Match entity canonical names/aliases to entity ids.
                rows = (
                    await session.execute(
                        select(db.Entity.id, db.Entity.canonical_name, db.Entity.aliases)
                    )
                ).all()
        except Exception:
            logger.warning("retrieval.entity_lookup_failed", exc_info=True)
            return {}, []

        name_to_id = {
            str(r.canonical_name): str(r.id)
            for r in rows
            if r.canonical_name
        }
        for r in rows:
            for alias in r.aliases or []:
                name_to_id.setdefault(str(alias), str(r.id))

        root_ids: list[str] = []
        for name in profile.entity_candidates:
            if name in name_to_id:
                root_ids.append(name_to_id[name])
        root_ids = list(dict.fromkeys(root_ids))

        neighbor_ids: set[str] = set()
        async with self.session_factory() as session:
            for root_id in root_ids:
                try:
                    nb = await self.graph.neighborhood(
                        session, root_id=root_id, hops=1
                    )
                    graph_nodes.extend(nb.nodes)
                    for n in nb.nodes:
                        neighbor_ids.add(n.node_id)
                except Exception:
                    logger.warning(
                        "retrieval.graph_hop_failed", root_id=root_id, exc_info=True
                    )
                    continue
            # Giai đoạn 5, B4: record the retrieval hit on the edges actually
            # walked. One batched UPDATE for all query-matched entities; failure
            # here must never cost the user their search results.
            if root_ids:
                try:
                    touched = await self.graph.touch_edges(session, node_ids=root_ids)
                    await session.commit()
                    if touched:
                        logger.debug("retrieval.edges_touched", count=touched)
                except Exception:
                    await session.rollback()
                    logger.warning("retrieval.edge_touch_failed", exc_info=True)

        # Map graph neighbours back to documents via semantic_units.entity_ids
        # → chunks.document_id. entity_ids is a JSONB array of entity uuids, so
        # the match is a containment test, never a substring scan of content.
        scores: dict[str, float] = {}
        if not neighbor_ids:
            return {}, graph_nodes
        neighbor_list = sorted(neighbor_ids)
        try:
            async with self.session_factory() as session:
                stmt = (
                    select(db.Chunk.document_id, db.SemanticUnit.entity_ids)
                    .join(db.Chunk, db.Chunk.id == db.SemanticUnit.chunk_id)
                    .where(
                        db.SemanticUnit.entity_ids.op("?|")(
                            sa.cast(neighbor_list, ARRAY(sa.Text))
                        )
                    )
                    .limit(top_k)
                )
                if session_id:
                    # Giai đoạn 4: scope via the owning document's session.
                    stmt = stmt.join(
                        db.Document, db.Document.id == db.Chunk.document_id
                    ).where(
                        sql_or_(
                            db.Document.session_id.is_(None),
                            db.Document.session_id == session_id,
                        )
                    )
                rows = (await session.execute(stmt)).all()
        except Exception:
            logger.warning("retrieval.graph_expansion_failed", exc_info=True)
            return {}, graph_nodes

        for document_id, entity_ids in rows:
            if document_id is None:
                continue
            hits = sum(1 for nid in neighbor_ids if nid in (entity_ids or []))
            if hits:
                key = str(document_id)
                scores[key] = max(scores.get(key, 0.0), min(1.0, 0.5 + hits * 0.1))
        return scores, graph_nodes

    async def _load_chunks(
        self,
        chunk_ids: list[str],
        query: str,
        scores: dict[str, float],
        *,
        session_id: str | None = None,
    ) -> list[dict]:
        """Load full chunk rows for the given ids and format for responses."""
        if not chunk_ids or self.session_factory is None:
            return []
        async with self.session_factory() as session:
            stmt = select(db.Document).where(db.Document.id.in_(chunk_ids))
            if session_id:
                stmt = stmt.where(
                    sql_or_(
                        db.Document.session_id.is_(None),
                        db.Document.session_id == session_id,
                    )
                )
            result = await session.execute(stmt)
            docs = {str(d.id): d for d in result.scalars()}
        items: list[dict] = []
        for cid in chunk_ids:
            doc = docs.get(cid)
            if doc is None:
                continue
            items.append(
                {
                    "chunk_id": cid,
                    "title": doc.title or "",
                    "content": doc.content,
                    "score": round(scores.get(cid, 0.0), 4),
                    "source_id": str(doc.source_id) if doc.source_id else None,
                    "tags": doc.tags or [],
                    "meta": doc.meta or {},
                    "doc_type": doc.doc_type,
                    "domain": doc.domain,
                    "chunk_role": doc.chunk_role,
                    "quality_tier": doc.quality_tier,
                    "quality_score": doc.quality_score,
                }
            )
        return items