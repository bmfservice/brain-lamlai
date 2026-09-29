"""Braind RAG Service.

Hybrid retrieval engine combining:
- Vector similarity (pgvector, cosine)
- Full-text search (PostgreSQL FTS via tsvector)
- Fuzzy matching (pg_trgm)
Then merges via Reciprocal Rank Fusion (RRF) for quality results.

Excerpt extraction: each result includes a focused ~280-char window
centered on the highest keyword-density region, so the agent sees
precise context instead of wall-of-text chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sqlalchemy import or_, select, text

from core.config import settings
from core.database import Document as DocumentModel
from core.logging import get_logger
from core.vllm import VLLMClient

logger = get_logger(__name__)

# Maximum length of the focused excerpt for agent consumption.
_MAX_EXCERPT = 300
# Window around the keyword match centre.
_EXCERPT_HALF = 140


@dataclass
class RetrievedChunk:
    """A retrieved document chunk with score and focused excerpt."""

    chunk_id: str
    title: str
    content: str
    score: float
    source_id: str | None = None
    tags: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    excerpt: str = ""  # Focused ~280-char window for the agent.

    def to_dict(self) -> dict:
        return {
            "chunk_id": str(self.chunk_id),
            "title": self.title,
            "content": self.content,  # Full content for writer/learn callers.
            "score": round(self.score, 4),
            "source_id": str(self.source_id) if self.source_id else None,
            "tags": self.tags,
            "meta": self.meta,
            "excerpt": self.excerpt,
        }


class RAGService:
    """Hybrid retrieval service."""

    def __init__(self, session_factory, vllm: VLLMClient, top_k: int | None = None):
        self.session_factory = session_factory
        self.vllm = vllm
        self.top_k = top_k or settings.top_k

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        tags: list[str] | None = None,
        exclude_source_types: list[str] | None = None,
        session_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks using hybrid search (vector + FTS + trgm).

        By default, generated writer artefacts are excluded from retrieval so
        they cannot recursively pollute chat or future content generation.
        Pass an empty list to explicitly include every source type.

        ``session_id`` (Giai đoạn 4) scopes results to that session's private
        knowledge plus the global corpus; ``None`` means system context (all
        rows visible — scheduler, digests).
        """
        top_k = top_k or self.top_k
        excluded_types = self.default_excludes(exclude_source_types)
        scores = await self.component_scores(
            query,
            top_k=top_k,
            tags=tags,
            excluded_types=excluded_types,
            session_id=session_id,
        )
        rrf = self._rrf(scores, top_k)
        return await self._load_chunks(query, rrf, top_k, session_id=session_id)

    async def component_scores(
        self,
        query: str,
        *,
        top_k: int | None = None,
        tags: list[str] | None = None,
        excluded_types: list[str] | None = None,
        session_id: str | None = None,
    ) -> dict[str, dict[str, float]]:
        """Run the three retrieval streams and return their raw score dicts.

        Exposed so the unified pipeline (Phase 8) can merge entity/graph
        streams on top of the same vector + FTS + trgm candidates.  Each
        stream is independent and best-effort (a failed stream degrades the
        merge, never fails the search).
        """
        top_k = top_k or self.top_k
        excluded_types = self.default_excludes(excluded_types)

        # Vector search
        vector_scores: dict[str, float] = {}
        try:
            embedding = await self.vllm.embed(query)
            vector_scores = await self._vector_search(
                embedding, top_k, tags, excluded_types, session_id=session_id
            )
        except Exception as exc:
            logger.warning("rag.vector_failed", error=str(exc))

        # Full-text search
        fts_scores: dict[str, float] = {}
        try:
            fts_scores = await self._fts_search(
                query, top_k, tags, excluded_types, session_id=session_id
            )
        except Exception as exc:
            logger.warning("rag.fts_failed", error=str(exc))

        # Fuzzy search (trgm)
        trgm_scores: dict[str, float] = {}
        try:
            trgm_scores = await self._trgm_search(
                query, top_k, tags, excluded_types, session_id=session_id
            )
        except Exception as exc:
            logger.warning("rag.trgm_failed", error=str(exc))

        return {"vector": vector_scores, "fts": fts_scores, "trgm": trgm_scores}

    @staticmethod
    def default_excludes(exclude_source_types: list[str] | None) -> list[str]:
        """Writer artifacts are excluded from ordinary retrieval by default."""
        return (
            ["writer_generated", "writer_duplicate_reference"]
            if exclude_source_types is None
            else exclude_source_types
        )

    @staticmethod
    def _rrf(
        scores: dict[str, dict[str, float]], top_k: int
    ) -> dict[str, float]:
        """Reciprocal Rank Fusion over the component score dicts."""
        rrf: dict[str, float] = {}
        for component in scores.values():
            ranked = sorted(component, key=component.get, reverse=True)
            for rank, chunk_id in enumerate(ranked[:top_k]):
                rrf[chunk_id] = rrf.get(chunk_id, 0.0) + 1.0 / (60 + rank)
        return rrf

    async def _load_chunks(
        self,
        query: str,
        rrf: dict[str, float],
        top_k: int,
        *,
        session_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """Load full chunk rows for the RRF-ordered id set.

        Session filter applied here too (defense-in-depth): ids that somehow
        leaked from another session are not hydrated into results.
        """
        if not rrf:
            return []
        top_ids = [
            cid for cid, _ in sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
        ]
        async with self.session_factory() as session:
            stmt = select(DocumentModel).where(DocumentModel.id.in_(top_ids))
            if session_id:
                stmt = stmt.where(
                    or_(DocumentModel.session_id.is_(None), DocumentModel.session_id == session_id)
                )
            result = await session.execute(stmt)
            docs = {str(d.id): d for d in result.scalars()}
            items: list[RetrievedChunk] = []
            for cid in top_ids:
                doc = docs.get(cid)
                if doc is None:
                    continue
                items.append(
                    RetrievedChunk(
                        chunk_id=cid,
                        title=doc.title or "",
                        content=doc.content,
                        score=rrf[cid],
                        source_id=str(doc.source_id) if doc.source_id else None,
                        tags=doc.tags or [],
                        meta=doc.meta or {},
                        excerpt=self.focused_excerpt(query, doc.content or ""),
                    )
                )
        return items

    @staticmethod
    def focused_excerpt(query: str, content: str, max_chars: int = _MAX_EXCERPT) -> str:
        """Return the highest keyword-density excerpt from a retrieved chunk.

        The hybrid retriever keeps full chunks for workflows needing them, but
        the agent receives this compact semantic neighborhood.  Candidate
        windows are centred on query-term occurrences; ties favour earlier,
        coherent paragraph boundaries.  A normalized whitespace result keeps
        the response compact without corrupting the stored source.
        """
        normalized = re.sub(r"\s+", " ", content or "").strip()
        if len(normalized) <= max_chars:
            return normalized

        terms = [
            term.lower()
            for term in re.findall(r"[\wÀ-ỹ]{2,}", query, flags=re.UNICODE)
            if len(term) >= 2
        ]
        # Preserve order while ignoring duplicated terms and common weak tokens.
        stop_words = {
            "của",
            "và",
            "các",
            "cho",
            "trong",
            "với",
            "những",
            "được",
            "là",
            "có",
            "từ",
            "một",
            "the",
            "and",
            "for",
            "with",
            "what",
        }
        terms = list(dict.fromkeys(t for t in terms if t not in stop_words))
        if not terms:
            return normalized[:max_chars].rstrip(" ,;:") + "…"

        lower = normalized.lower()
        centers: list[int] = []
        for term in terms:
            centers.extend(m.start() for m in re.finditer(re.escape(term), lower))
        if not centers:
            return normalized[:max_chars].rstrip(" ,;:") + "…"

        best_start = 0
        best_score = -1
        # Score every candidate window by summed whole-word query-term matches.
        for center in centers:
            start = max(0, min(len(normalized) - max_chars, center - _EXCERPT_HALF))
            end = min(len(normalized), start + max_chars)
            window = lower[start:end]
            score = sum(len(re.findall(rf"\b{re.escape(term)}\b", window)) for term in terms)
            if score > best_score:
                best_score = score
                best_start = start

        start = best_start
        end = min(len(normalized), start + max_chars)
        # Avoid visibly chopped words at both boundaries where practical.
        if start > 0:
            boundary = normalized.find(" ", start)
            if boundary != -1 and boundary < start + 35:
                start = boundary + 1
        if end < len(normalized):
            boundary = normalized.rfind(" ", end - 35, end)
            if boundary != -1:
                end = boundary
        excerpt = normalized[start:end].strip()
        if start > 0:
            excerpt = "…" + excerpt
        if end < len(normalized):
            excerpt += "…"
        return excerpt

    async def recent_knowledge(
        self, limit: int = 20, *, session_id: str | None = None
    ) -> list[RetrievedChunk]:
        """Return the most recently ingested documents (for digests / feeds)."""
        async with self.session_factory() as session:
            stmt = select(DocumentModel).order_by(DocumentModel.created_at.desc()).limit(limit)
            if session_id:
                stmt = stmt.where(
                    or_(DocumentModel.session_id.is_(None), DocumentModel.session_id == session_id)
                )
            result = await session.execute(stmt)
            return [
                RetrievedChunk(
                    chunk_id=str(d.id),
                    title=d.title or "",
                    content=d.content,
                    score=0.0,
                    source_id=str(d.source_id) if d.source_id else None,
                    tags=d.tags or [],
                    meta=d.meta or {},
                )
                for d in result.scalars()
            ]

    @staticmethod
    def _filter_clauses(
        tags: list[str] | None,
        excluded_types: list[str] | None,
        session_id: str | None = None,
    ) -> tuple[str, str, dict]:
        """Build parameterized tag/source-type/session filters.

        Tags and source types reach here from API input, so they are bound
        parameters rather than interpolated SQL. The session clause scopes to
        the caller's rows plus the global corpus (NULL session_id).
        """
        params: dict = {}
        tag_clause = ""
        if tags:
            tag_clause = "AND d.tags ?| CAST(:tags AS text[])"
            params["tags"] = list(tags)
        type_clause = ""
        if excluded_types:
            type_clause = "AND s.source_type <> ALL(CAST(:excluded_types AS text[]))"
            params["excluded_types"] = list(excluded_types)
        session_clause = ""
        if session_id:
            session_clause = (
                "AND (d.session_id IS NULL OR d.session_id = CAST(:session_id AS uuid))"
            )
            params["session_id"] = session_id
        return tag_clause, type_clause, params, session_clause

    async def _vector_search(
        self,
        embedding: list[float],
        top_k: int,
        tags: list[str] | None,
        excluded_types: list[str],
        *,
        session_id: str | None = None,
    ) -> dict[str, float]:
        tag_clause, type_clause, params, session_clause = self._filter_clauses(
            tags, excluded_types, session_id
        )
        params["embedding"] = "[" + ",".join(str(x) for x in embedding) + "]"
        params["top_k"] = top_k
        sql = f"""
            SELECT d.id::text,
                   1 - (d.embedding_vllm_1024 <=> CAST(:embedding AS vector)) AS score
            FROM documents d
            JOIN sources s ON s.id = d.source_id
            WHERE d.embedding_vllm_1024 IS NOT NULL
              AND d.status = 'active'
            {tag_clause}
            {type_clause}
            {session_clause}
            ORDER BY d.embedding_vllm_1024 <=> CAST(:embedding AS vector)
            LIMIT :top_k
        """
        async with self.session_factory() as session:
            result = await session.execute(text(sql), params)
            return {row[0]: float(row[1]) for row in result}

    async def _fts_search(
        self,
        query: str,
        top_k: int,
        tags: list[str] | None,
        excluded_types: list[str],
        *,
        session_id: str | None = None,
    ) -> dict[str, float]:
        tag_clause, type_clause, params, session_clause = self._filter_clauses(
            tags, excluded_types, session_id
        )
        params.update({"query": query, "top_k": top_k})
        sql = f"""
            SELECT d.id::text, ts_rank(d.fts_vector, plainto_tsquery('english', :query)) AS score
            FROM documents d
            JOIN sources s ON s.id = d.source_id
            WHERE d.fts_vector @@ plainto_tsquery('english', :query)
              AND d.status = 'active'
            {tag_clause}
            {type_clause}
            {session_clause}
            ORDER BY score DESC
            LIMIT :top_k
        """
        async with self.session_factory() as session:
            result = await session.execute(text(sql), params)
            return {row[0]: float(row[1]) for row in result}

    async def _trgm_search(
        self,
        query: str,
        top_k: int,
        tags: list[str] | None,
        excluded_types: list[str],
        *,
        session_id: str | None = None,
    ) -> dict[str, float]:
        tag_clause, type_clause, params, session_clause = self._filter_clauses(
            tags, excluded_types, session_id
        )
        params.update({"query": query, "top_k": top_k})
        sql = f"""
            SELECT d.id::text, similarity(d.content, :query) AS score
            FROM documents d
            JOIN sources s ON s.id = d.source_id
            WHERE d.content % :query
              AND d.status = 'active'
            {tag_clause}
            {type_clause}
            {session_clause}
            ORDER BY score DESC
            LIMIT :top_k
        """
        async with self.session_factory() as session:
            result = await session.execute(text(sql), params)
            return {row[0]: float(row[1]) for row in result}
