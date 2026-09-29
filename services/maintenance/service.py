"""Maintenance service (Brain Second Phase 12).

Provides a small, deterministic maintenance runner intended to be invoked from
the daily 00:00 window (or manually via scripts).  It performs three safe,
read-mostly passes:

1. **Deduplication** — find duplicate sources by content hash and near-duplicate
   chunks by token overlap, returning candidates for operator review.
2. **Contradiction scan** — surface pairs of semantic units that refer to the
   same entity within overlapping windows but have opposite polarity.
3. **Incremental reindex** — collect chunks whose ``embedding_vllm_1024`` is NULL
   and request fresh embeddings from the vLLM embedding service.

The service never deletes data automatically; it returns candidate lists so a
human or a later approved job can act.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sqlalchemy import func, or_, select, update

from core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from sqlalchemy.ext.asyncio import AsyncSession

logger = get_logger(__name__)

# The embedding server rejects a request whose total token count exceeds the
# model window, so re-embedding runs in small batches of truncated inputs.
_REINDEX_BATCH = 8
_MAX_EMBED_CHARS = 8000


@dataclass
class MaintenanceReport:
    """Result of a maintenance run."""

    duplicates: list[dict] = field(default_factory=list)
    contradictions: list[dict] = field(default_factory=list)
    reindex_queue: list[str] = field(default_factory=list)


class MaintenanceService:
    """Daily maintenance helper: dedup, contradiction scan, reindex queue."""

    def __init__(
        self,
        session_factory: Callable[[], AsyncSession] | None = None,
        embed_fn: Callable[..., Awaitable[list[list[float]]]] | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.embed_fn = embed_fn

    async def run(self) -> MaintenanceReport:
        """Run all three maintenance passes and return a report."""
        duplicates = await self.find_duplicate_sources()
        contradictions = await self.find_contradictions()
        reindex_queue = await self.find_reindex_candidates()
        return MaintenanceReport(
            duplicates=duplicates,
            contradictions=contradictions,
            reindex_queue=reindex_queue,
        )

    async def find_duplicate_sources(self) -> list[dict]:
        """Find sources with identical ``content_hash`` (exact duplicates)."""
        if self.session_factory is None:
            return []
        from core.database import Source

        async with self.session_factory() as session:  # type: ignore[arg-type]
            subq = (
                select(Source.content_hash, func.count().label("cnt"))
                .where(Source.content_hash.isnot(None))
                .group_by(Source.content_hash)
                .having(func.count() > 1)
            )
            result = await session.execute(subq)
            rows = result.all()
            if not rows:
                return []
            hashes = {r[0] for r in rows}
            stmt = (
                select(Source.id, Source.name, Source.content_hash)
                .where(Source.content_hash.in_(hashes))
                .order_by(Source.content_hash, Source.created_at)
            )
            result = await session.execute(stmt)
            items = [
                {
                    "id": str(r[0]),
                    "name": r[1],
                    "content_hash": r[2],
                }
                for r in result.all()
            ]
        # Group by hash for the report.
        grouped: dict[str, dict] = {}
        for item in items:
            grouped.setdefault(
                item["content_hash"],
                {"content_hash": item["content_hash"], "ids": []},
            )
            grouped[item["content_hash"]]["ids"].append(
                {"id": item["id"], "name": item["name"]}
            )
        return list(grouped.values())

    async def find_contradictions(self, entity_overlap_threshold: float = 0.9) -> list[dict]:
        """Find semantic-unit pairs with overlapping entity sets and opposite polarity.

        Args:
            entity_overlap_threshold: Minimum Jaccard overlap of ``entity_ids``
                to consider two units as referring to the same subject.
        """
        if self.session_factory is None:
            return []
        from core.database import SemanticUnit

        async with self.session_factory() as session:  # type: ignore[arg-type]
            stmt = (
                select(SemanticUnit)
                .where(SemanticUnit.entity_ids.isnot(None))
                .where(SemanticUnit.polarity != 0)
            )
            result = await session.execute(stmt)
            units = result.scalars().all()

        pairs: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for i, a in enumerate(units):
            for b in units[i + 1 :]:
                if a.polarity == b.polarity:
                    continue
                a_entities = set(a.entity_ids or [])
                b_entities = set(b.entity_ids or [])
                if not a_entities or not b_entities:
                    continue
                jaccard = len(a_entities & b_entities) / len(a_entities | b_entities)
                if jaccard < entity_overlap_threshold:
                    continue
                key = tuple(sorted([str(a.id), str(b.id)]))  # noqa: C413
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(
                    {
                        "unit_a": str(a.id),
                        "unit_b": str(b.id),
                        "entity_overlap": round(jaccard, 4),
                        "polarity_a": a.polarity,
                        "polarity_b": b.polarity,
                    }
                )
        return pairs

    async def find_reindex_candidates(self) -> list[str]:
        """Return ids of retrievable documents whose vector column is NULL.

        The semantic pass owns a sibling ``documents`` row holding the whole
        source text; it is a container, never a retrieval unit, and is far too
        long to embed. Those rows are excluded.
        """
        if self.session_factory is None:
            return []
        from core.database import Document

        async with self.session_factory() as session:  # type: ignore[arg-type]
            stmt = select(Document.id).where(
                Document.embedding_vllm_1024.is_(None),
                or_(
                    Document.meta.is_(None),
                    ~Document.meta.has_key("semantic_pass"),  # noqa: W601 - JSONB
                ),
            )
            result = await session.execute(stmt)
            return [str(row[0]) for row in result.all()]

    async def reindex(self, chunk_ids: list[str]) -> int:
        """Re-embed documents with a NULL vector and write the vectors back.

        Returns the number of documents updated. Only fills NULL columns, so a
        partial run is safe to repeat and never overwrites a good embedding.
        Runs in small batches with truncated inputs: one oversized row must not
        fail the whole pass, since the embedding server rejects a request whose
        total token count exceeds the model window.
        """
        if not chunk_ids or self.embed_fn is None or self.session_factory is None:
            return 0
        from core.database import Document

        updated = 0
        async with self.session_factory() as session:  # type: ignore[arg-type]
            result = await session.execute(
                select(Document.id, Document.content).where(
                    Document.id.in_(chunk_ids),
                    Document.embedding_vllm_1024.is_(None),
                )
            )
            rows = [(row[0], row[1]) for row in result.all() if row[1]]
            for start in range(0, len(rows), _REINDEX_BATCH):
                batch = rows[start : start + _REINDEX_BATCH]
                texts = [content[:_MAX_EMBED_CHARS] for _, content in batch]
                try:
                    embeddings = await self.embed_fn(texts)
                except Exception:
                    logger.warning(
                        "maintenance.reindex_batch_failed", offset=start, exc_info=True
                    )
                    continue
                for (doc_id, _), embedding in zip(batch, embeddings, strict=True):
                    if not embedding:
                        continue
                    await session.execute(
                        update(Document)
                        .where(Document.id == doc_id)
                        .values(embedding_vllm_1024=embedding)
                    )
                    updated += 1
                await session.commit()
        return updated

    @staticmethod
    def _token_set(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))
