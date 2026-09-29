"""Knowledge store anomaly and structure discovery (CLAUDE.md §8, §9).

Discovers orphaned nodes, missing links, overpopulated categories, and unreferenced units.
Stale or unresolvable items are moved to quarantine via QuarantineService (never permanently deleted).
"""

from __future__ import annotations

import uuid
from typing import Any

import sqlalchemy as sa
from sqlalchemy import func, select, update

from core import database as db
from core.logging import get_logger
from services.maintenance.quarantine import QuarantineService
from services.selforg.journal import (
    OP_QUARANTINE_ITEM,
    OP_REPARENT_NODE,
    ReorgJournal,
)
from services.selforg.merge import CategoryMerger
from services.selforg.split import CategorySplitter

logger = get_logger(__name__)


class KnowledgeDiscoverer:
    """Discovers structural gaps, orphans, and reorganization opportunities."""

    def __init__(
        self,
        session_factory,
        journal: ReorgJournal | None = None,
        quarantine: QuarantineService | None = None,
        merger: CategoryMerger | None = None,
        splitter: CategorySplitter | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.journal = journal or ReorgJournal(session_factory)
        self.quarantine = quarantine or QuarantineService(session_factory)
        self.merger = merger or CategoryMerger(session_factory, self.journal)
        self.splitter = splitter or CategorySplitter(session_factory, self.journal)

    async def discover_orphaned_terms(self) -> list[dict[str, Any]]:
        """Identify taxonomy terms whose parent_id points to a non-existent or inactive term."""
        async with self.session_factory() as session:
            # Terms with parent_id that does not exist in taxonomy_terms
            parent_subq = select(db.TaxonomyTerm.id)
            stmt = select(db.TaxonomyTerm).where(
                db.TaxonomyTerm.parent_id.is_not(None),
                db.TaxonomyTerm.parent_id.not_in(parent_subq),
            )
            orphans = (await session.execute(stmt)).scalars().all()
            return [
                {
                    "term_id": str(t.id),
                    "term": t.term,
                    "facet": t.facet,
                    "invalid_parent_id": str(t.parent_id),
                    "status": t.status,
                }
                for t in orphans
            ]

    async def discover_ambiguous_terms(self, confidence_floor: float = 0.4) -> list[dict[str, Any]]:
        """Find terms in proposed or ambiguous status that have not been promoted."""
        async with self.session_factory() as session:
            stmt = select(db.TaxonomyTerm).where(
                db.TaxonomyTerm.status.in_(["ambiguous", "proposed"]),
                db.TaxonomyTerm.confidence < confidence_floor,
                db.TaxonomyTerm.seen_count < 2,
            )
            rows = (await session.execute(stmt)).scalars().all()
            return [
                {
                    "term_id": str(r.id),
                    "term": r.term,
                    "facet": r.facet,
                    "status": r.status,
                    "confidence": r.confidence,
                    "seen_count": r.seen_count,
                }
                for r in rows
            ]

    async def reparent_node(
        self,
        term_id: uuid.UUID | str,
        new_parent_id: uuid.UUID | str | None,
        *,
        rationale: str | None = None,
    ) -> uuid.UUID:
        """Reparent a taxonomy term, recording the change in the reorg journal."""
        t_uuid = uuid.UUID(str(term_id)) if isinstance(term_id, str) else term_id
        p_uuid = (
            uuid.UUID(str(new_parent_id))
            if new_parent_id is not None and str(new_parent_id).strip()
            else None
        )

        async with self.session_factory() as session:
            term = await session.get(db.TaxonomyTerm, t_uuid)
            if term is None:
                raise ValueError(f"Term {t_uuid} not found")

            if p_uuid is not None:
                parent = await session.get(db.TaxonomyTerm, p_uuid)
                if parent is None:
                    raise ValueError(f"New parent term {p_uuid} not found")

            before_state = {
                "term_id": str(term.id),
                "term": term.term,
                "facet": term.facet,
                "original_parent_id": str(term.parent_id) if term.parent_id else None,
            }

            term.parent_id = p_uuid

            after_state = {
                "term_id": str(term.id),
                "term": term.term,
                "facet": term.facet,
                "new_parent_id": str(p_uuid) if p_uuid else None,
            }

            await session.commit()

        reparent_rationale = (
            rationale
            or f"Reparented node '{term.term}' to parent '{p_uuid or 'root'}'"
        )
        op_id = await self.journal.record_operation(
            operation_type=OP_REPARENT_NODE,
            target_type="taxonomy_terms",
            target_id=str(t_uuid),
            before_state=before_state,
            after_state=after_state,
            rationale=reparent_rationale,
        )

        logger.info(
            "selforg.reparent_node",
            op_id=str(op_id),
            term_id=str(t_uuid),
            new_parent_id=str(p_uuid),
        )
        return op_id

    async def quarantine_stale_item(
        self,
        *,
        table_name: str,
        row_id: str,
        reason: str = "stale_unresolvable",
        detail: str | None = None,
    ) -> uuid.UUID:
        """Quarantine a stale or unresolvable item via QuarantineService (enforcing HR-NO-PERMANENT-DELETE)."""
        qid = await self.quarantine.quarantine_item(
            table_name=table_name,
            row_id=row_id,
            reason=reason,
            detail=detail,
        )

        # Record in reorg journal
        op_id = await self.journal.record_operation(
            operation_type=OP_QUARANTINE_ITEM,
            target_type=table_name,
            target_id=row_id,
            before_state={"table_name": table_name, "row_id": row_id, "active": True},
            after_state={"quarantine_id": str(qid), "reason": reason, "active": False},
            rationale=detail or f"Quarantined {table_name}:{row_id} due to {reason}",
        )
        return op_id

    async def run_discovery_sweep(self) -> dict[str, Any]:
        """Execute a full discovery pass returning opportunities across merge, split, and hygiene."""
        merge_candidates = await self.merger.find_merge_candidates()
        split_candidates = await self.splitter.find_split_candidates()
        orphans = await self.discover_orphaned_terms()
        ambiguous = await self.discover_ambiguous_terms()

        return {
            "merge_candidates_count": len(merge_candidates),
            "merge_candidates": [
                {
                    "source_id": str(c.source_id),
                    "source_term": c.source_term,
                    "target_id": str(c.target_id),
                    "target_term": c.target_term,
                    "facet": c.facet,
                    "similarity": round(c.similarity, 3),
                    "rationale": c.rationale,
                }
                for c in merge_candidates
            ],
            "split_candidates_count": len(split_candidates),
            "split_candidates": [
                {
                    "term_id": str(s.term_id),
                    "term": s.term,
                    "facet": s.facet,
                    "dispersion": round(s.dispersion, 3),
                    "proposed_subterms": s.proposed_subterms,
                    "rationale": s.rationale,
                }
                for s in split_candidates
            ],
            "orphaned_terms_count": len(orphans),
            "orphaned_terms": orphans,
            "ambiguous_terms_count": len(ambiguous),
            "ambiguous_terms": ambiguous,
        }
