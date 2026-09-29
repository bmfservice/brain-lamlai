"""Split overly dispersed categories into sub-categories (CLAUDE.md §8, §9).

When a taxonomy category or term becomes overpopulated or broad (high semantic dispersion),
CategorySplitter evaluates and creates narrower sub-categories rooted at the parent term.
All dispersion thresholds are read dynamically from settings (`selforg_split_dispersion_threshold`).
"""

from __future__ import annotations

import datetime
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select

from core import database as db
from core.logging import get_logger
from core.textanalysis import normalize_label
from services.selforg.journal import OP_SPLIT_CATEGORY, ReorgJournal
from services.settings.resolver import SettingsResolver
from services.settings.store import SettingsStore

logger = get_logger(__name__)


@dataclass(frozen=True)
class SplitCandidate:
    """A broad or overpopulated category proposed for splitting."""

    term_id: uuid.UUID
    term: str
    facet: str
    seen_count: int
    dispersion: float
    proposed_subterms: list[str]
    rationale: str


class CategorySplitter:
    """Detects and splits overly dispersed taxonomy categories."""

    def __init__(
        self,
        session_factory,
        journal: ReorgJournal | None = None,
        *,
        settings_resolver: SettingsResolver | None = None,
        settings_store: SettingsStore | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.journal = journal or ReorgJournal(session_factory)
        self.settings_store = settings_store or (
            SettingsStore(session_factory) if session_factory else None
        )
        self.settings_resolver = settings_resolver or (
            SettingsResolver(session_factory, self.settings_store)
            if session_factory
            else None
        )

    async def _get_threshold(self) -> float:
        """Read selforg_split_dispersion_threshold dynamically from settings."""
        if self.settings_resolver is not None:
            try:
                val, _ = await self.settings_resolver.resolve("selforg_split_dispersion_threshold")
                if val is not None:
                    return float(val)
            except Exception as exc:
                logger.debug("selforg.split.resolve_setting_failed", error=str(exc))
        return 0.7

    async def find_split_candidates(
        self,
        *,
        facet: str | None = None,
        min_seen_count: int = 5,
        threshold: float | None = None,
        limit: int = 20,
    ) -> list[SplitCandidate]:
        """Find overpopulated terms with high dispersion or multi-token breadth."""
        dispersion_threshold = (
            threshold if threshold is not None else await self._get_threshold()
        )

        async with self.session_factory() as session:
            stmt = select(db.TaxonomyTerm).where(
                db.TaxonomyTerm.status == "active",
                db.TaxonomyTerm.merged_into_id.is_(None),
                db.TaxonomyTerm.seen_count >= min_seen_count,
            )
            if facet:
                stmt = stmt.where(db.TaxonomyTerm.facet == facet)
            terms = (await session.execute(stmt)).scalars().all()

            candidates: list[SplitCandidate] = []
            for t in terms:
                # Count existing child subterms
                children_count = (
                    await session.execute(
                        select(func.count(db.TaxonomyTerm.id)).where(
                            db.TaxonomyTerm.parent_id == t.id
                        )
                    )
                ).scalar() or 0

                # Compute structural dispersion based on token complexity and usage
                tokens = [p for p in t.term.replace("_", " ").split() if p]
                token_count = len(tokens)

                # Base dispersion: multi-token categories with high usage and few children
                # have high dispersion (need branching)
                dispersion = min(
                    1.0,
                    (token_count * 0.25) + (min(t.seen_count, 50) / 100.0) - (children_count * 0.1),
                )

                if dispersion >= dispersion_threshold:
                    # Propose subterms from token components if multi-word
                    proposed: list[str] = []
                    if token_count > 1:
                        for token in tokens:
                            sub_term = f"{t.term}_{token}" if len(token) > 2 else f"{t.term}_sub"
                            proposed.append(normalize_label(sub_term))
                    else:
                        proposed = [f"{t.term}_general", f"{t.term}_specific"]

                    rationale = (
                        f"Category '{t.term}' (facet: {t.facet}) has seen_count={t.seen_count} "
                        f"and dispersion={dispersion:.2f} >= {dispersion_threshold:.2f}"
                    )
                    candidates.append(
                        SplitCandidate(
                            term_id=t.id,
                            term=t.term,
                            facet=t.facet,
                            seen_count=t.seen_count,
                            dispersion=dispersion,
                            proposed_subterms=proposed,
                            rationale=rationale,
                        )
                    )
                    if len(candidates) >= limit:
                        break

        return candidates

    async def split_category(
        self,
        term_id: uuid.UUID | str,
        subterms: list[str],
        *,
        rationale: str | None = None,
    ) -> uuid.UUID:
        """Split a parent term by creating child terms rooted at parent_id.

        Records before/after states in the reorg journal.
        """
        p_uuid = uuid.UUID(str(term_id)) if isinstance(term_id, str) else term_id

        if not subterms:
            raise ValueError("subterms list cannot be empty")

        now = datetime.datetime.now(datetime.UTC)
        created_ids: list[uuid.UUID] = []

        async with self.session_factory() as session:
            parent = await session.get(db.TaxonomyTerm, p_uuid)
            if parent is None:
                raise ValueError(f"Parent term {p_uuid} not found")

            # Capture before state
            existing_children = (
                await session.execute(
                    select(db.TaxonomyTerm).where(db.TaxonomyTerm.parent_id == p_uuid)
                )
            ).scalars().all()

            before_state = {
                "parent_id": str(parent.id),
                "parent_term": parent.term,
                "facet": parent.facet,
                "seen_count": parent.seen_count,
                "existing_children": [
                    {"id": str(c.id), "term": c.term} for c in existing_children
                ],
            }

            # Create subterms
            for st in subterms:
                norm_sub = normalize_label(st)
                if not norm_sub:
                    continue

                sub_id = uuid.uuid4()
                created_ids.append(sub_id)

                subterm_obj = db.TaxonomyTerm(
                    id=sub_id,
                    facet=parent.facet,
                    term=norm_sub,
                    parent_id=parent.id,
                    origin="derived",
                    confidence=parent.confidence or 0.85,
                    status="active",
                    authority=parent.authority,
                    first_seen_at=now,
                    last_seen_at=now,
                    seen_count=0,
                )
                session.add(subterm_obj)

            after_state = {
                "parent_id": str(parent.id),
                "parent_term": parent.term,
                "created_subterm_ids": [str(sid) for sid in created_ids],
                "created_subterms": subterms,
            }

            await session.commit()

        # Record operation in reorg journal
        split_rationale = (
            rationale
            or f"Split broad category '{parent.term}' into {len(created_ids)} sub-categories"
        )
        op_id = await self.journal.record_operation(
            operation_type=OP_SPLIT_CATEGORY,
            target_type="taxonomy_terms",
            target_id=str(p_uuid),
            before_state=before_state,
            after_state=after_state,
            rationale=split_rationale,
        )

        logger.info(
            "selforg.split_category",
            op_id=str(op_id),
            parent_id=str(p_uuid),
            created_count=len(created_ids),
        )
        return op_id
