"""Merge near-duplicate taxonomy categories/nodes (CLAUDE.md §8, §9).

When two categories or terms learned by braind have significant semantic overlap,
CategoryMerger merges them, reparents child terms, reassigns aliases, and records
the complete before/after audit state in the reorg journal.
Thresholds are read dynamically from settings (`selforg_merge_threshold`), never hardcoded.
"""

from __future__ import annotations

import difflib
import uuid
from dataclasses import dataclass
from typing import Any

import sqlalchemy as sa
from sqlalchemy import func, select, update

from core import database as db
from core.logging import get_logger
from core.textanalysis import normalize_label
from services.selforg.journal import OP_MERGE_TERMS, ReorgJournal
from services.settings.resolver import SettingsResolver
from services.settings.store import SettingsStore

logger = get_logger(__name__)


@dataclass(frozen=True)
class MergeCandidate:
    """A pair of terms proposed for merging."""

    source_id: uuid.UUID
    source_term: str
    target_id: uuid.UUID
    target_term: str
    facet: str
    similarity: float
    rationale: str


class CategoryMerger:
    """Detects and merges near-duplicate taxonomy categories."""

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
        """Read selforg_merge_threshold dynamically from settings."""
        if self.settings_resolver is not None:
            try:
                val, _ = await self.settings_resolver.resolve("selforg_merge_threshold")
                if val is not None:
                    return float(val)
            except Exception as exc:
                logger.debug("selforg.merge.resolve_setting_failed", error=str(exc))
        return 0.9

    def _compute_string_similarity(self, a: str, b: str) -> float:
        """Normalized string similarity accounting for underscores/hyphens."""
        norm_a = normalize_label(a).replace("_", " ").strip()
        norm_b = normalize_label(b).replace("_", " ").strip()
        if norm_a == norm_b:
            return 1.0

        matcher = difflib.SequenceMatcher(None, norm_a, norm_b)
        ratio = matcher.ratio()

        # Token set overlap bonus (e.g. "machine learning" vs "learning machine")
        tokens_a = set(norm_a.split())
        tokens_b = set(norm_b.split())
        if tokens_a and tokens_b:
            token_jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
            return max(ratio, token_jaccard)
        return ratio

    async def find_merge_candidates(
        self,
        *,
        facet: str | None = None,
        threshold: float | None = None,
        limit: int = 50,
    ) -> list[MergeCandidate]:
        """Scan active terms in each facet and identify pairs meeting the merge threshold."""
        target_threshold = threshold if threshold is not None else await self._get_threshold()

        async with self.session_factory() as session:
            stmt = select(db.TaxonomyTerm).where(
                db.TaxonomyTerm.status == "active",
                db.TaxonomyTerm.merged_into_id.is_(None),
            )
            if facet:
                stmt = stmt.where(db.TaxonomyTerm.facet == facet)
            terms = (await session.execute(stmt)).scalars().all()

        # Group terms by facet
        by_facet: dict[str, list[db.TaxonomyTerm]] = {}
        for t in terms:
            by_facet.setdefault(t.facet, []).append(t)

        candidates: list[MergeCandidate] = []
        for f_name, term_list in by_facet.items():
            n = len(term_list)
            for i in range(n):
                for j in range(i + 1, n):
                    t1 = term_list[i]
                    t2 = term_list[j]

                    sim = self._compute_string_similarity(t1.term, t2.term)
                    if sim >= target_threshold:
                        # Pick target: higher seen_count or older term wins
                        if (t2.seen_count > t1.seen_count) or (
                            t2.seen_count == t1.seen_count and t2.first_seen_at < t1.first_seen_at
                        ):
                            source, target = t1, t2
                        else:
                            source, target = t2, t1

                        rationale = (
                            f"High semantic similarity ({sim:.2f} >= {target_threshold}) "
                            f"between '{source.term}' and '{target.term}' in facet '{f_name}'"
                        )
                        candidates.append(
                            MergeCandidate(
                                source_id=source.id,
                                source_term=source.term,
                                target_id=target.id,
                                target_term=target.term,
                                facet=f_name,
                                similarity=sim,
                                rationale=rationale,
                            )
                        )
                        if len(candidates) >= limit:
                            return candidates

        return candidates

    async def merge_terms(
        self,
        source_id: uuid.UUID | str,
        target_id: uuid.UUID | str,
        *,
        rationale: str | None = None,
    ) -> uuid.UUID:
        """Execute a merge of source_term into target_term.

        Updates status, reparents children, reassigns aliases, and records audit state.
        """
        s_uuid = uuid.UUID(str(source_id)) if isinstance(source_id, str) else source_id
        t_uuid = uuid.UUID(str(target_id)) if isinstance(target_id, str) else target_id

        if s_uuid == t_uuid:
            raise ValueError("Cannot merge a term into itself")

        async with self.session_factory() as session:
            source = await session.get(db.TaxonomyTerm, s_uuid)
            target = await session.get(db.TaxonomyTerm, t_uuid)

            if source is None:
                raise ValueError(f"Source term {s_uuid} not found")
            if target is None:
                raise ValueError(f"Target term {t_uuid} not found")

            # Capture before state
            child_terms = (
                await session.execute(
                    select(db.TaxonomyTerm).where(db.TaxonomyTerm.parent_id == s_uuid)
                )
            ).scalars().all()

            aliases = (
                await session.execute(
                    select(db.TaxonomyAlias).where(db.TaxonomyAlias.term_id == s_uuid)
                )
            ).scalars().all()

            before_state = {
                "source_term_id": str(source.id),
                "source_term": source.term,
                "source_facet": source.facet,
                "source_status": source.status,
                "source_merged_into_id": str(source.merged_into_id) if source.merged_into_id else None,
                "source_seen_count": source.seen_count,
                "source_authority": source.authority,
                "target_term_id": str(target.id),
                "target_term": target.term,
                "target_status": target.status,
                "target_seen_count": target.seen_count,
                "target_authority": target.authority,
                "reparented_children": [
                    {"id": str(c.id), "term": c.term, "original_parent_id": str(s_uuid)}
                    for c in child_terms
                ],
                "reassigned_aliases": [
                    {"id": str(a.id), "surface_form": a.surface_form, "original_term_id": str(s_uuid)}
                    for a in aliases
                ],
            }

            # Apply merge changes
            source.status = "merged"
            source.merged_into_id = target.id
            target.seen_count += source.seen_count
            if source.authority is not None:
                if target.authority is None or source.authority > target.authority:
                    target.authority = source.authority

            # Reparent child terms to target
            for child in child_terms:
                child.parent_id = target.id

            # Reassign aliases to target
            for alias in aliases:
                alias.term_id = target.id

            # Create an alias for source term pointing to target if not exists
            source_norm = normalize_label(source.term)
            existing_alias = (
                await session.execute(
                    select(db.TaxonomyAlias).where(
                        db.TaxonomyAlias.normalized_form == source_norm,
                        db.TaxonomyAlias.term_id == target.id,
                    )
                )
            ).scalar_one_or_none()

            if existing_alias is None and source.term != target.term:
                session.add(
                    db.TaxonomyAlias(
                        id=uuid.uuid4(),
                        term_id=target.id,
                        surface_form=source.term,
                        normalized_form=source_norm,
                        locale=None,
                    )
                )

            after_state = {
                "source_term_id": str(source.id),
                "source_status": source.status,
                "source_merged_into_id": str(target.id),
                "target_term_id": str(target.id),
                "target_seen_count": target.seen_count,
                "target_authority": target.authority,
                "reparented_children_count": len(child_terms),
                "reassigned_aliases_count": len(aliases),
            }

            await session.commit()

        # Record operation in append-only journal
        merge_rationale = (
            rationale
            or f"Merged near-duplicate term '{source.term}' into '{target.term}' in facet '{target.facet}'"
        )
        op_id = await self.journal.record_operation(
            operation_type=OP_MERGE_TERMS,
            target_type="taxonomy_terms",
            target_id=str(target_id),
            before_state=before_state,
            after_state=after_state,
            rationale=merge_rationale,
        )

        logger.info(
            "selforg.merged_terms",
            op_id=str(op_id),
            source=str(s_uuid),
            target=str(t_uuid),
        )
        return op_id
