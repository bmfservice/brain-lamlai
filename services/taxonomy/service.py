"""Empty-start, DB-backed dynamic taxonomy (CLAUDE.md 2.3, 6.2, 9).

Every label that used to live in a frozen ``frozenset`` in ``core/taxonomy.py`` is
resolved here against ``taxonomy_terms`` instead, so braind can mint a new
category the first time it meets a kind of content it has never seen.

Resolution chain for :meth:`TaxonomyService.resolve` (plan §3.2):

1. normalize the incoming label (``core.textanalysis.normalize_label``);
2. match an **active** term in the facet;
3. match an **alias** pointing at an active term (locale-aware);
4. apply enabled **rules** for the facet (``synonym`` / ``regex`` kinds map text
   onto terms; other kinds are consumed as text-analysis rules);
5. if the caller's model confidence is at or above ``taxonomy_propose_threshold``,
   insert (or re-visit) a ``proposed`` term and auto-promote it once
   ``seen_count`` reaches ``taxonomy_promote_min_hits``;
6. otherwise record the label as ``ambiguous`` and hand it to the quarantine
   service so nothing is silently dropped or invented as a de-facto category.

There is **no fallback default anywhere**: every threshold is read through the
settings layer, and an unknown setting key raises rather than silently using a
literal. Values are never invented for labels the brain cannot place — the
caller keeps the raw text and the resolution status side by side.
"""

from __future__ import annotations

import asyncio
import datetime
import re
import uuid
from typing import Any, Iterable

import sqlalchemy as sa
from sqlalchemy import func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from core import database as db
from core.logging import get_logger
from core.textanalysis import normalize_label
from services.maintenance.quarantine import QuarantineService
from services.settings.cache import SettingsCache
from services.settings.resolver import SettingsResolver
from services.settings.store import SettingsStore
from services.taxonomy.models import (
    ORIGIN_MODEL,
    REASON_AMBIGUOUS,
    STATUS_ACTIVE,
    STATUS_AMBIGUOUS,
    STATUS_PROPOSED,
    TaxonomyResolution,
)

logger = get_logger(__name__)

# ``TermResolution`` is the name the approved plan uses for the result object.
TermResolution = TaxonomyResolution

# Settings keys this service depends on. They exist as catalog definitions so an
# operator can retune taxonomy behaviour from the Settings UI without a restart
# (CLAUDE.md §9); listing them here keeps the key names in one place.
KEY_PROPOSE_THRESHOLD = "taxonomy_propose_threshold"
KEY_PROMOTE_MIN_HITS = "taxonomy_promote_min_hits"
KEY_ACCEPTANCE_CONFIDENCE = "taxonomy_acceptance_confidence"
KEY_AMBIGUITY_CONFIDENCE = "taxonomy_ambiguity_confidence"

# Rule kinds that map text onto terms during resolution. Every other kind
# (stopwords, language hints, structure markers, query suffixes) is free-form
# text-analysis data returned by :meth:`get_rules`.
TERM_MAPPING_KINDS: tuple[str, ...] = ("synonym", "regex")

# Facet names used across the pipeline. These are *names of dimensions*, not
# classification vocabularies: the labels inside each dimension are learned.
FACET_DOC_TYPE = "doc_type"
FACET_CHUNK_ROLE = "chunk_role"
FACET_ENTITY_TYPE = "entity_type"
FACET_AUDIENCE = "audience"
FACET_QUALITY_TIER = "quality_tier"
FACET_RELATION_TYPE = "relation_type"
FACET_TOPIC = "topic"


class TaxonomyService:
    """Empty-start dynamic taxonomy over ``taxonomy_*`` tables."""

    def __init__(
        self,
        session_factory,
        *,
        settings_store: SettingsStore | None = None,
        settings_resolver: SettingsResolver | None = None,
        settings_cache: SettingsCache | None = None,
        quarantine_service: QuarantineService | None = None,
    ) -> None:
        self.session_factory = session_factory
        # The settings layer is the only legal source of a threshold, so it is
        # built here rather than injected as optional: a taxonomy service without
        # settings would have to invent numbers.
        self.settings_store = settings_store or SettingsStore(session_factory)
        self.settings_resolver = settings_resolver or SettingsResolver(
            session_factory, self.settings_store
        )
        self.settings_cache = settings_cache
        self.quarantine_service = quarantine_service or QuarantineService(session_factory)

    # --- settings ---------------------------------------------------------

    async def _get_setting(self, key: str) -> Any:
        """Read one setting through cache -> resolver -> definition default.

        Raises if the key is not in the catalog: an absent threshold is a
        configuration error, never an excuse for a literal in business logic.
        """
        if self.settings_cache is not None:
            cached = self.settings_cache.get(key)
            if cached is not None:
                return cached

        value, _scope = await self.settings_resolver.resolve(key)
        if value is None:
            raise ValueError(f"setting {key!r} resolved to None; taxonomy accepts no default")
        if self.settings_cache is not None:
            self.settings_cache.set(key, value)
        return value

    async def _float_setting(self, key: str) -> float:
        return float(await self._get_setting(key))

    async def _int_setting(self, key: str) -> int:
        return int(await self._get_setting(key))

    # --- facets -----------------------------------------------------------

    async def ensure_facet(
        self,
        facet: str,
        label: str = "",
        *,
        growable: bool = True,
        authority_editable: bool = True,
    ) -> str:
        """Register a classification dimension if absent; return its normalized name.

        Creating a facet is how "tạo node loại nếu chưa có" happens for a brand
        new dimension of analysis — it is derived from the pipeline asking for it,
        never from a seeded list.
        """
        norm_facet = normalize_label(facet)
        if not norm_facet:
            return ""

        display_label = label.strip() if label else norm_facet.replace("_", " ").title()
        now = datetime.datetime.now(datetime.UTC)

        async with self.session_factory() as session:
            await session.execute(
                pg_insert(db.TaxonomyFacet)
                .values(
                    facet=norm_facet,
                    label=display_label,
                    growable=growable,
                    authority_editable=authority_editable,
                    created_at=now,
                    updated_at=now,
                )
                .on_conflict_do_nothing(index_elements=["facet"])
            )
            await session.commit()
        return norm_facet

    async def list_facets(self) -> list[dict[str, Any]]:
        """Every dimension braind currently classifies along, with its vocabulary size."""
        async with self.session_factory() as session:
            rows = (
                await session.execute(
                    sa.select(
                        db.TaxonomyFacet.facet,
                        db.TaxonomyFacet.label,
                        db.TaxonomyFacet.growable,
                        db.TaxonomyFacet.authority_editable,
                        func.count(db.TaxonomyTerm.id)
                        .filter(db.TaxonomyTerm.status == STATUS_ACTIVE)
                        .label("active_terms"),
                    )
                    .outerjoin(db.TaxonomyTerm, db.TaxonomyTerm.facet == db.TaxonomyFacet.facet)
                    .group_by(
                        db.TaxonomyFacet.facet,
                        db.TaxonomyFacet.label,
                        db.TaxonomyFacet.growable,
                        db.TaxonomyFacet.authority_editable,
                    )
                    .order_by(db.TaxonomyFacet.facet)
                )
            ).all()
        return [
            {
                "facet": r.facet,
                "label": r.label,
                "growable": r.growable,
                "authority_editable": r.authority_editable,
                "active_terms": r.active_terms,
            }
            for r in rows
        ]

    # --- core resolution --------------------------------------------------

    async def resolve(
        self,
        facet: str,
        value: object,
        *,
        confidence: float | None = None,
        evidence: str | None = None,
        chunk_id: uuid.UUID | str | None = None,
        locale: str | None = None,
        origin: str = ORIGIN_MODEL,
        quarantine: bool = True,
    ) -> TaxonomyResolution:
        """Place ``value`` inside ``facet`` without a closed vocabulary.

        Args:
            facet: dimension name (``doc_type``, ``entity_type``, ...).
            value: raw model output or user label; any case/spacing/hyphenation.
            confidence: the model's own confidence in ``value``. ``None`` means
                unmeasured: the term is still proposed, but its ``confidence``
                column stays NULL so it is never mistaken for a measurement.
            evidence: the span/reason that justifies the label (stored with the
                quarantined record and logged, so a human can review the call).
            chunk_id: the chunk the label came from, for provenance.
            locale: language of ``value``, used to prefer a locale-specific alias.
            quarantine: when False, an ambiguous label is recorded but not pushed
                to the quarantine table (callers that already gate on it).

        Returns:
            :class:`TaxonomyResolution` carrying the resolved term, its lifecycle
            status and the confidence actually used. ``term`` is always the
            normalized incoming label when nothing matched, so a caller can keep
            the model's own wording rather than a fabricated default.
        """
        norm_facet = normalize_label(facet)
        norm_value = normalize_label(value)

        if not norm_facet:
            raise ValueError("resolve() requires a non-empty facet")
        if not norm_value:
            # Nothing to place: report it as ambiguous without inventing a label.
            return TaxonomyResolution(
                term="",
                facet=norm_facet,
                status=STATUS_AMBIGUOUS,
                confidence=0.0,
                origin=ORIGIN_MODEL,
            )

        await self.ensure_facet(norm_facet)
        doc_id = uuid.UUID(str(chunk_id)) if chunk_id is not None else None

        async with self.session_factory() as session:
            existing = (
                await session.execute(
                    select(db.TaxonomyTerm).where(
                        db.TaxonomyTerm.facet == norm_facet,
                        db.TaxonomyTerm.term == norm_value,
                    )
                )
            ).scalar_one_or_none()

            # 1. exact active-term match
            if existing is not None and existing.status == STATUS_ACTIVE:
                return await self._bump(session, existing, resolution_confidence=1.0)

            # 2. alias -> active term
            alias_hit = await self._match_alias(session, norm_facet, norm_value, locale)
            if alias_hit is not None:
                return await self._bump(session, alias_hit, resolution_confidence=1.0)

            # 3. facet rules -> term
            rule_hit = await self._match_rules(session, norm_facet, norm_value, locale)
            if rule_hit is not None:
                return await self._bump(session, rule_hit, resolution_confidence=1.0)

            # 4. a proposed/ambiguous term with the same name still counts as a
            #    sighting: it accumulates towards the promote threshold.
            if existing is not None and existing.status in (STATUS_PROPOSED, STATUS_AMBIGUOUS):
                return await self._revisit(session, existing, confidence=confidence)

            # 5. brand new label -> propose or quarantine, gated by settings.
            propose_threshold = await self._float_setting(KEY_PROPOSE_THRESHOLD)
            unmeasured = confidence is None
            eff_confidence = float(propose_threshold if unmeasured else confidence)

            if eff_confidence >= propose_threshold:
                return await self._propose(
                    session,
                    facet=norm_facet,
                    term=norm_value,
                    confidence=None if unmeasured else eff_confidence,
                    reported=eff_confidence,
                    chunk_id=doc_id,
                    origin=origin,
                )

            return await self._mark_ambiguous(
                session,
                facet=norm_facet,
                term=norm_value,
                confidence=eff_confidence,
                evidence=evidence,
                chunk_id=doc_id,
                origin=origin,
                quarantine=quarantine,
                propose_threshold=propose_threshold,
            )

    async def resolve_many(
        self,
        facet: str,
        values: Iterable[object],
        **kwargs: Any,
    ) -> list[TaxonomyResolution]:
        """Resolve a batch of labels concurrently (CLAUDE.md 2.4).

        Results keep input order. Failures degrade to an ambiguous resolution
        for that one label instead of discarding the whole batch.
        """
        items = list(values)
        if not items:
            return []
        results = await asyncio.gather(
            *(self.resolve(facet, v, **kwargs) for v in items), return_exceptions=True
        )
        norm_facet = normalize_label(facet)
        out: list[TaxonomyResolution] = []
        for value, res in zip(items, results, strict=True):
            if isinstance(res, BaseException):
                logger.warning("taxonomy.resolve_failed", facet=norm_facet, error=str(res))
                out.append(
                    TaxonomyResolution(
                        term=normalize_label(value),
                        facet=norm_facet,
                        status=STATUS_AMBIGUOUS,
                        confidence=0.0,
                    )
                )
            else:
                out.append(res)
        return out

    # --- resolution internals --------------------------------------------

    async def _bump(
        self,
        session,
        term: db.TaxonomyTerm,
        *,
        resolution_confidence: float,
    ) -> TaxonomyResolution:
        """Record one more sighting of a term we already know."""
        await session.execute(
            update(db.TaxonomyTerm)
            .where(db.TaxonomyTerm.id == term.id)
            .values(seen_count=db.TaxonomyTerm.seen_count + 1, last_seen_at=func.now())
        )
        await session.commit()
        return TaxonomyResolution(
            term=term.term,
            facet=term.facet,
            status=term.status,
            confidence=resolution_confidence,
            authority=term.authority,
            origin=term.origin,
            term_id=term.id,
        )

    async def _revisit(
        self,
        session,
        term: db.TaxonomyTerm,
        *,
        confidence: float | None,
    ) -> TaxonomyResolution:
        """Re-observe a not-yet-active term; promote it once it has enough hits."""
        promote_min_hits = await self._int_setting(KEY_PROMOTE_MIN_HITS)
        seen = (term.seen_count or 0) + 1
        values: dict[str, Any] = {"seen_count": seen, "last_seen_at": func.now()}
        if confidence is not None:
            # Keep the strongest measured confidence for this label.
            values["confidence"] = max(term.confidence or 0.0, float(confidence))

        promoted = term.status != STATUS_ACTIVE and seen >= promote_min_hits
        if promoted:
            values["status"] = STATUS_ACTIVE

        await session.execute(update(db.TaxonomyTerm).where(db.TaxonomyTerm.id == term.id).values(**values))
        await session.commit()

        if promoted:
            logger.info(
                "taxonomy.term_promoted",
                term_id=str(term.id),
                facet=term.facet,
                term=term.term,
                seen_count=seen,
                reason="min_hits",
            )

        return TaxonomyResolution(
            term=term.term,
            facet=term.facet,
            status=STATUS_ACTIVE if promoted else term.status,
            confidence=float(confidence) if confidence is not None else (term.confidence or 0.0),
            authority=term.authority,
            origin=term.origin,
            term_id=term.id,
        )

    async def _propose(
        self,
        session,
        *,
        facet: str,
        term: str,
        confidence: float | None,
        reported: float,
        chunk_id: uuid.UUID | None,
        origin: str,
    ) -> TaxonomyResolution:
        """Insert a new ``proposed`` term, or promote it straight away if allowed."""
        promote_min_hits = await self._int_setting(KEY_PROMOTE_MIN_HITS)
        acceptance = await self._float_setting(KEY_ACCEPTANCE_CONFIDENCE)
        measured = None if confidence is None else float(confidence)

        # A label the model is already sure enough about to clear the acceptance
        # bar enters the vocabulary directly; anything less must earn its place
        # through repeated sightings.
        direct = measured is not None and measured >= acceptance
        initial_status = STATUS_ACTIVE if direct else STATUS_PROPOSED
        initial_hits = promote_min_hits if direct else 1
        new_id = uuid.uuid4()

        res = (
            (
                await session.execute(
                    pg_insert(db.TaxonomyTerm)
                    .values(
                        id=new_id,
                        facet=facet,
                        term=term,
                        origin=origin,
                        confidence=measured,
                        status=initial_status,
                        created_from_chunk_id=chunk_id,
                        seen_count=initial_hits,
                    )
                    .on_conflict_do_update(
                        index_elements=["facet", "term"],
                        set_={
                            "seen_count": db.TaxonomyTerm.seen_count + 1,
                            "last_seen_at": func.now(),
                            "confidence": func.coalesce(
                                pg_insert(db.TaxonomyTerm).excluded.confidence,
                                db.TaxonomyTerm.confidence,
                            ),
                        },
                    )
                    .returning(
                        db.TaxonomyTerm.id,
                        db.TaxonomyTerm.status,
                        db.TaxonomyTerm.authority,
                        db.TaxonomyTerm.seen_count,
                    )
                )
            )
            .fetchone()
        )
        await session.commit()

        term_id, status, authority, seen = res[0], res[1], res[2], res[3]

        # A quarantined label whose re-appearances now clear the bar is revived.
        if status == STATUS_AMBIGUOUS and seen >= promote_min_hits:
            await session.execute(
                update(db.TaxonomyTerm)
                .where(db.TaxonomyTerm.id == term_id)
                .values(status=STATUS_ACTIVE)
            )
            await session.commit()
            status = STATUS_ACTIVE
            logger.info(
                "taxonomy.term_promoted",
                term_id=str(term_id),
                facet=facet,
                term=term,
                seen_count=seen,
                reason="min_hits_after_ambiguity",
            )

        logger.info(
            "taxonomy.term_proposed",
            term_id=str(term_id),
            facet=facet,
            term=term,
            status=status,
            confidence=reported,
            seen_count=seen,
        )
        return TaxonomyResolution(
            term=term,
            facet=facet,
            status=status,
            confidence=reported,
            authority=authority,
            is_new=True,
            origin=origin,
            term_id=term_id,
        )

    async def _mark_ambiguous(
        self,
        session,
        *,
        facet: str,
        term: str,
        confidence: float,
        evidence: str | None,
        chunk_id: uuid.UUID | None,
        origin: str,
        quarantine: bool,
        propose_threshold: float,
    ) -> TaxonomyResolution:
        """Record a label that could not be placed, and flag it for review."""
        new_id = uuid.uuid4()
        res = (
            (
                await session.execute(
                    pg_insert(db.TaxonomyTerm)
                    .values(
                        id=new_id,
                        facet=facet,
                        term=term,
                        origin=origin,
                        confidence=confidence,
                        status=STATUS_AMBIGUOUS,
                        created_from_chunk_id=chunk_id,
                        seen_count=1,
                    )
                    .on_conflict_do_update(
                        index_elements=["facet", "term"],
                        set_={
                            "seen_count": db.TaxonomyTerm.seen_count + 1,
                            "last_seen_at": func.now(),
                        },
                    )
                    .returning(db.TaxonomyTerm.id)
                )
            )
            .fetchone()
        )
        await session.commit()
        term_id = res[0] if res else new_id

        logger.warning(
            "taxonomy.term_ambiguous",
            facet=facet,
            term=term,
            confidence=confidence,
            threshold=propose_threshold,
            evidence=evidence,
        )

        quarantine_id: uuid.UUID | None = None
        if quarantine and self.quarantine_service is not None:
            try:
                quarantine_id = await self.quarantine_service.quarantine_item(
                    table_name="taxonomy_terms",
                    row_id=str(term_id),
                    reason=REASON_AMBIGUOUS,
                    detail=(
                        f"label {term!r} for facet {facet!r} scored {confidence:.3f}, "
                        f"below {KEY_PROPOSE_THRESHOLD}={propose_threshold:.3f}"
                    ),
                    payload={
                        "facet": facet,
                        "term": term,
                        "confidence": confidence,
                        "evidence": evidence,
                        "chunk_id": str(chunk_id) if chunk_id else None,
                    },
                )
            except Exception as exc:  # quarantine must never swallow the label
                logger.error("taxonomy.quarantine_failed", error=str(exc))

        return TaxonomyResolution(
            term=term,
            facet=facet,
            status=STATUS_AMBIGUOUS,
            confidence=confidence,
            is_new=True,
            origin=origin,
            term_id=term_id,
            quarantine_id=quarantine_id,
        )

    async def _match_alias(
        self,
        session,
        facet: str,
        normalized: str,
        locale: str | None,
    ) -> db.TaxonomyTerm | None:
        """Follow an alias to its active term, preferring a locale-specific alias."""
        stmt = (
            select(db.TaxonomyAlias, db.TaxonomyTerm)
            .join(db.TaxonomyTerm, db.TaxonomyAlias.term_id == db.TaxonomyTerm.id)
            .where(
                db.TaxonomyTerm.facet == facet,
                db.TaxonomyAlias.normalized_form == normalized,
                db.TaxonomyTerm.status == STATUS_ACTIVE,
            )
        )
        rows = (await session.execute(stmt)).all()
        if not rows:
            return None
        if locale:
            for alias, term in rows:
                if alias.locale == locale:
                    return term
        for alias, term in rows:
            if alias.locale is None:
                return term
        return rows[0][1]

    async def _match_rules(
        self,
        session,
        facet: str,
        value: str,
        locale: str | None,
    ) -> db.TaxonomyTerm | None:
        """Apply term-mapping rules (``synonym`` / ``regex``) for the facet."""
        stmt = select(db.TaxonomyRule).where(
            db.TaxonomyRule.facet == facet,
            db.TaxonomyRule.enabled.is_(True),
            db.TaxonomyRule.kind.in_(TERM_MAPPING_KINDS),
        )
        if locale:
            stmt = stmt.where(
                or_(db.TaxonomyRule.locale.is_(None), db.TaxonomyRule.locale == locale)
            )
        rules = (await session.execute(stmt)).scalars().all()
        if not rules:
            return None

        target: str | None = None
        matched_rule: uuid.UUID | None = None
        for rule in rules:
            if rule.kind == "synonym":
                source, _, dest = rule.pattern.partition("->")
                if normalize_label(source) != value:
                    continue
                target = normalize_label(dest) or None
            else:  # regex: pattern -> term, or a pattern that names itself
                source, _, dest = rule.pattern.partition("=>")
                expression = source.strip()
                try:
                    if not re.search(expression, value, re.IGNORECASE):
                        continue
                except re.error:
                    logger.debug("taxonomy.rule_uncompilable", rule_id=str(rule.id))
                    continue
                target = normalize_label(dest) or None

            matched_rule = rule.id
            break

        if matched_rule is not None:
            await session.execute(
                update(db.TaxonomyRule)
                .where(db.TaxonomyRule.id == matched_rule)
                .values(hits=db.TaxonomyRule.hits + 1, updated_at=func.now())
            )
            await session.commit()

        if not target:
            return None

        term = (
            await session.execute(
                select(db.TaxonomyTerm).where(
                    db.TaxonomyTerm.facet == facet,
                    db.TaxonomyTerm.term == target,
                    db.TaxonomyTerm.status == STATUS_ACTIVE,
                )
            )
        ).scalar_one_or_none()
        return term

    # --- promotion / authority --------------------------------------------

    async def promote_term(
        self,
        term_id: uuid.UUID | str,
        *,
        min_hits: int | None = None,
        manual: bool = False,
    ) -> bool:
        """Promote a proposed/ambiguous term to ``active``.

        Automatic promotion requires ``seen_count`` to have reached
        ``taxonomy_promote_min_hits``; ``manual=True`` is the operator override
        from the Settings UI and is logged as such.
        """
        tid = uuid.UUID(str(term_id))
        threshold = (
            min_hits
            if min_hits is not None
            else await self._int_setting(KEY_PROMOTE_MIN_HITS)
        )

        async with self.session_factory() as session:
            term = (
                await session.execute(select(db.TaxonomyTerm).where(db.TaxonomyTerm.id == tid))
            ).scalar_one_or_none()
            if term is None:
                return False
            if term.status == STATUS_ACTIVE:
                return True
            if not manual and (term.seen_count or 0) < threshold:
                return False

            await session.execute(
                update(db.TaxonomyTerm)
                .where(db.TaxonomyTerm.id == tid)
                .values(status=STATUS_ACTIVE)
            )
            await session.commit()

        logger.info(
            "taxonomy.term_promoted",
            term_id=str(tid),
            facet=term.facet,
            term=term.term,
            manual=manual,
            seen_count=term.seen_count,
        )
        return True

    async def set_authority(self, term_id: uuid.UUID | str, authority: float) -> bool:
        """Write a term's authority when its facet allows it to be learned/edited."""
        tid = uuid.UUID(str(term_id))
        async with self.session_factory() as session:
            term = (
                await session.execute(select(db.TaxonomyTerm).where(db.TaxonomyTerm.id == tid))
            ).scalar_one_or_none()
            if term is None:
                return False
            facet = (
                await session.execute(
                    select(db.TaxonomyFacet).where(db.TaxonomyFacet.facet == term.facet)
                )
            ).scalar_one_or_none()
            if facet is not None and not facet.authority_editable:
                return False
            await session.execute(
                update(db.TaxonomyTerm).where(db.TaxonomyTerm.id == tid).values(authority=authority)
            )
            await session.commit()
        return True

    async def get_active_terms(self, facet: str, limit: int = 200) -> list[str]:
        """Active labels for a facet, hottest first — the learned vocabulary."""
        norm_facet = normalize_label(facet)
        if not norm_facet:
            return []
        async with self.session_factory() as session:
            stmt = (
                select(db.TaxonomyTerm.term)
                .where(
                    db.TaxonomyTerm.facet == norm_facet,
                    db.TaxonomyTerm.status == STATUS_ACTIVE,
                )
                .order_by(db.TaxonomyTerm.seen_count.desc(), db.TaxonomyTerm.term.asc())
                .limit(limit)
            )
            return list((await session.execute(stmt)).scalars().all())

    async def get_term_authority(self, facet: str, term: str) -> float | None:
        """Authority recorded for one learned label, or ``None`` when unknown.

        ``None`` is meaningful and must be preserved by callers: it means braind
        has not learned anything about this label yet, which is not the same as a
        neutral middle-of-the-scale guess.
        """
        norm_facet = normalize_label(facet)
        norm_term = normalize_label(term)
        if not norm_facet or not norm_term:
            return None
        async with self.session_factory() as session:
            return (
                await session.execute(
                    select(db.TaxonomyTerm.authority).where(
                        db.TaxonomyTerm.facet == norm_facet,
                        db.TaxonomyTerm.term == norm_term,
                    )
                )
            ).scalar_one_or_none()

    async def authority_map(self, facet: str) -> dict[str, float | None]:
        """Every learned label for a facet with its recorded authority."""
        norm_facet = normalize_label(facet)
        async with self.session_factory() as session:
            rows = (
                await session.execute(
                    select(db.TaxonomyTerm.term, db.TaxonomyTerm.authority).where(
                        db.TaxonomyTerm.facet == norm_facet,
                        db.TaxonomyTerm.status == STATUS_ACTIVE,
                    )
                )
            ).all()
        return {r.term: r.authority for r in rows}

    # --- aliases & rules --------------------------------------------------

    async def add_alias(
        self,
        term_id: uuid.UUID | str,
        surface_form: str,
        locale: str | None = None,
    ) -> uuid.UUID:
        """Attach a surface form to a term (the DB equivalent of the old alias dicts)."""
        tid = uuid.UUID(str(term_id))
        normalized = normalize_label(surface_form)
        alias_id = uuid.uuid4()
        if not normalized:
            return alias_id

        async with self.session_factory() as session:
            res = (
                await session.execute(
                    pg_insert(db.TaxonomyAlias)
                    .values(
                        id=alias_id,
                        term_id=tid,
                        surface_form=str(surface_form).strip(),
                        normalized_form=normalized,
                        locale=locale,
                    )
                    .on_conflict_do_update(
                        index_elements=["normalized_form", sa.text("COALESCE(locale, '')")],
                        set_={"term_id": tid, "surface_form": str(surface_form).strip()},
                    )
                    .returning(db.TaxonomyAlias.id)
                )
            ).fetchone()
            await session.commit()
        return res[0] if res else alias_id

    async def add_rule(
        self,
        facet: str,
        kind: str,
        pattern: str,
        *,
        locale: str | None = None,
        origin: str = ORIGIN_MODEL,
        enabled: bool = True,
    ) -> uuid.UUID:
        """Store a reusable text rule (stopword, marker, hint, synonym, regex)."""
        norm_facet = normalize_label(facet)
        now = datetime.datetime.now(datetime.UTC)
        rule_id = uuid.uuid4()
        if not norm_facet or not pattern.strip():
            return rule_id

        async with self.session_factory() as session:
            await session.execute(
                pg_insert(db.TaxonomyFacet)
                .values(
                    facet=norm_facet,
                    label=norm_facet.replace("_", " ").title(),
                    growable=True,
                    authority_editable=True,
                    created_at=now,
                    updated_at=now,
                )
                .on_conflict_do_nothing(index_elements=["facet"])
            )
            res = (
                await session.execute(
                    pg_insert(db.TaxonomyRule)
                    .values(
                        id=rule_id,
                        facet=norm_facet,
                        kind=kind,
                        pattern=pattern.strip(),
                        locale=locale,
                        origin=origin,
                        enabled=enabled,
                        hits=0,
                        created_at=now,
                        updated_at=now,
                    )
                    .on_conflict_do_update(
                        index_elements=[
                            "facet",
                            "kind",
                            "pattern",
                            sa.text("COALESCE(locale, '')"),
                        ],
                        set_={"enabled": enabled, "updated_at": now},
                    )
                    .returning(db.TaxonomyRule.id)
                )
            ).fetchone()
            await session.commit()
        return res[0] if res else rule_id

    async def get_rules(
        self,
        facet: str,
        kind: str | None = None,
        locale: str | None = None,
    ) -> list[dict[str, Any]]:
        """Enabled text rules for a facet — the only legal source for the
        stopwords, language hints, change markers and query suffixes that used to
        be tuples in module code."""
        norm_facet = normalize_label(facet)
        async with self.session_factory() as session:
            stmt = select(db.TaxonomyRule).where(
                db.TaxonomyRule.facet == norm_facet,
                db.TaxonomyRule.enabled.is_(True),
            )
            if kind:
                stmt = stmt.where(db.TaxonomyRule.kind == kind)
            if locale:
                stmt = stmt.where(
                    or_(db.TaxonomyRule.locale.is_(None), db.TaxonomyRule.locale == locale)
                )
            rows = (await session.execute(stmt.order_by(db.TaxonomyRule.hits.desc()))).scalars().all()
        return [
            {
                "id": str(r.id),
                "facet": r.facet,
                "kind": r.kind,
                "pattern": r.pattern,
                "locale": r.locale,
                "origin": r.origin,
                "hits": r.hits,
            }
            for r in rows
        ]

    async def get_patterns(
        self,
        facet: str,
        kind: str,
        locale: str | None = None,
    ) -> list[str]:
        """Just the pattern strings for one rule kind (the common call shape)."""
        return [r["pattern"] for r in await self.get_rules(facet, kind=kind, locale=locale)]

    async def term_patterns(
        self,
        term: str,
        *,
        facet: str,
        kind: str,
        locale: str | None = None,
    ) -> list[str]:
        """Rules scoped to a single learned term, e.g. markers for one freshness bucket.

        Terms that are still ``proposed`` count too: a category braind just
        learned must be usable immediately, not only after promotion.
        """
        norm_facet = normalize_label(facet)
        norm_term = normalize_label(term)
        async with self.session_factory() as session:
            stmt = select(db.TaxonomyRule.pattern).join(
                db.TaxonomyTerm,
                sa.and_(
                    db.TaxonomyTerm.facet == db.TaxonomyRule.facet,
                    db.TaxonomyTerm.term == norm_term,
                ),
            ).where(
                db.TaxonomyRule.facet == norm_facet,
                db.TaxonomyRule.kind == kind,
                db.TaxonomyRule.enabled.is_(True),
                db.TaxonomyTerm.status.in_([STATUS_ACTIVE, STATUS_PROPOSED]),
            )
            if locale:
                stmt = stmt.where(
                    or_(db.TaxonomyRule.locale.is_(None), db.TaxonomyRule.locale == locale)
                )
            return list((await session.execute(stmt)).scalars().all())

    async def matches_any(
        self,
        text: str,
        *,
        facet: str,
        kind: str,
        locale: str | None = None,
    ) -> bool:
        """True when any enabled regex rule of ``kind`` fires on ``text``."""
        if not text:
            return False
        for pattern in await self.get_patterns(facet, kind, locale=locale):
            try:
                if re.search(pattern, text, re.IGNORECASE):
                    return True
            except re.error:
                logger.debug("taxonomy.pattern_uncompilable", pattern=pattern[:80])
        return False

    async def terms_matching(
        self,
        facet: str,
        kind: str,
        text: str,
        *,
        locale: str | None = None,
    ) -> list[str]:
        """Active terms whose ``kind`` regex fires on ``text``.

        ``pattern`` is stored as ``<regex>=>=<term>``; a pattern without ``=>``
        contributes nothing (it is a bare marker, consumed by :meth:`matches_any`).
        """
        hits: list[str] = []
        for rule in await self.get_rules(facet, kind=kind, locale=locale):
            source, sep, dest = rule["pattern"].partition("=>")
            if not sep:
                continue
            try:
                if re.search(source.strip(), text or "", re.IGNORECASE):
                    label = normalize_label(dest)
                    if label and label not in hits:
                        hits.append(label)
            except re.error:
                logger.debug("taxonomy.pattern_uncompilable", pattern=source[:80])
        return hits

    async def count_terms(
        self,
        facet: str,
        *,
        status: str | None = None,
    ) -> int:
        """How many labels exist for a facet, optionally narrowed to one lifecycle state."""
        norm_facet = normalize_label(facet)
        async with self.session_factory() as session:
            stmt = select(func.count(db.TaxonomyTerm.id)).where(
                db.TaxonomyTerm.facet == norm_facet
            )
            if status:
                stmt = stmt.where(db.TaxonomyTerm.status == status)
            return int((await session.execute(stmt)).scalar() or 0)

    # --- dynamic prompts --------------------------------------------------

    async def build_classification_prompt(
        self,
        facet: str,
        *,
        max_terms: int = 40,
        instruction: str = "",
    ) -> str:
        """Prompt fragment describing the vocabulary learned so far for a facet.

        No enum is ever baked in: on an empty brain the fragment says so and asks
        the model to name the category itself, which is what makes a new category
        possible on the first document of a new kind.
        """
        norm_facet = await self.ensure_facet(normalize_label(facet))
        active_terms = await self.get_active_terms(norm_facet, limit=max_terms)
        dimension = (norm_facet or "").replace("_", " ")

        if active_terms:
            vocabulary = "; ".join(f'"{t}"' for t in active_terms)
            fragment = (
                f"Categories already learned for {dimension} (use one verbatim only when it "
                f"genuinely fits): {vocabulary}.\n"
                "If none of these describes the content, invent a new short snake_case "
                "category name for it and report your confidence in that choice."
            )
        else:
            fragment = (
                f"No categories exist yet for {dimension}. Name the category yourself as a "
                "short snake_case identifier describing what this content actually is, and "
                "report your confidence in it."
            )

        return f"{instruction}\n{fragment}" if instruction.strip() else fragment
