"""Nightly consolidation job (Giai đoạn 5, Layer 7 — sleep-time learning).

This is the neocortex doing its slow work while the system is idle. Complementary
Learning Systems says the fast store (LLM rulings) must be *replayed* into the
slow store (the distilled classifier) offline, or you never get the speedup. This
job is that replay.

It runs after the maintenance window (default 00:30, config ``consolidation_time``)
and performs independent, individually-counted steps — a failure in one never
blocks the others:

1. **Retrain the classifier** when the ledger has enough untrained decisions
   (``consolidation_min_new_samples``). Human overrides are up-weighted, so a
   correction folded in tonight shifts tomorrow's automatic rulings. After a fit
   the consumed rows are stamped ``trained_at`` so a quiet night is cheap.
2. **Weight-decay sweep** over active edges (B4): every edge's ``weight`` is
   recomputed from ``confidence × decay(idle) × reuse(use_count)``. Reused edges
   strengthen, abandoned ones fade — no per-edge special-casing.
3. **Contradiction quarantine** (B7): semantic-unit pairs the maintenance scan
   flagged as contradictory are moved to ``quarantined_knowledge`` (hidden,
   reviewable, never destroyed) instead of merely reported.

Designed for the user's RAM/CPU constraint: retraining only fires when there is
new signal, the decay sweep is a single read + batched write, and no step calls
the LLM or loads embeddings. The expensive category-dedup / node-compression
ideas from the blueprint are deliberately *not* here — they need an LLM budget
and would violate "no half-finished implementations"; this job stays cheap and
correct.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field

import sqlalchemy as sa

from core import database as db
from core.config import settings
from core.logging import get_logger
from services.consolidation.classifier import ConsolidatedRelationClassifier
from services.consolidation.dataset import (
    TrainingSet,
    count_untrained,
    load_training_set,
    stamp_trained,
)
from services.consolidation.weight import compute_weight

logger = get_logger(__name__)

# Maintenance's contradiction reason code, mirrored here so quarantine rows are
# self-describing without importing the maintenance module (no shared coupling).
_REASON_CONTRADICTION = "contradiction"


@dataclass
class ConsolidationReport:
    """Counts per step — surfaced by the scheduler log and the metrics gauge."""

    retrained: bool = False
    new_samples: int = 0
    model_version: str = ""
    edges_reweighted: int = 0
    flagged_review: int = 0
    quarantined: int = 0
    steps_failed: list[str] = field(default_factory=list)


class ConsolidationService:
    """Replays the fast store into the slow store and re-breathes edge weights."""

    def __init__(
        self,
        session_factory,
        classifier: ConsolidatedRelationClassifier | None,
        *,
        min_new_samples: int | None = None,
        half_life_days: float | None = None,
        maintenance=None,
    ) -> None:
        self.session_factory = session_factory
        self.classifier = classifier
        self.min_new_samples = (
            settings.consolidation_min_new_samples if min_new_samples is None else min_new_samples
        )
        self.half_life_days = (
            settings.consolidation_weight_half_life_days
            if half_life_days is None
            else half_life_days
        )
        self.review_weight_floor = float(settings.consolidation_review_weight_floor)
        # Optional: when wired, the job reuses maintenance's contradiction scan
        # rather than re-implementing it (the scan already exists and is tested).
        self.maintenance = maintenance

    async def run(self) -> ConsolidationReport:
        """Run every consolidation step. Each is isolated; one failure logs and
        continues so a partial night still does useful work."""
        report = ConsolidationReport()
        await self._step_retrain(report)
        await self._step_weight_sweep(report)
        await self._step_quarantine_contradictions(report)
        logger.info(
            "consolidation.job_done",
            retrained=report.retrained,
            new_samples=report.new_samples,
            model_version=report.model_version,
            edges_reweighted=report.edges_reweighted,
            flagged_review=report.flagged_review,
            quarantined=report.quarantined,
            failed=report.steps_failed,
        )
        return report

    # ------------------------------------------------------------------
    # Step 1 — retrain the distilled classifier from the ledger
    # ------------------------------------------------------------------

    async def _step_retrain(self, report: ConsolidationReport) -> None:
        if self.classifier is None:
            return
        try:
            async with self.session_factory() as session:
                untrained = await count_untrained(session)
                report.new_samples = untrained
                if untrained < self.min_new_samples:
                    logger.info(
                        "consolidation.retrain_skipped",
                        untrained=untrained,
                        threshold=self.min_new_samples,
                    )
                    return
                data: TrainingSet = await load_training_set(session)
                result = self.classifier.train(
                    data.features,
                    data.labels,
                    sample_weights=data.weights,
                )
                if not result.trained:
                    report.steps_failed.append(f"retrain:{result.reason}")
                    logger.warning("consolidation.retrain_failed", reason=result.reason)
                    return
                # Fold the consumed rows in and persist so a restart loads the
                # new model without retraining (persistence acceptance criterion).
                now = datetime.datetime.now(datetime.UTC)
                await stamp_trained(session, now)
                await session.commit()
            self.classifier.save()
            report.retrained = True
            report.model_version = result.version
            logger.info(
                "consolidation.retrained",
                version=result.version,
                n_samples=result.n_samples,
                n_classes=result.n_classes,
            )
        except Exception as exc:  # noqa: BLE001 — one step must not kill the night
            report.steps_failed.append(f"retrain:{str(exc)[:60]}")
            logger.error("consolidation.retrain_error", error=str(exc), exc_info=True)

    # ------------------------------------------------------------------
    # Step 2 — recompute every active edge's weight (decay + reuse)
    # ------------------------------------------------------------------

    async def _step_weight_sweep(self, report: ConsolidationReport) -> None:
        """Apply the B4 weight law to all active edges.

        Reads id/confidence/use_count/last_used_at/created_at, computes the new
        weight in Python (the law lives in one tested place), then issues one
        batched CASE-free UPDATE per edge via executemany. Edges are few relative
        to chunks, and this runs at 00:30, so a straightforward loop is fine and
        stays readable; it never grows per-file ingest cost (that is retrieval's
        O(log n) HNSW path, untouched here)."""
        try:
            now = datetime.datetime.now(datetime.UTC)
            async with self.session_factory() as session:
                rows = (
                    await session.execute(
                        sa.select(
                            db.KnowledgeLink.id,
                            db.KnowledgeLink.confidence,
                            db.KnowledgeLink.use_count,
                            db.KnowledgeLink.last_used_at,
                            db.KnowledgeLink.created_at,
                        ).where(db.KnowledgeLink.lifecycle == "active")
                    )
                ).all()
                if not rows:
                    return
                updates: list[dict] = []
                review_ids: list = []
                for link_id, confidence, use_count, last_used_at, created_at in rows:
                    anchor = last_used_at or created_at or now
                    days_unused = max(0.0, (now - anchor).total_seconds() / 86400.0)
                    new_weight = compute_weight(
                        confidence=float(confidence or 0.0),
                        days_unused=days_unused,
                        use_count=int(use_count or 0),
                        half_life_days=self.half_life_days,
                    )
                    updates.append({"id": link_id, "weight": new_weight})
                    # B7 self-monitoring: a decayed-to-near-nothing edge is stale
                    # or weakly supported — surface it for a human instead of
                    # letting it rank low forever. Same weight we just computed,
                    # so no extra read.
                    if new_weight < self.review_weight_floor:
                        review_ids.append(link_id)
                if updates:
                    await session.execute(
                        sa.update(db.KnowledgeLink)
                        .where(db.KnowledgeLink.id == sa.bindparam("id"))
                        .values(weight=sa.bindparam("weight")),
                        updates,
                    )
                if review_ids:
                    await session.execute(
                        sa.update(db.KnowledgeLink)
                        .where(
                            db.KnowledgeLink.id.in_(review_ids),
                            db.KnowledgeLink.lifecycle == "active",
                        )
                        .values(lifecycle="review")
                    )
                if updates or review_ids:
                    await session.commit()
                report.edges_reweighted = len(updates)
                report.flagged_review = len(review_ids)
                logger.info(
                    "consolidation.weight_sweep",
                    edges=len(updates),
                    flagged_review=len(review_ids),
                )
        except Exception as exc:  # noqa: BLE001
            report.steps_failed.append(f"weight:{str(exc)[:60]}")
            logger.error("consolidation.weight_error", error=str(exc), exc_info=True)

    # ------------------------------------------------------------------
    # Step 3 — quarantine contradictions (B7)
    # ------------------------------------------------------------------

    async def _step_quarantine_contradictions(self, report: ConsolidationReport) -> None:
        """Move contradiction-flagged semantic units to quarantine.

        Reuses maintenance's already-tested ``find_contradictions`` scan when
        wired; writes one ``quarantined_knowledge`` row per flagged unit (the
        payload is reversible — nothing is destroyed). Idempotent: a unit already
        quarantined for ``contradiction`` is skipped so repeated nights do not
        pile up duplicate rows."""
        if self.maintenance is None:
            return
        try:
            contradictions = await self.maintenance.find_contradictions()
            if not contradictions:
                return
            unit_ids: set[str] = set()
            for pair in contradictions:
                for key in ("unit_a", "unit_b"):
                    uid = pair.get(key)
                    if uid:
                        unit_ids.add(str(uid))
            if not unit_ids:
                return

            async with self.session_factory() as session:
                # Skip units already quarantined for this reason (idempotence).
                existing = set(
                    (
                        await session.execute(
                            sa.select(db.QuarantinedKnowledge.row_id).where(
                                db.QuarantinedKnowledge.table_name == "semantic_units",
                                db.QuarantinedKnowledge.reason == _REASON_CONTRADICTION,
                                db.QuarantinedKnowledge.row_id.in_(unit_ids),
                            )
                        )
                    ).scalars().all()
                )
                fresh = sorted(unit_ids - existing)
                if not fresh:
                    return
                session.add_all(
                    db.QuarantinedKnowledge(
                        table_name="semantic_units",
                        row_id=uid,
                        reason=_REASON_CONTRADICTION,
                        detail=(
                            "flagged by nightly contradiction scan "
                            "(overlapping entities, opposite polarity)"
                        ),
                    )
                    for uid in fresh
                )
                await session.commit()
                report.quarantined = len(fresh)
                logger.info("consolidation.quarantined_contradictions", count=len(fresh))
        except Exception as exc:  # noqa: BLE001
            report.steps_failed.append(f"quarantine:{str(exc)[:60]}")
            logger.error("consolidation.quarantine_error", error=str(exc), exc_info=True)
