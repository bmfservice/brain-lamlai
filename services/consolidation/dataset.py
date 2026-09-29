"""Training-set assembly + decision recording for the consolidated classifier.

The ``relation_decisions`` ledger (core.database.RelationDecision) is the
classifier's only teacher: every LLM ruling and every human override lands here
with the feature vector that produced it. This module turns that ledger into the
``(features, labels, weights)`` triple the classifier trains on.

Weighting policy — human overrides are ground truth:

- ``human_override=True``  → weight ``OVERRIDE_WEIGHT`` (default 10×)
- distilled LLM ruling     → weight 1.0

So a handful of corrections can pull the next model toward the human's intent
without drowning the (much larger) body of LLM-labelled examples. This is how an
override "feeds the next distillation" per the master blueprint, while still
taking effect *immediately* at decision time (handled in the override API, B5).
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlalchemy as sa

from core.database import RelationDecision
from core.logging import get_logger
from services.consolidation.features import REJECT_LABEL, featurize

logger = get_logger(__name__)

OVERRIDE_WEIGHT = 10.0
LLM_WEIGHT = 1.0


@dataclass(frozen=True)
class TrainingSet:
    """Materialized training data pulled from the ledger."""

    features: list[dict[str, float]]
    labels: list[str]
    weights: list[float]
    n_rows: int
    n_overrides: int
    latest_created_at: object | None = None

    @property
    def is_empty(self) -> bool:
        return self.n_rows == 0


async def load_training_set(session) -> TrainingSet:
    """Read every decision row and assemble the training triple.

    Rows already carry a stored feature JSONB; we re-featurize only if it is
    missing or malformed, so a schema tweak in ``features.py`` can be backfilled
    lazily without a migration. Untrained rows have ``trained_at IS NULL``; the
    consolidation job stamps them after a successful fit so the *next* run only
    counts new rows for its min-samples gate.
    """
    rows = (
        await session.execute(
            sa.select(RelationDecision).order_by(RelationDecision.created_at.asc())
        )
    ).scalars().all()

    features: list[dict[str, float]] = []
    labels: list[str] = []
    weights: list[float] = []
    overrides = 0
    latest = None

    for row in rows:
        label = (row.label or "").strip()
        if not label:
            continue
        feat = row.features if isinstance(row.features, dict) and row.features else None
        if feat is None:
            # Should not happen (features is NOT NULL), but degrade safely.
            logger.warning("consolidated_dataset.missing_features", id=str(row.id))
            continue
        features.append({k: float(v) for k, v in feat.items()})
        labels.append(label)
        if row.human_override:
            overrides += 1
            weights.append(OVERRIDE_WEIGHT)
        else:
            weights.append(LLM_WEIGHT)
        if latest is None or (row.created_at and row.created_at > latest):
            latest = row.created_at

    return TrainingSet(
        features=features,
        labels=labels,
        weights=weights,
        n_rows=len(labels),
        n_overrides=overrides,
        latest_created_at=latest,
    )


async def count_untrained(session) -> int:
    """How many decision rows have not yet been folded into a trained model.

    The consolidation job retrains only when this clears
    ``consolidation_min_new_samples`` — so a quiet night does not waste CPU.
    """
    value = await session.scalar(
        sa.select(sa.func.count())
        .select_from(RelationDecision)
        .where(RelationDecision.trained_at.is_(None))
    )
    return int(value or 0)


async def load_human_overrides(session) -> list[RelationDecision]:
    """Return the latest human ruling for each overridden pair, oldest→newest.

    Used at startup to repopulate the in-memory override registry so an exact-
    pair correction survives a restart *before* the next retrain folds it into
    the model. Ordered by ``created_at`` ascending so the caller can let the
    newest row win per pair.
    """
    rows = (
        await session.execute(
            sa.select(RelationDecision)
            .where(RelationDecision.human_override.is_(True))
            .order_by(RelationDecision.created_at.asc())
        )
    ).scalars().all()
    return list(rows)


async def stamp_trained(session, trained_at) -> int:
    """Mark all currently-untrained rows as consumed by a fit at ``trained_at``."""
    result = await session.execute(
        sa.update(RelationDecision)
        .where(RelationDecision.trained_at.is_(None))
        .values(trained_at=trained_at)
    )
    return int(result.rowcount or 0)


def build_features(*, proposed_type, signals, sentence) -> dict[str, float]:
    """Thin re-export so callers (semantic.py, override API) featurize through
    one name without importing the features module directly."""
    return featurize(proposed_type=proposed_type, signals=signals, sentence=sentence)


def decision_label(*, supported: bool, relation_type: str) -> str:
    """Map an LLM/consolidated ruling onto a ledger label.

    ``REJECT`` for "no relation asserted"; otherwise the taxonomy type. Kept
    here so the label vocabulary is defined once for both writers and readers.
    """
    if not supported:
        return REJECT_LABEL
    return (relation_type or REJECT_LABEL).strip().upper() or REJECT_LABEL
