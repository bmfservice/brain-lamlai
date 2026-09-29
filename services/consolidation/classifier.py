"""Consolidated relation classifier (Giai đoạn 5, Layer 4 — the neocortex).

The LLM verifier (Layer 5) is the hippocampus: accurate, expensive, slow. This
classifier is the *consolidated* tier — a small gradient-boosted model distilled
from the LLM's own past rulings (the ``relation_decisions`` ledger). Once it has
seen enough decisions it answers high-confidence candidates in microseconds
instead of a ~1s chat call, and the ratio of consolidated decisions rises over
time (the acceptance metric).

Why HistGradientBoostingClassifier (verified before building, per the user's
"research first" instruction):

- Benchmark on this machine: ~6.6 ms for a batch of 12 vs ~700–1000 ms per LLM
  candidate — ~100× cheaper, which is the entire point of the fast tier.
- ~45 KB on disk, CPU inference with negligible RAM — matches the "limit server
  RAM/CPU" constraint. GBDT beats an MLP at low sample counts (cold start) and
  needs almost no tuning.
- Native multiclass + ``predict_proba`` for the confidence threshold.

Cold start: with no trained model, :meth:`predict_batch` returns ``None`` for
every candidate, so all of them defer to the LLM — byte-for-byte the pre-Phase-5
behavior. No regression, no partial-model surprises.
"""

from __future__ import annotations

import contextlib
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from core.logging import get_logger
from services.consolidation.features import FEATURE_DIM, FEATURE_NAMES, to_vector

logger = get_logger(__name__)

MODEL_FILENAME = "consolidated_model.joblib"
META_FILENAME = "consolidated_model.json"


def _canonical_id(value) -> str:
    """Normalize a node id so the override registry matches across call sites.

    The API receives ids as path strings (any case/format); ``semantic`` keys on
    ``str(entity.id)``. Canonical UUIDs collapse those to one form; non-UUID ids
    (tests, synthetic) fall back to their raw string so they still round-trip."""
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, AttributeError, TypeError):
        return str(value)


@dataclass(frozen=True)
class ClassifierPrediction:
    """One consolidated-tier ruling on a candidate.

    ``decided`` is True only when the model's confidence clears the threshold —
    otherwise the caller must defer the candidate to the LLM. ``relation_type``
    may be ``REJECT`` (the model learned the sentence asserts no relation).
    """

    relation_type: str
    confidence: float
    decided: bool


@dataclass(frozen=True)
class TrainResult:
    """Outcome of one training run (for logging / metrics)."""

    trained: bool
    n_samples: int
    n_classes: int
    version: str
    reason: str = ""


class ConsolidatedRelationClassifier:
    """Distils LLM relation rulings into a small, fast, persistent model."""

    def __init__(
        self,
        model_dir: Path,
        *,
        confidence_threshold: float = 0.85,
        version_prefix: str = "consolidated",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.confidence_threshold = confidence_threshold
        self.version_prefix = version_prefix
        self._model = None
        self._meta: dict = {}
        self._loaded = False
        # Exact-pair human rulings: (source_id, target_id) → (label, confidence).
        # A human override is ground truth and wins over BOTH tiers immediately,
        # without waiting for tonight's retrain. Keyed by string ids so it is
        # independent of UUID/str representation at the call sites.
        self._overrides: dict[tuple[str, str], tuple[str, float]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """True once a trained model of the current feature dimension is held."""
        return self._model is not None and self._meta.get("feature_dim") == FEATURE_DIM

    @property
    def version(self) -> str:
        """Human-facing model version, e.g. ``consolidated-v3`` (empty when
        cold). Used as ``decided_by`` so every edge traces to the tier that
        ruled on it."""
        if not self.is_ready:
            return ""
        n = int(self._meta.get("train_count", 1))
        return f"{self.version_prefix}-v{n}"

    # ------------------------------------------------------------------
    # Human overrides (Layer 4.5 — ground truth, immediate)
    # ------------------------------------------------------------------

    def set_override(self, source_id: str, target_id: str, label: str, confidence: float) -> None:
        """Register a human ruling for an exact pair.

        Takes effect on the very next decision for that pair — no retrain, no
        restart. The override is also persisted to ``relation_decisions`` by the
        caller so tonight's distillation folds it into the model (and survives a
        reload via :meth:`load_overrides`)."""
        self._overrides[(_canonical_id(source_id), _canonical_id(target_id))] = (
            label,
            float(confidence),
        )

    def clear_override(self, source_id: str, target_id: str) -> bool:
        """Drop a human ruling for an exact pair. Returns True if one existed."""
        return (
            self._overrides.pop((_canonical_id(source_id), _canonical_id(target_id)), None)
            is not None
        )

    def lookup_override(self, source_id: str, target_id: str) -> tuple[str, float] | None:
        """Return ``(label, confidence)`` if a human ruled on this exact pair."""
        return self._overrides.get((_canonical_id(source_id), _canonical_id(target_id)))

    def override_count(self) -> int:
        return len(self._overrides)

    def overrides(self) -> list[tuple[str, str, str, float]]:
        """Snapshot of the registry as ``(source_id, target_id, label, conf)``.

        Sorted for stable API output; a copy, so callers cannot mutate state."""
        return sorted(
            (s, t, label, conf) for (s, t), (label, conf) in self._overrides.items()
        )

    def load_overrides(self, rows) -> int:
        """Bulk-load persisted overrides at startup.

        ``rows`` is an iterable of objects with ``.source_id``, ``.target_id``,
        ``.label``, ``.confidence`` — the human_override rows from the ledger,
        newest last so the latest ruling for a pair wins. Returns the count
        loaded into the registry."""
        loaded = 0
        for row in rows:
            try:
                self._overrides[
                    (_canonical_id(row.source_id), _canonical_id(row.target_id))
                ] = (
                    row.label,
                    float(row.confidence),
                )
                loaded += 1
            except (AttributeError, TypeError, ValueError):
                continue
        return loaded

    def load(self) -> bool:
        """Load a persisted model from disk. Safe to call at startup; returns
        False (and stays cold) when no model exists or it is incompatible.

        Persistence across restart is an acceptance criterion — this is what
        satisfies it (no retrain on boot)."""
        if self._loaded:
            return self.is_ready
        self._loaded = True
        model_path = self.model_dir / MODEL_FILENAME
        meta_path = self.model_dir / META_FILENAME
        if not model_path.exists():
            logger.info("consolidated_classifier.cold_start", path=str(model_path))
            return False
        try:
            import joblib

            blob = joblib.load(model_path)
            self._model = blob["model"]
            meta = dict(blob.get("meta", {}))
            if meta_path.exists():
                with contextlib.suppress(OSError, ValueError):
                    meta.update(json.loads(meta_path.read_text(encoding="utf-8")))
            self._meta = meta
        except Exception as exc:  # noqa: BLE001 — a corrupt model must not boot-fail
            logger.warning("consolidated_classifier.load_failed", error=str(exc)[:160])
            self._model = None
            self._meta = {}
            return False

        if self._meta.get("feature_dim") != FEATURE_DIM:
            logger.warning(
                "consolidated_classifier.feature_dim_mismatch",
                stored=self._meta.get("feature_dim"),
                current=FEATURE_DIM,
            )
            self._model = None
            return False
        logger.info(
            "consolidated_classifier.loaded",
            version=self.version,
            n_samples=self._meta.get("n_samples"),
        )
        return True

    def save(self) -> None:
        """Persist the trained model + metadata atomically enough for restart."""
        if self._model is None:
            return
        self.model_dir.mkdir(parents=True, exist_ok=True)
        try:
            import joblib

            joblib.dump(
                {"model": self._model, "meta": self._meta},
                self.model_dir / MODEL_FILENAME,
            )
            (self.model_dir / META_FILENAME).write_text(
                json.dumps(self._meta, default=str, indent=2), encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("consolidated_classifier.save_failed", error=str(exc)[:160])

    # ------------------------------------------------------------------
    # Inference (the fast tier)
    # ------------------------------------------------------------------

    def predict_batch(
        self, feature_dicts: list[dict[str, float]]
    ) -> list[ClassifierPrediction | None]:
        """Score many candidates in one vectorized pass (~ms).

        Returns a list aligned to the input; each entry is ``None`` when the
        model is cold or incompatible (→ caller defers to the LLM), else a
        :class:`ClassifierPrediction`. ``decided`` is False when confidence is
        below the threshold — the caller still defers those to the LLM.
        """
        if not feature_dicts or not self.is_ready:
            return [None] * len(feature_dicts)
        try:
            import numpy as np

            x = np.asarray([to_vector(f) for f in feature_dicts], dtype=float)
            if x.shape[1] != FEATURE_DIM:
                return [None] * len(feature_dicts)
            proba = self._model.predict_proba(x)
            preds = self._model.predict(x)
        except Exception as exc:  # noqa: BLE001 — inference must never fail an ingest
            logger.warning("consolidated_classifier.predict_failed", error=str(exc)[:160])
            return [None] * len(feature_dicts)

        out: list[ClassifierPrediction | None] = []
        for label, row in zip(preds, proba, strict=True):
            confidence = float(max(row)) if len(row) else 0.0
            out.append(
                ClassifierPrediction(
                    relation_type=str(label),
                    confidence=confidence,
                    decided=confidence >= self.confidence_threshold,
                )
            )
        return out

    # ------------------------------------------------------------------
    # Training (runs in the nightly consolidation job, not on the request path)
    # ------------------------------------------------------------------

    def train(
        self,
        feature_dicts: list[dict[str, float]],
        labels: list[str],
        *,
        sample_weights: list[float] | None = None,
    ) -> TrainResult:
        """Fit the model on the decision ledger.

        ``sample_weights`` lets human overrides out-vote distilled LLM labels
        (the override is ground truth). Requires ≥2 distinct labels — a
        single-class model cannot ``predict_proba`` and would be useless, so we
        skip rather than persist a degenerate model.
        """
        if len(feature_dicts) != len(labels) or not labels:
            return TrainResult(False, len(labels), 0, self.version, reason="empty_or_mismatched")
        classes = sorted(set(labels))
        if len(classes) < 2:
            return TrainResult(
                False, len(labels), len(classes), self.version, reason="single_class"
            )
        try:
            import numpy as np
            from sklearn.ensemble import HistGradientBoostingClassifier

            x = np.asarray([to_vector(f) for f in feature_dicts], dtype=float)
            y = np.asarray(labels, dtype=object)
            sw = np.asarray(sample_weights, dtype=float) if sample_weights else None
            model = HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.1,
                max_leaf_nodes=15,
                min_samples_leaf=5,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=10,
                random_state=42,
            )
            model.fit(x, y, sample_weight=sw)
        except Exception as exc:  # noqa: BLE001 — training failure must not crash the job
            logger.warning("consolidated_classifier.train_failed", error=str(exc)[:200])
            return TrainResult(False, len(labels), len(classes), self.version, reason=str(exc)[:80])

        train_count = int(self._meta.get("train_count", 0)) + 1
        self._model = model
        self._meta = {
            "feature_dim": FEATURE_DIM,
            "feature_names": list(FEATURE_NAMES),
            "classes": classes,
            "n_samples": len(labels),
            "train_count": train_count,
            "trained_at": time.time(),
            "threshold": self.confidence_threshold,
        }
        logger.info(
            "consolidated_classifier.trained",
            version=self.version,
            n_samples=len(labels),
            n_classes=len(classes),
        )
        return TrainResult(True, len(labels), len(classes), self.version)
