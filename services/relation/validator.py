"""Braind Relation Service (Brain Second Phase 7).

Implements the semantic-relation scoring pipeline (§11/§12) for the typed
knowledge graph.

Golden rules enforced here (master prompt §9/§10/§63):
- **Cosine/embedding similarity is ONE signal, never the only one.** A
  candidate whose only non-zero signal is embedding similarity is rejected —
  it is never persisted as an edge.
- **`RELATED_TO` is never the default.** When the exact type is unknown the
  validator downgrades to `POSSIBLE_RELATION` with low confidence, or rejects.
- **Every approved edge carries evidence + confidence + reason + model
  version** (§12), and is classed STRONG / MEDIUM / WEAK / SUSPECTED /
  INVALID (§49); only INVALID is auto-removable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.logging import get_logger

logger = get_logger(__name__)

# --- Edge-type taxonomy (§10) ---------------------------------------------
VALID_EDGE_TYPES: frozenset[str] = frozenset(
    {
        "IS_A",
        "INSTANCE_OF",
        "SAME_ENTITY_AS",
        "PART_OF",
        "HAS_PART",
        "PARENT_OF",
        "CHILD_OF",
        "DEPENDS_ON",
        "REQUIRES",
        "ENABLES",
        "CAUSES",
        "CAUSED_BY",
        "PRECEDES",
        "FOLLOWS",
        "OCCURS_BEFORE",
        "OCCURS_AFTER",
        "DERIVED_FROM",
        "EXTRACTED_FROM",
        "QUOTED_FROM",
        "REFERENCES",
        "SUPPORTS",
        "CONTRADICTS",
        "REFUTES",
        "SIMILAR_TO",
        "ALTERNATIVE_TO",
        "OPPOSITE_OF",
        "IMPLEMENTS",
        "USES",
        "PROVIDES",
        "HAS_PROPERTY",
        "HAS_VALUE",
        "SUPERSEDES",
        "SUPERSEDED_BY",
        "VERSION_OF",
        "RELATED_TO",
        "POSSIBLE_RELATION",
    }
)

# --- Relation strength classes (§49) --------------------------------------
POSSIBLE_RELATION = "POSSIBLE_RELATION"
STRONG = "STRONG"
MEDIUM = "MEDIUM"
WEAK = "WEAK"
SUSPECTED = "SUSPECTED"
INVALID = "INVALID"

# Weights per signal *source* — not one blind linear formula (§11). These are
# defaults; the learning pipeline (Phase 11) may learn per-domain weights.
_SIGNAL_WEIGHTS: dict[str, float] = {
    "semantic": 0.20,  # embedding agreement — deliberately capped
    "entity": 0.20,  # ≥1 resolved shared entity, type-matched
    "lexical": 0.10,  # term overlap
    "structural": 0.15,  # proximity in hierarchy / list / table
    "temporal": 0.05,  # ordering consistency
    "dependency": 0.15,  # syntactic evidence: "X causes Y"
    "explicit_reference": 0.10,  # refs, wikilinks, "unlike Y"
    "source": 0.05,  # source trust × recency × freshness
}


@dataclass(frozen=True)
class RelationCandidate:
    """A candidate edge between two nodes (stage-1 output).

    ``signals`` holds *independent*, pre-computed sub-scores in [0, 1]; the
    validator does the weighting/validation, it never computes similarity
    itself (that stays in embedding/lexical producers).

    ``evidence_text`` is the quoted supporting span (the sentence the pair
    co-occurs in) — kept alongside the provenance labels so every persisted
    edge carries *readable* evidence, not just a pointer.
    """

    source_id: str
    target_id: str
    proposed_type: str | None = None
    signals: dict[str, float] = field(default_factory=dict)
    evidence_sources: list[str] = field(default_factory=list)
    model_version: str = "relation-model-v1"
    evidence_text: str = ""
    rationale: str = ""

    @property
    def nonzero_signals(self) -> list[str]:
        return [k for k, v in self.signals.items() if (v or 0.0) > 0.0]


@dataclass(frozen=True)
class RelationVerdict:
    """Outcome of the validation → classification → confidence pipeline."""

    source_id: str
    target_id: str
    relation_type: str
    confidence: float
    strength: str  # STRONG | MEDIUM | WEAK | SUSPECTED | INVALID
    accepted: bool  # True → may be persisted as an edge
    reason: str
    evidence_count: int
    evidence_sources: list[str]
    model_version: str
    signal_breakdown: dict[str, float] = field(default_factory=dict)
    evidence_text: str = ""


class RelationValidator:
    """Stage 2-4: validate, classify, and score candidate relations.

    Stages (per §11):
      2. RELATION VALIDATION   — reject candidates with no trustable signal,
                                especially embedding-only candidates.
      3. RELATION CLASSIFICATION — pick a typed edge; unknown → POSSIBLE_RELATION.
      4. RELATION CONFIDENCE  — normalized weighted score + strength class.
    """

    def __init__(self, signal_weights: dict[str, float] | None = None):
        self.signal_weights = {**_SIGNAL_WEIGHTS, **(signal_weights or {})}

    # ------------------------------------------------------------------
    # Stage 2 — validation
    # ------------------------------------------------------------------

    def _validate(self, candidate: RelationCandidate) -> str | None:
        """Return a rejection reason, or ``None`` when the candidate passes
        stage 2.  Embedding-only candidates never pass."""
        nonzero = candidate.nonzero_signals
        if not nonzero:
            return "no evidence signals at all"
        if nonzero == ["semantic"]:
            # §9 — cosine similarity alone is NOT a relation.
            return "embedding/cosine similarity is the only signal (rejected §9)"
        if not candidate.evidence_sources:
            return "no evidence sources recorded (§12)"
        return None

    # ------------------------------------------------------------------
    # Stage 3 — classification
    # ------------------------------------------------------------------

    def _classify(self, candidate: RelationCandidate) -> tuple[str, bool]:
        """Return ``(relation_type, is_sure)``.

        Only taxonomy types are ever used.  An unknown proposal degrades to
        ``POSSIBLE_RELATION`` with a *flag* that forces low confidence — we
        never use ``RELATED_TO`` as a rest bucket (§10/§63).
        """
        proposed = (candidate.proposed_type or "").upper()
        if proposed in VALID_EDGE_TYPES:
            if proposed == POSSIBLE_RELATION:
                return proposed, False
            return proposed, True
        return POSSIBLE_RELATION, False

    # ------------------------------------------------------------------
    # Stage 4 — confidence + strength
    # ------------------------------------------------------------------

    def _score(self, candidate: RelationCandidate) -> float:
        """Normalized weighted sum of signals, clamped to [0, 1]."""
        raw = sum(
            self.signal_weights.get(key, 0.0) * max(0.0, min(1.0, value))
            for key, value in candidate.signals.items()
        )
        return max(0.0, min(1.0, raw))

    @staticmethod
    def _strength(confidence: float) -> str:
        """Map confidence to a §49 lifecycle class (only INVALID removable)."""
        if confidence >= 0.75:
            return STRONG
        if confidence >= 0.55:
            return MEDIUM
        if confidence >= 0.35:
            return WEAK
        if confidence >= 0.15:
            return SUSPECTED
        return INVALID

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def validate(self, candidate: RelationCandidate) -> RelationVerdict:
        """Full pipeline for one candidate edge."""
        # Stage 2 — validation.
        rejection = self._validate(candidate)
        if rejection:
            return RelationVerdict(
                source_id=candidate.source_id,
                target_id=candidate.target_id,
                relation_type=candidate.proposed_type or POSSIBLE_RELATION,
                confidence=0.0,
                strength=INVALID,
                accepted=False,
                reason=rejection,
                evidence_count=len(candidate.evidence_sources),
                evidence_sources=candidate.evidence_sources,
                model_version=candidate.model_version,
                signal_breakdown=dict(candidate.signals),
                evidence_text=candidate.evidence_text,
            )

        # Stage 3 — classification.
        relation_type, is_sure = self._classify(candidate)

        # Stage 4 — confidence.  Unsure types and weak signals are suppressed
        # (§10 POSSIBLE_RELATION / §63 low-confidence not a fact).
        confidence = self._score(candidate)
        if not is_sure:
            confidence = min(confidence, 0.25)
        strength = self._strength(confidence)

        accepted = strength not in {INVALID, SUSPECTED}
        reason = candidate.rationale or (
            f"validated multi-signal relation ({','.join(candidate.nonzero_signals)})"
            if accepted
            else (
                "low confidence — classify as possible relation, excluded from "
                "core facts until promoted"
            )
        )
        if relation_type == POSSIBLE_RELATION and accepted:
            reason = (
                f"specific edge type unknown; stored as POSSIBLE_RELATION with "
                f"confidence capped at {confidence:.2f}"
            )

        return RelationVerdict(
            source_id=candidate.source_id,
            target_id=candidate.target_id,
            relation_type=relation_type,
            confidence=round(confidence, 4),
            strength=strength,
            accepted=accepted,
            reason=reason,
            evidence_count=len(candidate.evidence_sources),
            evidence_sources=candidate.evidence_sources,
            model_version=candidate.model_version,
            signal_breakdown=dict(candidate.signals),
            evidence_text=candidate.evidence_text,
        )