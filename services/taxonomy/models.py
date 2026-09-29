"""Taxonomy resolution and data models for empty-start dynamic taxonomy."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from typing import Any


# --- Term lifecycle states -------------------------------------------------
# These are the states of the taxonomy *record*, not a classification
# vocabulary: which lifecycle a learned term is in is structural (it decides
# whether retrieval may surface the term), and it is fixed by the schema. The
# labels themselves — doc types, entity types, chunk roles, audiences — are
# learned into ``taxonomy_terms`` and are never enumerated here.
STATUS_ACTIVE = "active"
STATUS_PROPOSED = "proposed"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_MERGED = "merged"
STATUS_RETIRED = "retired"

TERM_STATUSES: frozenset[str] = frozenset(
    {STATUS_ACTIVE, STATUS_PROPOSED, STATUS_AMBIGUOUS, STATUS_MERGED, STATUS_RETIRED}
)

# Where a term came from. ``seed`` is deliberately absent: nothing is seeded.
ORIGIN_MODEL = "model"
ORIGIN_USER = "user"
ORIGIN_DERIVED = "derived"

# Quarantine reason reused from services.maintenance.quarantine for labels the
# resolver could not confidently classify.
REASON_AMBIGUOUS = "ambiguous"


@dataclass
class TaxonomyResolution:
    """Outcome of resolving a label/term against a dynamic taxonomy facet."""

    term: str
    facet: str
    status: str  # active | proposed | ambiguous | quarantined | merged | retired
    confidence: float
    authority: float | None = None
    is_new: bool = False
    origin: str = "model"
    term_id: uuid.UUID | None = None
    quarantine_id: uuid.UUID | None = None

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    @property
    def is_proposed(self) -> bool:
        return self.status == "proposed"

    @property
    def is_ambiguous(self) -> bool:
        return self.status == "ambiguous"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["term_id"] = str(self.term_id) if self.term_id else None
        d["quarantine_id"] = str(self.quarantine_id) if self.quarantine_id else None
        return d
