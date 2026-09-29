"""Braind Relation Service (Brain Second Phase 7).

Typed-edge generation + validation for the semantic knowledge graph.

Architecture (§11 — never a single blind linear formula):

    1. CANDIDATE GENERATION   multi-signal: co-occurrence, explicit refs,
                              wikilinks, shared entities, structural proximity,
                              template patterns.
    2. RELATION VALIDATION    reject candidates that only have weak/unverifiable
                              signals — especially cosine-similarity-only edges
                              (§9 — embedding is ONE signal, never the answer).
    3. RELATION CLASSIFICATION assign a typed edge from the §10 taxonomy;
                              ``POSSIBLE_RELATION`` is the low-confidence fallback
                              when the exact type is unknown (§10/§63).
    4. RELATION CONFIDENCE    confidence + evidence + reason per edge (§12).

No edge is persisted without evidence (golden rule §9/§63).  The validator is
pure/deterministic so it is unit-testable offline.
"""

from services.relation.validator import (
    INVALID,
    MEDIUM,
    POSSIBLE_RELATION,
    STRONG,
    SUSPECTED,
    VALID_EDGE_TYPES,
    WEAK,
    RelationCandidate,
    RelationValidator,
    RelationVerdict,
)

__all__ = [
    "INVALID",
    "MEDIUM",
    "POSSIBLE_RELATION",
    "RelationCandidate",
    "RelationValidator",
    "RelationVerdict",
    "STRONG",
    "SUSPECTED",
    "VALID_EDGE_TYPES",
    "WEAK",
]