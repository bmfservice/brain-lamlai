"""Feature-based reranker (Brain Second Phase 9).

Re-ranks the candidate chunks returned by the unified retrieval pipeline using
a small, interpretable feature vector.  Weights are config-driven (passed at
construction or read from ``core.config.settings`` when available) so the
system can be tuned without code changes.

Features (all normalised to [0, 1]):

- ``term_overlap``        — Jaccard-ish overlap between query terms and chunk text.
- ``entity_match``        — does the chunk mention one of the query entities?
- ``recency``             — newer chunks score higher (configurable half-life).
- ``source_trust``        — static boost from trusted source tags (e.g. docs > web).
- ``graph_distance``      — inverse graph distance from a query entity (0 = no graph).
- ``excerpt_quality``     — presence / density of a focused excerpt.
- ``role_match``          — does the passage do what the query asked for?
- ``source_authority``    — is this the kind of document worth trusting?

The final score is a weighted sum.  No LLM is invoked per re-ranking (§123
no per-query whole-dataset LLM where a heuristic suffices).
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass, field

from core import taxonomy as tax
from services.rag.preferences import RetrievalPreferences

_WEIGHT_DEFAULTS: dict[str, float] = {
    "term_overlap": 0.20,
    "entity_match": 0.20,
    "recency": 0.10,
    "source_trust": 0.08,
    "graph_distance": 0.07,
    "excerpt_quality": 0.05,
    # Classification carries a third of the weight: matching what a passage
    # *does* against what the query *asked for* separates a definition from a
    # nav label that happens to share the same words.
    "role_match": 0.18,
    "source_authority": 0.12,
}

_TOKEN_RE = re.compile(r"[\wÀ-ỹ]{2,}", re.UNICODE)

# Source tags that carry higher baseline trust.
_TRUSTED_TAGS = {"vault", "official", "docs", "internal"}


@dataclass(frozen=True)
class RankedChunk:
    """A chunk with its feature vector and final re-ranking score."""

    chunk_id: str
    content: str
    score: float
    features: dict[str, float] = field(default_factory=dict)


class FeatureReranker:
    """Deterministic feature-based reranker with config-driven weights."""

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = {**_WEIGHT_DEFAULTS, **(weights or {})}

    def rank(
        self,
        candidates: list[dict],
        *,
        query_terms: list[str],
        entity_candidates: tuple[str, ...] = (),
        graph_nodes: list = None,  # noqa: RUF013 - optional list
        chunk_distances: dict[str, int] | None = None,
        preferences: RetrievalPreferences | None = None,
    ) -> list[RankedChunk]:
        """Re-rank ``candidates`` and return them sorted by descending score.

        ``chunk_distances`` maps a candidate's ``chunk_id`` to its graph hop
        distance. Graph nodes are entity ids, which never match a chunk id, so
        the caller resolves the mapping; ``graph_nodes`` only bounds the
        default distance when no mapping is supplied.
        """
        if graph_nodes is None:
            graph_nodes = []
        node_distances = dict(chunk_distances or {})
        scored: list[RankedChunk] = []
        for cand in candidates:
            features = self._features(
                cand, query_terms, entity_candidates, node_distances, preferences
            )
            final = sum(self.weights.get(k, 0.0) * v for k, v in features.items())
            scored.append(
                RankedChunk(
                    chunk_id=str(cand.get("chunk_id", "")),
                    content=str(cand.get("content", "")),
                    score=round(final, 4),
                    features=features,
                )
            )
        return sorted(scored, key=lambda r: r.score, reverse=True)

    def _features(
        self,
        cand: dict,
        query_terms: list[str],
        entity_candidates: tuple[str, ...],
        node_distances: dict[str, int],
        preferences: RetrievalPreferences | None = None,
    ) -> dict[str, float]:
        content = str(cand.get("content", ""))
        content_lower = content.lower()
        query_set = {t.lower() for t in query_terms}
        chunk_tokens = set(_TOKEN_RE.findall(content_lower))
        overlap = 0.0
        if query_set and chunk_tokens:
            overlap = len(query_set & chunk_tokens) / len(query_set)

        entity_hits = sum(
            1 for e in entity_candidates if e.lower() in content_lower
        )
        entity_match = min(1.0, entity_hits * 0.5)

        # Recency: newer is better, default to neutral if no date present.
        meta = cand.get("meta")
        created = meta.get("created_at") if isinstance(meta, dict) else None
        if created and isinstance(created, datetime.datetime):
            age_days = max(0, (datetime.datetime.now(datetime.UTC) - created).days)
            recency = max(0.0, 1.0 - age_days / 365.0)
        else:
            recency = 0.5

        tags = cand.get("tags", []) or []
        source_trust = min(1.0, sum(1 for t in tags if t in _TRUSTED_TAGS) * 0.4)

        # Graph distance: closer neighbours score higher.
        distance = node_distances.get(str(cand.get("chunk_id", "") or ""))
        graph_distance = max(0.0, 1.0 - (distance or 5) / 5.0)

        excerpt = cand.get("excerpt", "") or cand.get("content", "")[:300]
        excerpt_quality = min(1.0, len(excerpt.strip()) / 300.0)

        # Classification features. Without preferences they fall back to the
        # taxonomy's standing authority, so a specification still outranks
        # marketing even when the query expressed no shape.
        role = cand.get("chunk_role")
        doc_type = cand.get("doc_type")
        if preferences is not None:
            role_match = preferences.role_bonus(role)
            source_authority = preferences.doc_type_bonus(doc_type)
        else:
            role_match = 0.0 if role in tax.NON_RETRIEVABLE_ROLES else 0.5
            source_authority = tax.DOC_TYPE_AUTHORITY.get(doc_type or "", 0.5)

        return {
            "term_overlap": round(overlap, 4),
            "entity_match": round(entity_match, 4),
            "recency": round(recency, 4),
            "source_trust": round(source_trust, 4),
            "graph_distance": round(graph_distance, 4),
            "excerpt_quality": round(excerpt_quality, 4),
            "role_match": round(role_match, 4),
            "source_authority": round(source_authority, 4),
        }
