"""Braind RAG - retrieval preferences derived from the query.

Maps a query onto the classification a passage must carry to answer it well.
Retrieval already knew *which words* a query used; this is what tells it which
*kind* of material to prefer — a "what is X" question wants a definition, "how
do I X" wants a procedure, and the writer wants specifications over sales copy
whatever the words match.

Preferences are soft: they reorder candidates, they never remove them. A wrong
guess costs a few ranks, not a missing answer. The one hard exclusion is
navigation, which cannot answer anything by construction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from core import taxonomy as tax
from services.rag.analyzer import QueryProfile, QueryType

# Query shapes that name the kind of passage they want.
_PROCEDURE_RE = re.compile(
    r"\b(how (do|to|can)|steps?|install|configure|set ?up|deploy|guide|tutorial|"
    r"làm (sao|thế nào)|cách|hướng dẫn|cài đặt|thiết lập|triển khai)\b",
    re.IGNORECASE,
)
_EXAMPLE_RE = re.compile(
    r"\b(example|sample|snippet|demo|show me|ví dụ|mẫu|minh họa)\b",
    re.IGNORECASE,
)
_DATA_RE = re.compile(
    r"\b(how (much|many)|price|cost|number|figure|statistic|rate|limit|size|"
    r"bao nhiêu|giá|chi phí|số liệu|thống kê|giới hạn)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RetrievalPreferences:
    """Which classifications this query should favour."""

    preferred_roles: tuple[str, ...] = ()
    preferred_doc_types: tuple[str, ...] = ()
    demoted_doc_types: tuple[str, ...] = ()
    excluded_roles: tuple[str, ...] = (tax.ROLE_NAVIGATION,)
    min_quality_tier: str = tax.TIER_LOW
    diagnostics: dict = field(default_factory=dict)

    def role_bonus(self, role: str | None) -> float:
        """Ranking multiplier contribution for a chunk's role, in [0, 1]."""
        if not role:
            return 0.5
        if role in self.excluded_roles:
            return 0.0
        if not self.preferred_roles:
            return 0.5
        if role == self.preferred_roles[0]:
            return 1.0
        return 0.75 if role in self.preferred_roles else 0.35

    def doc_type_bonus(self, doc_type: str | None) -> float:
        """Ranking multiplier contribution for a source's document type."""
        if not doc_type:
            return 0.5
        if doc_type in self.demoted_doc_types:
            return 0.1
        if doc_type in self.preferred_doc_types:
            return 1.0
        # Fall back to the taxonomy's standing authority so a specification
        # still outranks marketing even when the query names neither.
        return tax.DOC_TYPE_AUTHORITY.get(doc_type, 0.5)


def for_query(profile: QueryProfile) -> RetrievalPreferences:
    """Derive retrieval preferences from an analyzed query."""
    query = profile.query
    roles: list[str] = []
    doc_types: list[str] = []

    # Order matters: the first matching shape is the primary role, and a
    # question can legitimately look like more than one ("how much does X cost"
    # is data first, procedure second).
    if _DATA_RE.search(query):
        roles += [tax.ROLE_DATA, tax.ROLE_CLAIM]
        doc_types += [tax.DOC_DATA, tax.DOC_REFERENCE]
    if _PROCEDURE_RE.search(query):
        roles += [tax.ROLE_PROCEDURE, tax.ROLE_EXAMPLE]
        doc_types += [tax.DOC_TUTORIAL, tax.DOC_REFERENCE]
    if _EXAMPLE_RE.search(query):
        roles += [tax.ROLE_EXAMPLE, tax.ROLE_PROCEDURE]
        doc_types += [tax.DOC_TUTORIAL]
    if profile.asks_definition:
        roles += [tax.ROLE_DEFINITION, tax.ROLE_CONTEXT]
        doc_types += [tax.DOC_REFERENCE, tax.DOC_EXPLAINER]

    if profile.query_type in (QueryType.CONTRADICTION, QueryType.CROSS_DOCUMENT):
        roles += [tax.ROLE_CLAIM, tax.ROLE_DEFINITION]
        doc_types += [tax.DOC_ANALYSIS, tax.DOC_REFERENCE]
    elif profile.query_type in (
        QueryType.TEMPORAL_CURRENT,
        QueryType.TEMPORAL_HISTORICAL,
    ):
        roles += [tax.ROLE_CLAIM, tax.ROLE_DATA]
        doc_types += [tax.DOC_NEWS, tax.DOC_REFERENCE]

    return RetrievalPreferences(
        preferred_roles=_dedupe(roles),
        preferred_doc_types=_dedupe(doc_types),
        # Sales copy is never the best answer to a question. It stays
        # retrievable — the user may be asking about a product — but it sinks.
        demoted_doc_types=(tax.DOC_MARKETING,),
        diagnostics={"query_type": str(profile.query_type)},
    )


def for_writing(keywords: list[str] | None = None) -> RetrievalPreferences:
    """Preferences for grounding generated content.

    The writer needs source material it can state as fact, so specifications
    and data outrank opinion, and marketing copy is demoted hardest — writing
    new marketing from old marketing is how a knowledge base starts repeating
    itself.
    """
    return RetrievalPreferences(
        preferred_roles=(
            tax.ROLE_DEFINITION,
            tax.ROLE_CLAIM,
            tax.ROLE_DATA,
            tax.ROLE_PROCEDURE,
        ),
        preferred_doc_types=(
            tax.DOC_REFERENCE,
            tax.DOC_DATA,
            tax.DOC_EXPLAINER,
            tax.DOC_ANALYSIS,
        ),
        demoted_doc_types=(tax.DOC_MARKETING, tax.DOC_DISCUSSION),
        excluded_roles=(tax.ROLE_NAVIGATION, tax.ROLE_REFERENCE),
        min_quality_tier=tax.TIER_MEDIUM,
        diagnostics={"mode": "writer", "keywords": list(keywords or [])[:8]},
    )


def _dedupe(values: list[str]) -> tuple[str, ...]:
    """Preserve first-seen order while removing repeats."""
    return tuple(dict.fromkeys(values))
