"""Query analyzer (Brain Second Phase 8, golden cases C/E).

Classifies each search query so retrieval can adapt its strategy per §65's
eleven query types instead of running one blind blend:

- **exact** ("what is PostgreSQL")      — strongly rely on entity/FTS/vector
- **semantic** ("why does it scale")    — rely on vector + graph
- **multi_hop** (≥2 named entities)     — add graph expansion
- **temporal_current** ("GPU hiện tại") — current-fact priority (CASE C)
- **temporal_historical** ("GPU năm 2025") — time-scoped facts (CASE C)
- **contradiction** ("X vs Y", conflicting)   — surface both sides + SUPPORTS/CONTRADICTS
- **entity_alias** (query ≈ canonical/alias)  — expand to aliases (CASE E)
- **negative** (contains negation)      — never return "does not X" as positive (CASE B)
- **ambiguous** (polysemous/generic)    — ask-for-clarity or broad coverage
- **cross_document** ("compare A and B")— require ≥2 source documents
- **global** (intent sweep/questions)   — broad recall over the workspace

The analyzer is deterministic and pure — it never sends the query to an LLM
(§123 no whole-dataset / per-query LLM where a heuristic suffices).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from services.extract.temporal import TemporalWindow, extract_temporal_window

# Negation markers (CASE B) — reused from the claim extractor's vocabulary.
_NEGATION_RE = re.compile(
    r"\b(not|no|never|without|doesn't|don't|doesn t|does not|do not|cannot|can't|"
    r"unable|lack|absent|không phải|không|chưa)\b",
    re.IGNORECASE,
)

# Comparison / contradiction phrasing.
_CONTRAST_RE = re.compile(
    r"\b(vs\.?|versus|compared? to|contrast|differen[ct]|opposite|khác|"
    r"contradict|disagree)\b",
    re.IGNORECASE,
)

# Definition-style / exact-detail queries.
_EXACT_RE = re.compile(
    r"\b(what is|what are|who is|where is|when did|when was|when is|is gì|là gì|"
    r"nghĩa là|what does|define|definition|explain what)\b",
    re.IGNORECASE,
)
_WH_ANY_RE = re.compile(r"(^|\s)(what|why|how|which|who|where|when)\b", re.IGNORECASE)

# Cross-document comparison phrasing.
_CROSS_DOC_RE = re.compile(
    r"\b(difference|differences|compare|comparison|similarities|vs\.?|"
    r"versus|so sánh|khác biệt)\b",
    re.IGNORECASE,
)

# Upper-cased tokens and Title-Cased words are the candidate named entities.
_TOKEN_RE = re.compile(r"[\wÀ-ỹ]{2,}", re.UNICODE)
_STOP_WORDS = {
    "của", "và", "các", "cho", "trong", "với", "những", "được", "là", "có",
    "từ", "một", "the", "and", "for", "with", "what", "why", "how", "which",
    "who", "where", "when", "this", "that", "these", "those", "not", "no",
}

# Total named-entity count to consider a query multi-hop (＞1 named entity).
_MULTI_HOP_MIN_ENTITIES = 2


class QueryType(StrEnum):
    """The §65 query families the analyzer can dispatch."""

    EXACT = "exact"
    SEMANTIC = "semantic"
    MULTI_HOP = "multi_hop"
    TEMPORAL_CURRENT = "temporal_current"
    TEMPORAL_HISTORICAL = "temporal_historical"
    CONTRADICTION = "contradiction"
    ENTITY_ALIAS = "entity_alias"
    NEGATIVE = "negative"
    AMBIGUOUS = "ambiguous"
    CROSS_DOCUMENT = "cross_document"
    GLOBAL = "global"


@dataclass(frozen=True)
class QueryProfile:
    """Structured view of one search query (Phase 8 output)."""

    query: str
    query_type: QueryType
    terms: tuple[str, ...]  # deduped, stop-word-free lowercase terms
    entity_candidates: tuple[str, ...]  # title-cased / capitalized spans
    temporal: TemporalWindow
    negated: bool
    asks_definition: bool
    diagnostics: dict = field(default_factory=dict)


class QueryAnalyzer:
    """Deterministic query classification for the unified retriever."""

    def __init__(self, reference_year: int = 2026) -> None:
        self.reference_year = reference_year

    def analyze(self, query: str) -> QueryProfile:
        """Classify ``query`` and pull out its retrieval-relevant signals."""
        if not query or not query.strip():
            return QueryProfile(
                query=query or "",
                query_type=QueryType.GLOBAL,
                terms=(),
                entity_candidates=(),
                temporal=TemporalWindow(),
                negated=False,
                asks_definition=False,
                diagnostics={},
            )

        terms = tuple(self._terms(query))
        entity_candidates = tuple(self._entity_candidates(query))
        negated = _NEGATION_RE.search(query) is not None
        asks_definition = _EXACT_RE.search(query) is not None
        temporal = extract_temporal_window(query, reference_year=self.reference_year)

        query_type = self._classify(
            query=query,
            terms=terms,
            entity_count=len(entity_candidates),
            negated=negated,
            asks_definition=asks_definition,
            temporal=temporal,
        )
        return QueryProfile(
            query=query,
            query_type=query_type,
            terms=terms,
            entity_candidates=entity_candidates,
            temporal=temporal,
            negated=negated,
            asks_definition=asks_definition,
            diagnostics={"reference_year": self.reference_year},
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _terms(query: str) -> list[str]:
        seen: list[str] = []
        for token in _TOKEN_RE.findall(query):
            t = token.lower()
            if t not in _STOP_WORDS and t not in seen:
                seen.append(t)
        return seen

    @staticmethod
    def _entity_candidates(query: str) -> list[str]:
        """Extract named-entity candidates, joining consecutive title-cased tokens."""
        tokens = _TOKEN_RE.findall(query)
        candidates: list[str] = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.lower() in _STOP_WORDS or not (token[0].isupper() or _is_all_caps(token)):
                i += 1
                continue
            # Consume a run of capitalised tokens (e.g. "Docker Compose").
            run = [token]
            j = i + 1
            while j < len(tokens) and tokens[j][0:1].isupper():
                run.append(tokens[j])
                j += 1
            candidate = " ".join(run)
            if candidate not in candidates:
                candidates.append(candidate)
            i = j
        return candidates

    @staticmethod
    def _classify(
        *,
        query: str,
        terms: tuple[str, ...],
        entity_count: int,
        negated: bool,
        asks_definition: bool,
        temporal: TemporalWindow,
    ) -> QueryType:
        """§65 dispatch — most specific class wins, globals last."""
        if negated:
            return QueryType.NEGATIVE
        if _CONTRAST_RE.search(query):
            if _CROSS_DOC_RE.search(query) or entity_count >= 2:
                return QueryType.CROSS_DOCUMENT
            return QueryType.CONTRADICTION
        # Temporal signals are strong enough to override definition-style wording.
        if temporal.is_current:
            return QueryType.TEMPORAL_CURRENT
        if temporal.valid_from is not None:
            if temporal.valid_to is not None and temporal.valid_to >= QueryAnalyzer._REF_YEAR:
                return QueryType.TEMPORAL_CURRENT
            return QueryType.TEMPORAL_HISTORICAL
        if entity_count >= _MULTI_HOP_MIN_ENTITIES:
            return QueryType.MULTI_HOP
        if asks_definition:
            return QueryType.EXACT
        if _WH_ANY_RE.search(query) and len(terms) >= 3:
            return QueryType.SEMANTIC
        if entity_count == 1 and terms:
            return QueryType.ENTITY_ALIAS
        if _WH_ANY_RE.search(query) or len(terms) <= 2:
            return QueryType.GLOBAL
        return QueryType.SEMANTIC

    _REF_YEAR = 2026


def _is_all_caps(token: str) -> bool:
    return token.isupper() and any(c.isalpha() for c in token)


def analyze_query(query: str, *, reference_year: int = 2026) -> QueryProfile:
    """Synchronous convenience wrapper around :class:`QueryAnalyzer`."""
    return QueryAnalyzer(reference_year=reference_year).analyze(query)