"""Entity extractor — mentions with evidence spans (Phase 6.1).

Two strategies compose:
- **LLM-assisted** (primary when available): a mocked-safe ``chat_json``
  call returns ``{"entities": [{"name", "type", "start", "end"}]}``.
- **Heuristic fallback** (orthographic/dictionary patterns) so extraction
  works deterministically offline and in unit tests.

Every mention carries a byte span into the *original* document so the
downstream semantic_units table can attach provenance (§41) — never a
distance into the (possibly re-split) chunk text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Fallback pattern tables (used only when LLM is unavailable / fails)
# ---------------------------------------------------------------------------

# (regex pattern, entity_type) — ordered: first match wins.
_HEURISTIC_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Backtick-quoted identifiers are usually code/software/product names.
    (re.compile(r"`([A-Za-z][A-Za-z0-9_.\-/]{1,63})`"), "software"),
    # PascalCase (camelCase w/ leading cap) often a class/library/tool.
    (re.compile(r"\b([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+)\b"), "software"),
    # Uppercase acronyms (SQL, HTTP, NER, RAG …) as concepts/frameworks.
    (re.compile(r"\b([A-Z]{2,})\b"), "concept"),
    # Title-cased proper nouns (PostgreSQL, Docker, Obsidian …).
    (re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"), "organization"),
)

# Common Vietnamese/English stopwords to avoid shouting as entities.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "và",
        "của",
        "là",
        "với",
        "các",
        "cho",
        "một",
        "trong",
        "như",
        "này",
        "đó",
        "không",
        "and",
        "of",
        "for",
        "with",
        "that",
        "this",
        "from",
        "have",
        "has",
        "were",
        "was",
        "are",
        "not",
        "its",
        "his",
        "her",
    }
)

_ACRONYM_RE = re.compile(r"^[A-Z]{2,}$")


@dataclass(frozen=True)
class EntityMention:
    """A candidate entity found in a text span."""

    text: str
    entity_type: str
    start: int
    end: int

    @property
    def span(self) -> tuple[int, int]:
        return (self.start, self.end)


class EntityExtractor:
    """Extract entity-mention candidates from a text blob."""

    def __init__(self, vllm=None):
        self.vllm = vllm

    async def extract(self, text: str) -> list[EntityMention]:
        """Extract mentions, preferring LLM when available, else heuristics."""
        if self.vllm is not None:
            try:
                mentions = await self._extract_with_llm(text)
                if mentions:
                    return mentions
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("entity.extract_llm_failed", error=str(exc))
        return _heuristic_extract(text)

    async def _extract_with_llm(self, text: str) -> list[EntityMention]:
        """LLM round-trip for ``{"entities": [...]}`` with offset evidence."""
        prompt = (
            'Return JSON: {"entities": [{"name": "str", "type": "str", '
            '"start": "int", "end": "int"}]} — start/end are byte offsets '
            "into the text below.\n\n"
            f"TEXT:\n{text}"
        )
        payload = await self.vllm.chat_json(
            [{"role": "user", "content": prompt}],
            system="You extract entities with byte offsets.",
            temperature=0.0,
            max_tokens=600,
        )
        mentions: list[EntityMention] = []
        for ent in (payload or {}).get("entities") or []:
            name = str(ent.get("name", "")).strip()
            if not name:
                continue
            start = int(ent.get("start", text.find(name)))
            end = int(ent.get("end", start + len(name)))
            mentions.append(
                EntityMention(
                    text=name,
                    entity_type=str(ent.get("type", "concept")),
                    start=max(start, 0),
                    end=min(end, len(text)),
                )
            )
        return mentions


def extract_mentions(text: str) -> list[EntityMention]:
    """Convenience: extract mentions using deterministic heuristics only."""
    return _heuristic_extract(text)


def _heuristic_extract(text: str) -> list[EntityMention]:
    """Orthographic + dictionary extraction (offline fallback).

    Spans are computed against the *original* text so they remain valid for
    the semantic_units provenance contract.
    """
    mentions: list[EntityMention] = []
    seen: set[tuple[str, int, int]] = set()

    for pattern, entity_type in _HEURISTIC_PATTERNS:
        for match in pattern.finditer(text):
            raw = match.group(1)
            if _is_stopword(raw):
                continue
            start = match.start(1)
            span = (raw, start, start + len(raw))
            if span in seen:
                continue
            seen.add(span)
            # Skip acronyms already caught by the acronym pattern.
            if _ACRONYM_RE.match(raw) and entity_type != "concept":
                continue
            mentions.append(EntityMention(raw, entity_type, start, start + len(raw)))

    # De-duplicate overlapping mentions of the same name — keep the longest.
    mentions.sort(key=lambda m: (m.start, -(m.end - m.start)))
    deduped: list[EntityMention] = []
    for mention in mentions:
        if mention.text in {m.text for m in deduped}:
            continue
        deduped.append(mention)
    return deduped


def _is_stopword(token: str) -> bool:
    return token.strip(".,;:!?()[]{}'\"") in _STOPWORDS or token in _STOPWORDS