"""Claim / fact extraction (Brain Second P4-08, golden CASE B/C).

Splits a chunk into atomic ``SemanticUnitDraft``s, each carrying:

- ``polarity`` (+1 / -1 / 0) so "does **not** use CUDA 13" is never retrieved
  as a positive fact (CASE B — negation preserved).
- ``valid_from`` / ``valid_to`` from the temporal extractor (CASE C).
- byte ``span_start`` / ``span_end`` into the owning chunk text (provenance §41).
- ``fact_class`` (SOURCE_FACT | INFERRED_FACT | UNCERTAIN …, §41).

LLM-first via ``chat_json`` with a deterministic sentence-based heuristic
fallback — never sends every chunk to the LLM when a fast heuristic suffices
(§master "no LLM for every cleaning").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from services.extract.temporal import extract_temporal_window

# Negation markers flip polarity (CASE B).
_NEGATION_RE = re.compile(
    r"\b(not|no|never|without|doesn't|don't|doesn t|does not|do not|cannot|can't|"
    r"unable|lack|absent|never|không phải|không|chưa)\b",
    re.IGNORECASE,
)

# Hedges/uncertainty markers downgrade fact_class to UNCERTAIN.
_HEDGE_RE = re.compile(
    r"\b(maybe|might|possibly|perhaps|likely|unclear|unknown|seems|apparently|"
    r"có lẽ|chưa rõ)\b",
    re.IGNORECASE,
)

# Atomic sentence splitter (whitespace-friendly, keeps offsets).
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")

# Short sentence guard — atomic units must be meaningful.
_MIN_SENTENCE_CHARS = 24


@dataclass(frozen=True)
class SemanticUnitDraft:
    """One atomic claim ready to persist as a ``semantic_units`` row."""

    content: str
    unit_type: str  # fact | claim | definition | event | quote
    polarity: int  # +1 | -1 | 0
    fact_class: str  # SOURCE_FACT | INFERRED_FACT | UNCERTAIN | …
    span_start: int
    span_end: int
    valid_from: int | None = None
    valid_to: int | None = None
    entity_mentions: tuple[str, ...] = field(default_factory=tuple)
    confidence: float = 0.0


class ClaimExtractor:
    """Extract atomic claims from chunk text (LLM-first, heuristic fallback)."""

    def __init__(self, vllm=None) -> None:
        self.vllm = vllm

    async def extract(
        self,
        text: str,
        *,
        chunk_id: str | None = None,
    ) -> list[SemanticUnitDraft]:
        """Return atomic ``SemanticUnitDraft``s for ``text`` in offset order."""
        if not text or not text.strip():
            return []
        # LLM-assisted extraction only when available AND the text is long
        # enough to justify it (skip the LLM for tiny sentences).
        if self.vllm is not None and len(text) > 220:
            try:
                units = await self._llm_extract(text)
                if units:
                    return units
            except Exception:  # pragma: no cover - defensive
                pass
        return self._heuristic_extract(text)

    # ------------------------------------------------------------------

    async def _llm_extract(self, text: str) -> list[SemanticUnitDraft]:
        """LLM-first atomic extraction (golden CASE B/C aware)."""
        prompt = (
            "Split the following text into atomic factual claims. For each "
            "claim return JSON {claims:[{content, polarity(-1|0|1), "
            "fact_class(SOURCE_FACT|INFERRED_FACT|UNCERTAIN), "
            "valid_from, valid_to}]} — preserve negation exactly (polarity=-1 "
            "for negative statements) and capture any explicit year ranges.\n\n"
            f"TEXT:\n{text}"
        )
        data = await self.vllm.chat_json(
            [{"role": "user", "content": prompt}],
            system="You are a precise knowledge-extraction engine.",
            temperature=0.0,
            max_tokens=600,
        )
        drafts: list[SemanticUnitDraft] = []
        for item in (data.get("claims") if isinstance(data, dict) else None) or []:
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            polarity = int(item.get("polarity", 1) or 1)
            polarity = max(-1, min(1, polarity))
            fact_class = str(item.get("fact_class", "SOURCE_FACT"))
            start = text.find(content)
            end = start + len(content) if start >= 0 else 0
            if start < 0:
                start, end = 0, len(content)
            tw = extract_temporal_window(content)
            drafts.append(
                SemanticUnitDraft(
                    content=content,
                    unit_type="claim",
                    polarity=polarity,
                    fact_class=fact_class,
                    span_start=start,
                    span_end=end,
                    valid_from=tw.valid_from,
                    valid_to=tw.valid_to,
                    confidence=0.7 if polarity != 0 else 0.4,
                )
            )
        return drafts

    def _heuristic_extract(self, text: str) -> list[SemanticUnitDraft]:
        """Deterministic sentence-based fallback (CASE B/C aware)."""
        drafts: list[SemanticUnitDraft] = []
        position = 0
        for raw in _SENT_SPLIT_RE.split(text):
            stripped = raw.strip()
            if not stripped:
                position += len(raw) + 1
                continue
            local_start = text.find(stripped, position)
            start = local_start if local_start >= 0 else position
            end = start + len(stripped)
            position = end + 1
            if len(stripped) < _MIN_SENTENCE_CHARS:
                continue
            negated = _NEGATION_RE.search(stripped) is not None
            uncertain = _HEDGE_RE.search(stripped) is not None
            tw = extract_temporal_window(stripped)
            fact_class = "UNCERTAIN" if uncertain else "SOURCE_FACT"
            drafts.append(
                SemanticUnitDraft(
                    content=stripped,
                    unit_type="fact",
                    polarity=-1 if negated else (0 if uncertain else 1),
                    fact_class=fact_class,
                    span_start=start,
                    span_end=end,
                    valid_from=tw.valid_from,
                    valid_to=tw.valid_to,
                    confidence=0.5 if not uncertain else 0.3,
                )
            )
        return drafts


def extract_claims(text: str) -> list[SemanticUnitDraft]:
    """Synchronous convenience wrapper around the deterministic heuristic path."""
    return ClaimExtractor(vllm=None)._heuristic_extract(text)
