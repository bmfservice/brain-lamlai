"""Temporal extraction (Brain Second P4-10, golden CASE C).

Detects explicit year/date mentions and resolves them to a ``valid_from`` /
``valid_to`` window so retrieval can honour "current-fact priority" —
"GPU năm 2025?" must surface the RTX 3060 fact, "GPU hiện tại?" the RTX 5090.

Pure and deterministic: LLM-assisted refinement is a future hook; the
heuristic baseline never sends content to an LLM for every extraction
(§master "no LLM for every cleaning").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# "In 2025 ..." / "as of 2026" / "since 2024" / "valid 2025-2026"
_YEAR_RE = re.compile(
    r"\b(?:in|as of|since|from|by|valid(?:ity)?)\s+"
    r"(\d{4})\b"
    r"(?:\s*(?:-|to|–|until)\s*(\d{4}))?",
    re.IGNORECASE,
)
_BARE_YEAR_RE = re.compile(r"\b(19|20)(\d{2})\b")

# Phrases that mark a claim as the *current* state ("hiện tại", "now").
_CURRENT_RE = re.compile(r"\b(current|now|hiện tại|hiện tại)\b", re.IGNORECASE)


@dataclass(frozen=True)
class TemporalWindow:
    """Resolved validity window for a claim (CASE C)."""

    valid_from: int | None = None  # year
    valid_to: int | None = None  # year (None = open-ended / current)
    is_current: bool = False

    def is_open(self) -> bool:
        return self.valid_from is None and self.valid_to is None


def extract_temporal_window(text: str, *, reference_year: int = 2026) -> TemporalWindow:
    """Best-effort temporal window for ``text`` (CASE C).

    Two signals, in order:
    1. Explicit "in/as of/since YYYY[-YYYY]" → window from the phrase.
    2. "current"/"now"/"hiện tại" → open-ended current window.

    When both appear, the explicit year wins (it is the more specific signal).
    """
    if not text:
        return TemporalWindow()

    explicit = _YEAR_RE.search(text)
    if explicit:
        start = int(explicit.group(1))
        end_raw = explicit.group(2)
        if end_raw:
            return TemporalWindow(valid_from=start, valid_to=int(end_raw))
        # "since 2024" → open-ended current; "in 2025" → point-in-time window.
        is_since = bool(re.search(r"\bsince\b", explicit.group(0), re.IGNORECASE))
        if is_since:
            return TemporalWindow(valid_from=start, valid_to=None, is_current=True)
        return TemporalWindow(valid_from=start, valid_to=start)

    if _CURRENT_RE.search(text):
        return TemporalWindow(valid_from=None, valid_to=None, is_current=True)

    # Bare-year fallback: a single 4-digit year scoped to the sentence.
    bare = _BARE_YEAR_RE.search(text)
    if bare:
        year = int(bare.group(0))
        if 1900 <= year <= 2100:
            return TemporalWindow(valid_from=year, valid_to=year)
    return TemporalWindow()


def is_window_active_at(window: TemporalWindow, *, year: int) -> bool:
    """Whether a claim with ``window`` is *current* as of ``year`` (CASE C).

    Open-ended current windows (``is_current``) are always active.
    """
    if window.is_current and window.valid_to is None:
        return True
    if window.valid_from is not None and window.valid_to is not None:
        return window.valid_from <= year <= window.valid_to
    if window.valid_from is not None:
        return year >= window.valid_from
    return True  # no temporal signal → timeless
