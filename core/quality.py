"""Braind Core - content quality gate.

Everything entering the knowledge base passes through here first. The audit of
2026-08-20 found that a third of stored "knowledge" was search-engine snippets,
navigation menus, password-reset pages and raw PDF byte streams — material that
poisons retrieval for chat and the writer.

The gate is deterministic and LLM-free: it strips repeated site chrome, scores
how much of the text is real prose, and rejects what is structurally not
knowledge. Classification (what a document *is*) happens later, in
``services/ingest/classify.py``, and only for text that survives this stage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from core.textanalysis import contains_chinese

# Lines that are navigation chrome regardless of the site they came from.
_CHROME_PATTERNS = (
    r"^skip to (main )?content$",
    r"^(accept|allow) (all )?cookies?$",
    r"^(we use|this (site|website) uses) cookies",
    r"^(sign|log) ?(in|out)$",
    r"^subscribe( now)?$",
    r"^(share|tweet|print|email) this",
    r"^(read|xem) (more|thêm)$",
    r"^(all rights reserved|©.*)$",
    r"^powered by ",
    r"^(privacy|cookie) policy$",
    r"^terms (of (use|service)|and conditions)$",
    r"^(back to )?top$",
    r"^menu$",
    r"^(trang chủ|đăng nhập|đăng ký|liên hệ)$",
)
_CHROME_RE = re.compile("|".join(_CHROME_PATTERNS), re.IGNORECASE)

# A "prose line" ends like a sentence. Bare labels and menu entries do not.
_SENTENCE_END_RE = re.compile(r"[.!?…:;][\"')\]]*$")
# Sentence punctuation anywhere inside a block. A full stop between words marks
# prose; requiring it at the very end fails on text truncated mid-paragraph,
# which extraction does routinely.
_SENTENCE_PUNCT_RE = re.compile(r"\w[.!?…](\s|$)")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?…]+")
_WORD_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)
_URL_RE = re.compile(r"https?://\S+")
# Control characters that never appear in extracted prose — their presence means
# binary payload (PDF object streams) leaked through the extractor.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

REJECT_EMPTY = "empty"
REJECT_BINARY = "binary"
REJECT_TOO_SHORT = "too_short"
REJECT_CHINESE = "chinese"
REJECT_NOT_PROSE = "not_prose"
REJECT_LINK_FARM = "link_farm"
REJECT_BOILERPLATE = "boilerplate"


@dataclass(frozen=True)
class QualityReport:
    """Verdict and measurements for one candidate document."""

    accepted: bool
    text: str
    reason: str = ""
    prose_ratio: float = 0.0
    link_density: float = 0.0
    word_count: int = 0
    sentence_count: int = 0
    chrome_lines_removed: int = 0
    signals: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        """A 0-1 quality score used to rank sources during retrieval."""
        if not self.accepted:
            return 0.0
        # Prose ratio dominates; length is a weak secondary signal that saturates
        # quickly, so a focused 300-word note is not punished against a 5000-word
        # page of the same density.
        length_factor = min(1.0, self.word_count / 400.0)
        return round(
            max(0.0, min(1.0, 0.7 * self.prose_ratio + 0.3 * length_factor)), 4
        )


def strip_chrome(text: str, *, repeated: set[str] | None = None) -> tuple[str, int]:
    """Remove navigation chrome, returning ``(cleaned_text, lines_removed)``.

    ``repeated`` holds normalized lines already known to appear across other
    pages of the same site; they are dropped even when they look like prose,
    because a line that opens nineteen different documents is a site header.
    """
    repeated = repeated or set()
    kept: list[str] = []
    removed = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            kept.append("")
            continue
        if _CHROME_RE.match(line) or normalize_line(line) in repeated:
            removed += 1
            continue
        kept.append(raw)
    # Collapse the blank runs left behind by removed lines.
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    return cleaned, removed


def normalize_line(line: str) -> str:
    """Canonical form of a line for cross-page repetition matching."""
    return re.sub(r"\s+", " ", line.strip().lower())


def find_repeated_lines(pages: list[str], *, min_pages: int = 3) -> set[str]:
    """Return lines that appear on at least ``min_pages`` of ``pages``.

    Used by the site crawler: chrome is whatever the site repeats, which no
    fixed pattern list can know in advance.
    """
    if len(pages) < min_pages:
        return set()
    counts: dict[str, int] = {}
    for page in pages:
        for line in {normalize_line(ln) for ln in page.splitlines() if ln.strip()}:
            counts[line] = counts.get(line, 0) + 1
    return {line for line, count in counts.items() if count >= min_pages}


def reflow(text: str) -> list[str]:
    """Join wrapped lines back into paragraphs.

    HTML extraction emits one line per inline element, so a single sentence
    arrives split across several lines while a nav menu arrives as one short
    line each. Rejoining continuation lines is what makes the two shapes
    distinguishable: measuring raw lines scores a real article at 0.06.
    """
    blocks: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            blocks.append(" ".join(buffer).strip())
            buffer.clear()

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        buffer.append(line)
        if _SENTENCE_END_RE.search(line):
            flush()
    flush()
    return [b for b in blocks if b]


def measure(text: str) -> tuple[float, float, int, int]:
    """Return ``(prose_ratio, link_density, word_count, sentence_count)``.

    ``prose_ratio`` is the share of *words* sitting inside paragraph-shaped
    blocks. Weighting by words rather than by lines is what separates an
    article carrying a few nav labels from a menu carrying one stray sentence:
    both have many short lines, but only one keeps its mass in prose.

    A block counts as prose when it is long enough to carry a clause and
    contains sentence punctuation *anywhere* — requiring it at the end would
    score a whole article at zero whenever extraction truncates the final
    paragraph mid-sentence, which is common. A menu accumulates into one long
    block too, but never acquires a full stop.
    """
    blocks = reflow(text)
    if not blocks:
        return 0.0, 0.0, 0, 0

    words_total = 0
    words_prose = 0
    for block in blocks:
        count = len(_WORD_RE.findall(block))
        words_total += count
        if count >= 12 and _SENTENCE_PUNCT_RE.search(block):
            words_prose += count

    urls = len(_URL_RE.findall(text))
    # Per 100 words: stable across page length, unlike a per-block ratio that
    # swings with how aggressively extraction wrapped the text.
    link_density = urls / max(words_total / 100.0, 1.0)
    sentences = len([s for s in _SENTENCE_SPLIT_RE.split(text) if len(_WORD_RE.findall(s)) >= 4])
    ratio = words_prose / words_total if words_total else 0.0
    return ratio, link_density, words_total, sentences


def assess(
    text: str,
    *,
    min_words: int = 80,
    min_sentences: int = 3,
    min_prose_ratio: float = 0.25,
    max_link_density: float = 2.0,
    repeated_lines: set[str] | None = None,
    source_type: str = "",
) -> QualityReport:
    """Clean ``text`` and decide whether it is worth storing.

    Trusted local sources (vault notes, uploaded files, manual paste, chat
    turns) skip the prose thresholds: a config file or a table of figures is
    legitimately not prose, and the user put it there deliberately. Web-derived
    text gets the full gate.
    """
    if not text or not text.strip():
        return QualityReport(False, "", REJECT_EMPTY)

    if _CONTROL_RE.search(text):
        # Extraction leaked binary; ingesting it would store unsearchable noise
        # and pollute the semantic layer downstream.
        return QualityReport(False, "", REJECT_BINARY)

    cleaned, chrome_removed = strip_chrome(text, repeated=repeated_lines)
    if not cleaned:
        return QualityReport(False, "", REJECT_BOILERPLATE, chrome_lines_removed=chrome_removed)

    if contains_chinese(cleaned):
        return QualityReport(False, "", REJECT_CHINESE, chrome_lines_removed=chrome_removed)

    prose_ratio, link_density, words, sentences = measure(cleaned)
    signals: list[str] = []
    if chrome_removed:
        signals.append(f"chrome:{chrome_removed}")

    trusted = source_type in {"obsidian", "file", "manual", "chat", "vault"}
    if trusted:
        # Still reject the genuinely empty, but never judge structure.
        if words < 20:
            return QualityReport(
                False, "", REJECT_TOO_SHORT, prose_ratio, link_density, words, sentences,
                chrome_removed, signals,
            )
        signals.append("trusted_source")
        return QualityReport(
            True, cleaned, "", prose_ratio, link_density, words, sentences,
            chrome_removed, signals,
        )

    if words < min_words or sentences < min_sentences:
        # The SERP-snippet shape: a title plus two lines of blurb.
        return QualityReport(
            False, "", REJECT_TOO_SHORT, prose_ratio, link_density, words, sentences,
            chrome_removed, signals,
        )
    if link_density > max_link_density and prose_ratio < 0.5:
        # Dense links only condemn a page when there is little prose around
        # them. A well-sourced article cites heavily and must not be rejected
        # for it; a link directory has the links and nothing else.
        return QualityReport(
            False, "", REJECT_LINK_FARM, prose_ratio, link_density, words, sentences,
            chrome_removed, signals,
        )
    if prose_ratio < min_prose_ratio:
        return QualityReport(
            False, "", REJECT_NOT_PROSE, prose_ratio, link_density, words, sentences,
            chrome_removed, signals,
        )

    return QualityReport(
        True, cleaned, "", prose_ratio, link_density, words, sentences,
        chrome_removed, signals,
    )


def sanitize_model_text(text: str, *, fallback: str = "") -> str:
    """Return model-generated text, or ``fallback`` when it violates policy.

    The Han-ideograph rule applies to what the model writes, not only to what
    the user ingests: the audit found 27 source summaries and 52 document
    summaries stored in Chinese despite every input passing the gate.
    """
    candidate = (text or "").strip()
    if not candidate or contains_chinese(candidate):
        return fallback
    return candidate
