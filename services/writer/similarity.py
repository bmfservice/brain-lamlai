"""Braind Writer — pure-Python lexical similarity utilities.

Used for the external duplicate check: given generated content and a candidate
web page, compute an automated similarity signal (not a legal plagiarism
verdict). Everything is stdlib-only: Unicode NFKC normalization, word-level
and 5-word shingle overlap, plus a plain dot-product cosine over an embedded
vector when a vector is provided.

Design notes:
- Lexical overlap (containment + n-gram Jaccard) is the headline signal because
  it is deterministic and explainable for the user's "trùng" report.
- Semantic cosine (via local vLLM embeddings) is reported separately so the two are
  never conflated.
- _Stopwords here are normalized to ASCII so 'của' and 'tìm-kiếm' compare well.
"""

from __future__ import annotations

import math
import re
import unicodedata

# Stopwords chuẩn hóa ASCII (bỏ dấu) để so sánh chữ không bị trừng phạt.
_STOPWORDS = {
    "cua",
    "va",
    "trong",
    "la",
    "co",
    "khong",
    "mot",
    "nhung",
    "duoc",
    "cho",
    "voi",
    "theo",
    "cac",
    "nay",
    "do",
    "nhu",
    "ve",
    "ra",
    "vao",
    "the",
    "and",
    "with",
    "for",
    "from",
    "that",
    "this",
    "are",
    "was",
    "to",
    "of",
    "in",
    "on",
    "at",
    "a",
    "an",
    "by",
    "is",
    "be",
    "it",
}

_NOT_WORD = re.compile(r"[^a-z0-9]+")
_JUNK_TAG = re.compile(r"<[^>]+>")
_MD_TAIL = re.compile(r"[#>*`_\-\[\]()!]")
_WORD_RE = re.compile(r"[a-z0-9]{2,}")


def normalize_text(text: str) -> str:
    """Normalize text for comparison: NFKC, lowercase, words joined by '-'.

    Removes markup, punctuation and collapses whitespace. Vietnamese gets
    diacritic-folded to ASCII so dấu differences don't count against overlap.
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = unicodedata.normalize("NFD", t)
    # Combine accents back into one code point for non-Latin scripts, then
    # strip combining marks globally (turns 'ế' into 'e').
    t = "".join(ch for ch in unicodedata.normalize("NFC", t) if not unicodedata.combining(ch))
    t = t.lower()
    t = _JUNK_TAG.sub(" ", t)
    t = _MD_TAIL.sub(" ", t)
    t = _NOT_WORD.sub(" ", t)
    return " ".join(t.split())


def tokenize(text: str) -> list[str]:
    """Split normalized/raw text into lowercase word tokens."""
    if not text:
        return []
    norm = normalize_text(text)
    return re.findall(_WORD_RE.pattern, norm)


def shingles(tokens: list[str], n: int = 5) -> set[str]:
    """Build the set of contiguous n-word shingles (joined by '\x1f')."""
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def word_jaccard(a_tokens: list[str], b_tokens: list[str]) -> float:
    """Jaccard overlap on distinct word sets (0.0–1.0)."""
    sa, sb = set(a_tokens), set(b_tokens)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def shingle_containment(a_tokens: list[str], b_tokens: list[str], n: int = 5) -> float:
    """Fraction of A's shingles present in B — how much of A appears verbatim.

    This is the headline "trùng" signal: 1.0 means the generated text is fully
    copied from the candidate page; 0.0 means no shared phrase chains.
    """
    sa, sb = shingles(a_tokens, n), shingles(b_tokens, n)
    if not sa:
        return 0.0
    return len(sa & sb) / len(sa)


def combined_similarity(
    a_tokens: list[str],
    b_tokens: list[str],
    *,
    n: int = 5,
    shingle_weight: float = 0.7,
    word_weight: float = 0.3,
) -> float:
    """Score combining shingle containment + word Jaccard (0.0–1.0)."""
    if not a_tokens or not b_tokens:
        return 0.0
    sc = shingle_containment(a_tokens, b_tokens, n)
    wj = word_jaccard(a_tokens, b_tokens)
    return round(shingle_weight * sc + word_weight * wj, 4)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Plain-vector cosine similarity (pure Python, no numpy)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return round(dot / (na * nb), 4)


def longest_shared_fragment(a_tokens: list[str], b_tokens: list[str], min_len: int = 6) -> str:
    """Return a representative shared phrase (for matched_excerpt display)."""
    sa, sb = set(a_tokens), set(b_tokens)
    boundary = max(4, min_len)
    # Try increasingly explicit matches first.
    for n in (boundary, boundary - 1, 4, 3, 2):
        sa_ng, sb_ng = shingles(a_tokens, n), shingles(b_tokens, n)
        common = sa_ng & sb_ng
        if common:
            return " ".join(sorted(common, key=len, reverse=True)[0])
    inter = sa & sb
    return " ".join(sorted(inter, key=len, reverse=True)[:4]) if inter else ""


__all__ = [
    "normalize_text",
    "tokenize",
    "shingles",
    "word_jaccard",
    "shingle_containment",
    "combined_similarity",
    "cosine_similarity",
    "longest_shared_fragment",
]
