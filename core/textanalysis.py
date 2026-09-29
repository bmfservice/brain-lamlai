"""Braind Text Analysis — LLM-assisted summary + keyword extraction.

Dùng model local (qwen2.5) để:
- Tóm tắt nội dung (summary) → lưu vào source meta
- Trích từ khóa / chủ đề (keywords)
- Nhận diện chủ đề liên quan (topics) cho link graph
"""

from __future__ import annotations

import re

from core.logging import get_logger
from core.vllm import VLLMClient

logger = get_logger(__name__)

_SUMMARY_PROMPT = (
    "Tóm tắt nội dung dưới đây bằng tiếng Việt, 2-4 câu, nêu rõ ý chính. "
    "Chỉ trả về phần tóm tắt, không thêm lời dẫn."
)
_KEYWORDS_PROMPT = (
    "Trích ra tối đa 8 từ khóa / cụm từ quan trọng nhất từ nội dung (giữ nguyên ngôn ngữ gốc). "
    "Trả về dạng danh sách mỗi từ khóa trên một dòng, không đánh số."
)


def truncate(text: str, limit: int) -> str:
    return text[:limit] if text else ""


def normalize_label(label: object) -> str:
    """Canonical string normalization for labels, codes, tags and enum identifiers.

    Transforms whitespace and hyphens to standard underscores, strips
    leading/trailing whitespace, and lowercases. Returns empty string for None
    or non-string inputs.
    """
    if not isinstance(label, str):
        return ""
    return label.strip().lower().replace(" ", "_").replace("-", "_")


# Chinese ideograph ranges (CJK Unified Ideographs + Extension A).
# Deliberately EXCLUDES Japanese kana and Korean Hangul — only Han ideographs
# are prohibited per project policy.
#   U+3400–U+4DBF  CJK Unified Ideographs Extension A
#   U+4E00–U+9FFF  CJK Unified Ideographs
_CHINESE_RE = re.compile(r"[㐀-䶿一-鿿]")


def contains_chinese(text: str) -> bool:
    """Return ``True`` if ``text`` contains any Chinese (Han) ideograph.

    Matches CJK Unified Ideographs (``U+4E00–U+9FFF``) and Extension A
    (``U+3400–U+4DBF``). Japanese kana and Korean Hangul are NOT matched —
    only Han ideographs trigger the prohibition.

    Args:
        text: Arbitrary text to scan.

    Returns:
        ``True`` if at least one Han ideograph is present, else ``False``.
    """
    if not text:
        return False
    return _CHINESE_RE.search(text) is not None


# Detection of Han ideographs is the hard pre-ingest gate
# (``detect_language`` supplements — it never overrides ``contains_chinese``).
_HAN_RE = _CHINESE_RE
_KANA_RE = re.compile(r"[぀-ヿㇰ-ㇿ]")  # Hiragana / Katakana
_HANGUL_RE = re.compile(r"[가-힯ᄀ-ᇿ]")
_CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
_GREEK_RE = re.compile(r"[Ͱ-Ͽ]")
_ARABIC_RE = re.compile(r"[؀-ۿ]")

# Per-language stopword sets for Latin-script disambiguation. Frequencies are
# counted on the tokenized lowercase sample; the language with the largest
# stopword hit rate (normalized by its set size) wins.
_LATIN_STOPWORDS: dict[str, set[str]] = {
    "en": {"the", "and", "of", "to", "in", "is", "that", "for", "on", "with",
           "as", "this", "be", "by", "are", "it", "or", "from"},
    "vi": {"của", "và", "trong", "là", "có", "không", "một", "những", "được",
           "cho", "với", "theo", "các", "người", "này"},
    "fr": {"le", "la", "les", "de", "des", "un", "une", "et", "est", "que",
           "qui", "dans", "pour", "avec", "sur", "ces", "cette", "ce"},
    "de": {"der", "die", "das", "und", "ist", "den", "ein", "eine", "auf",
           "mit", "für", "von", "dem"},
}

# Primary, unambiguous script markers → language (highest confidence).
_SCRIPT_LANGUAGES: list[tuple[re.Pattern[str], str]] = [
    (_HAN_RE, "zh"),
    (_KANA_RE, "ja"),
    (_HANGUL_RE, "ko"),
    (_CYRILLIC_RE, "ru"),
    (_GREEK_RE, "el"),
    (_ARABIC_RE, "ar"),
]


def detect_language(text: str) -> tuple[str, float]:
    """Heuristically detect ``text``'s language -> ``(iso639_1, confidence)``.

    Process:
    1. Hard script markers (Han/kana/Hangul/Cyrillic/Greek/Arabic) win at
       confidence 0.95 — the script is unambiguous.
    2. Otherwise tokenize Latin words and score against ``_LATIN_STOPWORDS``.
       A language whose stopwords cover ≥50% of its unique tokens and fit the
       token mix is chosen; confidence scales with the coverage rate.
    3. Fallback: ``"en"`` at low confidence (0.4). Matching is lossy by design —
       this is a gating hint, not a classifier of record.

    ``contains_chinese`` alone remains the authoritative Han rejection gate;
    callers must run it before accepting content.

    Args:
        text: Sample text to classify.

    Returns:
        A ``(iso_code, confidence)`` tuple with confidence in ``[0, 1]``.
    """
    sample = text[:4000]
    if not sample:
        return "en", 0.0

    for pattern, iso in _SCRIPT_LANGUAGES:
        if pattern.search(sample):
            return iso, 0.95

    words = [w for w in re.findall(r"[A-Za-zÀ-ỹ]+", sample.lower()) if w]
    if not words:
        return "en", 0.4

    unique = set(words)
    best_iso, best_score, best_hits = "en", 0.0, 0
    for iso, stops in _LATIN_STOPWORDS.items():
        hits = unique & stops
        # Score: fraction of this language's stopword set present, weighted by
        # how much of the sample's unique vocabulary the language explains.
        coverage = len(hits) / max(1, len(stops))
        vocab_share = len(hits) / len(unique)
        score = coverage * 0.6 + vocab_share * 0.4
        if score > best_score:
            best_iso, best_score, best_hits = iso, score, len(hits)

    if best_hits:
        return best_iso, min(0.95, 0.5 + 0.2 * best_score)
    return "en", 0.4


class TextAnalyzer:
    """Analyze ingested content using the local LLM (small, fast)."""

    def __init__(self, vllm: VLLMClient, max_chars: int = 6000):
        self.vllm = vllm
        self.max_chars = max_chars

    async def summarize(self, text: str, title: str = "") -> str:
        try:
            out = await self.vllm.chat(
                [{"role": "user", "content": truncate(text, self.max_chars)}],
                system=_SUMMARY_PROMPT,
                temperature=0.2,
                max_tokens=300,
            )
            summary = (out or "").strip().strip('"')
            # Escape/clean markdown
            summary = re.sub(r"\s+", " ", summary)
            return summary[:800] or title[:200]
        except Exception as exc:
            logger.warning("analyzer.summary_failed", error=str(exc))
            return title[:200]

    async def extract_keywords(
        self,
        text: str,
        fallback_tags: list[str] | None = None,
    ) -> list[str]:
        try:
            out = await self.vllm.chat(
                [{"role": "user", "content": truncate(text, self.max_chars)}],
                system=_KEYWORDS_PROMPT,
                temperature=0.1,
                max_tokens=120,
            )
            raw = out or ""
            items = [
                line.strip().strip("-• *").strip()
                for line in raw.splitlines()
                if line.strip() and len(line.strip()) < 40
            ]
            # Also capture a single-line comma list
            if not items and "," in raw:
                items = [x.strip() for x in raw.split(",") if x.strip()][:8]
            result = [w for w in items if w][:8]
            if result:
                return result
        except Exception as exc:
            logger.warning("analyzer.keywords_failed", error=str(exc))
        return [str(t) for t in (fallback_tags or [])][:8]

    async def detect_topic(self, text: str) -> str:
        """Quick single-word topic guess from first N chars (heuristic)."""
        head = truncate(text, 1200).lower()
        freq: dict[str, int] = {}
        stopwords = {
            "của",
            "và",
            "trong",
            "là",
            "có",
            "không",
            "một",
            "những",
            "được",
            "cho",
            "với",
            "theo",
            "các",
            "the",
            "and",
            "of",
            "to",
            "for",
        }
        words = re.findall(r"[a-zA-ZÀ-ỹ]{4,}", head)
        for w in words:
            if w.lower() not in stopwords:
                freq[w.lower()] = freq.get(w.lower(), 0) + 1
        if not freq:
            return ""
        return max(freq, key=freq.get)
