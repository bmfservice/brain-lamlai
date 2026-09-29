"""Braind Ingest - document and passage classification.

Assigns the closed-vocabulary labels from ``core/taxonomy.py`` that retrieval
filters and ranks on: what kind of document this is, which domain it belongs
to, who it is written for, and what each passage does inside it.

Design constraints that shaped this module:

- **One LLM round-trip per document.** Deep classification was originally going
  to be several calls, but a 300-page site crawl already takes ~25 minutes; a
  per-facet call would have pushed it past two hours. One JSON response carries
  every document-level facet.
- **Passage roles are heuristic.** A per-chunk LLM call would multiply cost by
  the chunk count. Structure (numbered steps, tables, code, "X is a Y") is a
  reliable enough signal, and a wrong role only reorders results — it never
  hides a chunk, because role is a ranking feature and not a hard filter.
- **Model output is untrusted.** Everything is coerced onto the vocabulary; an
  unrecognised label becomes the default rather than a new de-facto category.
  This is exactly how ``entity_type`` reached 288 distinct values.
"""

from __future__ import annotations

import re

from core import taxonomy as tax
from core.logging import get_logger
from core.quality import QualityReport
from core.textanalysis import truncate
from core.vllm import VLLMClient

logger = get_logger(__name__)

_CLASSIFY_PROMPT = """You classify documents for a knowledge base.

Return ONLY a JSON object with these keys:
- "doc_type": one of reference, tutorial, explainer, news, analysis,
  discussion, marketing, data, personal, other
- "domain": 1-3 words naming the subject area in English, lowercase
  (e.g. "databases", "legal compliance", "web marketing")
- "audience": one of general, technical, business, academic, legal
- "language": ISO 639-1 code of the main language (e.g. "en", "vi")
- "summary": 2-3 sentences stating what this document actually says.
  Same language as the document. Never Chinese.
- "keywords": 4-8 specific noun phrases from the text. No generic words
  like "web", "search", "guide", "information".
- "key_claims": up to 3 short sentences stating the document's most
  concrete, checkable assertions.

Judge doc_type by what the writing DOES, not its subject:
- reference: specifications, API docs, standards, parameter tables
- tutorial: ordered instructions for accomplishing a task
- explainer: explains a concept without instructing
- news: reports a dated event or release
- analysis: argues a position, reviews, or presents a case study
- discussion: forum thread, Q&A, comment section
- marketing: sells a product or service
- data: mostly figures, tables, or datasets
- personal: private notes or conversation

No prose outside the JSON."""

# Structural markers for passage roles. Ordered by specificity: the first
# pattern that fits wins, so a numbered list of steps is a procedure even
# though it also contains sentences.
_STEP_RE = re.compile(r"^\s*(?:\d+[.)]\s|step\s+\d+|bước\s+\d+)", re.IGNORECASE | re.MULTILINE)
_IMPERATIVE_RE = re.compile(
    r"^\s*(?:run|install|open|click|create|add|set|configure|chạy|cài|mở|tạo|thêm)\b",
    re.IGNORECASE | re.MULTILINE,
)
_DEFINITION_RE = re.compile(
    r"\b(?:is a|is an|is the|are a|refers to|means|stands for|được gọi là|là một|là các)\b",
    re.IGNORECASE,
)
_CODE_RE = re.compile(r"```|^\s{4,}\S|^\s*[\w./-]+\s*[=:]\s*\S", re.MULTILINE)
_EXAMPLE_RE = re.compile(r"\b(?:for example|e\.g\.|such as|ví dụ|chẳng hạn)\b", re.IGNORECASE)
_TABLE_RE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_NUMBER_HEAVY_RE = re.compile(r"\d")
_URL_RE = re.compile(r"https?://\S+")
_WORD_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)
_CLAIM_RE = re.compile(
    r"\b(?:should|must|recommend|conclude|found that|results show|nên|phải|khuyến nghị)\b",
    re.IGNORECASE,
)

# Keywords the old pipeline emitted for nearly every web source; they carry no
# retrieval signal and crowded out the specific terms.
_KEYWORD_STOPLIST = frozenset(
    {
        "web", "search", "guide", "information", "content", "page", "site", "website",
        "article", "post", "blog", "news", "data", "thông tin", "nội dung", "trang",
        "bài viết", "tìm kiếm", "hướng dẫn",
    }
)


class DocumentClassification:
    """Document-level labels plus the derived metadata ingest stores."""

    __slots__ = (
        "doc_type",
        "domain",
        "audience",
        "language",
        "quality_tier",
        "quality_score",
        "summary",
        "keywords",
        "key_claims",
    )

    def __init__(
        self,
        *,
        doc_type: str = tax.DOC_OTHER,
        domain: str = "",
        audience: str = tax.AUDIENCE_GENERAL,
        language: str = "en",
        quality_tier: str = tax.TIER_MEDIUM,
        quality_score: float = 0.0,
        summary: str = "",
        keywords: list[str] | None = None,
        key_claims: list[str] | None = None,
    ) -> None:
        self.doc_type = doc_type
        self.domain = domain
        self.audience = audience
        self.language = language
        self.quality_tier = quality_tier
        self.quality_score = quality_score
        self.summary = summary
        self.keywords = keywords or []
        self.key_claims = key_claims or []

    @property
    def authority(self) -> float:
        """How much this document is worth as grounding for generated text."""
        return tax.DOC_TYPE_AUTHORITY.get(self.doc_type, 0.5)

    def as_meta(self) -> dict:
        """The subset stored in ``meta`` alongside the dedicated columns."""
        return {
            "summary": self.summary,
            "keywords": self.keywords,
            "key_claims": self.key_claims,
            "doc_type": self.doc_type,
            "domain": self.domain,
            "audience": self.audience,
            "authority": self.authority,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"DocumentClassification(type={self.doc_type}, domain={self.domain!r}, "
            f"tier={self.quality_tier}, kw={len(self.keywords)})"
        )


def classify_chunk_role(text: str, *, chunk_kind: str = "") -> str:
    """Return the taxonomy role for one passage.

    Structural, not semantic: it reads how the text is laid out. A procedure
    has numbered steps or imperatives, a definition has a copula, a table has
    pipes. Cheap enough to run on every chunk of a 300-page crawl.
    """
    body = (text or "").strip()
    if not body:
        return tax.ROLE_CONTEXT

    words = _WORD_RE.findall(body)
    word_count = len(words)

    if chunk_kind == "table" or len(_TABLE_RE.findall(body)) >= 2:
        return tax.ROLE_DATA
    # A short block that is mostly links is a see-also list, not prose.
    if word_count and len(_URL_RE.findall(body)) >= 3 and word_count < 120:
        return tax.ROLE_REFERENCE
    if _STEP_RE.search(body) or _IMPERATIVE_RE.search(body):
        return tax.ROLE_PROCEDURE
    if _CODE_RE.search(body) or _EXAMPLE_RE.search(body):
        return tax.ROLE_EXAMPLE
    # Definitions state what something is, near the start of the passage.
    if _DEFINITION_RE.search(body[:400]):
        return tax.ROLE_DEFINITION
    if _CLAIM_RE.search(body):
        return tax.ROLE_CLAIM
    # Mostly digits with little prose: a figures block that escaped the table
    # check (extraction often loses pipe characters).
    digits = len(_NUMBER_HEAVY_RE.findall(body))
    if word_count and digits / max(len(body), 1) > 0.15:
        return tax.ROLE_DATA
    # Too short to answer anything, and not matched above: chrome that survived
    # the quality gate.
    if word_count < 12:
        return tax.ROLE_NAVIGATION
    return tax.ROLE_CONTEXT


def clean_keywords(raw: object, *, limit: int = 8) -> list[str]:
    """Normalize model keyword output, dropping the generic ones."""
    if isinstance(raw, str):
        candidates = re.split(r"[,\n;]", raw)
    elif isinstance(raw, list):
        candidates = [str(item) for item in raw]
    else:
        return []
    seen: dict[str, str] = {}
    for candidate in candidates:
        term = re.sub(r"\s+", " ", str(candidate)).strip().strip("-•*·\"'")
        if not term or len(term) < 3 or len(term) > 60:
            continue
        key = term.lower()
        if key in _KEYWORD_STOPLIST or key in seen:
            continue
        seen[key] = term
        if len(seen) >= limit:
            break
    return list(seen.values())


def _clean_claims(raw: object, *, limit: int = 3) -> list[str]:
    if not isinstance(raw, list):
        return []
    claims: list[str] = []
    for item in raw:
        claim = re.sub(r"\s+", " ", str(item)).strip()
        if 20 <= len(claim) <= 300:
            claims.append(claim)
        if len(claims) >= limit:
            break
    return claims


class DocumentClassifier:
    """Classify a document in one LLM round-trip, with a heuristic fallback."""

    def __init__(self, vllm: VLLMClient | None = None, max_chars: int = 6000) -> None:
        self.vllm = vllm
        self.max_chars = max_chars

    async def classify(
        self,
        text: str,
        *,
        title: str = "",
        source_type: str = "",
        quality: QualityReport | None = None,
        fallback_keywords: list[str] | None = None,
    ) -> DocumentClassification:
        """Return the document's classification.

        ``quality`` supplies the tier and score: those are measurements from
        the gate, never the model's self-assessment, so they stay reproducible.
        """
        score = quality.score if quality else 0.0
        result = DocumentClassification(
            doc_type=_source_type_prior(source_type),
            quality_tier=tax.quality_tier(score),
            quality_score=score,
            summary="",
            keywords=clean_keywords(fallback_keywords or []),
        )
        if self.vllm is None or not text.strip():
            return result

        prompt = f"Title: {title}\n\n{truncate(text, self.max_chars)}"
        try:
            payload = await self.vllm.chat_json(
                [{"role": "user", "content": prompt}],
                system=_CLASSIFY_PROMPT,
                temperature=0.1,
                max_tokens=700,
            )
        except Exception as exc:
            logger.warning("classify.llm_failed", error=str(exc))
            return result

        if not isinstance(payload, dict):
            return result

        result.doc_type = tax.coerce(payload.get("doc_type"), tax.DOC_TYPES, result.doc_type)
        result.audience = tax.coerce(
            payload.get("audience"), tax.AUDIENCES, tax.AUDIENCE_GENERAL
        )
        result.domain = _clean_domain(payload.get("domain"))
        result.language = _clean_language(payload.get("language"))
        # sanitize_model_text is applied by the caller, which owns the fallback
        # text; here we only shape what the model returned.
        result.summary = re.sub(r"\s+", " ", str(payload.get("summary") or "")).strip()[:800]
        keywords = clean_keywords(payload.get("keywords"))
        result.keywords = keywords or result.keywords
        result.key_claims = _clean_claims(payload.get("key_claims"))
        return result


def _source_type_prior(source_type: str) -> str:
    """Best guess before the model speaks, from where the text came."""
    return {
        "obsidian": tax.DOC_PERSONAL,
        "vault": tax.DOC_PERSONAL,
        "chat": tax.DOC_PERSONAL,
        "manual": tax.DOC_PERSONAL,
        "daily": tax.DOC_PERSONAL,
        "rss": tax.DOC_NEWS,
        "file": tax.DOC_OTHER,
    }.get(source_type, tax.DOC_OTHER)


def _clean_domain(raw: object) -> str:
    """Normalize the free-text domain into a short lowercase label."""
    domain = re.sub(r"[^\w\s/-]", "", str(raw or "")).strip().lower()
    domain = re.sub(r"\s+", " ", domain)
    words = domain.split()
    if not words or len(words) > 4:
        return " ".join(words[:4]) if words else ""
    return " ".join(words)[:64]


def _clean_language(raw: object) -> str:
    """Accept only a two-letter ISO 639-1 code."""
    code = str(raw or "").strip().lower()[:2]
    return code if len(code) == 2 and code.isalpha() else "en"
