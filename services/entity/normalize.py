"""Braind Entity - name normalization and type coercion.

The audit of 2026-08-20 found 5553 entities in a state that made the knowledge
graph unusable: 545 names contained embedded newlines, 774 were three
characters or shorter (``WW``, ``CZ``, ``TB``), 19 were bare stopwords, 227
groups differed only in letter case, and ``entity_type`` had drifted to 288
distinct free-form values with ``Improve`` and ``Fun`` filed as organizations.

This module is the single place that decides whether a mention deserves to be
an entity at all, what its canonical form is, and which taxonomy type it takes.
"""

from __future__ import annotations

import re

from core import taxonomy as tax

# Names that are grammar, not entities. The heuristic extractor emits these
# from title-cased sentence openers.
_NAME_STOPWORDS: frozenset[str] = frozenset(
    {
        "all", "use", "include", "improve", "come", "pride", "fun", "next", "back",
        "home", "more", "read", "learn", "share", "click", "view", "see", "get",
        "new", "top", "best", "free", "help", "about", "contact", "search", "menu",
        "the", "this", "that", "these", "those", "and", "for", "with", "from",
        "và", "của", "là", "các", "cho", "một", "trong", "như", "này", "đó", "không",
    }
)

# Acronyms short enough to be noise unless they are known technical terms.
_KNOWN_SHORT: frozenset[str] = frozenset(
    {"ai", "ml", "db", "os", "id", "ui", "ux", "js", "go", "c", "r", "sql", "api",
     "rag", "llm", "gpu", "cpu", "ssl", "tls", "dns", "http", "css", "seo", "vin",
     "dot", "dmv", "epa", "pdf", "csv", "xml", "yes", "no"}
)

# Free-form types the old pipeline produced, mapped onto the closed vocabulary.
_TYPE_ALIASES: dict[str, str] = {
    "software": tax.ENTITY_PRODUCT,
    "tool": tax.ENTITY_PRODUCT,
    "library": tax.ENTITY_PRODUCT,
    "framework": tax.ENTITY_TECHNOLOGY,
    "language": tax.ENTITY_TECHNOLOGY,
    "protocol": tax.ENTITY_TECHNOLOGY,
    "format": tax.ENTITY_TECHNOLOGY,
    "method": tax.ENTITY_CONCEPT,
    "technique": tax.ENTITY_CONCEPT,
    "discipline": tax.ENTITY_CONCEPT,
    "field": tax.ENTITY_CONCEPT,
    "topic": tax.ENTITY_CONCEPT,
    "company": tax.ENTITY_ORGANIZATION,
    "brand": tax.ENTITY_ORGANIZATION,
    "team": tax.ENTITY_ORGANIZATION,
    "institution": tax.ENTITY_ORGANIZATION,
    "agency": tax.ENTITY_ORGANIZATION,
    "people": tax.ENTITY_PERSON,
    "author": tax.ENTITY_PERSON,
    "place": tax.ENTITY_LOCATION,
    "country": tax.ENTITY_LOCATION,
    "city": tax.ENTITY_LOCATION,
    "standard": tax.ENTITY_DOCUMENT,
    "law": tax.ENTITY_DOCUMENT,
    "regulation": tax.ENTITY_DOCUMENT,
    "publication": tax.ENTITY_DOCUMENT,
    "paper": tax.ENTITY_DOCUMENT,
    "measure": tax.ENTITY_METRIC,
    "kpi": tax.ENTITY_METRIC,
    "unit": tax.ENTITY_METRIC,
}

_WS_RE = re.compile(r"\s+")
_EDGE_PUNCT_RE = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)
_ACRONYM_RE = re.compile(r"^[A-Z0-9]{2,8}$")
_HAS_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)


def clean_name(raw: str) -> str:
    """Return the canonical display form of an entity name.

    Collapses embedded whitespace — extraction splits a heading across lines
    and produced names like ``Egress\\nAnomaly`` — and trims punctuation that
    the span boundaries dragged in.
    """
    name = _WS_RE.sub(" ", str(raw or "")).strip()
    name = _EDGE_PUNCT_RE.sub("", name)
    return name[:255]


def normalize_name(raw: str) -> str:
    """Return the matching key used to detect duplicate entities.

    Case- and punctuation-folded, so ``PostgreSQL``, ``postgresql`` and
    ``PostgreSQL.`` collapse to one entity instead of three.
    """
    return clean_name(raw).lower()


def is_valid_name(raw: str) -> bool:
    """Whether a cleaned name is worth storing as an entity."""
    name = clean_name(raw)
    if not name or not _HAS_LETTER_RE.search(name):
        return False
    if normalize_name(name) in _NAME_STOPWORDS:
        return False
    if len(name) <= 3 and name.lower() not in _KNOWN_SHORT:
        # Two- and three-letter fragments (WW, CZ, TB) are extraction noise
        # unless they are a technical term we recognise.
        return False
    # A "name" that is really a sentence came from a bad span.
    return len(name.split()) <= 6


def coerce_entity_type(raw: str, *, name: str = "") -> str:
    """Map any extractor output onto the closed entity vocabulary."""
    candidate = str(raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    if candidate in tax.ENTITY_TYPES:
        return candidate
    mapped = _TYPE_ALIASES.get(candidate)
    if mapped:
        return mapped
    # An all-caps short token is a technical acronym far more often than it is
    # an organization, which is how "SRU" and "TB" became companies.
    if name and _ACRONYM_RE.match(name):
        return tax.ENTITY_TECHNOLOGY
    return tax.ENTITY_CONCEPT


def describe(name: str, entity_type: str, context: str = "") -> str:
    """Build a short, deterministic description for an entity.

    Every entity had a NULL description, which left the entity embedding index
    empty and entity vector search returning nothing. A sentence built from the
    name, its type and the sentence it was found in is enough to embed.
    """
    label = entity_type.replace("_", " ")
    snippet = _WS_RE.sub(" ", context).strip()
    if snippet:
        snippet = snippet[:240].rsplit(" ", 1)[0] if len(snippet) > 240 else snippet
        return f"{name} — {label}. {snippet}"
    return f"{name} — {label}."


class EntityIndex:
    """Match entity names inside free text.

    ``semantic_units.entity_ids`` was empty on all 4917 rows because the claim
    extractor never filled ``entity_mentions`` — the field existed but nothing
    wrote it, so the unit→entity join, and with it the whole graph expansion,
    had no data. This index closes that gap by scanning claim text for the
    names already resolved for the document.

    Names are matched longest-first on word boundaries, so "vector database"
    wins over "database" and "AI" does not match inside "SAID".
    """

    __slots__ = ("_by_key", "_pattern")

    def __init__(self, names_to_ids: dict[str, str]) -> None:
        self._by_key = {normalize_name(name): eid for name, eid in names_to_ids.items() if name}
        # Longest first so the most specific name claims the span.
        ordered = sorted((k for k in self._by_key if k), key=len, reverse=True)
        self._pattern = (
            re.compile(
                r"(?<!\w)(" + "|".join(re.escape(k) for k in ordered) + r")(?!\w)",
                re.IGNORECASE | re.UNICODE,
            )
            if ordered
            else None
        )

    def __len__(self) -> int:
        return len(self._by_key)

    def find(self, text: str, *, limit: int = 12) -> list[str]:
        """Return ids of entities mentioned in ``text``, in first-seen order."""
        if not text or self._pattern is None:
            return []
        found: dict[str, None] = {}
        for match in self._pattern.finditer(text):
            entity_id = self._by_key.get(match.group(1).lower())
            if entity_id is not None:
                found.setdefault(entity_id, None)
                if len(found) >= limit:
                    break
        return list(found)
