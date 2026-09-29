"""Braind Core - knowledge taxonomy.

The closed vocabularies every classified row must use. Free-form labels were
what let ``entity_type`` explode into 288 distinct values with ``Improve`` and
``Fun`` filed as organizations, and left ``auto_topic`` holding stopwords like
``your``. A consumer can only filter on a value it can predict.

Everything here is data, not behaviour: the classifier maps model output onto
these terms, and retrieval reads them back.
"""

from __future__ import annotations

from core.textanalysis import normalize_label

# --- Document type: what kind of writing this is --------------------------
# Chosen so the writer and chat can prefer different shapes of source: a spec
# answers "what are the limits", a tutorial answers "how do I", marketing
# answers neither and should rank last.
DOC_REFERENCE = "reference"  # API docs, specifications, standards
DOC_TUTORIAL = "tutorial"  # how-to, guide, walkthrough
DOC_EXPLAINER = "explainer"  # conceptual explanation, background
DOC_NEWS = "news"  # dated reporting, announcements, releases
DOC_ANALYSIS = "analysis"  # opinion, review, case study, research
DOC_DISCUSSION = "discussion"  # forum thread, Q&A, comments
DOC_MARKETING = "marketing"  # product pages, sales copy, landing pages
DOC_DATA = "data"  # tables, figures, datasets, financials
DOC_PERSONAL = "personal"  # own notes, chat turns, vault entries
DOC_OTHER = "other"

DOC_TYPES: frozenset[str] = frozenset(
    {
        DOC_REFERENCE,
        DOC_TUTORIAL,
        DOC_EXPLAINER,
        DOC_NEWS,
        DOC_ANALYSIS,
        DOC_DISCUSSION,
        DOC_MARKETING,
        DOC_DATA,
        DOC_PERSONAL,
        DOC_OTHER,
    }
)

# How much each type is worth as grounding for generated text. The writer
# ranks by this; chat uses it to break ties between otherwise equal chunks.
DOC_TYPE_AUTHORITY: dict[str, float] = {
    DOC_REFERENCE: 1.0,
    DOC_DATA: 0.9,
    DOC_TUTORIAL: 0.85,
    DOC_EXPLAINER: 0.8,
    DOC_ANALYSIS: 0.7,
    DOC_NEWS: 0.6,
    DOC_PERSONAL: 0.6,
    DOC_DISCUSSION: 0.4,
    DOC_MARKETING: 0.2,
    DOC_OTHER: 0.5,
}

# --- Chunk role: what a passage does inside its document ------------------
# A "what is X" question wants a definition; "how do I X" wants a procedure.
# Without this the retriever can only match words, not intent.
ROLE_DEFINITION = "definition"  # states what something is
ROLE_PROCEDURE = "procedure"  # ordered steps, instructions
ROLE_EXAMPLE = "example"  # worked example, code sample
ROLE_DATA = "data"  # table, figures, measurements
ROLE_CLAIM = "claim"  # assertion, finding, recommendation
ROLE_CONTEXT = "context"  # background, narrative prose
ROLE_REFERENCE = "reference"  # link lists, citations, see-also
ROLE_NAVIGATION = "navigation"  # menus, breadcrumbs — excluded from retrieval

CHUNK_ROLES: frozenset[str] = frozenset(
    {
        ROLE_DEFINITION,
        ROLE_PROCEDURE,
        ROLE_EXAMPLE,
        ROLE_DATA,
        ROLE_CLAIM,
        ROLE_CONTEXT,
        ROLE_REFERENCE,
        ROLE_NAVIGATION,
    }
)

# Roles that never answer a question on their own.
NON_RETRIEVABLE_ROLES: frozenset[str] = frozenset({ROLE_NAVIGATION})

# --- Entity type: closed set, unlike the 288 free-form values found --------
ENTITY_PERSON = "person"
ENTITY_ORGANIZATION = "organization"
ENTITY_PRODUCT = "product"  # named software, hardware, services
ENTITY_TECHNOLOGY = "technology"  # languages, protocols, formats, techniques
ENTITY_LOCATION = "location"
ENTITY_EVENT = "event"
ENTITY_CONCEPT = "concept"  # abstract ideas, disciplines, methods
ENTITY_DOCUMENT = "document"  # named standards, laws, publications
ENTITY_METRIC = "metric"  # named measures, KPIs, units

ENTITY_TYPES: frozenset[str] = frozenset(
    {
        ENTITY_PERSON,
        ENTITY_ORGANIZATION,
        ENTITY_PRODUCT,
        ENTITY_TECHNOLOGY,
        ENTITY_LOCATION,
        ENTITY_EVENT,
        ENTITY_CONCEPT,
        ENTITY_DOCUMENT,
        ENTITY_METRIC,
    }
)

# --- Audience -------------------------------------------------------------
AUDIENCE_GENERAL = "general"
AUDIENCE_TECHNICAL = "technical"
AUDIENCE_BUSINESS = "business"
AUDIENCE_ACADEMIC = "academic"
AUDIENCE_LEGAL = "legal"

AUDIENCES: frozenset[str] = frozenset(
    {
        AUDIENCE_GENERAL,
        AUDIENCE_TECHNICAL,
        AUDIENCE_BUSINESS,
        AUDIENCE_ACADEMIC,
        AUDIENCE_LEGAL,
    }
)

# --- Quality tier ---------------------------------------------------------
# Derived from the quality gate's measurements, not from the model: a score is
# reproducible, a model's self-assessment is not.
TIER_HIGH = "high"
TIER_MEDIUM = "medium"
TIER_LOW = "low"

QUALITY_TIERS: frozenset[str] = frozenset({TIER_HIGH, TIER_MEDIUM, TIER_LOW})


def quality_tier(score: float) -> str:
    """Map a 0-1 quality score onto a tier."""
    if score >= 0.7:
        return TIER_HIGH
    if score >= 0.4:
        return TIER_MEDIUM
    return TIER_LOW


def coerce(value: object, allowed: frozenset[str], default: str) -> str:
    """Return ``value`` if it is in ``allowed``, else ``default``.

    Model output is untrusted: it invents labels, changes case, and answers in
    the wrong language. Anything unrecognised collapses to the default rather
    than entering the database as a new de-facto category.
    """
    if not isinstance(value, str):
        return default
    candidate = normalize_label(value)
    return candidate if candidate in allowed else default
