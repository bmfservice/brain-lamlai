"""Braind Entity Service (Brain Second Phase 6).

Entity extraction + resolution for the semantic knowledge layer.

Design contract (§87 governance, golden CASE E/F):
- **Never resolve or merge by embedding/cosine similarity alone.** Identity
  evidence is orthographic (canonical/alias), contextual, or relation-based;
  similarity only promotes a *candidate* to REVIEW_REQUIRED at most.
- **No silent auto-merge.** Merges are either dictionary/alias-driven (AUTO)
  or require review. Related-but-distinct entities (Docker vs Docker Compose,
  PostgreSQL vs the PostgreSQL logo) are kept separate and a relation hint is
  returned instead of collapsing them.
"""

from services.entity.extractor import (
    EntityExtractor,
    EntityMention,
    extract_mentions,
)
from services.entity.resolver import (
    AUTO_MERGE,
    DO_NOT_MERGE,
    REVIEW_REQUIRED,
    EntityResolution,
    EntityResolver,
)

__all__ = [
    "AUTO_MERGE",
    "DO_NOT_MERGE",
    "EntityExtractor",
    "EntityMention",
    "EntityResolution",
    "EntityResolver",
    "REVIEW_REQUIRED",
    "extract_mentions",
]