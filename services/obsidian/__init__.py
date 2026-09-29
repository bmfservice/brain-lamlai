"""Braind Obsidian display layer (ingest §3).

Renders the semantic knowledge layer into an Obsidian-readable vault subtree:
one note per entity, one note plus a JSON Canvas per document, edges as
`relationships:` frontmatter and `[[wikilinks]]`.

Design contract:
- **Export only.** Nothing here reads notes back into the database. Reverse-sync
  (watch → re-validate → update) is a separate concern with a separate risk
  profile and is deliberately not implemented here.
- **Every generated file is marked** with `braind_generated: true` so the vault
  scanner — which is the *input* side of this system — can never re-ingest this
  output as if a human had written it.
- **Deletion is guarded twice**: only inside the export directory, and only for
  files carrying that marker.
"""

from services.obsidian.exporter import ExportResult, ObsidianExporter
from services.obsidian.render import (
    GENERATED_MARKER,
    Claim,
    DocumentNote,
    EntityNote,
    Relationship,
    Section,
    render_canvas,
    render_document_note,
    render_entity_note,
    slugify,
    wikilink,
)

__all__ = [
    "GENERATED_MARKER",
    "Claim",
    "DocumentNote",
    "EntityNote",
    "ExportResult",
    "ObsidianExporter",
    "Relationship",
    "Section",
    "render_canvas",
    "render_document_note",
    "render_entity_note",
    "slugify",
    "wikilink",
]
