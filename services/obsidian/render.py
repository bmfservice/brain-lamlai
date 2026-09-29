"""Obsidian display layer — pure rendering (ingest §3).

Turns the semantic layer into Obsidian's own vocabulary: notes with YAML
frontmatter, `[[wikilink]]` edges, and JSON Canvas graphs. Nothing here touches
the database or the filesystem, so the exact bytes that will land in the vault
can be asserted in a unit test.

Three rules shape everything below.

**Every generated file is marked.** `braind_generated: true` is the first key of
every frontmatter block. The vault scanner is the *input* side of this system;
without a marker it would re-ingest this output as if a human had written it.
The marker is the guard that survives a file being moved out of the export
directory, so it is not optional and never conditional.

**Filenames come from stable keys only.** A re-ingest deletes and recreates the
semantic `documents` row, and chunk/unit UUIDs are minted fresh each pass. Only
`entities.normalized_name` (globally unique) and `sources.path` survive. Slugs
are derived from those, so re-exporting overwrites in place instead of littering
the vault with orphans.

**Output is deterministic.** Same input, same bytes — no timestamps, no
randomness, no set iteration order, and a fixed radial Canvas layout. Obsidian
watches these files and git tracks them; a re-export that changes nothing should
produce a diff that shows nothing.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

# Frontmatter key that marks a file as ours. Anything carrying it is skipped by
# the vault scanner, so it must be written on every single generated file.
GENERATED_MARKER = "braind_generated"

# Canvas geometry. Fixed so the layout is reproducible; Obsidian will happily
# let the user rearrange nodes, but our regenerated file must not churn.
_CANVAS_DOC_WIDTH = 420
_CANVAS_DOC_HEIGHT = 240
_CANVAS_ENTITY_WIDTH = 260
_CANVAS_ENTITY_HEIGHT = 120
_CANVAS_RADIUS_BASE = 420
_CANVAS_RADIUS_STEP = 26

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")
_WIKILINK_UNSAFE = re.compile(r"[\[\]|#^]")


def slugify(value: str, *, fallback: str = "untitled") -> str:
    """Return a filesystem- and wikilink-safe slug.

    Obsidian resolves `[[links]]` by filename, so the slug is an identifier, not
    decoration: two entities that slugify to the same string would silently
    become one note. Callers pass an already-unique key (a normalized entity
    name, a source path) and this only makes it safe to write down.
    """
    slug = _SLUG_STRIP.sub("-", str(value or "").lower()).strip("-")
    # Filesystems cap component length; 120 leaves room for the extension and
    # keeps long section paths from producing unopenable files.
    return slug[:120] or fallback


def wikilink(target: str) -> str:
    """Render `[[target]]` with the characters that would break it removed.

    A `]` or `|` inside a link silently truncates it in Obsidian, which would
    point the edge at the wrong note rather than fail loudly.
    """
    return f"[[{_WIKILINK_UNSAFE.sub(' ', str(target or '')).strip()}]]"


@dataclass(frozen=True)
class Relationship:
    """One typed edge as it appears in a note's frontmatter."""

    relation_type: str
    target_slug: str
    target_name: str
    confidence: float = 0.0
    direction: str = "out"  # out | in


@dataclass(frozen=True)
class EntityNote:
    """Everything needed to render one entity note."""

    entity_id: str
    canonical_name: str
    slug: str
    entity_type: str
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    resolution_state: str = "ambiguous"
    confidence: float = 0.0
    domain: str | None = None
    relationships: list[Relationship] = field(default_factory=list)
    mentioned_in: list[tuple[str, str]] = field(default_factory=list)  # (slug, title)


@dataclass(frozen=True)
class Claim:
    """One semantic unit as rendered inside a document note."""

    content: str
    polarity: int = 1
    fact_class: str = "SOURCE_FACT"
    confidence: float = 0.0
    entity_slugs: list[tuple[str, str]] = field(default_factory=list)  # (slug, name)


@dataclass(frozen=True)
class Section:
    """One chunk rendered as a heading plus its claims."""

    section_path: str
    source_location: str = ""
    chunk_role: str = ""
    content: str = ""
    claims: list[Claim] = field(default_factory=list)


@dataclass(frozen=True)
class DocumentNote:
    """Everything needed to render one document note and its Canvas."""

    source_id: str
    document_id: str
    title: str
    slug: str
    source_type: str = ""
    doc_type: str | None = None
    domain: str | None = None
    language: str | None = None
    quality_tier: str | None = None
    tags: list[str] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    entities: list[EntityNote] = field(default_factory=list)


def _yaml_scalar(value: object) -> str:
    """Render one scalar for frontmatter.

    Hand-rolled rather than `yaml.dump` because the output must be byte-stable
    across library versions and readable in a diff. Only the shapes this module
    actually emits are supported.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    text = str(value if value is not None else "")
    # Quote anything that YAML would otherwise reinterpret — a bare `[[x]]`
    # parses as a nested sequence, and a leading digit can become a number.
    return json.dumps(text, ensure_ascii=False)


def render_frontmatter(fields: dict[str, object]) -> str:
    """Render a YAML frontmatter block, marker first.

    Key order is the insertion order of ``fields`` except that the generated
    marker is forced to the top, where a human skimming the file — and the
    scanner guard — will see it immediately.
    """
    lines = ["---", f"{GENERATED_MARKER}: true"]
    for key, value in fields.items():
        if key == GENERATED_MARKER or value is None:
            continue
        if isinstance(value, list):
            if not value:
                continue
            lines.append(f"{key}:")
            for item in value:
                if isinstance(item, dict):
                    first = True
                    for sub_key, sub_value in item.items():
                        prefix = "  - " if first else "    "
                        lines.append(f"{prefix}{sub_key}: {_yaml_scalar(sub_value)}")
                        first = False
                else:
                    lines.append(f"  - {_yaml_scalar(item)}")
            continue
        lines.append(f"{key}: {_yaml_scalar(value)}")
    lines.append("---")
    return "\n".join(lines)


def render_entity_note(note: EntityNote) -> str:
    """Render one entity as an Obsidian note.

    Unconfirmed entities are exported alongside confirmed ones — most of the
    corpus is unconfirmed, and hiding it would make the vault a misleading view
    of what the system actually knows. Instead `resolution_state` is in the
    frontmatter and a `braind/review` tag makes the backlog filterable.
    """
    needs_review = note.resolution_state != "resolved"
    tags = ["braind/entity", f"braind/type/{slugify(note.entity_type)}"]
    if needs_review:
        tags.append("braind/review")

    relationships = [
        {
            "type": rel.relation_type,
            "target": wikilink(rel.target_slug),
            "confidence": round(rel.confidence, 3),
            "direction": rel.direction,
        }
        for rel in note.relationships
    ]

    frontmatter = render_frontmatter(
        {
            "node_id": note.entity_id,
            "node_type": "entity",
            "entity_type": note.entity_type,
            "path": f"entities/{note.slug}",
            "resolution_state": note.resolution_state,
            "confidence": round(note.confidence, 3),
            "domain": note.domain,
            "aliases": note.aliases,
            "relationships": relationships,
            "tags": tags,
        }
    )

    body = [f"# {note.canonical_name}", ""]
    if note.description:
        body += [note.description, ""]

    if needs_review:
        body += [
            f"> [!warning] Needs review — `resolution_state: {note.resolution_state}`",
            "> This entity was extracted but never confirmed as a distinct identity.",
            "",
        ]

    if note.relationships:
        body.append("## Relationships")
        body.append("")
        for rel in note.relationships:
            arrow = "→" if rel.direction == "out" else "←"
            body.append(
                f"- {arrow} **{rel.relation_type}** {wikilink(rel.target_slug)} "
                f"— {rel.target_name} (confidence {rel.confidence:.2f})"
            )
        body.append("")

    if note.mentioned_in:
        body.append("## Mentioned in")
        body.append("")
        for slug, title in note.mentioned_in:
            body.append(f"- {wikilink(slug)} — {title}")
        body.append("")

    return f"{frontmatter}\n\n" + "\n".join(body).rstrip() + "\n"


def render_claim(claim: Claim) -> str:
    """Render one semantic unit as a bullet.

    A negated unit is prefixed with an explicit **NOT**. `polarity = -1` exists
    precisely so `"does not use CUDA 13"` is never read as a positive fact; a
    renderer that dropped the sign would reintroduce that bug in the one view a
    human actually reads.
    """
    text = claim.content.strip()
    marker = "**NOT** " if claim.polarity == -1 else ""
    links = " ".join(wikilink(slug) for slug, _ in claim.entity_slugs)
    suffix = f" {links}" if links else ""
    return f"- {marker}{text}{suffix}"


def render_document_note(note: DocumentNote) -> str:
    """Render one source document as a note: sections as headings, units as claims."""
    frontmatter = render_frontmatter(
        {
            "node_id": note.source_id,
            "node_type": "document",
            "document_id": note.document_id,
            "path": f"documents/{note.slug}",
            "source_type": note.source_type,
            "doc_type": note.doc_type,
            "domain": note.domain,
            "language": note.language,
            "quality_tier": note.quality_tier,
            "entities": [wikilink(e.slug) for e in note.entities],
            "tags": ["braind/document", *[f"braind/tag/{slugify(t)}" for t in note.tags]],
        }
    )

    body = [f"# {note.title}", ""]
    if not note.sections:
        body += ["*No semantic sections were extracted for this document.*", ""]

    for section in note.sections:
        heading = section.section_path or section.source_location or "Section"
        body.append(f"## {heading}")
        body.append("")
        if section.source_location:
            body.append(f"*{section.source_location}*")
            body.append("")
        if section.claims:
            body += [render_claim(c) for c in section.claims]
            body.append("")
        elif section.content:
            # No claims were extracted, so the chunk text is the only content
            # this section can show. Without it the note would be an empty
            # heading, which reads as "nothing here" rather than "not analysed".
            body.append(section.content.strip())
            body.append("")

    if note.entities:
        body.append("## Entities")
        body.append("")
        for entity in note.entities:
            state = "" if entity.resolution_state == "resolved" else " *(review)*"
            body.append(f"- {wikilink(entity.slug)} — {entity.entity_type}{state}")
        body.append("")

    return f"{frontmatter}\n\n" + "\n".join(body).rstrip() + "\n"


def render_canvas(note: DocumentNote, *, export_dirname: str = "braind") -> str:
    """Render a JSON Canvas: the document at the centre, its entities around it.

    Edges are co-mention — the entity appears in a claim of this document — with
    typed knowledge-graph edges overlaid and labelled where they exist. Building
    the Canvas from typed edges alone would produce an almost empty graph: the
    corpus has thousands of entities and only a couple of dozen typed edges, so
    co-mention is what carries the picture.

    Layout is a fixed radial arrangement, so regenerating an unchanged document
    yields identical bytes.
    """
    doc_path = f"{export_dirname}/documents/{note.slug}.md"
    nodes: list[dict] = [
        {
            "id": f"doc-{note.slug}",
            "type": "file",
            "file": doc_path,
            "x": -_CANVAS_DOC_WIDTH // 2,
            "y": -_CANVAS_DOC_HEIGHT // 2,
            "width": _CANVAS_DOC_WIDTH,
            "height": _CANVAS_DOC_HEIGHT,
            "color": "5",
        }
    ]
    edges: list[dict] = []

    count = len(note.entities)
    # A wider ring for more entities, so labels do not overlap at any size.
    radius = _CANVAS_RADIUS_BASE + _CANVAS_RADIUS_STEP * max(0, count - 6)
    for index, entity in enumerate(note.entities):
        angle = (2 * math.pi * index / count) if count else 0.0
        nodes.append(
            {
                "id": f"entity-{entity.slug}",
                "type": "file",
                "file": f"{export_dirname}/entities/{entity.slug}.md",
                "x": round(radius * math.cos(angle)) - _CANVAS_ENTITY_WIDTH // 2,
                "y": round(radius * math.sin(angle)) - _CANVAS_ENTITY_HEIGHT // 2,
                "width": _CANVAS_ENTITY_WIDTH,
                "height": _CANVAS_ENTITY_HEIGHT,
                "color": "3" if entity.resolution_state != "resolved" else "4",
            }
        )
        edges.append(
            {
                "id": f"edge-doc-{entity.slug}",
                "fromNode": f"doc-{note.slug}",
                "fromSide": "right",
                "toNode": f"entity-{entity.slug}",
                "toSide": "left",
                "label": "mentions",
            }
        )

    exported = {e.slug for e in note.entities}
    for entity in note.entities:
        for rel in entity.relationships:
            # Only draw an edge both of whose ends are on this Canvas; a link to
            # an entity this document never mentions would dangle.
            if rel.target_slug not in exported or rel.target_slug == entity.slug:
                continue
            edges.append(
                {
                    "id": f"edge-{entity.slug}-{rel.relation_type}-{rel.target_slug}",
                    "fromNode": f"entity-{entity.slug}",
                    "fromSide": "bottom",
                    "toNode": f"entity-{rel.target_slug}",
                    "toSide": "top",
                    "label": rel.relation_type,
                }
            )

    return json.dumps({"nodes": nodes, "edges": edges}, indent=2, ensure_ascii=False) + "\n"
