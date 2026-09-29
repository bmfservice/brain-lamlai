"""Obsidian display layer — database reads and vault writes (ingest §3).

Assembles the render-layer dataclasses from the semantic tables and writes them
into the vault. The split matters: `render.py` is pure and decides *what the
bytes are*, this module decides *what the data is* and *where it lands*.

Writes are atomic (temp file + `os.replace`) because Obsidian and the vault
scanner both watch this directory live. A half-written note would be read by one
of them before it was finished.

The exporter only ever writes inside its own export directory, and only ever
deletes files that carry the generated marker. A user's note that ends up in the
export tree is left alone — the cost of an unexpected leftover file is a tidy-up,
the cost of deleting someone's writing is unrecoverable.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select

from core import database as db
from core.logging import get_logger
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
)
from services.relation.validator import VALID_EDGE_TYPES

logger = get_logger(__name__)

# A document that mentions more entities than this produces an unreadable Canvas
# and a note that is mostly link list. The cap is on the *rendered* view only —
# nothing in the database is hidden by it.
_MAX_ENTITIES_PER_DOCUMENT = 40


@dataclass
class ExportResult:
    """What one export run wrote, for the API response and the log."""

    documents: int = 0
    entities: int = 0
    canvases: int = 0
    pruned: int = 0
    errors: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


class ObsidianExporter:
    """Render the semantic layer into an Obsidian-readable vault subtree."""

    def __init__(self, export_dir: Path) -> None:
        self.export_dir = Path(export_dir)
        self.entities_dir = self.export_dir / "entities"
        self.documents_dir = self.export_dir / "documents"

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def _write(self, path: Path, content: str) -> bool:
        """Write ``content`` atomically; return True when the file changed.

        Skipping an unchanged write is not just an optimization: Obsidian and
        the vault scanner both watch this tree, and rewriting identical bytes
        would wake them on every ingest for no reason.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            try:
                if path.read_text(encoding="utf-8") == content:
                    return False
            except OSError:  # pragma: no cover - unreadable file is rewritten
                pass
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)
        return True

    def _is_generated(self, path: Path) -> bool:
        """Whether a file carries our marker — the only files we may delete."""
        try:
            head = path.read_text(encoding="utf-8", errors="replace")[:512]
        except OSError:  # pragma: no cover - defensive
            return False
        return f"{GENERATED_MARKER}: true" in head

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    async def _entity_notes(
        self, session, entity_ids: list[str]
    ) -> dict[str, EntityNote]:
        """Load entities by id, with their typed edges attached."""
        uuids: list[uuid.UUID] = []
        for raw in entity_ids:
            try:
                uuids.append(uuid.UUID(str(raw)))
            except (TypeError, ValueError):
                continue
        if not uuids:
            return {}

        rows = (
            await session.execute(select(db.Entity).where(db.Entity.id.in_(uuids)))
        ).scalars().all()
        notes: dict[str, EntityNote] = {}
        name_by_id: dict[str, str] = {}
        for row in rows:
            slug = slugify(row.normalized_name or row.canonical_name, fallback="entity")
            key = str(row.id)
            name_by_id[key] = row.canonical_name
            notes[key] = EntityNote(
                entity_id=key,
                canonical_name=row.canonical_name,
                slug=slug,
                entity_type=row.entity_type,
                description=row.description or "",
                aliases=[str(a) for a in (row.aliases or []) if str(a) != row.canonical_name],
                resolution_state=row.resolution_state,
                confidence=float(row.confidence or 0.0),
                domain=(row.meta or {}).get("domain"),
            )

        edges = await self._entity_edges(session, list(notes))
        for entity_id, rels in edges.items():
            note = notes.get(entity_id)
            if note is None:
                continue
            resolved = [
                Relationship(
                    relation_type=rel.relation_type,
                    target_slug=notes[rel.target_slug].slug,
                    target_name=name_by_id.get(rel.target_slug, ""),
                    confidence=rel.confidence,
                    direction=rel.direction,
                )
                for rel in rels
                if rel.target_slug in notes
            ]
            notes[entity_id] = EntityNote(
                **{**note.__dict__, "relationships": resolved}
            )
        return notes

    async def _entity_edges(
        self, session, entity_ids: list[str]
    ) -> dict[str, list[Relationship]]:
        """Typed entity→entity edges among ``entity_ids``.

        Self-loops are excluded: the §5 resolution review writes an edge from an
        entity to itself as a "this name collides across domains" marker, which
        is a review record, not a relationship between two things.
        """
        uuids = [uuid.UUID(e) for e in entity_ids]
        if not uuids:
            return {}
        rows = (
            await session.execute(
                select(
                    db.KnowledgeLink.source_id,
                    db.KnowledgeLink.target_id,
                    db.KnowledgeLink.relation_type,
                    db.KnowledgeLink.confidence,
                ).where(
                    db.KnowledgeLink.source_kind == "entity",
                    db.KnowledgeLink.target_kind == "entity",
                    db.KnowledgeLink.lifecycle == "active",
                    db.KnowledgeLink.source_id != db.KnowledgeLink.target_id,
                    db.KnowledgeLink.source_id.in_(uuids),
                    db.KnowledgeLink.target_id.in_(uuids),
                )
            )
        ).all()

        out: dict[str, list[Relationship]] = {}
        for source_id, target_id, relation_type, confidence in rows:
            # The renderer must never invent an edge label; an unrecognised type
            # falls back to the taxonomy's own catch-all.
            label = (
                relation_type
                if relation_type in VALID_EDGE_TYPES
                else "POSSIBLE_RELATION"
            )
            out.setdefault(str(source_id), []).append(
                Relationship(label, str(target_id), "", float(confidence or 0.0), "out")
            )
            out.setdefault(str(target_id), []).append(
                Relationship(label, str(source_id), "", float(confidence or 0.0), "in")
            )
        return out

    async def _document_note(self, session, source) -> DocumentNote | None:
        """Assemble one source's note from its semantic document, chunks and units."""
        semantic_doc = (
            await session.execute(
                select(db.Document)
                .where(
                    db.Document.source_id == source.id,
                    db.Document.meta.has_key("semantic_pass"),  # noqa: W601 - JSONB
                )
                .limit(1)
            )
        ).scalars().first()
        if semantic_doc is None:
            return None

        chunks = (
            await session.execute(
                select(db.Chunk)
                .where(db.Chunk.document_id == semantic_doc.id)
                # `order` is the pre-order index; `parent_id` is effectively
                # unpopulated (fragment pruning nulls it), so section_path plus
                # this ordering is the only reliable hierarchy signal.
                .order_by(db.Chunk.order.nulls_last(), db.Chunk.start_offset)
            )
        ).scalars().all()
        if not chunks:
            return None

        chunk_ids = [c.id for c in chunks]
        units = (
            await session.execute(
                select(db.SemanticUnit)
                .where(db.SemanticUnit.chunk_id.in_(chunk_ids))
                .order_by(db.SemanticUnit.span_start)
            )
        ).scalars().all()

        referenced: list[str] = []
        for unit in units:
            for raw in unit.entity_ids or []:
                if str(raw) not in referenced:
                    referenced.append(str(raw))
        entity_notes = await self._entity_notes(session, referenced[:_MAX_ENTITIES_PER_DOCUMENT])

        units_by_chunk: dict[uuid.UUID, list[db.SemanticUnit]] = {}
        for unit in units:
            units_by_chunk.setdefault(unit.chunk_id, []).append(unit)

        sections = [
            Section(
                section_path=chunk.section_path or "",
                source_location=chunk.source_location or "",
                chunk_role=chunk.chunk_role or "",
                content=chunk.content or "",
                claims=[
                    Claim(
                        content=unit.content,
                        polarity=int(unit.polarity),
                        fact_class=unit.fact_class,
                        confidence=float(unit.confidence or 0.0),
                        entity_slugs=[
                            (entity_notes[str(eid)].slug, entity_notes[str(eid)].canonical_name)
                            for eid in (unit.entity_ids or [])
                            if str(eid) in entity_notes
                        ],
                    )
                    for unit in units_by_chunk.get(chunk.id, [])
                ],
            )
            for chunk in chunks
        ]

        return DocumentNote(
            source_id=str(source.id),
            document_id=str(semantic_doc.id),
            title=source.name or "Untitled",
            # `sources.path` survives re-ingest (the row is updated in place)
            # while every semantic id is regenerated, so it is the only safe
            # basis for a stable filename.
            slug=slugify(f"{source.name}-{source.path}", fallback=str(source.id)),
            source_type=source.source_type or "",
            doc_type=source.doc_type,
            domain=source.domain,
            language=source.language,
            quality_tier=source.quality_tier,
            tags=[str(t) for t in (source.tags or [])],
            sections=sections,
            entities=list(entity_notes.values()),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def export_source(self, session, source_id: uuid.UUID) -> ExportResult:
        """Export one source: its document note, Canvas, and every entity it mentions."""
        result = ExportResult()
        source = (
            await session.execute(select(db.Source).where(db.Source.id == source_id))
        ).scalars().first()
        if source is None:
            return result

        note = await self._document_note(session, source)
        if note is None:
            logger.info(
                "obsidian.export_skipped",
                source=str(source_id),
                reason="no_semantic_layer",
            )
            return result

        if self._write(self.documents_dir / f"{note.slug}.md", render_document_note(note)):
            result.documents += 1
        canvas = render_canvas(note, export_dirname=self.export_dir.name)
        if self._write(self.documents_dir / f"{note.slug}.canvas", canvas):
            result.canvases += 1

        for entity in note.entities:
            enriched = EntityNote(
                **{**entity.__dict__, "mentioned_in": [(note.slug, note.title)]}
            )
            if self._write(self.entities_dir / f"{entity.slug}.md", render_entity_note(enriched)):
                result.entities += 1

        logger.info(
            "obsidian.export_source",
            source=str(source_id),
            slug=note.slug,
            entities=len(note.entities),
            sections=len(note.sections),
        )
        return result

    async def export_all(self, session, *, prune: bool = True) -> ExportResult:
        """Rebuild the whole export tree, then remove notes with no live row."""
        total = ExportResult()
        sources = (
            await session.execute(select(db.Source).order_by(db.Source.created_at))
        ).scalars().all()

        written_documents: set[str] = set()
        written_entities: set[str] = set()
        for source in sources:
            try:
                note = await self._document_note(session, source)
                if note is None:
                    continue
                written_documents.add(note.slug)
                if self._write(
                    self.documents_dir / f"{note.slug}.md", render_document_note(note)
                ):
                    total.documents += 1
                canvas = render_canvas(note, export_dirname=self.export_dir.name)
                if self._write(self.documents_dir / f"{note.slug}.canvas", canvas):
                    total.canvases += 1
                for entity in note.entities:
                    written_entities.add(entity.slug)
                    enriched = EntityNote(
                        **{**entity.__dict__, "mentioned_in": [(note.slug, note.title)]}
                    )
                    if self._write(
                        self.entities_dir / f"{entity.slug}.md", render_entity_note(enriched)
                    ):
                        total.entities += 1
            except Exception as exc:  # pragma: no cover - one bad source must not stop the run
                logger.warning(
                    "obsidian.export_source_failed", source=str(source.id), error=str(exc)
                )
                total.errors.append(f"{source.id}: {exc}")

        if prune:
            total.pruned = self._prune(written_documents, written_entities)

        logger.info(
            "obsidian.export_all",
            sources=len(sources),
            documents=total.documents,
            entities=total.entities,
            canvases=total.canvases,
            pruned=total.pruned,
            errors=len(total.errors),
        )
        return total

    def _prune(self, documents: set[str], entities: set[str]) -> int:
        """Delete generated files whose row no longer exists.

        Two guards, because deletion is the one irreversible thing this module
        does: the file must live under the export directory, *and* it must carry
        our own generated marker. A note a human wrote and filed here is left in
        place — an unexpected leftover file costs a tidy-up, deleting someone's
        writing costs their work.
        """
        removed = 0
        for directory, keep, suffixes in (
            (self.documents_dir, documents, (".md", ".canvas")),
            (self.entities_dir, entities, (".md",)),
        ):
            if not directory.is_dir():
                continue
            for path in sorted(directory.iterdir()):
                if not path.is_file() or path.suffix not in suffixes:
                    continue
                if path.stem in keep:
                    continue
                # A .canvas has no frontmatter to mark, so it is trusted only by
                # virtue of sitting beside a document note we know we generate.
                if path.suffix == ".md" and not self._is_generated(path):
                    logger.info("obsidian.prune_skipped_foreign", path=str(path))
                    continue
                path.unlink()
                removed += 1
        return removed
