"""Braind Ingest Service.

Watches Obsidian vault + other sources, parses markdown frontmatter,
chunks, embeds and stores into PostgreSQL + pgvector.
"""

from __future__ import annotations

import datetime
import hashlib
import re
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

import frontmatter as fm
from sqlalchemy import delete, func, or_, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

# Avoid circular import on type annotation only
from core import database as db
from core import make_json_safe, quality
from core.chunking import HierarchicalChunker, SemanticChunk, TextChunker
from core.config import settings
from core.logging import get_logger
from core.textanalysis import TextAnalyzer
from core.vllm import VLLMClient
from services.ingest.classify import DocumentClassifier, classify_chunk_role
from services.ingest.freshness import LanguageClassifier
from services.ingest.segment import TopicShiftSegmenter
from services.obsidian.render import GENERATED_MARKER

logger = get_logger(__name__)

# Stopwords chung cho heuristic keyword detection
_STOPWORDS = {
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
    "này",
    "đó",
    "như",
    "về",
    "ra",
    "vào",
    "the",
    "and",
    "with",
    "for",
    "from",
    "that",
    "this",
    "are",
    "was",
}


def _is_flat_document():
    """Match only the legacy flat chunk rows, never the semantic document.

    The semantic pass owns a sibling ``documents`` row tagged
    ``meta.semantic_pass``; deleting it cascades away chunks, semantic units
    and entity edges, so vault re-sync must leave it to the semantic pass.
    """
    return or_(
        db.Document.meta.is_(None),
        ~db.Document.meta.has_key("semantic_pass"),  # noqa: W601 - JSONB operator
    )


@dataclass
class _RChunk:
    """One retrieval-facing chunk destined for the ``documents`` table.

    ``chunk_kind`` records how the boundary was decided, in descending order of
    fidelity to the document's own structure:

    - ``structure`` — a structure-aware node (paragraph, list or table under a
      heading) that already fits the embedding budget.
    - ``topic_shift`` — a sub-slice of an oversized heading-less node whose
      boundary an LLM chose by reading the sentences (§2).
    - ``sentence_pack`` — the same node packed at sentence boundaries because
      the LLM was unavailable or returned nothing usable.
    - ``fallback`` — the whole-document fixed-size safety net, reached only when
      the structure-aware pass produced no prose node at all.
    """

    text: str
    section_path: str
    chunk_kind: str


# Structure-aware node kinds that carry retrievable prose. Heading-only
# ``section`` nodes are skipped because their text already lives in section_path.
_CONTENT_KINDS = {"paragraph", "list", "table"}

# Only prose is worth topic-segmenting. A list or table is already one unit; the
# LLM would cut it between rows, which destroys the thing that made it readable.
_SEGMENTABLE_KINDS = {"paragraph"}


@dataclass
class IngestResult:
    """Result of an ingestion pass."""

    source: str
    added: int = 0
    updated: int = 0
    deleted: int = 0
    skipped: int = 0
    duplicate_of: str | None = None
    job_id: str | None = None
    characters_read: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.added + self.updated + self.deleted + self.skipped + len(self.errors)

    def __str__(self) -> str:
        return (
            f"added={self.added} updated={self.updated} "
            f"deleted={self.deleted} skipped={self.skipped} errors={len(self.errors)}"
        )


class IngestService:
    """Handles ingesting markdown files (Obsidian vault) + text into brain.
    
    Now supports both V1 (legacy) and V2 (enhanced semantic) chunking.
    """

    def __init__(
        self,
        session_factory,
        vllm: VLLMClient,
        vault_dir: Path | None = None,
        semantic: object | None = None,
        job_tracker: object | None = None,
        obsidian: object | None = None,
    ):
        self.session_factory = session_factory
        self.vllm = vllm
        self.vault_dir = Path(vault_dir or settings.vault_dir)
        self.chunker = TextChunker(settings.chunk_size, settings.chunk_overlap)
        # Structure-aware primary chunker for the retrieval-facing documents
        # table (§1/§2). TextChunker above is retained only as the logged
        # fixed-size fallback in _retrieval_chunks.
        self.hchunker = HierarchicalChunker(chunk_size=settings.chunk_size)
        # LLM topic-shift segmenter for oversized heading-less nodes (§2). It
        # never cuts on embedding cosine; it falls back to sentence packing.
        self.segmenter = TopicShiftSegmenter(vllm)
        self.tanalyzer = TextAnalyzer(vllm)
        # One LLM round-trip per document produces every classification facet.
        self.classifier = DocumentClassifier(vllm)
        # Phase 2: language hint classifies content so ``meta`` carries
        # ``detected_language`` (contains_chinese stays the hard reject).
        self.lang_classifier = LanguageClassifier(vllm=vllm)
        self._file_hashes: dict[Path, str] = {}
        # Brain Second semantic pass (optional collaborator): populates the
        # Phase-3 tables after the legacy document pipeline commits (§22).
        self.semantic = semantic
        # Phase 7: optional stage-level job tracer (non-fatal observability).
        self.job_tracker = job_tracker
        # Obsidian display layer (§3): renders committed knowledge into the
        # vault. Optional — ingestion must work with no vault to write to.
        self.obsidian = obsidian

    async def _retrieval_chunks(self, content: str, title: str) -> tuple[list[_RChunk], bool]:
        """Structure-aware chunks for the ``documents`` table (§1/§2).

        Three stages, each a strictly weaker signal than the one before:

        1. ``HierarchicalChunker`` cuts on the document's own headings and block
           structure. It runs with ``compact=True, prune=False``: coalescing
           merges the one-block-per-table-cell fragments that Office and PDF
           extraction emit, while skipping the lossy prune half so no prose is
           dropped from a table retrieval must still be able to return.
        2. Any surviving prose node too long to be one topic (no internal
           heading to cut on) goes to the LLM topic-shift segmenter — a model
           reading the sentences, never an embedding-cosine threshold.
        3. The fixed-size ``TextChunker`` runs only when stage 1 produced no
           prose node at all, and says so in the log.

        Returns ``(chunks, fallback_used)``.
        """
        nodes = self.hchunker.chunk(content, title=title, compact=True, prune=False)
        kept = [n for n in nodes if n.chunk_kind in _CONTENT_KINDS and n.content.strip()]

        chunks: list[_RChunk] = []
        segmented_nodes = 0
        for node in kept:
            pieces = await self._segment_node(node, title=title)
            if pieces is None:
                chunks.append(
                    _RChunk(
                        text=node.content,
                        section_path=node.section_path,
                        chunk_kind="structure",
                    )
                )
                continue
            segmented_nodes += 1
            chunks.extend(pieces)

        fallback_used = False
        if not chunks:
            fallback_used = True
            chunks = [
                _RChunk(text=piece, section_path="", chunk_kind="fallback")
                for piece in self.chunker.chunk(content)
            ]
            logger.info(
                "ingest.fixedsize_fallback",
                title=title[:60],
                reason="no structure-aware chunks",
                pieces=len(chunks),
            )

        characters_chunked = sum(len(c.text) for c in chunks)
        characters_read = len(content)
        logger.info(
            "ingest.chunked",
            title=title[:60],
            characters_read=characters_read,
            characters_chunked=characters_chunked,
            coverage=round(characters_chunked / characters_read, 3) if characters_read else 0.0,
            chunk_kind="fallback" if fallback_used else "structure",
            chunks=len(chunks),
            segmented_nodes=segmented_nodes,
        )
        return chunks, fallback_used

    async def _segment_node(self, node: SemanticChunk, *, title: str) -> list[_RChunk] | None:
        """Topic-segment one oversized heading-less prose node (§2).

        Returns ``None`` when the node should stay whole — it is short enough,
        it is a list or table whose rows must not be separated, segmentation is
        disabled, or the segmenter found no boundary worth taking.
        """
        if not settings.segment_enabled:
            return None
        if node.chunk_kind not in _SEGMENTABLE_KINDS:
            return None
        if len(node.content) <= settings.segment_trigger_chars:
            return None

        segments = await self.segmenter.segment(
            node.content, title=node.section_path or title
        )
        if len(segments) <= 1:
            # One segment means the model saw a single topic; keep the original
            # node rather than relabelling identical text.
            return None
        return [
            _RChunk(
                text=seg.text,
                section_path=node.section_path,
                chunk_kind=(
                    "topic_shift" if seg.method == "llm_topic_shift" else "sentence_pack"
                ),
            )
            for seg in segments
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ingest_text(
        self,
        content: str,
        *,
        title: str = "",
        source_type: str = "manual",
        path: str | None = None,
        url: str | None = None,
        tags: list[str] | None = None,
        session_id: str | None = None,
        fingerprint_hash: str | None = None,
        meta: dict | None = None,
    ) -> IngestResult:
        """Ingest a single text blob (from RSS, web, manual)."""
        return await self._ingest_text(
            content,
            title=title,
            source_type=source_type,
            path=path,
            url=url,
            tags=tags,
            session_id=session_id,
            fingerprint_hash=fingerprint_hash,
            meta=meta,
        )

    async def _ingest_text(
        self,
        content: str,
        *,
        title: str = "",
        source_type: str = "manual",
        path: str | None = None,
        url: str | None = None,
        tags: list[str] | None = None,
        session_id: str | None = None,
        fingerprint_hash: str | None = None,
        meta: dict | None = None,
    ) -> IngestResult:
        """Legacy V1 ingestion (original implementation)."""
        title = title or (path and Path(path).stem) or "Untitled"
        result = IngestResult(source=title)

        jin = None
        if self.job_tracker is not None:
            jin = await self.job_tracker.new_run()
            result.job_id = str(jin)

        # --- Quality gate: strip chrome, reject what is not knowledge -------
        # Replaces the old "len < 120 and no Han ideographs" pair, which let in
        # search-result snippets, navigation menus and PDF byte streams.
        report = quality.assess(content, source_type=source_type)
        if not report.accepted:
            result.skipped += 1
            logger.info(
                "ingest.rejected",
                title=title[:60],
                reason=report.reason,
                prose=round(report.prose_ratio, 2),
                words=report.word_count,
            )
            return result
        content = report.text
        try:
            dup = await self.dedup_check(content, threshold=0.92)
            if dup.get("count", 0) and dup.get("duplicates"):
                d = dup["duplicates"][0]
                result.skipped += 1
                result.duplicate_of = str(d["id"])
                logger.info(
                    "ingest.skip_similar",
                    title=title[:60],
                    sim=round(d["similarity"], 3),
                    dup_title=str(d["title"])[:60],
                )
                return result
        except Exception as exc:
            logger.warning("ingest.dedup_check_failed", error=str(exc))

        try:
            async with self.session_factory() as session:
                # --- Dedup: skip if this exact content already exists ---------
                content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                existing = (
                    (await session.execute(select(db.Source).where(db.Source.path == content_hash)))
                    .scalars()
                    .first()
                )
                if existing is not None:
                    result.skipped += 1
                    result.duplicate_of = str(existing.id)
                    logger.info("ingest.dedup", title=title[:60], dup_id=str(existing.id))
                    return result

                # --- Classification: one LLM round-trip for every facet -----
                classification = await self.classifier.classify(
                    content,
                    title=title,
                    source_type=source_type,
                    quality=report,
                    fallback_keywords=tags,
                )
                # The Han-ideograph rule applies to model output too: 27 stored
                # summaries were in Chinese despite every input passing the gate.
                classification.summary = quality.sanitize_model_text(
                    classification.summary, fallback=title[:200]
                )
                keywords = classification.keywords
                if tags:
                    # Merge explicit tags ahead of extracted ones (dedupe).
                    keywords = list(dict.fromkeys(list(tags) + keywords))[:10]

                safe_meta = make_json_safe(
                    {
                        **classification.as_meta(),
                        "keywords": keywords,
                        "quality": {
                            "prose_ratio": round(report.prose_ratio, 4),
                            "word_count": report.word_count,
                            "chrome_lines_removed": report.chrome_lines_removed,
                        },
                        **(meta or {}),
                    }
                )

                # --- Store source ----------------------------------------------
                # Giai đoạn 4: session scoping. A request-supplied uuid marks
                # the row private to one visitor; None stays system/global.
                try:
                    _sid = uuid.UUID(session_id) if session_id else None
                except (ValueError, AttributeError):
                    _sid = None
                source = db.Source(
                    name=title,
                    source_type=source_type,
                    path=content_hash,
                    url=url,
                    meta=safe_meta,
                    content_hash=content_hash,
                    tags=tags or keywords,
                    doc_type=classification.doc_type,
                    domain=classification.domain or None,
                    audience=classification.audience,
                    quality_tier=classification.quality_tier,
                    quality_score=classification.quality_score,
                    language=classification.language,
                    session_id=_sid,
                    fingerprint_hash=fingerprint_hash,
                )
                session.add(source)
                await session.flush()
                result.added += 1

                # --- Chunk + embed + store ------------------------------------
                # Structure-aware primary chunking (§1/§2); fixed-size only as a
                # logged fallback inside _retrieval_chunks.
                chunks, _ = await self._retrieval_chunks(content, title)
                result.characters_read = len(content)
                for i, rc in enumerate(chunks):
                    embedding = None
                    try:
                        embedding = await self.vllm.embed(rc.text)
                    except Exception as exc:
                        logger.warning("ingest.embed_failed", error=str(exc))
                    doc = db.Document(
                        source_id=source.id,
                        title=title,
                        content=rc.text,
                        content_type="markdown",
                        chunk_index=i,
                        chunk_count=len(chunks),
                        embedding_vllm_1024=embedding,
                        tags=tags or keywords,
                        # Giai đoạn 4: denormalized session scope for the
                        # retrieval streams (no join needed).
                        session_id=source.session_id,
                        meta={
                            "source_type": source_type,
                            "section_path": rc.section_path,
                            "chunk_kind": rc.chunk_kind,
                            **safe_meta,
                        },
                        # Denormalized so retrieval filters without a join, plus
                        # the per-passage role a source-level label cannot carry.
                        doc_type=classification.doc_type,
                        domain=classification.domain or None,
                        chunk_role=classify_chunk_role(rc.text),
                        quality_tier=classification.quality_tier,
                        quality_score=classification.quality_score,
                    )
                    session.add(doc)

                # --- Auto-link: related docs share keywords --------------------
                links = await self._auto_link(session, keywords or tags, current_id=source.id)
                if links:
                    source.meta = {**(source.meta or {}), "links": links}

                # Phase 8: record an immutable ingest version (reason="ingest").
                await self._record_version(session, source.id, content_hash)

                await session.commit()
                if jin is not None:
                    await self.job_tracker.finish_run(
                        jin, "legacy_ingest", status="ok", source_id=source.id
                    )
                logger.info(
                    "ingest.complete",
                    title=title[:60],
                    chunks=len(chunks),
                    keywords=keywords[:5],
                    links=len(links),
                )

                # --- Brain Second semantic pass (independent + retryable §22).
                # Runs AFTER the legacy commit so a semantic failure never rolls
                # back the stored source; only the semantic tables are affected.
                await self._run_semantic_pass(
                    session,
                    jin=jin,
                    source_id=source.id,
                    content=content,
                    title=title,
                    source_type=source_type,
                    source_tags=keywords,
                    domain=classification.domain or None,
                    session_id=session_id,
                )
        except Exception as exc:
            if jin is not None:
                with suppress(Exception):  # tracing must never mask the original
                    await self.job_tracker.finish_run(
                        jin, "legacy_ingest", status="failed", error=str(exc)[:2000]
                    )
            logger.error("ingest.error", error=str(exc), exc_info=True)
            result.errors.append(str(exc))
        return result

    async def _record_version(
        self,
        session: AsyncSession,
        source_id: uuid.UUID,
        content_hash: str | None,
        *,
        reason: str = "ingest",
    ) -> None:
        """Append one immutable ``DocumentVersion`` for a source (§18).

        Reads the highest version number for the source and inserts the next one
        under the unique (source_id, version) index. Idempotent-by-monotonic: a
        re-ingest of the same source always records a newer version rather than
        overwriting history. Non-fatal on failure (versioning must not abort the
        ingest that already committed).
        """
        try:
            last = (
                (
                    await session.execute(
                        select(func.max(db.DocumentVersion.version)).where(
                            db.DocumentVersion.source_id == source_id
                        )
                    )
                ).scalar_one_or_none()
                or 0
            )
            session.add(
                db.DocumentVersion(
                    source_id=source_id,
                    version=int(last) + 1,
                    content_hash=content_hash,
                    reason=reason,
                )
            )
        except Exception:  # pragma: no cover - defensive
            logger.warning("ingest.version_record_failed", exc_info=True)

    async def _auto_link(
        self,
        session: AsyncSession,
        keywords: list[str] | None,
        *,
        current_id,
        relation_type: str = "related",
        limit: int = 8,
    ) -> list[dict]:
        """Find related sources sharing the same tags/keywords and persist the
        edges as first-class rows in ``knowledge_links`` (upsert on the unique
        (source_id, target_id, relation_type) index so re-ingests never create
        duplicates). Also returns the link summary for embedding in meta.
        """
        if not keywords:
            return []
        try:
            kwset = {str(k).lower() for k in keywords}
            rows = (
                (
                    await session.execute(
                        select(db.Source)
                        .where(db.Source.id != current_id)
                        .order_by(db.Source.created_at.desc())
                        .limit(500)
                    )
                )
                .scalars()
                .all()
            )

            links: list[dict] = []
            seen: set[tuple[str, str]] = set()
            now = datetime.datetime.now(datetime.UTC)

            for src in rows:
                src_kw = set(str(k).lower() for k in self._source_keywords(src))
                shared = kwset & src_kw
                if not shared:
                    continue
                weight = len(shared) / max(len(kwset), len(src_kw), 1)
                a, b = str(current_id), str(src.id)
                key = tuple(sorted((a, b)))
                if key in seen:
                    continue
                seen.add(key)

                # Upsert the traction edge into the normalized graph table.
                # Giai đoạn 3: legacy keyword edges used to bypass the relation
                # validator entirely (confidence defaulted to 1.0 with empty
                # evidence). They now carry honest evidence — the shared
                # keywords themselves — and a proportionate confidence.
                await session.execute(
                    pg_insert(db.KnowledgeLink)
                    .values(
                        source_id=a,
                        target_id=b,
                        relation_type=relation_type,
                        weight=round(weight, 4),
                        confidence=round(weight, 4),
                        evidence=[
                            f"shared_keywords:{','.join(sorted(shared)[:8])}"
                        ],
                        evidence_count=1,
                        model_version="keyword-link-v1",
                        reason=(
                            "keyword co-occurrence between sources "
                            f"({len(shared)} shared tags)"
                        ),
                        source_kind="source",
                        target_kind="source",
                        meta={"shared": sorted(shared)[:8], "created_at": now.isoformat()},
                    )
                    .on_conflict_do_nothing(
                        index_elements=[
                            "source_id",
                            "target_id",
                            "relation_type",
                        ]
                    )
                )

                links.append(
                    {
                        "source_id": b,
                        "title": src.name[:120],
                        "shared": sorted(shared)[:4],
                        "weight": round(weight, 2),
                    }
                )
                if len(links) >= limit:
                    break

            return links
        except Exception as exc:
            logger.warning("ingest.link_failed", error=str(exc))
            return []

    @staticmethod
    def _source_keywords(source) -> list[str]:
        """Collect normalized source tags, keywords, and topic labels safely."""
        keywords: set[str] = set()
        tags = getattr(source, "tags", None)
        if isinstance(tags, list):
            keywords.update(str(tag).strip().lower() for tag in tags if str(tag).strip())

        meta = getattr(source, "meta", None) or {}
        for key in ("keywords", "tags", "auto_topic"):
            value = meta.get(key)
            if isinstance(value, list):
                keywords.update(str(item).strip().lower() for item in value if str(item).strip())
            elif isinstance(value, str) and value.strip():
                keywords.add(value.strip().lower())
        return sorted(keywords)

    @staticmethod
    def _extract_keywords_heuristic(text: str, limit: int = 6) -> list[str]:
        """Fast English/Vietnamese keyword extraction (fallback if LLM down)."""
        words = re.findall(r"[A-Za-zÀ-ỹ][A-Za-zÀ-ỹ\-]{3,}", text.lower())
        stop = _STOPWORDS
        freq: dict[str, int] = {}
        for w in words:
            if w not in stop:
                freq[w] = freq.get(w, 0) + 1
        ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        return [w for w, _ in ranked[:limit]]

    async def dedup_check(self, content: str, threshold: float = 0.92) -> dict:
        """Check whether similar content already exists using embedding cosine.

        Args:
            content: Candidate plain-text content to compare.
            threshold: Minimum cosine similarity considered a duplicate.

        Returns:
            Matching document metadata and the number of matches.
        """
        try:
            embedding = await self.vllm.embed(content[:4000])
            vector_literal = "[" + ",".join(str(value) for value in embedding) + "]"
            async with self.session_factory() as sess:
                rows = (
                    await sess.execute(
                        text(
                            "SELECT id, title, "
                            "1 - (embedding_vllm_1024 <=> CAST(:vec AS vector)) AS sim "
                            "FROM documents WHERE embedding_vllm_1024 IS NOT NULL "
                            "ORDER BY embedding_vllm_1024 <=> CAST(:vec AS vector) LIMIT 5"
                        ),
                        {"vec": vector_literal},
                    )
                ).all()
            dups = [
                {
                    "id": str(r[0]),
                    "title": r[1],
                    "similarity": float(r[2]),
                }
                for r in rows
                if float(r[2]) >= threshold
            ]
            return {"duplicates": dups, "count": len(dups)}
        except Exception as exc:
            logger.warning("ingest.dedup_failed", error=str(exc))
            return {"duplicates": [], "count": 0, "error": str(exc)}

    # ------------------------------------------------------------------
    # Vault (Obsidian) sync
    # ------------------------------------------------------------------

    async def sync_vault(self) -> IngestResult:
        """Full sync of the Obsidian vault directory."""
        result = IngestResult(source=str(self.vault_dir))
        if not self.vault_dir.exists():
            logger.warning("vault.missing", path=str(self.vault_dir))
            return result

        for path in sorted(self.vault_dir.rglob("*.md")):
            if ".obsidian" in path.parts:
                continue
            if settings.obsidian_export_dirname in path.parts:
                # The Obsidian display layer writes generated notes here. They
                # are derived from the knowledge base, so ingesting them would
                # feed the system its own output back as source material.
                continue
            try:
                await self._ingest_file(path, result)
            except Exception as exc:
                logger.error("vault.file_error", path=str(path), error=str(exc))
                result.errors.append(f"{path}: {exc}")

        logger.info(
            "vault.sync_done",
            added=result.added,
            updated=result.updated,
            errors=len(result.errors),
        )
        return result

    async def _ingest_file(self, path: Path, result: IngestResult) -> None:
        rel = path.relative_to(self.vault_dir)
        content = path.read_text(encoding="utf-8", errors="replace")
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        if f"{GENERATED_MARKER}: true" in content[:512]:
            # A note written by the Obsidian display layer. The directory guard
            # in sync_vault already covers the normal case; this catches a
            # generated note that was moved or copied into a user folder. It runs
            # *before* the hash is recorded so the file is re-checked every sync
            # rather than being remembered as "seen".
            logger.debug("vault.skip_generated", path=str(rel))
            return

        jin = None
        if self.job_tracker is not None:
            jin = await self.job_tracker.new_run()
            result.job_id = str(jin)

        # Skip unchanged files using the content hash.
        file_id = str(path)
        last_hash = self._file_hashes.get(path)
        if last_hash == content_hash:
            return
        self._file_hashes[path] = content_hash

        # Parse frontmatter
        try:
            post = fm.loads(content)
            meta = dict(post.metadata or {})
            body = post.content
            title = str(meta.pop("title", path.stem))
            tags = meta.pop("tags", []) or []
            if isinstance(tags, str):
                tags = [tags]
            source_type = str(meta.pop("type", "obsidian"))
        except Exception:
            body = content
            title = path.stem
            tags = []
            source_type = "obsidian"
            meta = {}

        # Vault notes are trusted input: the gate rejects only the empty, the
        # binary and Han ideographs, never a note's structure.
        report = quality.assess(body, source_type=source_type)
        if not report.accepted:
            result.skipped += 1
            logger.info(
                "ingest.rejected", title=title[:60], path=str(rel), reason=report.reason
            )
            return
        body = report.text

        # Chunk + embed + store — structure-aware primary (§1/§2).
        chunks, _ = await self._retrieval_chunks(body, title)
        if not chunks:
            return
        result.characters_read += len(body)

        async with self.session_factory() as session:
            # Find or create source
            source = (
                (await session.execute(select(db.Source).where(db.Source.path == file_id)))
                .scalars()
                .first()
            )

            # Smart analysis (only when file is new or updated)
            classification = await self.classifier.classify(
                body,
                title=title,
                source_type=source_type,
                quality=report,
                fallback_keywords=tags,
            )
            classification.summary = quality.sanitize_model_text(
                classification.summary, fallback=title[:200]
            )
            keywords = classification.keywords
            safe_meta = make_json_safe(
                {
                    "relative_path": str(rel),
                    **classification.as_meta(),
                    "keywords": keywords,
                    **meta,
                }
            )

            is_new = source is None
            if source is None:
                source = db.Source(
                    name=title,
                    source_type=source_type,
                    path=file_id,
                    meta=safe_meta,
                    content_hash=content_hash,
                    tags=tags,
                    doc_type=classification.doc_type,
                    domain=classification.domain or None,
                    audience=classification.audience,
                    quality_tier=classification.quality_tier,
                    quality_score=classification.quality_score,
                    language=classification.language,
                )
                session.add(source)
                await session.flush()
                result.added += 1
            else:
                source.name = title
                source.meta = safe_meta
                source.content_hash = content_hash
                source.tags = tags
                source.doc_type = classification.doc_type
                source.domain = classification.domain or None
                source.audience = classification.audience
                source.quality_tier = classification.quality_tier
                source.quality_score = classification.quality_score
                source.language = classification.language
                result.updated += 1

            # Populate parent-child links when the note declares a `parent`
            # in frontmatter or the filename matches a `[[...]]` wikilink title.
            parent_id = await self._resolve_parent(session, source, title, meta or {}, content)
            if parent_id is not None and source.parent_id != parent_id:
                source.parent_id = parent_id

            # Remove only the flat chunk rows. The semantic document (and the
            # chunks/units/entities cascading from it) is replaced inside the
            # semantic pass's own transaction, so a failed pass leaves the
            # previous semantic layer intact instead of wiping it.
            await session.execute(
                delete(db.Document).where(
                    db.Document.source_id == source.id,
                    _is_flat_document(),
                )
            )

            for i, rc in enumerate(chunks):
                try:
                    embedding = await self.vllm.embed(rc.text)
                except Exception:
                    embedding = None  # Still store even if embedding fails
                doc = db.Document(
                    source_id=source.id,
                    title=title,
                    content=rc.text,
                    content_type="markdown",
                    chunk_index=i,
                    chunk_count=len(chunks),
                    embedding_vllm_1024=embedding,
                    tags=tags,
                    meta={
                        "relative_path": str(rel),
                        "section_path": rc.section_path,
                        "chunk_kind": rc.chunk_kind,
                        **safe_meta,
                    },
                    doc_type=classification.doc_type,
                    domain=classification.domain or None,
                    chunk_role=classify_chunk_role(rc.text),
                    quality_tier=classification.quality_tier,
                    quality_score=classification.quality_score,
                )
                session.add(doc)

            # Auto-link to related sources (shared keywords)
            links = await self._auto_link(session, keywords or tags, current_id=source.id)
            if links:
                source.meta = {**(source.meta or {}), "links": links}

            # Phase 8: record an immutable version for the file ingest.
            await self._record_version(session, source.id, content_hash)

            await session.commit()
            if jin is not None:
                await self.job_tracker.finish_run(
                    jin, "vault_ingest", status="ok", source_id=source.id
                )
            logger.debug(
                "vault.file_ingested",
                file=str(rel),
                chunks=len(chunks),
                added=is_new,
                parent=str(parent_id) if parent_id else None,
                links=len(links),
            )

            # --- Brain Second semantic pass (independent + retryable §22).
            # Mirrors the text-ingest chain: the legacy commit is preserved and
            # the semantic collaborator re-runs so vault re-syncs never wipe the
            # derived semantic document (documented in INGESTION_ARCHITECTURE.md).
            await self._run_semantic_pass(
                session,
                jin=jin,
                source_id=source.id,
                content=body,
                title=title,
                source_type=source_type,
                source_tags=list(keywords or tags),
                domain=classification.domain or None,
                session_id=None,  # vault sync is a system background op (shared KB)
            )

    async def _run_semantic_pass(
        self,
        session: AsyncSession,
        *,
        jin: str | None,
        source_id: uuid.UUID,
        content: str,
        title: str,
        source_type: str,
        source_tags: list[str],
        domain: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Run the optional Brain Second semantic pass after a legacy commit (§22).

        Best-effort: a semantic failure only touches the semantic tables, never
        the already-committed legacy source. Shared by the text-ingest and
        vault-sync paths so both keep the derived semantic layer in sync.
        """
        if self.semantic is None or source_id is None:
            return
        try:
            stats = await self.semantic.semantic_pass(
                session,
                source_id=source_id,
                content=content,
                title=title,
                source_type=source_type,
                source_tags=source_tags,
                domain=domain,
                session_id=session_id,
            )
            if stats.get("chunks", 0) > 0:
                await session.commit()
                if jin is not None and self.job_tracker is not None:
                    await self.job_tracker.finish_run(
                        jin, "semantic_pass", status="ok", source_id=source_id
                    )
                logger.info("ingest.semantic_pass", stats=stats)
                await self._export_obsidian(session, source_id)
            else:
                await session.rollback()
                if jin is not None and self.job_tracker is not None:
                    await self.job_tracker.finish_run(
                        jin, "semantic_pass", status="failed",
                        error="no chunks produced", source_id=source_id,
                    )
                logger.warning(
                    "ingest.semantic_pass_failed",
                    title=title[:60],
                    stats=stats,
                )
        except Exception as exc:
            await session.rollback()
            if jin is not None and self.job_tracker is not None:
                await self.job_tracker.finish_run(
                    jin, "semantic_pass", status="failed",
                    error=str(exc)[:2000], source_id=source_id,
                )
            logger.warning(
                "ingest.semantic_pass_error",
                title=title[:60],
                error=str(exc),
            )

    async def _export_obsidian(self, session: AsyncSession, source_id: uuid.UUID) -> None:
        """Render this source into the Obsidian display layer (§3).

        Best-effort in the same sense as the semantic pass itself: the knowledge
        is already committed, and a vault that is briefly out of date is a far
        smaller problem than an ingest that fails because a file could not be
        written. The next export brings it back in line.
        """
        if self.obsidian is None or not settings.obsidian_export_enabled:
            return
        try:
            await self.obsidian.export_source(session, source_id)
        except Exception as exc:
            logger.warning(
                "obsidian.export_failed", source=str(source_id), error=str(exc)
            )

    async def _resolve_parent(
        self,
        session: AsyncSession,
        source,
        title: str,
        meta: dict,
        content: str,
    ) -> uuid.UUID | None:
        """Resolve an explicit or inferred parent for a vault note.

        Priority: ``parent`` frontmatter > Obsidian ``[[wikilink]]`` in the body
        > a folder hierarchy hint. Returns the parent source id or None.
        """
        target_name: str | None = None

        declared = meta.get("parent") or meta.get("parent_note")
        if declared:
            target_name = str(declared).strip().rstrip(".md").replace("/", "")

        if not target_name:
            for m in re.finditer(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]", content):
                name = m.group(1).strip()
                if not name:
                    continue
                # Prefer a title match; otherwise pick the first wikilink.
                if name.lower() == title.lower():
                    continue
                target_name = name
                break

        if not target_name:
            return None

        parent = (
            (await session.execute(select(db.Source).where(db.Source.name.ilike(target_name))))
            .scalars()
            .first()
        )
        if parent is None or parent.id == source.id:
            return None

        # Persist the explicit parent-child edge. Giai đoạn 3: carries the
        # provenance (frontmatter/wikilink/folder) as its evidence instead of
        # relying on column defaults.
        await session.execute(
            pg_insert(db.KnowledgeLink)
            .values(
                source_id=str(parent.id),
                target_id=str(source.id),
                relation_type="PARENT_OF",
                weight=1.0,
                confidence=1.0,
                evidence=[f"declared_parent:{target_name}"],
                evidence_count=1,
                model_version="vault-structure-v1",
                reason="explicit parent declared in the vault note",
                source_kind="source",
                target_kind="source",
                meta={"title": title[:120]},
            )
            .on_conflict_do_nothing(index_elements=["source_id", "target_id", "relation_type"])
        )
        return parent.id

    async def get_stats(self, session: AsyncSession | None = None) -> dict:
        """Return stats about the knowledge base."""

        async def _stats(sess: AsyncSession) -> dict:
            from sqlalchemy import func as safunc

            doc_count = (await sess.execute(safunc.count(db.Document.id))).scalar()
            src_count = (await sess.execute(safunc.count(db.Source.id))).scalar()

            # Per-source-type breakdown
            by_type_rows = (
                await sess.execute(
                    select(db.Source.source_type, safunc.count(db.Source.id)).group_by(
                        db.Source.source_type
                    )
                )
            ).all()
            by_type = {t or "manual": c for t, c in by_type_rows}

            # Recent sources (labels for memory view)
            recent_rows = (
                await sess.execute(
                    select(db.Source.name, db.Source.id, db.Source.meta, db.Source.created_at)
                    .order_by(db.Source.created_at.desc())
                    .limit(30)
                )
            ).all()
            recent = [
                {
                    "id": str(r[1]),
                    "title": str(r[0])[:80],
                    "summary": ((r[2] or {}).get("summary") or "")[:200],
                    "keywords": (r[2] or {}).get("keywords") or [],
                    "links": (r[2] or {}).get("links") or [],
                    "created_at": r[3].isoformat() if r[3] else None,
                }
                for r in recent_rows
            ]
            return {
                "sources": src_count,
                "chunks": doc_count,
                "by_type": by_type,
                "recent": recent,
            }

        if session is not None:
            return await _stats(session)
        async with self.session_factory() as sess:
            return await _stats(sess)


def new_ingest_service(
    session_factory,
    vllm: VLLMClient,
    vault_dir: Path | None = None,
) -> IngestService:
    """Create an ingest service using the shared vLLM client."""
    return IngestService(session_factory, vllm, vault_dir)
