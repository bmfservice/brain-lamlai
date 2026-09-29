"""One-time local-model cleanup for existing Braind knowledge.

Runs against the configured PostgreSQL database. It applies the approved balanced
policy: remove certain placeholder/too-short records, re-embed missing vectors,
fill incomplete metadata with the local vLLM model, remove near-duplicate
sources while retaining the richer record, then rebuild normalized graph links.

Run a dry audit first:
    PYTHONPATH=. .venv/bin/python scripts/cleanup_knowledge.py --dry-run

Apply changes:
    PYTHONPATH=. .venv/bin/python scripts/cleanup_knowledge.py --apply
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field

from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from core import make_json_safe
from core.config import settings
from core.database import KnowledgeLink, Source, create_engine, init_db
from core.logging import get_logger
from core.textanalysis import TextAnalyzer
from core.vllm import VLLMClient
from services.ingest.service import IngestService

logger = get_logger(__name__)

MIN_CONTENT_CHARS = 120
DUPLICATE_SIMILARITY = 0.92
PLACEHOLDER_TITLES = {
    "mẫu note kiến thức",
    "mẫu ghi chú hằng ngày",
    "mẫu dự án",
}


@dataclass
class CleanupReport:
    """Counts and audited record labels produced by a cleanup pass."""

    dry_run: bool
    deleted: int = 0
    deleted_titles: list[str] = field(default_factory=list)
    metadata_enriched: int = 0
    embeddings_repaired: int = 0
    duplicates_deleted: int = 0
    duplicate_pairs: list[str] = field(default_factory=list)
    links_created: int = 0
    errors: list[str] = field(default_factory=list)

    def print_summary(self) -> None:
        """Print a stable, concise cleanup report for operators."""
        prefix = "DRY RUN" if self.dry_run else "APPLIED"
        print(f"{prefix} cleanup summary")
        print(f"  deleted: {self.deleted}")
        print(f"  metadata_enriched: {self.metadata_enriched}")
        print(f"  embeddings_repaired: {self.embeddings_repaired}")
        print(f"  duplicates_deleted: {self.duplicates_deleted}")
        print(f"  links_created_or_confirmed: {self.links_created}")
        if self.deleted_titles:
            print("  deleted_titles:")
            for title in self.deleted_titles:
                print(f"    - {title}")
        if self.duplicate_pairs:
            print("  duplicate_pairs:")
            for pair in self.duplicate_pairs:
                print(f"    - {pair}")
        if self.errors:
            print("  errors:")
            for error in self.errors:
                print(f"    - {error}")


def source_text(source: Source) -> str:
    """Return bounded source content by joining its already-loaded documents."""
    chunks = sorted(source.documents, key=lambda doc: doc.chunk_index)
    return "\n\n".join(chunk.content for chunk in chunks if chunk.content).strip()[:6000]


def source_quality_rank(source: Source) -> tuple[int, int, int]:
    """Rank duplicate candidates; higher rank means the source should survive."""
    content_size = sum(len(doc.content or "") for doc in source.documents)
    metadata_size = len((source.meta or {}).get("summary") or "")
    tag_count = len(source.tags or [])
    return (content_size, metadata_size, tag_count)


def needs_metadata(source: Source) -> bool:
    """Return whether the source lacks useful graph/search metadata."""
    meta = source.meta or {}
    return not (
        (meta.get("summary") or "").strip()
        and isinstance(meta.get("keywords"), list)
        and meta.get("keywords")
        and source.tags
    )


async def load_sources(session: AsyncSession) -> list[Source]:
    """Load all sources and documents needed for one cleanup operation."""
    from sqlalchemy.orm import selectinload

    result = await session.execute(
        select(Source)
        .options(selectinload(Source.documents))
        .order_by(Source.created_at.asc())
    )
    return list(result.scalars().unique())


async def delete_low_quality_sources(
    session: AsyncSession,
    sources: Sequence[Source],
    report: CleanupReport,
) -> set[str]:
    """Delete approved placeholder and too-short sources using DB cascades."""
    deleted_ids: set[str] = set()
    for source in sources:
        content = source_text(source)
        title = (source.name or "").strip()
        title_key = title.lower()
        is_placeholder = title_key in PLACEHOLDER_TITLES
        is_too_short = len(content) < MIN_CONTENT_CHARS
        if not (is_placeholder or is_too_short):
            continue

        reason = "placeholder" if is_placeholder else f"content<{MIN_CONTENT_CHARS}"
        label = f"[{reason}] {title} ({len(content)} chars)"
        report.deleted += 1
        report.deleted_titles.append(label)
        deleted_ids.add(str(source.id))
        if not report.dry_run:
            await session.delete(source)
    if not report.dry_run:
        await session.flush()
    return deleted_ids


async def enrich_metadata(
    sources: Sequence[Source],
    analyzer: TextAnalyzer,
    report: CleanupReport,
) -> None:
    """Fill summary, keywords, topic, tags, and a stable hash with vLLM."""
    for source in sources:
        if not needs_metadata(source):
            continue
        content = source_text(source)
        if len(content) < MIN_CONTENT_CHARS:
            continue
        try:
            summary = await analyzer.summarize(content, title=source.name)
            keywords = await analyzer.extract_keywords(content, fallback_tags=source.tags)
            topic = await analyzer.detect_topic(content)
            tags = list(
                dict.fromkeys(
                    str(value).strip().lower()
                    for value in [*(source.tags or []), *keywords, topic]
                    if str(value).strip()
                )
            )[:10]
            meta = {
                **(source.meta or {}),
                "summary": summary,
                "keywords": keywords,
                "auto_topic": topic or None,
                "cleaned_with_model": settings.llm_model,
            }
            report.metadata_enriched += 1
            if not report.dry_run:
                source.meta = make_json_safe(meta)
                source.tags = tags
                source.content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                for document in source.documents:
                    document.tags = tags
                    document.meta = make_json_safe(
                        {**(document.meta or {}), "source_type": source.source_type, **meta}
                    )
        except Exception as exc:
            error = f"metadata {source.name[:80]}: {exc}"
            logger.warning("cleanup.metadata_failed", error=error)
            report.errors.append(error)


async def repair_embeddings(
    sources: Sequence[Source],
    vllm: VLLMClient,
    report: CleanupReport,
) -> None:
    """Generate the missing vector embeddings with the configured local model."""
    for source in sources:
        for document in source.documents:
            if document.embedding_vllm_1024 is not None:
                continue
            try:
                embedding = await vllm.embed(document.content)
                report.embeddings_repaired += 1
                if not report.dry_run:
                    document.embedding_vllm_1024 = embedding
            except Exception as exc:
                error = f"embedding {source.name[:80]}: {exc}"
                logger.warning("cleanup.embedding_failed", error=error)
                report.errors.append(error)


async def remove_near_duplicates(
    session: AsyncSession,
    sources: Sequence[Source],
    report: CleanupReport,
) -> set[str]:
    """Remove cosine near-duplicates, retaining the richer/older source deterministically."""
    survivors = [source for source in sources if source.documents and str(source.id) not in set()]
    removed_ids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()

    for source in survivors:
        if str(source.id) in removed_ids or not source.documents:
            continue
        document = source.documents[0]
        if document.embedding is None:
            continue
        vector = "[" + ",".join(str(value) for value in document.embedding) + "]"
        rows = (
            await session.execute(
                text(
                    "SELECT d.source_id::text, "
                    "1 - (d.embedding <=> CAST(:vec AS vector)) AS similarity "
                    "FROM documents d WHERE d.embedding IS NOT NULL "
                    "AND d.source_id <> :source_id "
                    "ORDER BY d.embedding <=> CAST(:vec AS vector) LIMIT 8"
                ),
                {"vec": vector, "source_id": str(source.id)},
            )
        ).all()
        for other_id, similarity in rows:
            if float(similarity) < DUPLICATE_SIMILARITY or str(other_id) in removed_ids:
                continue
            pair_key = tuple(sorted((str(source.id), str(other_id))))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            other = next((item for item in survivors if str(item.id) == str(other_id)), None)
            if other is None or not other.documents:
                continue

            keep, discard = (source, other)
            if source_quality_rank(other) > source_quality_rank(source):
                keep, discard = other, source
            if (
                source_quality_rank(other) == source_quality_rank(source)
                and other.created_at < source.created_at
            ):
                keep, discard = other, source
            if str(discard.id) in removed_ids:
                continue

            report.duplicates_deleted += 1
            report.duplicate_pairs.append(
                f"{discard.name[:70]} -> {keep.name[:70]} (cosine={float(similarity):.3f})"
            )
            removed_ids.add(str(discard.id))
            if not report.dry_run:
                await session.delete(discard)
                await session.flush()
    return removed_ids


async def rebuild_links(
    session: AsyncSession,
    ingest: IngestService,
    report: CleanupReport,
) -> None:
    """Recreate related edges from clean metadata without retaining stale graph rows."""
    if not report.dry_run:
        await session.execute(delete(KnowledgeLink).where(KnowledgeLink.relation_type == "related"))
        await session.flush()

    sources = await load_sources(session)
    for source in sources:
        keywords = ingest._source_keywords(source)
        if not keywords:
            continue
        links = await ingest._auto_link(session, keywords, current_id=source.id, limit=12)
        report.links_created += len(links)
        if not report.dry_run and links:
            source.meta = {**(source.meta or {}), "links": links}


async def main(apply: bool) -> None:
    """Execute an audit-only or transactional cleanup pass."""
    engine = create_engine()
    report = CleanupReport(dry_run=not apply)
    try:
        await init_db(engine)
        vllm = VLLMClient()
        if not await vllm.is_available():
            raise RuntimeError(
                "vLLM chat and embedding endpoints must be available before cleanup"
            )
        analyzer = TextAnalyzer(vllm)

        async with AsyncSession(engine, expire_on_commit=False) as session:
            sources = await load_sources(session)
            deleted = await delete_low_quality_sources(session, sources, report)
            retained = [source for source in sources if str(source.id) not in deleted]
            await enrich_metadata(retained, analyzer, report)
            await repair_embeddings(retained, vllm, report)
            duplicate_ids = await remove_near_duplicates(session, retained, report)
            retained = [source for source in retained if str(source.id) not in duplicate_ids]
            await rebuild_links(session, IngestService(None, vllm), report)
            if apply:
                await session.commit()
            else:
                await session.rollback()
    finally:
        await vllm.close()
        await engine.dispose()

    report.print_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean existing Braind knowledge with vLLM.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Persist cleanup changes. Without this flag, run an audit-only dry run.",
    )
    args = parser.parse_args()
    asyncio.run(main(apply=args.apply))
