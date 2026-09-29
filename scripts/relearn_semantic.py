"""Re-learn all existing knowledge with the current Brain-Second semantic code.

For every source it:
  1. reconstructs the full text from the already-stored legacy document chunks,
  2. deletes that source's *derived* semantic documents (``meta.semantic_pass``)
     — chunks / semantic_units / search_evidence cascade away — leaving the
     original ``sources`` and ``documents`` untouched,
  3. re-runs ``SemanticIngestService.semantic_pass`` with the current pipeline
     (hierarchical chunks, language + freshness stamps, concepts, search
     validation), and
  4. appends an immutable ``DocumentVersion`` (reason="update").

Entities and ``knowledge_links`` are upserted by canonical name / unique pair,
so a re-run merges instead of duplicating them.

Run a dry audit first (prints what would change, touches nothing):
    PYTHONPATH=. .venv/bin/python scripts/relearn_semantic.py --dry-run

Apply:
    PYTHONPATH=. .venv/bin/python scripts/relearn_semantic.py --apply

Operator-only: may mutate PostgreSQL. Make a backup before ``--apply``.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
from dataclasses import dataclass, field

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from core.database import (
    Document,
    DocumentVersion,
    Source,
    create_engine,
    init_db,
)
from core.logging import get_logger
from core.vllm import VLLMClient
from services.ingest.search_validate import SearchValidator
from services.ingest.semantic import SemanticIngestService
from services.ingest.service import IngestService
from services.tools.web import WebSearchTool

logger = get_logger(__name__)

MIN_CHARS = 50


@dataclass
class RelearnReport:
    dry_run: bool
    learned: int = 0
    skipped_empty: int = 0
    failed: int = 0
    failures: list[str] = field(default_factory=list)
    chunk_total: int = 0
    unit_total: int = 0


def reconstruct_text(source: Source) -> str:
    """Join the source's stored flat chunks back into a full text.

    The semantic pass keeps its own sibling ``documents`` row holding the whole
    text; including it here would duplicate the entire source.
    """
    chunks = sorted(
        (
            d
            for d in source.documents
            if d.content and not (d.meta or {}).get("semantic_pass")
        ),
        key=lambda d: d.chunk_index or 0,
    )
    return "\n\n".join(chunk.content for chunk in chunks).strip()


async def delete_semantic_documents(session: AsyncSession, source_id) -> None:
    """Remove derived semantic documents; chunks/units/evidence cascade."""
    await session.execute(
        delete(Document).where(
            Document.source_id == source_id,
            Document.meta["semantic_pass"].astext == "true",
        )
    )
    await session.flush()


async def learn_source(
    session: AsyncSession,
    semantic: SemanticIngestService,
    source: Source,
    report: RelearnReport,
) -> None:
    content = reconstruct_text(source)
    if len(content) < MIN_CHARS:
        report.skipped_empty += 1
        return
    try:
        # Nested transaction: a failure in one source rolls back only its own
        # work, never poisoning the rest of the run. ``semantic_pass`` swallows
        # DB errors and returns ``stats['error']``, but the transaction is
        # already aborted — so we must raise inside the savepoint to roll it
        # back cleanly and keep the outer transaction usable.
        async with session.begin_nested():
            if not report.dry_run:
                await delete_semantic_documents(session, source.id)
            stats = await semantic.semantic_pass(
                session,
                source_id=source.id,
                content=content,
                title=source.name or "",
                source_type=source.source_type,
                source_tags=source.tags or [],
            )
            if stats.get("error"):
                raise RuntimeError(f"semantic pass failed: {stats['error']}")
        report.learned += 1
        report.chunk_total += stats.get("chunks") or 0
        report.unit_total += stats.get("units") or 0
        if report.dry_run:
            return
        if not stats.get("error"):
            # Record an immutable version for the re-learn (upsert-safe).
            try:
                oldest = (
                    await session.execute(
                        select(DocumentVersion.version)
                        .where(DocumentVersion.source_id == source.id)
                        .order_by(DocumentVersion.version.desc())
                        .limit(1)
                    )
                ).scalar_one_or_none()
                session.add(
                    DocumentVersion(
                        source_id=source.id,
                        version=int(oldest or 0) + 1,
                        content_hash=source.content_hash,
                        reason="update",
                        meta={
                            "relearned_at": datetime.datetime.now(
                                datetime.UTC
                            ).isoformat()
                        },
                    )
                )
            except Exception:  # pragma: no cover - defensive
                logger.warning("relearn.version_failed", source=str(source.id))
    except Exception as exc:  # keep going; report the failure
        report.failed += 1
        report.failures.append(f"{source.name or source.id}: {exc}")
        logger.warning("relearn.source_failed", source=str(source.id), error=str(exc))


async def main(apply: bool, source_filter: str | None = None) -> None:
    report = RelearnReport(dry_run=not apply)
    engine = create_engine()
    vllm = VLLMClient()
    try:
        await init_db(engine)
        if not await vllm.is_available():
            raise RuntimeError(
                "vLLM chat and embedding endpoints must be available to re-learn"
            )
        semantic = SemanticIngestService(vllm, 800)
        ingest = IngestService(session_factory=None, vllm=vllm, semantic=None)
        semantic.search_validator = SearchValidator(vllm, WebSearchTool(ingest))

        async with AsyncSession(engine, expire_on_commit=False) as session:
            source_query = select(Source).options(selectinload(Source.documents))
            if source_filter:
                source_query = source_query.where(Source.name == source_filter)
            sources = (
                (await session.execute(source_query)).scalars().unique().all()
            )
            for source in sources:
                await learn_source(session, semantic, source, report)
            await session.flush()
            if apply:
                await session.commit()
            else:
                await session.rollback()
    finally:
        await vllm.close()
        await engine.dispose()

    prefix = "DRY RUN" if not apply else "APPLIED"
    print(f"{prefix} relearn summary")
    print(f"  sources_to_relearn: {report.learned}")
    print(f"  skipped_empty: {report.skipped_empty}")
    print(f"  chunks: {report.chunk_total}")
    print(f"  units: {report.unit_total}")
    print(f"  failed: {report.failed}")
    for failure in report.failures[:20]:
        print(f"    - {failure}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Re-learn all knowledge with the current semantic pipeline."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Persist changes. Without this, run an audit-only dry run.",
    )
    parser.add_argument(
        "--source",
        metavar="NAME",
        default=None,
        help="Re-learn only the source whose name matches exactly (default: all).",
    )
    args = parser.parse_args()
    asyncio.run(main(apply=args.apply, source_filter=args.source))