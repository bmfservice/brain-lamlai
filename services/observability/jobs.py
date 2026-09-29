"""Ingestion job observability + quality report (Phase 7).

``IngestionJobTracker`` records one ``ingestion_jobs`` row per pipeline stage
with duration/status/tool/model and an optional error string. Failures are
recorded, never fatal — a failed stage row can be re-marked and re-run through
the retry endpoint, and the underlying source document is never deleted.
"""

from __future__ import annotations

import datetime
import time
import uuid
from contextlib import asynccontextmanager

import sqlalchemy as sa

from core import database as db
from core.logging import get_logger

logger = get_logger(__name__)


class IngestionJobTracker:
    """Append-only, non-fatal stage tracing for one ingestion run."""

    def __init__(self, session_factory):
        self.session_factory = session_factory

    async def new_run(self) -> int:
        """Begin a run; returns a unique ``ingestion_id``."""
        return uuid.uuid4()

    async def finish_run(self, ingestion_id, stage, *, status="ok", error=None,
                         source_id=None, tool=None, model=None) -> None:
        """Record a stage outcome for a run without a surrounding context manager.

        Pairs with ``new_run`` for call sites where wrapping the whole block is
        impractical; anything recorded is non-fatal (never raises).
        """
        await self._record(
            ingestion_id, stage, status=status, error=error,
            source_id=source_id, tool=tool, model=model,
        )

    @asynccontextmanager
    async def stage(self, ingestion_id, stage, *, source_id=None, tool=None, model=None):
        """Record one stage within an ingestion run.

        Times the block and persists a running→ok/failed row. A stage failure is
        captured (non-fatal) and re-raised so the caller's own error handling
        still runs — nothing is swallowed silently.
        """
        start = time.monotonic()
        status = "ok"
        error = None
        try:
            yield
        except Exception as exc:  # capture, then re-raise
            status = "failed"
            error = str(exc)[:2000]
            raise
        finally:
            await self._record(
                ingestion_id,
                stage,
                status=status,
                error=error,
                source_id=source_id,
                tool=tool,
                model=model,
                duration_ms=int((time.monotonic() - start) * 1000),
            )

    async def _record(self, ingestion_id, stage, *, status, error=None,
                      source_id=None, tool=None, model=None, duration_ms=0) -> None:
        try:
            async with self.session_factory() as session:
                session.add(
                    db.IngestionJob(
                        ingestion_id=ingestion_id,
                        source_id=source_id,
                        stage=stage,
                        status=status,
                        finished_at=datetime.datetime.now(datetime.UTC),
                        duration_ms=duration_ms,
                        tool=tool,
                        model=model,
                        error=error,
                    )
                )
                await session.commit()
        except Exception:  # pragma: no cover - tracing must never break ingest
            logger.warning("ingest.job_trace_failed", stage=stage, exc_info=True)


async def _utility_queries(session) -> dict:
    """Cheap quality counts over the semantic-layer + ingest tables."""

    async def scalar(q):
        result = await session.execute(q)
        return result.scalar_one_or_none() or 0

    return {
        "document_count": await scalar(sa.select(sa.func.count(db.Document.id))),
        "source_count": await scalar(sa.select(sa.func.count(db.Source.id))),
        "chunk_count": await scalar(sa.select(sa.func.count(db.Chunk.id))),
        "unit_count": await scalar(sa.select(sa.func.count(db.SemanticUnit.id))),
        "entity_count": await scalar(sa.select(sa.func.count(db.Entity.id))),
        "concept_count": await scalar(
            sa.select(sa.func.count(db.Entity.id)).where(
                db.Entity.entity_type == "concept"
            )
        ),
        "link_count": await scalar(sa.select(sa.func.count(db.KnowledgeLink.id))),
        # §5 — how much entity resolution is still waiting on a human. Without
        # this the report could not distinguish a clean corpus from one where
        # every ingest queued another ambiguous name.
        "needs_review_count": await scalar(
            sa.select(sa.func.count(db.Entity.id)).where(
                db.Entity.resolution_state.in_(("review", "ambiguous"))
            )
        ),
        # Same-name-different-domain collisions the resolver refused to merge.
        # Distinct from the count above: those are entities never confirmed,
        # these are specific pairs a human has to rule on.
        "resolution_conflict_count": await scalar(
            sa.select(sa.func.count(db.KnowledgeLink.id)).where(
                db.KnowledgeLink.lifecycle == "review",
                db.KnowledgeLink.meta.has_key("entity_resolution_review"),  # noqa: W601
            )
        ),
        "conflicted_count": await scalar(
            sa.select(sa.func.count(db.SearchEvidence.id)).where(
                db.SearchEvidence.verdict == "CONFLICTED"
            )
        ),
        # Near-duplicate skips are not persisted as a separate column (dedup is
        # the unique content_hash + skip path), so duplicate_count is omitted.
        "failed_stage_count": await scalar(
            sa.select(sa.func.count(db.IngestionJob.id)).where(
                db.IngestionJob.status == "failed"
            )
        ),
    }


async def build_quality_report(session_factory) -> dict:
    """Assemble the /api/ingest/report payload (quality + freshness breakdown)."""
    async with session_factory() as session:
        report = await _utility_queries(session)

        # Group by the select-list position: SQLAlchemy emits a separate bind
        # parameter for each occurrence of the JSONB key, so repeating the
        # expression in GROUP BY leaves PostgreSQL unable to match the two
        # (GroupingError).
        freshness: dict[str, int] = {}
        for name, count in (
            await session.execute(
                sa.select(db.SemanticUnit.meta["freshness"].astext, sa.func.count())
                .group_by(sa.literal_column("1"))
            )
        ).all():
            if name:
                freshness[str(name)] = int(count)
        report["freshness_breakdown"] = freshness

        # Recent stage trace (latest 25).
        rows = (
            await session.execute(
                sa.select(db.IngestionJob)
                .order_by(db.IngestionJob.started_at.desc())
                .limit(25)
            )
        ).scalars().all()
        report["recent_jobs"] = [
            {
                "ingestion_id": str(j.ingestion_id),
                "stage": j.stage,
                "status": j.status,
                "duration_ms": j.duration_ms,
                "error": j.error,
            }
            for j in rows
            if j is not None
        ]
        return report


async def retry_failed_stage(session_factory, job_id: uuid.UUID) -> dict:
    """Mark a failed stage row as re-queued (idempotent retry bookmark).

    The underlying work is never destroyed — this only flags the row so an
    operator/consumer can re-run that stage against the already-committed
    source. Returns the updated row summary or raises a typed error if missing.
    """
    async with session_factory() as session:
        row = (
            await session.execute(
                sa.select(db.IngestionJob).where(db.IngestionJob.id == job_id)
            )
        ).scalar_one_or_none()
        if row is None:
            raise LookupError(f"ingestion job {job_id} not found")
        row.status = "queued"
        await session.commit()
        return {
            "job_id": str(row.id),
            "ingestion_id": str(row.ingestion_id),
            "stage": row.stage,
            "status": row.status,
        }