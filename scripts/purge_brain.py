#!/usr/bin/env python3
"""Erase every trace of ingested knowledge, leaving an empty brain.

This is what "start the brain from scratch" means: sources, documents, chunks,
semantic units, entities, tags, graph edges, learning runs, retrieval feedback,
generated content and chat history are all deleted. What survives is the parts
that are configuration rather than knowledge — the UI registry, the scheduler's
task definitions, the audit log and the alembic version — plus the Obsidian
vault on disk, which is the user's own data and is re-synced on next start.

DESTRUCTIVE AND IRREVERSIBLE. Requires an explicit ``--apply`` and refuses to
run without a database dump newer than an hour, because a stale backup would
not contain what is about to be destroyed.

    PYTHONPATH=. .venv/bin/python scripts/purge_brain.py --dry-run
    PYTHONPATH=. .venv/bin/python scripts/purge_brain.py --apply
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import sys
from pathlib import Path

from sqlalchemy import text

from core.config import settings
from core.database import create_engine, create_session

# A dump older than this cannot contain the rows this script is about to
# delete, so it is not a usable rollback.
_BACKUP_MAX_AGE_MINUTES = 60

# Ordered parent-last so a partial run never orphans rows. Most of these
# cascade anyway; being explicit keeps the report honest about what went.
_KNOWLEDGE_TABLES: tuple[str, ...] = (
    "search_evidence",
    "semantic_units",
    "chunks",
    "document_versions",
    "knowledge_links",
    "retrieval_feedback",
    "feedback_events",
    "dirty_flags",
    "learn_urls",
    "learn_runs",
    "ingestion_jobs",
    "maintenance_runs",
    "entities",
    "tags",
    "documents",
    "sources",
    "generated_content",
    "news_items",
    "messages",
    "conversations",
)

# Configuration, not knowledge. Never touched.
_PRESERVED_TABLES: tuple[str, ...] = (
    "registry_entries",
    "scheduled_tasks",
    "scheduled_task_runs",
    "audit_logs",
    "embedding_versions",
    "model_records",
    "freshness_states",
    "alembic_version",
)


def _recent_backup() -> Path | None:
    """Return the newest dump written within the freshness window."""
    backups = Path(settings.data_dir) / "backups"
    if not backups.is_dir():
        return None
    cutoff = datetime.datetime.now().timestamp() - _BACKUP_MAX_AGE_MINUTES * 60
    dumps = [p for p in backups.glob("*.dump") if p.stat().st_mtime >= cutoff]
    return max(dumps, key=lambda p: p.stat().st_mtime) if dumps else None


async def _counts(session, tables: tuple[str, ...]) -> dict[str, int]:
    """Row count per table, skipping tables this database does not have."""
    counts: dict[str, int] = {}
    for table in tables:
        exists = (
            await session.execute(
                text("SELECT to_regclass(:name) IS NOT NULL"), {"name": table}
            )
        ).scalar_one()
        if not exists:
            continue
        counts[table] = (
            await session.execute(text(f"SELECT count(*) FROM {table}"))  # noqa: S608
        ).scalar_one()
    return counts


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="Report what would be deleted.")
    group.add_argument("--apply", action="store_true", help="Actually delete everything.")
    parser.add_argument(
        "--skip-backup-check",
        action="store_true",
        help=f"Proceed without a dump newer than {_BACKUP_MAX_AGE_MINUTES} minutes.",
    )
    args = parser.parse_args()

    if args.apply and not args.skip_backup_check:
        backup = _recent_backup()
        if backup is None:
            print(
                f"Refusing to run: no database dump under {settings.data_dir}/backups "
                f"newer than {_BACKUP_MAX_AGE_MINUTES} minutes.\n"
                "Take one first:\n"
                "  pg_dump -Fc braind -f data/backups/braind_pre_purge_$(date +%F_%H%M%S).dump",
                file=sys.stderr,
            )
            return 1
        print(f"Rollback point: {backup.name}")

    engine = create_engine()
    session_factory = create_session(engine)
    try:
        async with session_factory() as session:
            before = await _counts(session, _KNOWLEDGE_TABLES)
            preserved = await _counts(session, _PRESERVED_TABLES)

        total = sum(before.values())
        print(f"\n{'table':26s} {'rows':>10s}")
        print("-" * 38)
        for table, count in before.items():
            if count:
                print(f"{table:26s} {count:10d}")
        print("-" * 38)
        print(f"{'TOTAL TO DELETE':26s} {total:10d}\n")

        print("preserved (configuration, not knowledge):")
        for table, count in preserved.items():
            print(f"  {table:24s} {count:8d}")

        if args.dry_run:
            print("\nDry run — nothing was deleted. Re-run with --apply.")
            return 0

        if not total:
            print("\nNothing to delete; the brain is already empty.")
            return 0

        present = [t for t in _KNOWLEDGE_TABLES if t in before]
        async with session_factory() as session:
            # One statement, one transaction: a half-purged brain would be
            # worse than either state. RESTART IDENTITY resets sequences so the
            # new brain does not inherit the old one's counters.
            await session.execute(
                text(f"TRUNCATE {', '.join(present)} RESTART IDENTITY CASCADE")  # noqa: S608
            )
            await session.commit()

        async with session_factory() as session:
            after = await _counts(session, _KNOWLEDGE_TABLES)
        remaining = sum(after.values())
        print(f"\nDeleted {total} rows. Remaining in knowledge tables: {remaining}.")
        if remaining:
            print("WARNING: some rows survived:", {k: v for k, v in after.items() if v})
            return 1
        print(
            "\nThe brain is empty. Restart the API to re-sync the Obsidian vault:\n"
            "  systemctl --user restart braind.service"
        )
        return 0
    finally:
        await engine.dispose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
