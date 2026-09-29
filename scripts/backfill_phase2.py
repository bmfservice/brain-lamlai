"""Backfill script — Phase 2 schema hygiene.

Run once after restart (init_db already created the columns/indexes):

1. Populate ``Source.content_hash`` from the sha256 of the first 4k bytes.
2. Promote ``meta['keywords']`` / ``meta['tags']`` into the new ``tags`` JSONB column.
3. Re-run ``_auto_link`` for every source so the ``knowledge_links`` table is populated.
4. Detect parent-child hints from frontmatter and wikilinks, set ``parent_id`` and
   create ``parent_child`` edges.

All operations are idempotent (ON CONFLICT DO NOTHING) so it is safe to re-run.

This script MUTATES PostgreSQL. It requires an explicit ``--apply`` and refuses
to run without a recent backup unless ``--skip-backup-check`` is passed.

    PYTHONPATH=. .venv/bin/python scripts/backfill_phase2.py --apply
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import json
import re
import sys
from pathlib import Path

import frontmatter
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from core.database import Source, create_engine, init_db
from core.vllm import VLLMClient
from services.ingest.service import IngestService

# A dump older than this is treated as no backup at all.
_BACKUP_MAX_AGE_HOURS = 24


async def backfill_content_hash(session: AsyncSession) -> int:
    """Fill the content_hash column for all sources that lack it."""
    rows = (
        await session.execute(
            select(Source).where(Source.content_hash.is_(None))
        )
    ).scalars().all()
    count = 0
    for src in rows:
        seed = (src.path or src.name or str(src.id))[:4096]
        src.content_hash = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        count += 1
    await session.commit()
    return count


async def backfill_tags(session: AsyncSession) -> int:
    """Promote meta keywords/tags into the dedicated ``tags`` column."""
    rows = (
        await session.execute(
            select(Source).where(Source.tags.is_(None) | (Source.tags == []))
        )
    ).scalars().all()
    count = 0
    for src in rows:
        meta = src.meta or {}
        tags: list[str] = []
        for k in ("keywords", "tags", "auto_topic"):
            v = meta.get(k)
            if isinstance(v, list):
                tags.extend(str(x).strip().lower() for x in v if str(x).strip())
            elif isinstance(v, str) and v.strip():
                tags.append(v.strip().lower())
        if tags:
            src.tags = list(dict.fromkeys(tags))[:30]
            count += 1
    await session.commit()
    return count


async def backfill_links(session: AsyncSession, vllm: VLLMClient) -> int:
    """Recompute link edges using the real ingest helper."""
    ingest = IngestService(session_factory=None, vllm=vllm)
    rows = (
        await session.execute(
            select(Source).order_by(Source.created_at.desc()).limit(1000)
        )
    ).scalars().all()
    created = 0
    for src in rows:
        kws = IngestService._source_keywords(src)
        if not kws:
            continue
        links = await ingest._auto_link(
            session, kws, current_id=src.id, limit=12
        )
        created += len(links)
    await session.commit()
    return created


async def backfill_parents(session: AsyncSession) -> int:
    """Detect parent-child from frontmatter ``parent`` or wikilinks."""
    rows = (
        await session.execute(
            select(Source).where(Source.parent_id.is_(None))
        )
    ).scalars().all()
    created = 0
    for src in rows:
        meta = src.meta or {}
        declared = meta.get("parent") or meta.get("parent_note")
        if not declared and src.path:
            # Try to read the vault file again (cheap fallback).
            try:
                post = frontmatter.loads(
                    Path(src.path).read_text(encoding="utf-8", errors="replace")
                )
                declared = post.metadata.get("parent") or post.metadata.get("parent_note")
            except (OSError, ValueError):
                declared = None

        target_name = None
        if declared:
            target_name = str(declared).strip().rstrip(".md")
        elif src.path:
            try:
                content = Path(src.path).read_text(encoding="utf-8", errors="replace")
                for m in re.finditer(r"\[\[([^\]|#]+)", content):
                    target_name = m.group(1).strip()
                    break
            except OSError:
                pass

        if not target_name:
            continue

        parent = (
            await session.execute(
                select(Source).where(Source.name.ilike(target_name))
            )
        ).scalars().first()
        if parent is None or parent.id == src.id:
            continue

        src.parent_id = parent.id
        await session.execute(
            text(
                "INSERT INTO knowledge_links (source_id, target_id, relation_type, weight, meta) "
                "VALUES (:s, :t, 'parent_child', 1.0, :meta) "
                "ON CONFLICT DO NOTHING"
            ),
            {
                "s": str(parent.id),
                "t": str(src.id),
                "meta": json.dumps({"title": src.name[:120]}),
            },
        )
        created += 1
    await session.commit()
    return created


def _recent_backup() -> Path | None:
    """Return the newest DB dump written within the freshness window."""
    backups = Path(settings.data_dir) / "backups"
    if not backups.is_dir():
        return None
    cutoff = datetime.datetime.now().timestamp() - _BACKUP_MAX_AGE_HOURS * 3600
    dumps = [p for p in backups.glob("*.dump") if p.stat().st_mtime >= cutoff]
    return max(dumps, key=lambda p: p.stat().st_mtime) if dumps else None


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true", help="Actually write the backfill to PostgreSQL."
    )
    parser.add_argument(
        "--skip-backup-check",
        action="store_true",
        help=f"Proceed without a dump newer than {_BACKUP_MAX_AGE_HOURS}h.",
    )
    args = parser.parse_args()

    if not args.apply:
        print(
            "Refusing to run: this script mutates PostgreSQL.\n"
            "Take a backup (pg_dump -Fc) and re-run with --apply."
        )
        return 1

    if not args.skip_backup_check:
        backup = _recent_backup()
        if backup is None:
            print(
                f"Refusing to run: no database dump under {settings.data_dir}/backups "
                f"newer than {_BACKUP_MAX_AGE_HOURS}h.\n"
                "Take one, or pass --skip-backup-check if you accept the risk.",
                file=sys.stderr,
            )
            return 1
        print(f"Using backup: {backup.name}")

    engine = create_engine()
    await init_db(engine)

    async with AsyncSession(engine, expire_on_commit=False) as session:
        vllm = VLLMClient()
        h = await backfill_content_hash(session)
        t = await backfill_tags(session)
        links = await backfill_links(session, vllm)
        p = await backfill_parents(session)
        print(f"Backfill done — content_hash={h}, tags={t}, links={links}, parents={p}")
    await engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
