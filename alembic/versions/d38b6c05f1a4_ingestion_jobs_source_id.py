"""rename_ingestion_jobs_document_id_to_source_id

Revision ID: d38b6c05f1a4
Revises: c4d7e91a2b60
Create Date: 2026-08-20 10:55:00.000000

``ingestion_jobs.document_id`` had a FK to ``documents.id``, but every caller
passed a ``sources.id`` — an ingest stage belongs to a source, not to one of its
chunk rows. On live PostgreSQL that raised a ForeignKeyViolationError on every
vault-sync stage record (the tracker swallowed it, so the column stayed empty).

This renames the column to ``source_id`` and repoints the FK at ``sources``.
Existing values are discarded because they were never persisted.
"""
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d38b6c05f1a4"
down_revision: str | Sequence[str] | None = "c4d7e91a2b60"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Repoint the stage trace at sources."""
    op.execute(
        "ALTER TABLE ingestion_jobs "
        "DROP CONSTRAINT IF EXISTS ingestion_jobs_document_id_fkey"
    )
    op.execute("DROP INDEX IF EXISTS idx_ingestion_jobs_document_id")
    op.execute("ALTER TABLE ingestion_jobs RENAME COLUMN document_id TO source_id")
    op.execute("UPDATE ingestion_jobs SET source_id = NULL")
    op.execute(
        "ALTER TABLE ingestion_jobs "
        "ADD CONSTRAINT ingestion_jobs_source_id_fkey "
        "FOREIGN KEY (source_id) REFERENCES sources(id) ON DELETE SET NULL"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_source_id "
        "ON ingestion_jobs(source_id)"
    )


def downgrade() -> None:
    """Restore the documents-scoped column."""
    op.execute(
        "ALTER TABLE ingestion_jobs "
        "DROP CONSTRAINT IF EXISTS ingestion_jobs_source_id_fkey"
    )
    op.execute("DROP INDEX IF EXISTS idx_ingestion_jobs_source_id")
    op.execute("ALTER TABLE ingestion_jobs RENAME COLUMN source_id TO document_id")
    op.execute("UPDATE ingestion_jobs SET document_id = NULL")
    op.execute(
        "ALTER TABLE ingestion_jobs "
        "ADD CONSTRAINT ingestion_jobs_document_id_fkey "
        "FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE SET NULL"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_document_id "
        "ON ingestion_jobs(document_id)"
    )
