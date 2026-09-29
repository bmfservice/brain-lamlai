"""add_chunks_content

Revision ID: c4d7e91a2b60
Revises: 9f2c1e4a7b3d
Create Date: 2026-08-20 10:05:00.000000

The ``chunks`` table stored only byte offsets into the owning ``documents``
row, plus a 120-character preview in ``meta.content``. When the owning
document was replaced (every vault re-sync), the chunk text became
unrecoverable. This revision adds a nullable ``chunks.content`` column and
backfills it from the offsets against the still-present document text.

Additive and reversible: existing rows keep working with a NULL content until
they are re-ingested; the backfill recovers everything still resolvable.
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c4d7e91a2b60"
down_revision: str | Sequence[str] | None = "9f2c1e4a7b3d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add chunks.content and backfill it from document offsets."""
    op.add_column("chunks", sa.Column("content", sa.Text(), nullable=True))
    # substring() is 1-indexed; start_offset is a 0-indexed character offset.
    op.execute(
        """
        UPDATE chunks c
        SET content = substring(d.content FROM c.start_offset + 1
                                FOR GREATEST(c.end_offset - c.start_offset, 0))
        FROM documents d
        WHERE c.document_id = d.id
          AND c.content IS NULL
          AND c.end_offset > c.start_offset
        """
    )


def downgrade() -> None:
    """Drop the content column (offsets and meta preview remain)."""
    op.drop_column("chunks", "content")
