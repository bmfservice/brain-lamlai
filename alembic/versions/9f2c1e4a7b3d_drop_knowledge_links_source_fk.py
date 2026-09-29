"""drop_knowledge_links_source_fk

Revision ID: 9f2c1e4a7b3d
Revises: x1f3a2b9cdd1p
Create Date: 2026-08-19 13:00:00.000000

Polymorphic graph edges. ``knowledge_links.source_id``/``target_id`` are the
node ids named by ``source_kind``/``target_kind`` (source|chunk|unit|entity|tag).
The legacy FK to ``sources.id`` only permits source→source edges, so the
semantic layer's entity/chunk/unit edges violate it on live PostgreSQL (mocked
tests never enforce FKs, which is why this only surfaced during the re-learn).

This revision drops the two FKs so entity/chunk/unit/tag edges can be stored.
Referential integrity is enforced by the writer. The downgrade re-adds the FKs
(only valid if no non-source rows exist — a data check is performed first).

Backup was taken before this change (``braind_pre_relearn_20260819_110431.dump``).
"""
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9f2c1e4a7b3d"
down_revision: str | Sequence[str] | None = "x1f3a2b9cdd1p"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Drop the source-only FKs so polymorphic edges can be stored."""
    op.execute(
        "ALTER TABLE knowledge_links "
        "DROP CONSTRAINT IF EXISTS knowledge_links_source_id_fkey"
    )
    op.execute(
        "ALTER TABLE knowledge_links "
        "DROP CONSTRAINT IF EXISTS knowledge_links_target_id_fkey"
    )


def downgrade() -> None:
    """Re-add the FKs, but only if no non-source rows exist (data check)."""
    op.execute(
        "DO $$ "
        "BEGIN "
        "  IF EXISTS ("
        "    SELECT 1 FROM knowledge_links "
        "    WHERE source_kind <> 'source' OR target_kind <> 'source'"
        "  ) THEN "
        "    RAISE EXCEPTION "
        "      'cannot re-add FK: knowledge_links has non-source edges'; "
        "  END IF; "
        "END $$"
    )
    op.execute(
        "ALTER TABLE knowledge_links "
        "ADD CONSTRAINT knowledge_links_source_id_fkey "
        "FOREIGN KEY (source_id) REFERENCES sources(id) ON DELETE CASCADE"
    )
    op.execute(
        "ALTER TABLE knowledge_links "
        "ADD CONSTRAINT knowledge_links_target_id_fkey "
        "FOREIGN KEY (target_id) REFERENCES sources(id) ON DELETE CASCADE"
    )
