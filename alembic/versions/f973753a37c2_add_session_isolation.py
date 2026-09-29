"""add_session_isolation

Revision ID: f973753a37c2
Revises: f7c3a1e9d240
Create Date: 2026-09-10 13:46:50.346017

Giai đoạn 4 — anonymous multi-user session scoping (no auth layer).

Every knowledge row gains a nullable ``session_id``: NULL means the row is
system/global (background jobs, RSS, vault sync, seed data) and is visible to
every session; a set value scopes the row to exactly one visitor session.
Retrieval filters on ``(session_id IS NULL OR session_id = :current)`` — the
disjunction, not an equality, so global knowledge stays shared.

Also adds ``quarantined_knowledge``: the nightly maintenance window moves
contradicted/duplicate-flagged rows here instead of deleting them, so nothing
a human wrote is ever destroyed without review.

Additive only — existing rows keep their values (NULL → global).
"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'f973753a37c2'
down_revision: str | Sequence[str] | None = 'f7c3a1e9d240'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add session scoping columns + indexes and the quarantine table."""
    # --- session scoping -------------------------------------------------
    op.add_column(
        "sources", sa.Column("session_id", UUID(as_uuid=True), nullable=True)
    )
    op.add_column(
        "sources",
        sa.Column("fingerprint_hash", sa.String(64), nullable=True),
    )
    op.add_column(
        "documents", sa.Column("session_id", UUID(as_uuid=True), nullable=True)
    )
    op.add_column(
        "conversations", sa.Column("session_id", UUID(as_uuid=True), nullable=True)
    )
    op.add_column(
        "conversations",
        sa.Column("fingerprint_hash", sa.String(64), nullable=True),
    )
    op.add_column(
        "messages", sa.Column("session_id", UUID(as_uuid=True), nullable=True)
    )

    op.create_index(
        "idx_sources_session_id", "sources", ["session_id"]
    )
    op.create_index(
        "idx_documents_session_id", "documents", ["session_id"]
    )
    op.create_index(
        "idx_conversations_session_id", "conversations", ["session_id"]
    )
    op.create_index(
        "idx_messages_session_id", "messages", ["session_id"]
    )

    # --- quarantine (never destroy without review) -----------------------
    op.create_table(
        "quarantined_knowledge",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id",
            UUID(as_uuid=True),
            nullable=True,
        ),
        sa.Column("table_name", sa.String(60), nullable=False),
        sa.Column("row_id", sa.String(64), nullable=False),
        sa.Column("reason", sa.String(40), nullable=False),
        sa.Column("detail", sa.Text()),
        sa.Column("payload", JSONB, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index(
        "idx_quarantined_session", "quarantined_knowledge", ["session_id"]
    )
    op.create_index(
        "idx_quarantined_reason", "quarantined_knowledge", ["reason"]
    )


def downgrade() -> None:
    """Reverse in inverse dependency order."""
    op.drop_index("idx_quarantined_reason", table_name="quarantined_knowledge")
    op.drop_index("idx_quarantined_session", table_name="quarantined_knowledge")
    op.drop_table("quarantined_knowledge")

    op.drop_index("idx_messages_session_id", table_name="messages")
    op.drop_index("idx_conversations_session_id", table_name="conversations")
    op.drop_index("idx_documents_session_id", table_name="documents")
    op.drop_index("idx_sources_session_id", table_name="sources")

    op.drop_column("messages", "session_id")
    op.drop_column("conversations", "fingerprint_hash")
    op.drop_column("conversations", "session_id")
    op.drop_column("documents", "session_id")
    op.drop_column("sources", "fingerprint_hash")
    op.drop_column("sources", "session_id")
