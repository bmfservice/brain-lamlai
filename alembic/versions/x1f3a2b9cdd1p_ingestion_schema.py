"""ingestion_schema

Revision ID: x1f3a2b9cdd1p
Revises: b2f9c4a7d310
Create Date: 2026-08-19 12:00:00.000000

Ingestion upgrade, Phase 1 (additive). Adds chunk provenance columns
(``root_id``, ``order``, ``page_number``, ``section_number``,
``source_location``, ``content_fingerprint``), the ``search_evidence`` and
``ingestion_jobs`` tables, and the seeded ``freshness_states`` taxonomy.

All additive; the downgrade drops only what this revision created. Mirrors the
idempotent DDL block in ``core/database.py init_db()``.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "x1f3a2b9cdd1p"
down_revision: str | Sequence[str] | None = "b2f9c4a7d310"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade - additive ingestion schema (provenance + observability + search evidence)."""
    # 1) Additive chunk provenance columns (all nullable).
    op.execute(
        "ALTER TABLE chunks "
        "ADD COLUMN IF NOT EXISTS root_id UUID, "
        'ADD COLUMN IF NOT EXISTS "order" INTEGER, '
        "ADD COLUMN IF NOT EXISTS page_number INTEGER, "
        "ADD COLUMN IF NOT EXISTS section_number INTEGER, "
        "ADD COLUMN IF NOT EXISTS source_location TEXT, "
        "ADD COLUMN IF NOT EXISTS content_fingerprint TEXT"
    )
    op.create_index(
        "idx_chunks_content_fingerprint", "chunks",
        ["document_id", "content_fingerprint"], if_not_exists=True,
    )

    # 2) search_evidence.
    op.create_table(
        "search_evidence",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                server_default=sa.text("gen_random_uuid()")),
        sa.Column("document_id", UUID(as_uuid=True), sa.ForeignKey(
            "documents.id", ondelete="CASCADE"), nullable=True),
        sa.Column("chunk_id", UUID(as_uuid=True), sa.ForeignKey(
            "chunks.id", ondelete="CASCADE"), nullable=True),
        sa.Column("unit_id", UUID(as_uuid=True), sa.ForeignKey(
            "semantic_units.id", ondelete="CASCADE"), nullable=True),
        sa.Column("search_query", sa.Text, nullable=False),
        sa.Column("web_source_url", sa.Text, nullable=True),
        sa.Column("web_snippet", sa.Text, nullable=True),
        sa.Column("web_fetched_text", sa.Text, nullable=True),
        sa.Column("verdict", sa.String(30), nullable=False, server_default="UNVERIFIED"),
        sa.Column("conflict_sentence", sa.Text, nullable=True),
        sa.Column("external_update_candidate", sa.Text, nullable=True),
        sa.Column("similarity_input_vs_web", sa.Float, nullable=True),
        sa.Column("model_version", sa.String(100), nullable=True),
        sa.Column("meta", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", TIMESTAMP(timezone=True), nullable=False,
                server_default=sa.text("now()")),
    )
    op.create_index("idx_search_evidence_node", "search_evidence", ["chunk_id"])
    op.create_index(
        "idx_search_evidence_chunk_query", "search_evidence",
        ["chunk_id", "search_query"], unique=True,
    )
    op.create_index("idx_search_evidence_verdict", "search_evidence", ["verdict"])

    # 3) ingestion_jobs.
    op.create_table(
        "ingestion_jobs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                server_default=sa.text("gen_random_uuid()")),
        sa.Column("ingestion_id", UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", UUID(as_uuid=True), sa.ForeignKey(
            "documents.id", ondelete="SET NULL"), nullable=True),
        sa.Column("stage", sa.String(60), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="running"),
        sa.Column("started_at", TIMESTAMP(timezone=True), nullable=False,
                server_default=sa.text("now()")),
        sa.Column("finished_at", TIMESTAMP(timezone=True), nullable=True),
        sa.Column("duration_ms", sa.Integer, nullable=True),
        sa.Column("tool", sa.String(100), nullable=True),
        sa.Column("model", sa.String(255), nullable=True),
        sa.Column("tokens", sa.Integer, nullable=True),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("meta", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.create_index("idx_ingestion_jobs_ingestion_id", "ingestion_jobs", ["ingestion_id"])
    op.create_index("idx_ingestion_jobs_document_id", "ingestion_jobs", ["document_id"])
    op.create_index("idx_ingestion_jobs_status", "ingestion_jobs", ["status"])

    # 4) freshness_states (seeded taxonomy).
    op.create_table(
        "freshness_states",
        sa.Column("name", sa.String(40), primary_key=True),
        sa.Column("table_name", sa.String(60), nullable=False,
                server_default="semantic_units"),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("llm_probe", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("created_at", TIMESTAMP(timezone=True), nullable=False,
                server_default=sa.text("now()")),
    )
    op.execute(
        "INSERT INTO freshness_states (name, llm_probe, description) "
        "VALUES "
        "  ('STATIC', FALSE, 'Evergreen reference material; never web-revalidated'), "
        "  ('LONG_LIVED', FALSE, 'Changes slowly (years); revalidate only on schedule'), "
        "  ('TIME_SENSITIVE', TRUE, 'Changes quickly; primary search-validation target'), "
        "  ('REALTIME', TRUE, 'Live/current; highest search-validation priority') "
        "ON CONFLICT (name) DO NOTHING"
    )


def downgrade() -> None:
    """Downgrade - drop only what this revision created."""
    op.drop_table("freshness_states")
    op.drop_index("idx_ingestion_jobs_status", table_name="ingestion_jobs")
    op.drop_index("idx_ingestion_jobs_document_id", table_name="ingestion_jobs")
    op.drop_index("idx_ingestion_jobs_ingestion_id", table_name="ingestion_jobs")
    op.drop_table("ingestion_jobs")
    op.drop_index("idx_search_evidence_verdict", table_name="search_evidence")
    op.drop_index("idx_search_evidence_chunk_query", table_name="search_evidence")
    op.drop_index("idx_search_evidence_node", table_name="search_evidence")
    op.drop_table("search_evidence")
    op.drop_index("idx_chunks_content_fingerprint", table_name="chunks")
    for column in (
        "content_fingerprint",
        "source_location",
        "section_number",
        "page_number",
        "order",
        "root_id",
    ):
        quoted = '"order"' if column == "order" else column
        op.execute(f"ALTER TABLE chunks DROP COLUMN IF EXISTS {quoted}")