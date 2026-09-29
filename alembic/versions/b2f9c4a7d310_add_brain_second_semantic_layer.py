"""add_brain_second_semantic_layer

Revision ID: b2f9c4a7d310
Revises: 16dbaf38ae11
Create Date: 2026-08-17 15:30:00.000000

Phase 3 of the Brain Second upgrade: semantic chunking, entities, tags,
document/embedding versioning, maintenance state, model registry, feedback,
and the typed-edge extension of ``knowledge_links``. All additive; the
downgrade drops only what this revision created.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b2f9c4a7d310"
down_revision: str | Sequence[str] | None = "16dbaf38ae11"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _typed_edges_upgrade() -> None:
    """Add the typed-edge columns to ``knowledge_links`` (additive)."""
    op.execute(
        "ALTER TABLE knowledge_links "
        "ADD COLUMN IF NOT EXISTS source_kind TEXT NOT NULL DEFAULT 'source', "
        "ADD COLUMN IF NOT EXISTS target_kind TEXT NOT NULL DEFAULT 'source', "
        "ADD COLUMN IF NOT EXISTS confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0, "
        "ADD COLUMN IF NOT EXISTS evidence JSONB NOT NULL DEFAULT '[]'::jsonb, "
        "ADD COLUMN IF NOT EXISTS evidence_count INTEGER NOT NULL DEFAULT 0, "
        "ADD COLUMN IF NOT EXISTS reason TEXT, "
        "ADD COLUMN IF NOT EXISTS model_version TEXT, "
        "ADD COLUMN IF NOT EXISTS lifecycle TEXT NOT NULL DEFAULT 'active', "
        "ADD COLUMN IF NOT EXISTS valid_from TIMESTAMPTZ, "
        "ADD COLUMN IF NOT EXISTS valid_to TIMESTAMPTZ"
    )
    op.create_index(
        "idx_knowledge_links_confidence",
        "knowledge_links",
        ["confidence"],
        if_not_exists=True,
    )


def _typed_edges_downgrade() -> None:
    """Remove the typed-edge columns from ``knowledge_links``."""
    op.drop_index("idx_knowledge_links_confidence", table_name="knowledge_links")
    for column in (
        "valid_to",
        "valid_from",
        "lifecycle",
        "model_version",
        "reason",
        "evidence_count",
        "evidence",
        "confidence",
        "target_kind",
        "source_kind",
    ):
        op.execute(f"ALTER TABLE knowledge_links DROP COLUMN IF EXISTS {column}")


def upgrade() -> None:
    """Upgrade schema - add the Brain Second semantic knowledge layer."""
    _typed_edges_upgrade()

    # chunks
    op.create_table(
        "chunks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                server_default=sa.text("gen_random_uuid()")),
        sa.Column("document_id", UUID(as_uuid=True), sa.ForeignKey(
            "documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("parent_id", UUID(as_uuid=True), sa.ForeignKey(
            "chunks.id", ondelete="SET NULL"), nullable=True),
        sa.Column("section_path", sa.Text, nullable=True),
        sa.Column("hierarchy_level", sa.Integer, nullable=False, server_default="0"),
        sa.Column("start_offset", sa.Integer, nullable=False, server_default="0"),
        sa.Column("end_offset", sa.Integer, nullable=False, server_default="0"),
        sa.Column("prev_chunk_id", UUID(as_uuid=True), sa.ForeignKey(
            "chunks.id", ondelete="SET NULL"), nullable=True),
        sa.Column("next_chunk_id", UUID(as_uuid=True), sa.ForeignKey(
            "chunks.id", ondelete="SET NULL"), nullable=True),
        sa.Column("context_summary", sa.Text, nullable=True),
        sa.Column("embedding_vllm_1024", Vector(1024), nullable=True),
        sa.Column("embedding_version", sa.String(100), nullable=True),
        sa.Column("meta", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", TIMESTAMP(timezone=True), nullable=False,
                server_default=sa.text("now()")),
        sa.Column("updated_at", TIMESTAMP(timezone=True), nullable=False,
                server_default=sa.text("now()")),
    )
    op.create_index("idx_chunks_document_id", "chunks", ["document_id"])
    op.create_index("idx_chunks_parent_id", "chunks", ["parent_id"])
    op.create_index("idx_chunks_section_path", "chunks", ["section_path"])

    # semantic_units
    op.create_table(
        "semantic_units",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                server_default=sa.text("gen_random_uuid()")),
        sa.Column("chunk_id", UUID(as_uuid=True), sa.ForeignKey(
            "chunks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("unit_type", sa.String(50), nullable=False, server_default="fact"),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("polarity", sa.Integer, nullable=False, server_default="1"),
        sa.Column("valid_from", TIMESTAMP(timezone=True), nullable=True),
        sa.Column("valid_to", TIMESTAMP(timezone=True), nullable=True),
        sa.Column("entity_ids", JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("confidence", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("span_start", sa.Integer, nullable=False, server_default="0"),
        sa.Column("span_end", sa.Integer, nullable=False, server_default="0"),
        sa.Column("embedding_vllm_1024", Vector(1024), nullable=True),
        sa.Column("fact_class", sa.String(40), nullable=False, server_default="SOURCE_FACT"),
        sa.Column("meta", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", TIMESTAMP(timezone=True), nullable=False,
                server_default=sa.text("now()")),
        sa.CheckConstraint("polarity IN (-1, 0, 1)", name="ck_units_polarity"),
    )
    op.create_index("idx_units_chunk_id", "semantic_units", ["chunk_id"])
    op.create_index("idx_units_valid_to", "semantic_units", ["valid_to"])
    op.create_index("idx_units_fact_class", "semantic_units", ["fact_class"])

    # entities
    op.create_table(
        "entities",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                server_default=sa.text("gen_random_uuid()")),
        sa.Column("canonical_name", sa.String(255), nullable=False),
        sa.Column("aliases", JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("entity_type", sa.String(50), nullable=False, server_default="concept"),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("embedding_vllm_1024", Vector(1024), nullable=True),
        sa.Column("resolution_state", sa.String(20), nullable=False, server_default="ambiguous"),
        sa.Column("merge_policy", sa.String(20), nullable=False, server_default="review"),
        sa.Column("confidence", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("meta", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", TIMESTAMP(timezone=True), nullable=False,
                server_default=sa.text("now()")),
        sa.Column("updated_at", TIMESTAMP(timezone=True), nullable=False,
                server_default=sa.text("now()")),
    )
    op.create_index("uq_entities_canonical_name", "entities", ["canonical_name"], unique=True)
    op.create_index("idx_entities_type", "entities", ["entity_type"])
    op.create_index("idx_entities_resolution_state", "entities", ["resolution_state"])

    # tags
    op.create_table(
        "tags",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("parent_id", UUID(as_uuid=True), sa.ForeignKey(
            "tags.id", ondelete="SET NULL"), nullable=True),
        sa.Column("synonyms", JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("domain", sa.String(100), nullable=True),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.Column("created_at", TIMESTAMP(timezone=True), nullable=False,
                server_default=sa.text("now()")),
        sa.Column("updated_at", TIMESTAMP(timezone=True), nullable=False,
                server_default=sa.text("now()")),
    )
    op.create_index("uq_tags_name", "tags", ["name", "parent_id"], unique=True)
    op.create_index("idx_tags_parent_id", "tags", ["parent_id"])

    # document_versions
    op.create_table(
        "document_versions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                server_default=sa.text("gen_random_uuid()")),
        sa.Column("source_id", UUID(as_uuid=True), sa.ForeignKey(
            "sources.id", ondelete="CASCADE"), nullable=False),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=True),
        sa.Column("diff_summary", sa.Text, nullable=True),
        sa.Column("reason", sa.String(50), nullable=True),
        sa.Column("meta", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", TIMESTAMP(timezone=True), nullable=False,
                server_default=sa.text("now()")),
    )
    op.create_index(
        "uq_document_versions_source_version", "document_versions",
        ["source_id", "version"], unique=True,
    )
    op.create_index("idx_document_versions_source_id", "document_versions", ["source_id"])

    # embedding_versions
    op.create_table(
        "embedding_versions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                server_default=sa.text("gen_random_uuid()")),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("dimensions", sa.Integer, nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
        sa.Column("config", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", TIMESTAMP(timezone=True), nullable=False,
                server_default=sa.text("now()")),
    )
    op.create_index("idx_embedding_versions_status", "embedding_versions", ["status"])
    op.create_index(
        "uq_embedding_versions_model_dim",
        "embedding_versions",
        ["model_name", "dimensions"],
        unique=True,
    )

    # maintenance_runs
    op.create_table(
        "maintenance_runs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                server_default=sa.text("gen_random_uuid()")),
        sa.Column("scope", sa.String(50), nullable=False, server_default="all"),
        sa.Column("status", sa.String(20), nullable=False, server_default="running"),
        sa.Column("started_at", TIMESTAMP(timezone=True), nullable=False,
                server_default=sa.text("now()")),
        sa.Column("finished_at", TIMESTAMP(timezone=True), nullable=True),
        sa.Column("summary", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("meta", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.create_index("idx_maintenance_runs_status", "maintenance_runs", ["status"])
    op.create_index("idx_maintenance_runs_started", "maintenance_runs", ["started_at"])

    # dirty_flags
    op.create_table(
        "dirty_flags",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                server_default=sa.text("gen_random_uuid()")),
        sa.Column("entity_kind", sa.String(50), nullable=False),
        sa.Column("entity_id", UUID(as_uuid=True), nullable=False),
        sa.Column("dirty_type", sa.String(50), nullable=False),
        sa.Column("reason", sa.Text, nullable=True),
        sa.Column("processed_at", TIMESTAMP(timezone=True), nullable=True),
        sa.Column("created_at", TIMESTAMP(timezone=True), nullable=False,
                server_default=sa.text("now()")),
    )
    op.create_index("idx_dirty_flags_kind_id", "dirty_flags", ["entity_kind", "entity_id"])
    op.create_index(
        "idx_dirty_flags_unprocessed",
        "dirty_flags",
        ["processed_at"],
        postgresql_where=sa.text("processed_at IS NULL"),
    )

    # model_records
    op.create_table(
        "model_records",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                server_default=sa.text("gen_random_uuid()")),
        sa.Column("role", sa.String(50), nullable=False),
        sa.Column("model", sa.String(255), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
        sa.Column("version", sa.String(100), nullable=True),
        sa.Column("params", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", TIMESTAMP(timezone=True), nullable=False,
                server_default=sa.text("now()")),
    )
    op.create_index("idx_model_records_role_status", "model_records", ["role", "status"])
    op.create_index("uq_model_records_role_model", "model_records", ["role", "model"], unique=True)

    # feedback_events
    op.create_table(
        "feedback_events",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                server_default=sa.text("gen_random_uuid()")),
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("query", sa.Text, nullable=False),
        sa.Column("retrieved_ids", JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("chosen_id", UUID(as_uuid=True), nullable=True),
        sa.Column("label", sa.Integer, nullable=True),
        sa.Column("source", sa.String(50), nullable=True),
        sa.Column("user_id", UUID(as_uuid=True), nullable=True),
        sa.Column("embedding_version", sa.String(100), nullable=True),
        sa.Column("created_at", TIMESTAMP(timezone=True), nullable=False,
                server_default=sa.text("now()")),
    )
    op.create_index("idx_feedback_events_created", "feedback_events", ["created_at"])
    op.create_index("idx_feedback_events_query", "feedback_events", ["query"])

    # Semantic-layer vector indexes (HNSW, same strategy as documents).
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_embedding_vllm_1024 "
        "ON chunks USING hnsw (embedding_vllm_1024 vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 200)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_units_embedding_vllm_1024 "
        "ON semantic_units USING hnsw (embedding_vllm_1024 vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 200)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_entities_embedding_vllm_1024 "
        "ON entities USING hnsw (embedding_vllm_1024 vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 200)"
    )


def downgrade() -> None:
    """Downgrade schema - drop the Brain Second semantic knowledge layer."""
    _typed_edges_downgrade()

    op.drop_index("idx_feedback_events_query", table_name="feedback_events")
    op.drop_index("idx_feedback_events_created", table_name="feedback_events")
    op.drop_table("feedback_events")

    op.drop_index("uq_model_records_role_model", table_name="model_records")
    op.drop_index("idx_model_records_role_status", table_name="model_records")
    op.drop_table("model_records")

    op.drop_index("idx_dirty_flags_unprocessed", table_name="dirty_flags")
    op.drop_index("idx_dirty_flags_kind_id", table_name="dirty_flags")
    op.drop_table("dirty_flags")

    op.drop_index("idx_maintenance_runs_started", table_name="maintenance_runs")
    op.drop_index("idx_maintenance_runs_status", table_name="maintenance_runs")
    op.drop_table("maintenance_runs")

    op.drop_index("uq_embedding_versions_model_dim", table_name="embedding_versions")
    op.drop_index("idx_embedding_versions_status", table_name="embedding_versions")
    op.drop_table("embedding_versions")

    op.drop_index("idx_document_versions_source_id", table_name="document_versions")
    op.drop_index(
        "uq_document_versions_source_version", table_name="document_versions"
    )
    op.drop_table("document_versions")

    op.drop_index("idx_tags_parent_id", table_name="tags")
    op.drop_index("uq_tags_name", table_name="tags")
    op.drop_table("tags")

    op.drop_index("idx_entities_resolution_state", table_name="entities")
    op.drop_index("idx_entities_type", table_name="entities")
    op.drop_index("uq_entities_canonical_name", table_name="entities")
    op.drop_table("entities")

    op.drop_index("idx_units_fact_class", table_name="semantic_units")
    op.drop_index("idx_units_valid_to", table_name="semantic_units")
    op.drop_index("idx_units_chunk_id", table_name="semantic_units")
    op.drop_table("semantic_units")

    op.drop_index("idx_chunks_section_path", table_name="chunks")
    op.drop_index("idx_chunks_parent_id", table_name="chunks")
    op.drop_index("idx_chunks_document_id", table_name="chunks")
    op.drop_table("chunks")