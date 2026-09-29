"""deep_ingest_classification

Revision ID: e51c8a3f7b92
Revises: d38b6c05f1a4
Create Date: 2026-08-20 15:10:00.000000

Adds the classification columns the deep-ingest upgrade writes and that
retrieval filters on. Before this, every classification signal lived in JSONB
``meta``, which cannot be indexed per key and left ``auto_topic`` holding
stopwords like "your" with nothing able to validate it.

Columns use closed vocabularies from ``core/taxonomy.py``:

- ``sources``: doc_type, domain, audience, quality_tier, quality_score, language
- ``documents``: doc_type, domain, chunk_role, quality_tier, quality_score
- ``chunks``: chunk_role
- ``entities``: normalized_name (case/whitespace-folded, for merging duplicates)

All columns are nullable and additive; existing rows keep working unclassified
until they are re-ingested.
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e51c8a3f7b92"
down_revision: str | Sequence[str] | None = "d38b6c05f1a4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_SOURCE_COLUMNS = (
    ("doc_type", sa.String(32)),
    ("domain", sa.String(64)),
    ("audience", sa.String(32)),
    ("quality_tier", sa.String(16)),
    ("quality_score", sa.Float()),
    ("language", sa.String(8)),
)

_DOCUMENT_COLUMNS = (
    ("doc_type", sa.String(32)),
    ("domain", sa.String(64)),
    ("chunk_role", sa.String(32)),
    ("quality_tier", sa.String(16)),
    ("quality_score", sa.Float()),
)


def upgrade() -> None:
    """Add classification columns and their indexes."""
    for name, coltype in _SOURCE_COLUMNS:
        op.add_column("sources", sa.Column(name, coltype, nullable=True))
    for name, coltype in _DOCUMENT_COLUMNS:
        op.add_column("documents", sa.Column(name, coltype, nullable=True))
    op.add_column("chunks", sa.Column("chunk_role", sa.String(32), nullable=True))
    op.add_column("entities", sa.Column("normalized_name", sa.String(255), nullable=True))

    op.create_index("idx_sources_doc_type", "sources", ["doc_type"])
    op.create_index("idx_sources_domain", "sources", ["domain"])
    op.create_index("idx_sources_quality_tier", "sources", ["quality_tier"])
    op.create_index("idx_documents_doc_type", "documents", ["doc_type"])
    op.create_index("idx_documents_chunk_role", "documents", ["chunk_role"])
    op.create_index("idx_documents_quality_tier", "documents", ["quality_tier"])
    op.create_index("idx_chunks_role", "chunks", ["chunk_role"])
    op.create_index("idx_entities_normalized_name", "entities", ["normalized_name"])


def downgrade() -> None:
    """Drop the classification columns."""
    op.drop_index("idx_entities_normalized_name", table_name="entities")
    op.drop_index("idx_chunks_role", table_name="chunks")
    op.drop_index("idx_documents_quality_tier", table_name="documents")
    op.drop_index("idx_documents_chunk_role", table_name="documents")
    op.drop_index("idx_documents_doc_type", table_name="documents")
    op.drop_index("idx_sources_quality_tier", table_name="sources")
    op.drop_index("idx_sources_domain", table_name="sources")
    op.drop_index("idx_sources_doc_type", table_name="sources")

    op.drop_column("entities", "normalized_name")
    op.drop_column("chunks", "chunk_role")
    for name, _ in _DOCUMENT_COLUMNS:
        op.drop_column("documents", name)
    for name, _ in _SOURCE_COLUMNS:
        op.drop_column("sources", name)
