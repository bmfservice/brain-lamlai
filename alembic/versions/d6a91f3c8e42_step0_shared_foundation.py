"""Step 0 shared database foundation.

Revision ID: d6a91f3c8e42
Revises: 7c1e9d4a2f80
Create Date: 2026-09-11

Adds the empty-start dynamic taxonomy, capability registry, self-organization
journal, GPU resource samples, durable agent runs, and learned agent domains.
No taxonomy terms, facets, capabilities, or other runtime data are seeded.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

revision: str = "d6a91f3c8e42"
down_revision: str | Sequence[str] | None = "7c1e9d4a2f80"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "taxonomy_facets",
        sa.Column("facet", sa.String(120), primary_key=True),
        sa.Column("label", sa.String(255), nullable=False),
        sa.Column("growable", sa.Boolean(), nullable=False),
        sa.Column("authority_editable", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )

    op.create_table(
        "taxonomy_terms",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "facet",
            sa.String(120),
            sa.ForeignKey("taxonomy_facets.facet", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("term", sa.String(255), nullable=False),
        sa.Column(
            "parent_id",
            UUID(as_uuid=True),
            sa.ForeignKey("taxonomy_terms.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("origin", sa.String(40), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("status", sa.String(40), nullable=False),
        sa.Column(
            "created_from_chunk_id",
            UUID(as_uuid=True),
            sa.ForeignKey("chunks.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "merged_into_id",
            UUID(as_uuid=True),
            sa.ForeignKey("taxonomy_terms.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("authority", sa.Float(), nullable=True),
        sa.Column(
            "first_seen_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("seen_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
    )
    op.create_index(
        "uq_taxonomy_terms_facet_term",
        "taxonomy_terms",
        ["facet", "term"],
        unique=True,
    )
    op.create_index("idx_taxonomy_terms_status", "taxonomy_terms", ["status"])
    op.create_index("idx_taxonomy_terms_parent", "taxonomy_terms", ["parent_id"])

    op.create_table(
        "taxonomy_aliases",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "term_id",
            UUID(as_uuid=True),
            sa.ForeignKey("taxonomy_terms.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("surface_form", sa.String(255), nullable=False),
        sa.Column("normalized_form", sa.String(255), nullable=False),
        sa.Column("locale", sa.String(80), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index(
        "uq_taxonomy_aliases_normalized_locale",
        "taxonomy_aliases",
        ["normalized_form", sa.text("COALESCE(locale, '')")],
        unique=True,
    )
    op.create_index("idx_taxonomy_aliases_term", "taxonomy_aliases", ["term_id"])

    op.create_table(
        "taxonomy_rules",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "facet",
            sa.String(120),
            sa.ForeignKey("taxonomy_facets.facet", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("kind", sa.String(80), nullable=False),
        sa.Column("pattern", sa.Text(), nullable=False),
        sa.Column("locale", sa.String(80), nullable=True),
        sa.Column("origin", sa.String(40), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("hits", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index(
        "uq_taxonomy_rules_facet_kind_pattern_locale",
        "taxonomy_rules",
        ["facet", "kind", "pattern", sa.text("COALESCE(locale, '')")],
        unique=True,
    )
    op.create_index("idx_taxonomy_rules_enabled", "taxonomy_rules", ["enabled"])

    op.create_table(
        "capabilities",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("kind", sa.String(120), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column(
            "configuration",
            JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index(
        "uq_capabilities_kind_name",
        "capabilities",
        ["kind", "name"],
        unique=True,
    )
    op.create_index("idx_capabilities_enabled", "capabilities", ["enabled"])

    op.create_table(
        "reorg_operations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("operation_type", sa.String(80), nullable=False),
        sa.Column("status", sa.String(40), nullable=False),
        sa.Column("target_type", sa.String(80), nullable=False),
        sa.Column("target_id", sa.String(255), nullable=False),
        sa.Column("before_state", JSONB(), nullable=True),
        sa.Column("after_state", JSONB(), nullable=True),
        sa.Column("rationale", sa.Text(), nullable=True),
        sa.Column(
            "rollback_of_id",
            UUID(as_uuid=True),
            sa.ForeignKey("reorg_operations.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("idx_reorg_operations_status", "reorg_operations", ["status"])
    op.create_index(
        "idx_reorg_operations_target",
        "reorg_operations",
        ["target_type", "target_id"],
    )

    op.create_table(
        "gpu_resource_samples",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("gpu_index", sa.Integer(), nullable=False),
        sa.Column("total_vram_bytes", sa.BigInteger(), nullable=False),
        sa.Column("used_vram_bytes", sa.BigInteger(), nullable=False),
        sa.Column("free_vram_bytes", sa.BigInteger(), nullable=False),
        sa.Column("utilization_percent", sa.Float(), nullable=True),
        sa.Column("temperature_celsius", sa.Float(), nullable=True),
        sa.Column(
            "sampled_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index(
        "idx_gpu_resource_samples_gpu_time",
        "gpu_resource_samples",
        ["gpu_index", "sampled_at"],
    )

    op.create_table(
        "agent_runs",
        sa.Column("run_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("root_task_id", sa.String(36), nullable=False),
        sa.Column("state", sa.String(40), nullable=False),
        sa.Column(
            "checkpoint_json",
            JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "budget_json",
            JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index("uq_agent_runs_root_task", "agent_runs", ["root_task_id"], unique=True)
    op.create_index("idx_agent_runs_state", "agent_runs", ["state"])

    op.create_table(
        "agent_domains",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "agent_id",
            sa.String(50),
            sa.ForeignKey("agents.agent_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "facet",
            sa.String(120),
            sa.ForeignKey("taxonomy_facets.facet", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "term_id",
            UUID(as_uuid=True),
            sa.ForeignKey("taxonomy_terms.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("origin", sa.String(40), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index(
        "uq_agent_domains_agent_term",
        "agent_domains",
        ["agent_id", "term_id"],
        unique=True,
    )
    op.create_index("idx_agent_domains_facet", "agent_domains", ["facet"])


def downgrade() -> None:
    op.drop_index("idx_agent_domains_facet", table_name="agent_domains")
    op.drop_index("uq_agent_domains_agent_term", table_name="agent_domains")
    op.drop_table("agent_domains")

    op.drop_index("idx_agent_runs_state", table_name="agent_runs")
    op.drop_index("uq_agent_runs_root_task", table_name="agent_runs")
    op.drop_table("agent_runs")

    op.drop_index("idx_gpu_resource_samples_gpu_time", table_name="gpu_resource_samples")
    op.drop_table("gpu_resource_samples")

    op.drop_index("idx_reorg_operations_target", table_name="reorg_operations")
    op.drop_index("idx_reorg_operations_status", table_name="reorg_operations")
    op.drop_table("reorg_operations")

    op.drop_index("idx_capabilities_enabled", table_name="capabilities")
    op.drop_index("uq_capabilities_kind_name", table_name="capabilities")
    op.drop_table("capabilities")

    op.drop_index("idx_taxonomy_rules_enabled", table_name="taxonomy_rules")
    op.drop_index("uq_taxonomy_rules_facet_kind_pattern_locale", table_name="taxonomy_rules")
    op.drop_table("taxonomy_rules")

    op.drop_index("idx_taxonomy_aliases_term", table_name="taxonomy_aliases")
    op.drop_index("uq_taxonomy_aliases_normalized_locale", table_name="taxonomy_aliases")
    op.drop_table("taxonomy_aliases")

    op.drop_index("idx_taxonomy_terms_parent", table_name="taxonomy_terms")
    op.drop_index("idx_taxonomy_terms_status", table_name="taxonomy_terms")
    op.drop_index("uq_taxonomy_terms_facet_term", table_name="taxonomy_terms")
    op.drop_table("taxonomy_terms")

    op.drop_table("taxonomy_facets")
