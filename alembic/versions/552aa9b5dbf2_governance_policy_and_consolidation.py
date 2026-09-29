"""governance_policy_and_consolidation

Revision ID: 552aa9b5dbf2
Revises: f973753a37c2
Create Date: 2026-09-10 18:26:13.689709

Giai đoạn 5 — governance policy engine + consolidated-learning dataset.

Three additive pieces:

1. ``policy_rules`` — declarative allow/deny/ask_first rules (DB-backed, so
   they are editable via REST without a restart). Evaluated by the governance
   engine before every mutating action; deny beats allow at equal priority.
2. ``relation_decisions`` — the training ledger. Every relation decision
   (LLM verdict, consolidated-classifier verdict, or human override) is
   recorded with its feature vector so the consolidated classifier can be
   retrained periodically. Human overrides carry ``human_override=true`` and
   get priority weighting in training.
3. ``knowledge_links.use_count`` / ``last_used_at`` — reuse tracking so the
   consolidation sweep can apply time-decay to edges nobody reads and a reuse
   bonus to edges retrieval keeps hitting.

Additive only: all new columns are nullable/defaulted, existing rows untouched.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '552aa9b5dbf2'
down_revision: str | Sequence[str] | None = 'f973753a37c2'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create policy/decision tables and add edge reuse-tracking columns."""
    op.create_table(
        "policy_rules",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("scope", sa.String(60), nullable=False),  # file_import|response|tool|category
        sa.Column("rule_type", sa.String(12), nullable=False),  # allow|deny|ask_first
        sa.Column("pattern", sa.String(255), nullable=False),  # mime/ext/action pattern
        sa.Column("reason", sa.Text()),
        sa.Column("priority", sa.Integer, nullable=False, server_default="100"),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.Column("enabled", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime, server_default=sa.text("NOW()")),
    )
    op.create_index("idx_policy_rules_scope", "policy_rules", ["scope"])
    op.create_index("idx_policy_rules_enabled", "policy_rules", ["enabled"])

    op.create_table(
        "relation_decisions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("source_id", UUID(as_uuid=True), nullable=False),
        sa.Column("target_id", UUID(as_uuid=True), nullable=False),
        sa.Column("features", JSONB, nullable=False),  # engineered feature vector
        sa.Column("label", sa.String(50), nullable=False),  # relation_type decided
        sa.Column("decided_by", sa.String(60), nullable=False),  # llm|consolidated|human
        sa.Column("confidence", sa.Float, nullable=False, server_default="0"),
        sa.Column("rationale", sa.Text()),
        sa.Column("human_override", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("trained_at", sa.DateTime(timezone=True)),  # NULL = not yet trained on
        sa.Column("created_at", sa.DateTime, server_default=sa.text("NOW()")),
    )
    op.create_index("idx_relation_decisions_trained", "relation_decisions", ["trained_at"])
    op.create_index("idx_relation_decisions_override", "relation_decisions", ["human_override"])

    op.add_column(
        "knowledge_links",
        sa.Column("use_count", sa.Integer, nullable=False, server_default="0"),
    )
    op.add_column(
        "knowledge_links",
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("idx_knowledge_links_last_used", "knowledge_links", ["last_used_at"])


def downgrade() -> None:
    """Reverse in inverse dependency order."""
    op.drop_index("idx_knowledge_links_last_used", table_name="knowledge_links")
    op.drop_column("knowledge_links", "last_used_at")
    op.drop_column("knowledge_links", "use_count")

    op.drop_index("idx_relation_decisions_override", table_name="relation_decisions")
    op.drop_index("idx_relation_decisions_trained", table_name="relation_decisions")
    op.drop_table("relation_decisions")

    op.drop_index("idx_policy_rules_enabled", table_name="policy_rules")
    op.drop_index("idx_policy_rules_scope", table_name="policy_rules")
    op.drop_table("policy_rules")
