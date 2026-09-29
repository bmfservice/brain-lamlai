"""add_hierarchical_agent_office

Revision ID: f7c3a1e9d240
Revises: 23aa8b8ec552
Create Date: 2026-08-22 12:30:00.000000

Extends the multi-agent tables into a real hierarchical workflow: agent
level/parent/capabilities, structured task fields + explicit lifecycle state
and task_type, plus reports, audit events, and provenance-tagged memory.
Additive only — existing columns and rows are preserved.
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "f7c3a1e9d240"
down_revision: str | Sequence[str] | None = "23aa8b8ec552"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # --- agents: hierarchy metadata ---
    op.add_column("agents", sa.Column("level", sa.String(20), nullable=True))
    op.add_column("agents", sa.Column("parent_agent", sa.String(50), nullable=True))
    op.add_column("agents", sa.Column("capabilities", sa.JSON, nullable=True))
    op.add_column("agents", sa.Column("current_task", sa.String(36), nullable=True))

    # --- agent_tasks: structured task contract + explicit lifecycle ---
    op.add_column("agent_tasks", sa.Column("task_type", sa.String(20), server_default="subtask"))
    op.add_column("agent_tasks", sa.Column("state", sa.String(30), server_default="created"))
    op.add_column("agent_tasks", sa.Column("department", sa.String(50), nullable=True))
    op.add_column("agent_tasks", sa.Column("root_task_id", sa.String(36), nullable=True))
    op.add_column("agent_tasks", sa.Column("objective", sa.Text, nullable=True))
    op.add_column("agent_tasks", sa.Column("instructions", sa.Text, nullable=True))
    op.add_column("agent_tasks", sa.Column("expected_output", sa.Text, nullable=True))
    op.add_column("agent_tasks", sa.Column("required_output", sa.Text, nullable=True))
    op.add_column("agent_tasks", sa.Column("acceptance_criteria", sa.JSON, nullable=True))
    op.add_column("agent_tasks", sa.Column("input_data", sa.JSON, nullable=True))
    op.add_column("agent_tasks", sa.Column("constraints", sa.JSON, nullable=True))
    op.add_column("agent_tasks", sa.Column("dependencies", sa.JSON, nullable=True))
    op.add_column(
        "agent_tasks",
        sa.Column("confirmation_required", sa.Boolean, server_default="false"),
    )
    op.add_column("agent_tasks", sa.Column("rework_count", sa.Integer, server_default="0"))
    op.create_index("idx_agent_tasks_root", "agent_tasks", ["root_task_id"])
    op.create_index("idx_agent_tasks_state", "agent_tasks", ["state"])
    op.create_index("idx_agent_tasks_type", "agent_tasks", ["task_type"])

    # --- agent_reports: specialist + department reports ---
    op.create_table(
        "agent_reports",
        sa.Column("report_id", sa.String(36), primary_key=True),
        sa.Column("task_id", sa.String(36), nullable=False),
        sa.Column("root_task_id", sa.String(36), nullable=True),
        sa.Column("department_task_id", sa.String(36), nullable=True),
        sa.Column("agent_id", sa.String(50), nullable=False),
        sa.Column("report_type", sa.String(20), nullable=False),  # specialist | department
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("confidence", sa.Float, server_default="0.5"),
        sa.Column("payload", sa.JSON, nullable=False),
        sa.Column("created_at", sa.DateTime, server_default=sa.text("NOW()")),
    )
    op.create_index("idx_agent_reports_task", "agent_reports", ["task_id"])
    op.create_index("idx_agent_reports_root", "agent_reports", ["root_task_id"])

    # --- agent_audit_events: workflow audit trail ---
    op.create_table(
        "agent_audit_events",
        sa.Column("event_id", sa.String(36), primary_key=True),
        sa.Column("action", sa.String(50), nullable=False),
        sa.Column("actor", sa.String(50), nullable=True),  # agent_id or 'user'
        sa.Column("task_id", sa.String(36), nullable=True),
        sa.Column("parent_task_id", sa.String(36), nullable=True),
        sa.Column("root_task_id", sa.String(36), nullable=True),
        sa.Column("previous_status", sa.String(30), nullable=True),
        sa.Column("new_status", sa.String(30), nullable=True),
        sa.Column("reason", sa.Text, nullable=True),
        sa.Column("event_metadata", sa.JSON, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.text("NOW()")),
    )
    op.create_index("idx_agent_audit_root", "agent_audit_events", ["root_task_id"])
    op.create_index("idx_agent_audit_task", "agent_audit_events", ["task_id"])
    op.create_index("idx_agent_audit_action", "agent_audit_events", ["action"])

    # --- agent_memory: provenance-tagged learning ---
    op.create_table(
        "agent_memory",
        sa.Column("memory_id", sa.String(36), primary_key=True),
        sa.Column("root_task_id", sa.String(36), nullable=True),
        sa.Column("task_id", sa.String(36), nullable=True),
        sa.Column("department", sa.String(50), nullable=True),
        sa.Column("agent_id", sa.String(50), nullable=True),
        sa.Column("knowledge_type", sa.String(30), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("source", sa.String(100), nullable=True),
        sa.Column("confidence", sa.Float, server_default="0.5"),
        sa.Column("evidence", sa.JSON, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.text("NOW()")),
    )
    op.create_index("idx_agent_memory_root", "agent_memory", ["root_task_id"])
    op.create_index("idx_agent_memory_type", "agent_memory", ["knowledge_type"])


def downgrade() -> None:
    op.drop_table("agent_memory")
    op.drop_table("agent_audit_events")
    op.drop_table("agent_reports")
    op.drop_index("idx_agent_tasks_type", table_name="agent_tasks")
    op.drop_index("idx_agent_tasks_state", table_name="agent_tasks")
    op.drop_index("idx_agent_tasks_root", table_name="agent_tasks")
    for col in (
        "rework_count", "confirmation_required", "dependencies", "constraints",
        "input_data", "acceptance_criteria", "required_output", "expected_output",
        "instructions", "objective", "root_task_id", "department", "state", "task_type",
    ):
        op.drop_column("agent_tasks", col)
    for col in ("current_task", "capabilities", "parent_agent", "level"):
        op.drop_column("agents", col)
