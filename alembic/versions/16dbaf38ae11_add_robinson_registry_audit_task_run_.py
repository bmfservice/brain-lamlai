"""add_robinson_registry_audit_task_run_tables

Revision ID: 16dbaf38ae11
Revises:
Create Date: 2026-08-17 14:10:48.562869

"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '16dbaf38ae11'
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema - add Robinson registry, audit, and scheduled_task_runs tables."""
    # registry_entries
    op.create_table(
        'registry_entries',
        sa.Column('id', UUID(as_uuid=True), primary_key=True,
                server_default=sa.text('gen_random_uuid()')),
        sa.Column('kind', sa.String(50), nullable=False),
        sa.Column('key', sa.String(100), nullable=False),
        sa.Column('label', sa.String(255), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('icon', sa.String(100), nullable=True),
        sa.Column('href', sa.String(255), nullable=True),
        sa.Column('action', sa.String(100), nullable=True),
        sa.Column('group_name', sa.String(100), nullable=True),
        sa.Column('sort_order', sa.Integer, nullable=False, server_default='0'),
        sa.Column('enabled', sa.Boolean, nullable=False, server_default='true'),
        sa.Column('meta', JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column('created_at', TIMESTAMP(timezone=True), nullable=False,
                server_default=sa.text('now()')),
        sa.Column('updated_at', TIMESTAMP(timezone=True), nullable=False,
                server_default=sa.text('now()')),
    )
    op.create_index(
        'uq_registry_entries_kind_key',
        'registry_entries',
        ['kind', 'key'],
        unique=True,
    )
    op.create_index('idx_registry_entries_kind_enabled', 'registry_entries', ['kind', 'enabled'])
    op.create_index('idx_registry_entries_sort', 'registry_entries', ['kind', 'sort_order'])

    # audit_logs
    op.create_table(
        'audit_logs',
        sa.Column('id', UUID(as_uuid=True), primary_key=True,
                server_default=sa.text('gen_random_uuid()')),
        sa.Column('request_id', sa.String(64), nullable=True),
        sa.Column('actor', sa.String(255), nullable=True),
        sa.Column('service', sa.String(100), nullable=False, server_default='braind'),
        sa.Column('module', sa.String(100), nullable=True),
        sa.Column('action', sa.String(100), nullable=False),
        sa.Column('status', sa.String(50), nullable=False, server_default='ok'),
        sa.Column('duration_ms', sa.Integer, nullable=True),
        sa.Column('message', sa.Text, nullable=True),
        sa.Column('meta', JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column('created_at', TIMESTAMP(timezone=True), nullable=False,
                server_default=sa.text('now()')),
    )
    op.create_index('idx_audit_logs_created', 'audit_logs', ['created_at'])
    op.create_index('idx_audit_logs_action', 'audit_logs', ['action'])
    op.create_index('idx_audit_logs_request_id', 'audit_logs', ['request_id'])

    # scheduled_task_runs
    op.create_table(
        'scheduled_task_runs',
        sa.Column('id', UUID(as_uuid=True), primary_key=True,
                server_default=sa.text('gen_random_uuid()')),
        sa.Column('task_name', sa.String(255), nullable=False),
        sa.Column('task_type', sa.String(50), nullable=True),
        sa.Column('status', sa.String(50), nullable=False, server_default='running'),
        sa.Column('started_at', TIMESTAMP(timezone=True), nullable=False,
                server_default=sa.text('now()')),
        sa.Column('finished_at', TIMESTAMP(timezone=True), nullable=True),
        sa.Column('duration_ms', sa.Integer, nullable=True),
        sa.Column('message', sa.Text, nullable=True),
        sa.Column('meta', JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.create_index('idx_scheduled_task_runs_name', 'scheduled_task_runs', ['task_name'])
    op.create_index('idx_scheduled_task_runs_started', 'scheduled_task_runs', ['started_at'])


def downgrade() -> None:
    """Downgrade schema - drop Robinson tables."""
    op.drop_index('idx_scheduled_task_runs_started', table_name='scheduled_task_runs')
    op.drop_index('idx_scheduled_task_runs_name', table_name='scheduled_task_runs')
    op.drop_table('scheduled_task_runs')

    op.drop_index('idx_audit_logs_request_id', table_name='audit_logs')
    op.drop_index('idx_audit_logs_action', table_name='audit_logs')
    op.drop_index('idx_audit_logs_created', table_name='audit_logs')
    op.drop_table('audit_logs')

    op.drop_index('idx_registry_entries_sort', table_name='registry_entries')
    op.drop_index('idx_registry_entries_kind_enabled', table_name='registry_entries')
    op.drop_index('uq_registry_entries_kind_key', table_name='registry_entries')
    op.drop_table('registry_entries')