"""add_approval_system_tables

Revision ID: 23aa8b8ec552
Revises: a8d4b421d508
Create Date: 2026-08-21 13:55:50.771226

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '23aa8b8ec552'
down_revision: str | Sequence[str] | None = 'a8d4b421d508'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create approval_requests table
    op.create_table(
        'approval_requests',
        sa.Column('request_id', sa.String(36), primary_key=True),
        sa.Column('task_id', sa.String(36), nullable=True),
        sa.Column('agent_id', sa.String(50), nullable=False),
        sa.Column('action_type', sa.String(50), nullable=False),
        sa.Column('action_data', sa.JSON, nullable=False),
        sa.Column('status', sa.String(20), server_default='pending'),
        sa.Column('priority', sa.Integer, server_default='2'),
        sa.Column('requested_at', sa.DateTime, server_default=sa.text('NOW()')),
        sa.Column('reviewed_at', sa.DateTime, nullable=True),
        sa.Column('reviewed_by', sa.String(100), nullable=True),
        sa.Column('review_notes', sa.Text, nullable=True),
        sa.Column('expires_at', sa.DateTime, nullable=True),
        sa.ForeignKeyConstraint(['task_id'], ['agent_tasks.task_id']),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.agent_id']),
    )
    
    # Create approval_audit table
    op.create_table(
        'approval_audit',
        sa.Column('audit_id', sa.String(36), primary_key=True),
        sa.Column('request_id', sa.String(36), nullable=False),
        sa.Column('action', sa.String(50), nullable=False),
        sa.Column('actor', sa.String(100), nullable=True),
        sa.Column('notes', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.text('NOW()')),
        sa.ForeignKeyConstraint(['request_id'], ['approval_requests.request_id']),
    )
    
    # Create indexes
    op.create_index('idx_approval_requests_status', 'approval_requests', ['status'])
    op.create_index('idx_approval_requests_agent', 'approval_requests', ['agent_id'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('idx_approval_requests_agent')
    op.drop_index('idx_approval_requests_status')
    op.drop_table('approval_audit')
    op.drop_table('approval_requests')
