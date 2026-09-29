"""add_multi_agent_tables

Revision ID: a8d4b421d508
Revises: e51c8a3f7b92
Create Date: 2026-08-21 13:17:12.290049

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'a8d4b421d508'
down_revision: str | Sequence[str] | None = 'e51c8a3f7b92'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create agents table
    op.create_table(
        'agents',
        sa.Column('agent_id', sa.String(50), primary_key=True),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('role', sa.String(20), nullable=False),  # CEO, LEADER, SPECIALIST
        sa.Column('department', sa.String(50), nullable=True),
        sa.Column('specialization', sa.String(50), nullable=True),
        sa.Column('avatar', sa.String(10), nullable=True),
        sa.Column('color', sa.String(7), nullable=True),
        sa.Column('system_prompt', sa.Text, nullable=True),
        sa.Column('temperature', sa.Float, server_default='0.7'),
        sa.Column('can_delegate', sa.Boolean, server_default='false'),
        sa.Column('can_use_tools', sa.Boolean, server_default='false'),
        sa.Column('available_tools', sa.JSON, nullable=True),
        sa.Column('position', sa.JSON, nullable=True),  # {"x": 100, "y": 200}
        sa.Column('status', sa.String(20), server_default='active'),  # active, busy, offline
        sa.Column('created_at', sa.DateTime, server_default=sa.text('NOW()')),
        sa.Column(
            'updated_at', sa.DateTime, server_default=sa.text('NOW()'), onupdate=sa.text('NOW()')
        ),
    )
    
    # Create agent_tasks table
    op.create_table(
        'agent_tasks',
        sa.Column('task_id', sa.String(36), primary_key=True),  # UUID as string
        sa.Column('parent_task_id', sa.String(36), nullable=True),
        sa.Column('user_request', sa.Text, nullable=True),
        sa.Column('task_description', sa.Text, nullable=False),
        sa.Column('assigned_to', sa.String(50), nullable=False),
        sa.Column('assigned_by', sa.String(50), nullable=True),
        # pending, in_progress, completed, failed
        sa.Column('status', sa.String(20), server_default='pending'),
        sa.Column('priority', sa.Integer, server_default='1'),
        sa.Column('result', sa.JSON, nullable=True),
        sa.Column('error', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.text('NOW()')),
        sa.Column('started_at', sa.DateTime, nullable=True),
        sa.Column('completed_at', sa.DateTime, nullable=True),
        sa.ForeignKeyConstraint(['assigned_to'], ['agents.agent_id']),
        sa.ForeignKeyConstraint(['assigned_by'], ['agents.agent_id']),
        sa.ForeignKeyConstraint(['parent_task_id'], ['agent_tasks.task_id']),
    )
    
    # Create agent_conversations table
    op.create_table(
        'agent_conversations',
        sa.Column('conversation_id', sa.String(36), primary_key=True),  # UUID as string
        sa.Column('task_id', sa.String(36), nullable=True),
        sa.Column('from_agent', sa.String(50), nullable=False),
        sa.Column('to_agent', sa.String(50), nullable=False),
        # delegate, report, question, answer
        sa.Column('message_type', sa.String(20), nullable=False),
        sa.Column('content', sa.Text, nullable=False),
        sa.Column('created_at', sa.DateTime, server_default=sa.text('NOW()')),
        sa.ForeignKeyConstraint(['task_id'], ['agent_tasks.task_id']),
        sa.ForeignKeyConstraint(['from_agent'], ['agents.agent_id']),
        sa.ForeignKeyConstraint(['to_agent'], ['agents.agent_id']),
    )
    
    # Create indexes for better query performance
    op.create_index('idx_agent_tasks_status', 'agent_tasks', ['status'])
    op.create_index('idx_agent_tasks_assigned_to', 'agent_tasks', ['assigned_to'])
    op.create_index('idx_agent_conversations_task', 'agent_conversations', ['task_id'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('idx_agent_conversations_task')
    op.drop_index('idx_agent_tasks_assigned_to')
    op.drop_index('idx_agent_tasks_status')
    op.drop_table('agent_conversations')
    op.drop_table('agent_tasks')
    op.drop_table('agents')
