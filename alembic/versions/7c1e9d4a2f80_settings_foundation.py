"""settings_foundation

Revision ID: 7c1e9d4a2f80
Revises: 552aa9b5dbf2
Create Date: 2026-09-11 00:00:00.000000

Brain Second Phase 1a — the settings foundation.

Four additive tables that make settings a DB-backed, typed, versioned source of
truth for BILL's behavior instead of literals in code:

1. ``setting_definitions`` — the typed schema of every setting key (data type,
   bounds, allowed values, risk level, restart/sensitivity flags, source and
   rationale). Seeded from ``services/settings/catalog.py``.
2. ``setting_values`` — the current value of a key at one priority scope
   (system_hard > security > project > task > user_preference > temporary >
   learned). ``scope_ref`` narrows a row to a session/task/preference id and is
   NULL for global rows.
3. ``setting_history`` — an append-only version log: every write records the
   real previous value, who changed it and why, so a rollback creates a new
   version and never erases the trail.
4. ``settings_snapshots`` — an immutable frozen view of resolved settings for a
   task/session. Created here so the schema lands in one migration; the resolver
   that populates it is Phase 1b.

The unique index on ``setting_values`` is an *expression* index over
``COALESCE(scope_ref, '')``: Postgres treats NULLs as distinct in a plain
UNIQUE(key, scope, scope_ref) constraint, so without the COALESCE two global
rows (scope_ref IS NULL) for the same key/scope would both be allowed
(Phase 1a risk R12). ``op.create_index`` cannot express a COALESCE target, so
the index is created with raw SQL that matches the ORM declaration in
``core/database.py::SettingValue`` byte for byte.

Additive only: no existing table is touched, so no real data can be affected.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '7c1e9d4a2f80'
down_revision: str | Sequence[str] | None = '552aa9b5dbf2'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the four settings tables and the COALESCE unique index."""
    op.create_table(
        "setting_definitions",
        sa.Column("key", sa.String(120), primary_key=True),
        sa.Column("category", sa.String(40), nullable=False),
        # A key may surface under more than one UI sidebar without duplicating
        # the row (Phase 0 decision Q6).
        sa.Column(
            "also_in_categories", JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")
        ),
        # bool|int|float|str|enum|json|string_list
        sa.Column("data_type", sa.String(16), nullable=False),
        # JSONB carries every data_type safely in one column.
        sa.Column("default_value", JSONB, nullable=False),
        sa.Column("min_value", sa.Float, nullable=True),
        sa.Column("max_value", sa.Float, nullable=True),
        sa.Column("allowed_values", JSONB, nullable=True),
        sa.Column("description", sa.String(255), nullable=False),
        sa.Column("detailed_description", sa.Text, nullable=True),
        # low|medium|high|critical
        sa.Column("risk_level", sa.String(12), nullable=False, server_default=sa.text("'low'")),
        sa.Column("requires_restart", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("is_sensitive", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("is_critical", sa.Boolean, nullable=False, server_default=sa.text("false")),
        # e.g. {"pattern": "...", "gt": 0} — enforced by schema._apply_rules.
        sa.Column("validation_rules", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        # appendix_a|derived_phase3|derived_phase4|derived_phase5|derived_phase7
        sa.Column("source", sa.String(60), nullable=False),
        sa.Column("rationale", sa.Text, nullable=True),
        sa.Column("sort_order", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")
        ),
    )
    op.create_index("idx_setting_definitions_category", "setting_definitions", ["category"])
    op.create_index("idx_setting_definitions_critical", "setting_definitions", ["is_critical"])

    op.create_table(
        "setting_values",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        # Logical FK to setting_definitions.key — deliberately not a physical
        # ForeignKey so seed/rollback order is never constrained by it.
        sa.Column("key", sa.String(120), nullable=False),
        # system_hard|security|project|task|user_preference|temporary|learned
        sa.Column("scope", sa.String(24), nullable=False),
        # NULL = global row; otherwise a session/task/preference id.
        sa.Column("scope_ref", sa.String(120), nullable=True),
        sa.Column("value", JSONB, nullable=False),
        # seed|api|system|learned|migration
        sa.Column("updated_by", sa.String(60), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")
        ),
    )
    # Expression index: COALESCE so two global rows (scope_ref IS NULL) collide
    # as intended. op.create_index cannot express a COALESCE target (R12).
    op.execute(
        "CREATE UNIQUE INDEX uq_setting_values_key_scope_ref "
        "ON setting_values (key, scope, COALESCE(scope_ref, ''))"
    )
    op.create_index("idx_setting_values_key", "setting_values", ["key"])
    op.create_index("idx_setting_values_scope", "setting_values", ["scope"])

    op.create_table(
        "setting_history",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("key", sa.String(120), nullable=False),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("scope", sa.String(24), nullable=False),
        sa.Column("scope_ref", sa.String(120), nullable=True),
        # NULL previous_value = the first time this key/scope was created.
        sa.Column("previous_value", JSONB, nullable=True),
        sa.Column("new_value", JSONB, nullable=False),
        sa.Column(
            "changed_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")
        ),
        sa.Column("changed_by", sa.String(60), nullable=False),
        sa.Column("reason", sa.Text, nullable=True),
        # Reserved for the Phase 1b snapshot link; not populated in 1a.
        sa.Column("snapshot_id", UUID(as_uuid=True), nullable=True),
    )
    op.create_index(
        "idx_setting_history_key_version", "setting_history", ["key", "version"]
    )
    op.create_index("idx_setting_history_changed_at", "setting_history", ["changed_at"])

    # Created in 1a so the schema lands once; populated by the Phase 1b resolver.
    op.create_table(
        "settings_snapshots",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("task_id", UUID(as_uuid=True), nullable=True),
        sa.Column("session_id", UUID(as_uuid=True), nullable=True),
        # The frozen, resolved settings a task keeps using even if settings
        # change underneath it mid-run.
        sa.Column("values", JSONB, nullable=False),
        # key -> the scope that won, rendered by the Phase 6 Effective View.
        sa.Column("resolved_from", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")
        ),
    )
    op.create_index("idx_settings_snapshots_task", "settings_snapshots", ["task_id"])
    op.create_index("idx_settings_snapshots_session", "settings_snapshots", ["session_id"])


def downgrade() -> None:
    """Reverse in inverse dependency order (snapshots -> history -> values -> definitions)."""
    op.drop_index("idx_settings_snapshots_session", table_name="settings_snapshots")
    op.drop_index("idx_settings_snapshots_task", table_name="settings_snapshots")
    op.drop_table("settings_snapshots")

    op.drop_index("idx_setting_history_changed_at", table_name="setting_history")
    op.drop_index("idx_setting_history_key_version", table_name="setting_history")
    op.drop_table("setting_history")

    op.drop_index("idx_setting_values_scope", table_name="setting_values")
    op.drop_index("idx_setting_values_key", table_name="setting_values")
    op.execute("DROP INDEX IF EXISTS uq_setting_values_key_scope_ref")
    op.drop_table("setting_values")

    op.drop_index("idx_setting_definitions_critical", table_name="setting_definitions")
    op.drop_index("idx_setting_definitions_category", table_name="setting_definitions")
    op.drop_table("setting_definitions")
