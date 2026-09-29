"""Tests for 7-level Priority Hierarchy resolver and snapshots (Brain Second Phase 1b).

Verifies:
- All 7 tiers of resolution:
  SYSTEM_HARD > SECURITY > PROJECT > TASK > USER_PREFERENCE > TEMPORARY > LEARNED > default
- Scenario #1: Learned Behavior cannot override System Hard Rule.
- Scenario #2: Task Override beats User Preference.
- Scope reference targeting (task_id, user_id, session_id).
- Snapshot creation, immutability, and retrieval.
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio

from services.settings.resolver import SettingsResolver
from services.settings.schema import SettingScope
from services.settings.snapshot import SettingsSnapshotService

pytest_plugins = ["tests.integration.conftest"]


@pytest_asyncio.fixture
async def resolver(session_factory, store):
    return SettingsResolver(session_factory, store=store)


@pytest_asyncio.fixture
async def snapshot_service(session_factory, resolver):
    return SettingsSnapshotService(session_factory, resolver)


# ---------------------------------------------------------------------------
# Resolution Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fallback_to_default_when_no_values(resolver, seeded):
    """When no values are set, resolver falls back to catalog default."""
    val, scope = await resolver.resolve("allow_assumption")
    assert val is False
    # allow_assumption is seeded at PROJECT scope by seed_settings
    assert scope == SettingScope.PROJECT


@pytest.mark.asyncio
async def test_scenario_1_learned_cannot_override_system_hard(resolver, store, seeded):
    """Scenario #1: Learned Behavior CANNOT override System Hard Rule.

    SYSTEM_HARD (rank 0) strictly dominates LEARNED (rank 6).
    """
    key = "reasoning_depth"

    # Set SYSTEM_HARD scope value
    await store.set_value(
        key,
        "deep",
        scope=SettingScope.SYSTEM_HARD,
        changed_by="admin",
        reason="system hard guard",
    )

    # Set LEARNED scope value attempt to override
    await store.set_value(
        key,
        "low",
        scope=SettingScope.LEARNED,
        changed_by="learned_engine",
        reason="learned habit",
    )

    val, winning_scope = await resolver.resolve(key)
    assert val == "deep"
    assert winning_scope == SettingScope.SYSTEM_HARD


@pytest.mark.asyncio
async def test_scenario_2_task_override_beats_user_preference(resolver, store, seeded):
    """Scenario #2: Task Override beats User Preference.

    TASK (rank 3) strictly beats USER_PREFERENCE (rank 4).
    """
    key = "reasoning_depth"
    task_id = "task-alpha-123"
    user_id = "user-bob"

    # Remove the seeded PROJECT row so TASK and USER_PREFERENCE compete directly
    await store.delete_value(key, scope=SettingScope.PROJECT, changed_by="test")

    # User prefers 'low'
    await store.set_value(
        key,
        "low",
        scope=SettingScope.USER_PREFERENCE,
        scope_ref=user_id,
        changed_by="bob",
        reason="user pref",
    )

    # Task specifically overrides to 'deep'
    await store.set_value(
        key,
        "deep",
        scope=SettingScope.TASK,
        scope_ref=task_id,
        changed_by="orchestrator",
        reason="complex task requires deep reasoning",
    )

    # When resolving with both task_id and user_id, TASK wins
    val, winning_scope = await resolver.resolve(key, task_id=task_id, user_id=user_id)
    assert val == "deep"
    assert winning_scope == SettingScope.TASK

    # When resolving with user_id only (no task_id), USER_PREFERENCE wins
    val_user_only, user_scope = await resolver.resolve(key, user_id=user_id)
    assert val_user_only == "low"
    assert user_scope == SettingScope.USER_PREFERENCE


@pytest.mark.asyncio
async def test_all_7_tiers_resolution_precedence(resolver, store, seeded):
    """Verify precedence order step by step through all 7 tiers.

    SYSTEM_HARD (0) > SECURITY (1) > PROJECT (2) > TASK (3) > USER_PREFERENCE (4) >
    TEMPORARY (5) > LEARNED (6) > default.
    """
    key = "tool_timeout_seconds"  # int, default 30, bounds 1..600
    task_id = "task-tier-test"
    user_id = "user-tier-test"
    session_id = "session-tier-test"

    # Remove seeded PROJECT row so we can start from the bottom
    await store.delete_value(key, scope=SettingScope.PROJECT, changed_by="test")

    # 0. Fallback to default (30)
    val, scope = await resolver.resolve(key)
    assert val == 30
    assert scope == "default"

    # 1. Tier 6: LEARNED = 35 beats default
    await store.set_value(key, 35, scope=SettingScope.LEARNED, changed_by="learner")
    val, scope = await resolver.resolve(key)
    assert val == 35
    assert scope == SettingScope.LEARNED

    # 2. Tier 5: TEMPORARY = 40 beats LEARNED
    await store.set_value(
        key, 40, scope=SettingScope.TEMPORARY, scope_ref=session_id, changed_by="session"
    )
    val, scope = await resolver.resolve(key, session_id=session_id)
    assert val == 40
    assert scope == SettingScope.TEMPORARY

    # 3. Tier 4: USER_PREFERENCE = 45 beats TEMPORARY and LEARNED
    await store.set_value(
        key, 45, scope=SettingScope.USER_PREFERENCE, scope_ref=user_id, changed_by="user"
    )
    val, scope = await resolver.resolve(
        key, session_id=session_id, user_id=user_id
    )
    assert val == 45
    assert scope == SettingScope.USER_PREFERENCE

    # 4. Tier 3: TASK = 50 beats USER_PREFERENCE, TEMPORARY, and LEARNED
    await store.set_value(
        key, 50, scope=SettingScope.TASK, scope_ref=task_id, changed_by="task"
    )
    val, scope = await resolver.resolve(
        key, task_id=task_id, user_id=user_id, session_id=session_id
    )
    assert val == 50
    assert scope == SettingScope.TASK

    # 5. Tier 2: PROJECT = 55 beats TASK and lower
    await store.set_value(
        key, 55, scope=SettingScope.PROJECT, changed_by="project_admin"
    )
    val, scope = await resolver.resolve(
        key, task_id=task_id, user_id=user_id, session_id=session_id
    )
    assert val == 55
    assert scope == SettingScope.PROJECT

    # 6. Tier 1: SECURITY = 60 beats PROJECT and lower
    await store.set_value(key, 60, scope=SettingScope.SECURITY, changed_by="sec_admin")
    val, scope = await resolver.resolve(
        key, task_id=task_id, user_id=user_id, session_id=session_id
    )
    assert val == 60
    assert scope == SettingScope.SECURITY

    # 7. Tier 0: SYSTEM_HARD = 65 beats SECURITY and all lower tiers
    await store.set_value(key, 65, scope=SettingScope.SYSTEM_HARD, changed_by="system")
    val, scope = await resolver.resolve(
        key, task_id=task_id, user_id=user_id, session_id=session_id
    )
    assert val == 65
    assert scope == SettingScope.SYSTEM_HARD


@pytest.mark.asyncio
async def test_sub_tiers_without_project_override(resolver, store, seeded):
    """Verify relative precedence among lower tiers:
    TASK > USER_PREFERENCE > TEMPORARY > LEARNED.
    """
    key = "when_missing_information"
    # enum: ask | assume | abort | escalate | retry | warn_user | stop
    task_id = "task-sub-tier"
    user_id = "user-sub-tier"
    session_id = "session-sub-tier"

    # Delete the PROJECT row to test lower tiers
    await store.delete_value(key, scope=SettingScope.PROJECT, changed_by="test")

    # Tier 6: LEARNED = 'warn_user'
    await store.set_value(key, "warn_user", scope=SettingScope.LEARNED, changed_by="learner")
    val, scope = await resolver.resolve(key)
    assert val == "warn_user"
    assert scope == SettingScope.LEARNED

    # Tier 5: TEMPORARY = 'stop' beats LEARNED
    await store.set_value(
        key, "stop", scope=SettingScope.TEMPORARY, scope_ref=session_id, changed_by="session"
    )
    val, scope = await resolver.resolve(key, session_id=session_id)
    assert val == "stop"
    assert scope == SettingScope.TEMPORARY

    # Tier 4: USER_PREFERENCE = 'use_best_effort' beats TEMPORARY and LEARNED
    await store.set_value(
        key,
        "use_best_effort",
        scope=SettingScope.USER_PREFERENCE,
        scope_ref=user_id,
        changed_by="user",
    )
    val, scope = await resolver.resolve(key, session_id=session_id, user_id=user_id)
    assert val == "use_best_effort"
    assert scope == SettingScope.USER_PREFERENCE

    # Tier 3: TASK = 'ask' beats USER_PREFERENCE, TEMPORARY, and LEARNED
    await store.set_value(
        key, "ask", scope=SettingScope.TASK, scope_ref=task_id, changed_by="task"
    )
    val, scope = await resolver.resolve(
        key, task_id=task_id, user_id=user_id, session_id=session_id
    )
    assert val == "ask"
    assert scope == SettingScope.TASK


@pytest.mark.asyncio
async def test_scope_ref_isolation(resolver, store, seeded):
    """Values scoped with specific scope_ref only apply to that specific reference."""
    key = "reasoning_depth"
    task_1 = "task-uuid-1"
    task_2 = "task-uuid-2"

    # Delete project scope so task can win if matched
    await store.delete_value(key, scope=SettingScope.PROJECT, changed_by="test")

    await store.set_value(
        key, "normal", scope=SettingScope.TASK, scope_ref=task_1, changed_by="test"
    )

    # task_1 gets "normal"
    val1, scope1 = await resolver.resolve(key, task_id=task_1)
    assert val1 == "normal"
    assert scope1 == SettingScope.TASK

    # task_2 falls back to default because task_1's override does not apply
    val2, scope2 = await resolver.resolve(key, task_id=task_2)
    assert val2 == "deep"  # default from spec
    assert scope2 == "default"


@pytest.mark.asyncio
async def test_resolve_all_returns_full_map(resolver, seeded):
    """resolve_all returns both resolved values and resolved_from for all catalog keys."""
    values, resolved_from = await resolver.resolve_all()
    assert len(values) >= 116
    assert len(resolved_from) == len(values)
    assert "reasoning_depth" in values
    assert resolved_from["reasoning_depth"] == "project"


# ---------------------------------------------------------------------------
# Snapshot Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_snapshot_creation_and_retrieval(snapshot_service, seeded):
    """create_snapshot writes a row and get_snapshot reads frozen values."""
    task_id = uuid.uuid4()
    session_id = uuid.uuid4()

    snapshot_id = await snapshot_service.create_snapshot(
        task_id=task_id, session_id=session_id
    )
    assert isinstance(snapshot_id, uuid.UUID)

    frozen_values = await snapshot_service.get_snapshot(snapshot_id)
    assert frozen_values is not None
    assert isinstance(frozen_values, dict)
    assert "reasoning_depth" in frozen_values
    assert frozen_values["reasoning_depth"] == "deep"

    record = await snapshot_service.get_snapshot_record(snapshot_id)
    assert record is not None
    assert record["id"] == snapshot_id
    assert record["task_id"] == task_id
    assert record["session_id"] == session_id
    assert "reasoning_depth" in record["values"]
    assert record["resolved_from"]["reasoning_depth"] == "project"


@pytest.mark.asyncio
async def test_snapshot_immutability(snapshot_service, store, seeded):
    """Modifying settings in DB AFTER a snapshot is taken does NOT affect the snapshot."""
    task_id = uuid.uuid4()
    key = "reasoning_depth"

    # 1. Take snapshot when reasoning_depth is 'deep'
    snapshot_id = await snapshot_service.create_snapshot(task_id=task_id)
    initial_values = await snapshot_service.get_snapshot(snapshot_id)
    assert initial_values[key] == "deep"

    # 2. Modify reasoning_depth in DB at SYSTEM_HARD scope to 'normal'
    await store.set_value(
        key,
        "normal",
        scope=SettingScope.SYSTEM_HARD,
        changed_by="admin",
        reason="system update after snapshot",
    )

    # 3. Read snapshot again: it MUST still return the frozen 'deep' value
    frozen_after = await snapshot_service.get_snapshot(snapshot_id)
    assert frozen_after[key] == "deep"

    # 4. A newly created snapshot DOES see the updated value
    new_snapshot_id = await snapshot_service.create_snapshot(task_id=task_id)
    new_values = await snapshot_service.get_snapshot(new_snapshot_id)
    assert new_values[key] == "normal"


@pytest.mark.asyncio
async def test_get_nonexistent_snapshot_returns_none(snapshot_service):
    """Querying a non-existent snapshot_id returns None."""
    random_id = uuid.uuid4()
    assert await snapshot_service.get_snapshot(random_id) is None
    assert await snapshot_service.get_snapshot_record(random_id) is None
