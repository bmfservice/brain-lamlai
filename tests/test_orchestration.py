"""Tests for Phase 2: Task Orchestrator & State Machine (tests/test_orchestration.py)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from services.agents.domain import TaskState
from services.agents.state_machine import (
    CANONICAL_TRANSITIONS,
    IllegalTransition,
    TaskStateMachine,
)
from services.orchestration.service import TaskOrchestrationService
from services.settings.snapshot import create_default_snapshot

# ----------------------------------------------------------------------
# 1. State Machine: 11 Canonical States & Transitions
# ----------------------------------------------------------------------


def test_all_11_canonical_states_exist():
    """Verify all 11 canonical task states from the Brain Second spec are defined."""
    expected_states = {
        "pending",
        "planning",
        "planned",
        "executing",
        "waiting_approval",
        "waiting_input",
        "verifying",
        "repairing",
        "completed",
        "failed",
        "cancelled",
    }
    actual_states = {s.value for s in CANONICAL_TRANSITIONS}
    assert expected_states == actual_states


def test_canonical_happy_path_transitions():
    """Test standard happy path:
    PENDING -> PLANNING -> PLANNED -> EXECUTING -> VERIFYING -> COMPLETED.
    """
    service = TaskOrchestrationService()
    task = service.create_task("Run market research analysis")
    assert task.state == TaskState.PENDING

    task = service.start_planning(task.task_id)
    assert task.state == TaskState.PLANNING
    assert task.settings_snapshot is not None

    task = service.plan_task(task.task_id, plan_data={"steps": ["analyze", "summarize"]})
    assert task.state == TaskState.PLANNED

    task = service.start_execution(task.task_id)
    assert task.state == TaskState.EXECUTING
    assert task.started_at is not None

    task = service.start_verification(task.task_id)
    assert task.state == TaskState.VERIFYING

    task = service.complete_task(task.task_id, result={"deliverable": "done"})
    assert task.state == TaskState.COMPLETED
    assert task.completed_at is not None
    assert task.is_terminal


def test_waiting_approval_and_input_transitions():
    """Test transitions into and out of WAITING_APPROVAL and WAITING_INPUT."""
    service = TaskOrchestrationService()
    task = service.create_task("Sensitive operation")

    service.start_planning(task.task_id)
    service.plan_task(task.task_id)
    service.start_execution(task.task_id)

    # Transition into WAITING_APPROVAL
    service.transition_task(task.task_id, TaskState.WAITING_APPROVAL)
    assert task.state == TaskState.WAITING_APPROVAL

    # Approved back to EXECUTING
    service.transition_task(task.task_id, TaskState.EXECUTING)
    assert task.state == TaskState.EXECUTING

    # Transition into WAITING_INPUT
    service.request_user_input(task.task_id, "Please confirm the target brand")
    assert task.state == TaskState.WAITING_INPUT
    assert task.pending_input == "Please confirm the target brand"

    # Input provided, resumes to EXECUTING
    service.provide_user_input(task.task_id, "Brand X")
    assert task.state == TaskState.EXECUTING
    assert task.pending_input is None


def test_repair_loop_transitions():
    """Test transitions involving REPAIRING state."""
    service = TaskOrchestrationService()
    task = service.create_task("Code refactoring")

    service.start_planning(task.task_id)
    service.plan_task(task.task_id)
    service.start_execution(task.task_id)
    service.start_verification(task.task_id)
    assert task.state == TaskState.VERIFYING

    # Verification fails -> REPAIRING
    service.start_repair(task.task_id, errors=["SyntaxError on line 10"])
    assert task.state == TaskState.REPAIRING

    # Repair re-executes -> EXECUTING
    service.retry_execution(task.task_id)
    assert task.state == TaskState.EXECUTING

    # Re-verify -> VERIFYING -> COMPLETED
    service.start_verification(task.task_id)
    service.complete_task(task.task_id)
    assert task.state == TaskState.COMPLETED


def test_all_11_states_legal_transitions_coverage():
    """Verify that every one of the 11 states can perform its legal moves."""
    legal_transitions = [
        (TaskState.PENDING, TaskState.PLANNING),
        (TaskState.PENDING, TaskState.CANCELLED),
        (TaskState.PENDING, TaskState.FAILED),
        (TaskState.PLANNING, TaskState.PLANNED),
        (TaskState.PLANNING, TaskState.WAITING_INPUT),
        (TaskState.PLANNING, TaskState.FAILED),
        (TaskState.PLANNING, TaskState.CANCELLED),
        (TaskState.PLANNED, TaskState.EXECUTING),
        (TaskState.PLANNED, TaskState.WAITING_APPROVAL),
        (TaskState.PLANNED, TaskState.FAILED),
        (TaskState.PLANNED, TaskState.CANCELLED),
        (TaskState.EXECUTING, TaskState.VERIFYING),
        (TaskState.EXECUTING, TaskState.WAITING_APPROVAL),
        (TaskState.EXECUTING, TaskState.WAITING_INPUT),
        (TaskState.EXECUTING, TaskState.COMPLETED),
        (TaskState.EXECUTING, TaskState.FAILED),
        (TaskState.EXECUTING, TaskState.CANCELLED),
        (TaskState.WAITING_APPROVAL, TaskState.EXECUTING),
        (TaskState.WAITING_APPROVAL, TaskState.PLANNING),
        (TaskState.WAITING_APPROVAL, TaskState.FAILED),
        (TaskState.WAITING_APPROVAL, TaskState.CANCELLED),
        (TaskState.WAITING_INPUT, TaskState.EXECUTING),
        (TaskState.WAITING_INPUT, TaskState.PLANNING),
        (TaskState.WAITING_INPUT, TaskState.FAILED),
        (TaskState.WAITING_INPUT, TaskState.CANCELLED),
        (TaskState.VERIFYING, TaskState.COMPLETED),
        (TaskState.VERIFYING, TaskState.REPAIRING),
        (TaskState.VERIFYING, TaskState.FAILED),
        (TaskState.VERIFYING, TaskState.CANCELLED),
        (TaskState.REPAIRING, TaskState.EXECUTING),
        (TaskState.REPAIRING, TaskState.VERIFYING),
        (TaskState.REPAIRING, TaskState.FAILED),
        (TaskState.REPAIRING, TaskState.CANCELLED),
    ]

    for source, target in legal_transitions:
        msg = f"Expected legal: {source} -> {target}"
        assert TaskStateMachine.can_transition(source, target), msg


def test_illegal_transitions_rejected():
    """Verify illegal state transitions raise IllegalTransition."""
    illegal_moves = [
        (TaskState.PENDING, TaskState.COMPLETED),
        (TaskState.PENDING, TaskState.VERIFYING),
        (TaskState.PLANNING, TaskState.COMPLETED),
        (TaskState.PLANNED, TaskState.COMPLETED),
        (TaskState.COMPLETED, TaskState.EXECUTING),
        (TaskState.COMPLETED, TaskState.PLANNING),
        (TaskState.FAILED, TaskState.EXECUTING),
        (TaskState.FAILED, TaskState.PENDING),
        (TaskState.CANCELLED, TaskState.EXECUTING),
        (TaskState.CANCELLED, TaskState.VERIFYING),
    ]

    for source, target in illegal_moves:
        assert not TaskStateMachine.can_transition(source, target)
        with pytest.raises(IllegalTransition):
            TaskStateMachine.assert_transition(source, target)


# ----------------------------------------------------------------------
# 2. Settings Snapshot Creation and Attachment
# ----------------------------------------------------------------------


def test_snapshot_attached_at_planning_state():
    """Verify SettingsSnapshot is attached when task enters PLANNING state."""
    service = TaskOrchestrationService()
    task = service.create_task(
        "Generate financial report",
        settings_overrides={"max_retry_count": 4, "when_uncertain": "ask"},
    )
    assert task.settings_snapshot is None
    assert task.snapshot_id is None

    # Enter PLANNING
    service.start_planning(task.task_id)

    assert task.settings_snapshot is not None
    assert task.snapshot_id == task.settings_snapshot.snapshot_id
    # Overrides honored in snapshot
    assert task.settings_snapshot.get("max_retry_count") == 4
    assert task.settings_snapshot.get("when_uncertain") == "ask"
    # Catalog default available
    assert task.settings_snapshot.get("quality_gate_enabled") is True


def test_snapshot_immutability():
    """Verify snapshot values cannot be mutated to affect other tasks."""
    snapshot = create_default_snapshot(overrides={"key_a": "value_1"})
    vals = snapshot.values
    vals["key_a"] = "mutated_value"
    # Internal state remains original
    assert snapshot.get("key_a") == "value_1"


# ----------------------------------------------------------------------
# 3. Tool Execution Gating: Sensitive Tools & Confirmation
# ----------------------------------------------------------------------


def test_sensitive_tool_triggers_waiting_approval():
    """Sensitive tools (e.g. shell_exec) pause task in WAITING_APPROVAL."""
    service = TaskOrchestrationService()
    task = service.create_task("Run script")
    service.start_planning(task.task_id)
    service.plan_task(task.task_id)
    service.start_execution(task.task_id)

    decision = service.check_tool_execution(
        task.task_id,
        tool_name="shell_exec",
        tool_args={"cmd": "ls -la"},
    )
    assert not decision.allowed
    assert decision.needs_approval
    assert task.state == TaskState.WAITING_APPROVAL
    assert task.pending_approval is not None
    assert task.pending_approval["tool_name"] == "shell_exec"


def test_tool_require_confirmation_setting():
    """When tool_require_confirmation=True in snapshot, any tool pauses for approval."""
    service = TaskOrchestrationService()
    task = service.create_task(
        "Search knowledge",
        settings_overrides={"tool_require_confirmation": True},
    )
    service.start_planning(task.task_id)
    service.plan_task(task.task_id)
    service.start_execution(task.task_id)

    decision = service.check_tool_execution(
        task.task_id,
        tool_name="web_search",  # normally safe
    )
    assert not decision.allowed
    assert decision.needs_approval
    assert task.state == TaskState.WAITING_APPROVAL


def test_approve_and_reject_tool():
    """Test approving or rejecting a paused tool call."""
    service = TaskOrchestrationService()
    task = service.create_task("Data migration")
    service.start_planning(task.task_id)
    service.plan_task(task.task_id)
    service.start_execution(task.task_id)

    service.check_tool_execution(task.task_id, "delete_file", is_sensitive=True)
    assert task.state == TaskState.WAITING_APPROVAL

    # Approve
    service.approve_tool(task.task_id, reviewer="lead_dev", notes="Verified safe")
    assert task.state == TaskState.EXECUTING
    assert task.pending_approval is None

    # Pause again and reject
    service.check_tool_execution(task.task_id, "rm_rf", is_sensitive=True)
    assert task.state == TaskState.WAITING_APPROVAL

    service.reject_tool(task.task_id, reviewer="lead_dev", reason="Too dangerous")
    assert task.state == TaskState.FAILED
    assert "Too dangerous" in str(task.error)


# ----------------------------------------------------------------------
# 4. Task Cancellation & Rollback Mechanism
# ----------------------------------------------------------------------


def test_task_cancellation_and_rollback():
    """Verify task cancellation records reason and executes rollback hooks."""
    service = TaskOrchestrationService()
    task = service.create_task("Deploy service")
    service.start_planning(task.task_id)
    service.plan_task(task.task_id)
    service.start_execution(task.task_id)

    reverted_items = []

    # Register temporary change with revert callback
    service.record_temporary_change(
        task.task_id,
        change_type="created_temp_table",
        details={"table": "temp_staging_123"},
        revert_fn=lambda: reverted_items.append("table_dropped"),
    )

    # Create a physical temporary file and register it for cleanup
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        temp_file_path = tf.name
    service.register_rollback_action(task.task_id, temp_file_path)
    assert Path(temp_file_path).exists()

    # Cancel task
    service.cancel_task(task.task_id, reason="User requested abort")

    assert task.state == TaskState.CANCELLED
    assert task.cancellation_reason == "User requested abort"
    assert task.rolled_back is True
    assert "table_dropped" in reverted_items
    assert not Path(temp_file_path).exists()  # File cleaned up
    assert len(task.rollback_log) == 2


def test_task_failure_triggers_rollback():
    """Verify task failure triggers rollback of temporary changes."""
    service = TaskOrchestrationService()
    task = service.create_task("Database migration")
    service.start_planning(task.task_id)
    service.plan_task(task.task_id)
    service.start_execution(task.task_id)

    revert_called = []
    service.register_rollback_action(task.task_id, lambda: revert_called.append(True))

    service.fail_task(task.task_id, error="Connection timed out")

    assert task.state == TaskState.FAILED
    assert task.error == "Connection timed out"
    assert task.rolled_back is True
    assert revert_called == [True]
