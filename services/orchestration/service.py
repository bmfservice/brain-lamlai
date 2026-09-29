"""Task Orchestration Service (Brain Second Phase 2).

Manages the complete lifecycle of tasks across the 11 canonical states:
PENDING, PLANNING, PLANNED, EXECUTING, WAITING_APPROVAL, WAITING_INPUT,
VERIFYING, REPAIRING, COMPLETED, FAILED, CANCELLED.

Key capabilities:
- State transitions validated via TaskStateMachine.
- Immutable SettingsSnapshot attached at PLANNING state.
- Sensitive tool & tool_require_confirmation detection leading to WAITING_APPROVAL.
- Rollback execution when tasks transition to FAILED or CANCELLED.
"""

from __future__ import annotations

import inspect
import os
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from core.logging import get_logger
from services.agents.domain import TaskState, TaskType, to_canonical_state
from services.agents.state_machine import TaskStateMachine
from services.settings.snapshot import (
    SettingsSnapshot,
    create_default_snapshot,
    resolve_settings_snapshot,
)

logger = get_logger(__name__)

DEFAULT_SENSITIVE_TOOLS: frozenset[str] = frozenset(
    {
        "shell_exec",
        "file_delete",
        "delete_file",
        "database_drop",
        "system_reboot",
        "send_email",
        "execute_code",
        "rm_rf",
        "write_system_config",
        "publish_external",
    }
)


@dataclass
class ToolExecutionDecision:
    """Decision on whether a tool can execute immediately or requires human approval."""

    allowed: bool
    needs_approval: bool
    reason: str
    tool_name: str
    approval_id: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestratedTask:
    """Task entity tracked by TaskOrchestrationService."""

    task_id: str
    description: str
    state: TaskState = TaskState.PENDING
    task_type: TaskType = TaskType.CANONICAL
    priority: int = 2
    user_request: str | None = None

    # Settings snapshot frozen at PLANNING state
    settings_snapshot: SettingsSnapshot | None = None
    snapshot_id: str | None = None
    settings_overrides: dict[str, Any] = field(default_factory=dict)

    # Deliverables and execution state
    plan: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    cancellation_reason: str | None = None

    # Approval and interactive input
    pending_approval: dict[str, Any] | None = None
    pending_input: str | None = None

    # Rollback tracking
    rollback_actions: list[Any] = field(default_factory=list)
    rollback_log: list[dict[str, Any]] = field(default_factory=list)
    rolled_back: bool = False

    # Timing and audit
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_terminal(self) -> bool:
        return TaskStateMachine.is_terminal(self.state)

    def to_dict(self) -> dict[str, Any]:
        s_val = self.state.value if isinstance(self.state, TaskState) else self.state
        tt_val = self.task_type.value if isinstance(self.task_type, TaskType) else self.task_type
        return {
            "task_id": self.task_id,
            "description": self.description,
            "state": s_val,
            "task_type": tt_val,
            "priority": self.priority,
            "user_request": self.user_request,
            "snapshot_id": self.snapshot_id,
            "plan": self.plan,
            "result": self.result,
            "error": self.error,
            "cancellation_reason": self.cancellation_reason,
            "pending_approval": self.pending_approval,
            "pending_input": self.pending_input,
            "rolled_back": self.rolled_back,
            "rollback_log": self.rollback_log,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "history": self.history,
        }


class TaskOrchestrationService:
    """Orchestrates task lifecycle, state transitions, tool gates, and rollbacks."""

    def __init__(
        self,
        session_factory: Any = None,
        settings_store: Any = None,
        approval_service: Any = None,
        sensitive_tools: set[str] | frozenset[str] | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.settings_store = settings_store
        self.approval_service = approval_service
        self.sensitive_tools = (
            frozenset(sensitive_tools) if sensitive_tools is not None else DEFAULT_SENSITIVE_TOOLS
        )
        self._tasks: dict[str, OrchestratedTask] = {}

    # ------------------------------------------------------------------
    # Task Creation & Retrieval
    # ------------------------------------------------------------------

    def create_task(
        self,
        description: str,
        task_id: str | None = None,
        user_request: str | None = None,
        priority: int = 2,
        task_type: TaskType = TaskType.CANONICAL,
        settings_overrides: Mapping[str, Any] | None = None,
    ) -> OrchestratedTask:
        """Create a new task in PENDING state."""
        tid = task_id or str(uuid.uuid4())
        task = OrchestratedTask(
            task_id=tid,
            description=description,
            state=TaskState.PENDING,
            task_type=task_type,
            priority=priority,
            user_request=user_request or description,
            settings_overrides=dict(settings_overrides or {}),
        )
        task.history.append(
            {
                "from_state": None,
                "to_state": TaskState.PENDING.value,
                "timestamp": datetime.now(UTC).isoformat(),
                "reason": "Task created",
            }
        )
        self._tasks[tid] = task
        logger.info("orchestration.task_created", task_id=tid, state=TaskState.PENDING.value)
        return task

    def get_task(self, task_id: str) -> OrchestratedTask | None:
        """Retrieve task by ID."""
        return self._tasks.get(task_id)

    # ------------------------------------------------------------------
    # State Transitions & Lifecycle Management
    # ------------------------------------------------------------------

    def transition_task(
        self,
        task_id: str,
        target_state: TaskState | str,
        reason: str | None = None,
    ) -> OrchestratedTask:
        """Transition task to target state after asserting legality via state machine."""
        task = self._tasks.get(task_id)
        if not task:
            raise KeyError(f"Task with ID {task_id} not found")

        current_state = task.state
        target_enum = to_canonical_state(target_state)

        # Validate transition
        TaskStateMachine.assert_transition(current_state, target_enum, task.task_type)

        now = datetime.now(UTC)
        task.state = target_enum
        curr_val = (
            current_state.value if isinstance(current_state, TaskState) else str(current_state)
        )
        task.history.append(
            {
                "from_state": curr_val,
                "to_state": target_enum.value,
                "timestamp": now.isoformat(),
                "reason": reason or f"Transition to {target_enum.value}",
            }
        )

        # Lifecycle hooks per state
        if target_enum == TaskState.PLANNING:
            self._on_enter_planning(task)

        elif target_enum == TaskState.EXECUTING:
            if task.started_at is None:
                task.started_at = now

        elif target_enum in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED):
            task.completed_at = now
            if target_enum in (TaskState.FAILED, TaskState.CANCELLED):
                self._execute_rollback(task)

        logger.info(
            "orchestration.task_transitioned",
            task_id=task.task_id,
            from_state=curr_val,
            to_state=target_enum.value,
            reason=reason,
        )
        return task

    def _on_enter_planning(self, task: OrchestratedTask) -> None:
        """Attach SettingsSnapshot when task enters PLANNING state."""
        if task.settings_snapshot is None:
            snapshot = create_default_snapshot(
                task_id=task.task_id,
                overrides=task.settings_overrides,
            )
            task.settings_snapshot = snapshot
            task.snapshot_id = snapshot.snapshot_id
            logger.info(
                "orchestration.snapshot_attached",
                task_id=task.task_id,
                snapshot_id=snapshot.snapshot_id,
            )

    async def attach_resolved_snapshot(
        self,
        task_id: str,
        user_overrides: Mapping[str, Any] | None = None,
    ) -> SettingsSnapshot:
        """Resolve and attach DB-backed settings snapshot."""
        task = self._tasks.get(task_id)
        if not task:
            raise KeyError(f"Task with ID {task_id} not found")

        snapshot = await resolve_settings_snapshot(
            store=self.settings_store,
            task_id=task.task_id,
            task_overrides=task.settings_overrides,
            user_overrides=user_overrides,
        )
        task.settings_snapshot = snapshot
        task.snapshot_id = snapshot.snapshot_id
        return snapshot

    # ------------------------------------------------------------------
    # Convenient State Progression Helpers
    # ------------------------------------------------------------------

    def start_planning(
        self,
        task_id: str,
        settings_overrides: Mapping[str, Any] | None = None,
    ) -> OrchestratedTask:
        """Transition task from PENDING to PLANNING, creating settings snapshot."""
        task = self._tasks.get(task_id)
        if task and settings_overrides:
            task.settings_overrides.update(settings_overrides)
        return self.transition_task(task_id, TaskState.PLANNING, "Started planning")

    def plan_task(
        self,
        task_id: str,
        plan_data: dict[str, Any] | None = None,
    ) -> OrchestratedTask:
        """Transition task from PLANNING to PLANNED."""
        task = self._tasks.get(task_id)
        if task and plan_data:
            task.plan = plan_data
        return self.transition_task(task_id, TaskState.PLANNED, "Plan formulated")

    def start_execution(self, task_id: str) -> OrchestratedTask:
        """Transition task from PLANNED to EXECUTING."""
        return self.transition_task(task_id, TaskState.EXECUTING, "Execution started")

    def start_verification(self, task_id: str) -> OrchestratedTask:
        """Transition task from EXECUTING to VERIFYING."""
        return self.transition_task(task_id, TaskState.VERIFYING, "Verification started")

    def start_repair(
        self,
        task_id: str,
        errors: list[str] | None = None,
    ) -> OrchestratedTask:
        """Transition task from VERIFYING to REPAIRING."""
        reason = f"Repair triggered: {', '.join(errors)}" if errors else "Repair loop triggered"
        return self.transition_task(task_id, TaskState.REPAIRING, reason)

    def retry_execution(self, task_id: str) -> OrchestratedTask:
        """Transition task from REPAIRING back to EXECUTING (or VERIFYING)."""
        return self.transition_task(task_id, TaskState.EXECUTING, "Re-executing repaired task")

    def complete_task(
        self,
        task_id: str,
        result: dict[str, Any] | None = None,
    ) -> OrchestratedTask:
        """Complete task with final deliverable result."""
        task = self._tasks.get(task_id)
        if task and result:
            task.result = result
        return self.transition_task(task_id, TaskState.COMPLETED, "Task completed successfully")

    def fail_task(
        self,
        task_id: str,
        error: str | None = None,
    ) -> OrchestratedTask:
        """Fail task, record error, and trigger rollback."""
        task = self._tasks.get(task_id)
        if task:
            task.error = error or "Task execution failed"
        return self.transition_task(task_id, TaskState.FAILED, f"Task failed: {error}")

    def cancel_task(
        self,
        task_id: str,
        reason: str | None = None,
    ) -> OrchestratedTask:
        """Cancel task, record reason, and trigger rollback."""
        task = self._tasks.get(task_id)
        if task:
            task.cancellation_reason = reason or "Task cancelled by user"
        return self.transition_task(task_id, TaskState.CANCELLED, f"Cancelled: {reason}")

    # ------------------------------------------------------------------
    # Tool Execution Gating & Sensitive Tool Detection
    # ------------------------------------------------------------------

    def check_tool_execution(
        self,
        task_id: str,
        tool_name: str,
        tool_args: dict[str, Any] | None = None,
        is_sensitive: bool = False,
    ) -> ToolExecutionDecision:
        """Check if a tool call requires confirmation or can proceed immediately."""
        task = self._tasks.get(task_id)
        if not task:
            raise KeyError(f"Task with ID {task_id} not found")

        args = tool_args or {}
        snapshot = task.settings_snapshot or create_default_snapshot(task_id=task.task_id)

        # Check global confirmation requirement
        global_require_confirmation = bool(snapshot.get("tool_require_confirmation", False))
        tool_specific_require = bool(snapshot.get(f"tool_require_confirmation_{tool_name}", False))
        is_sensitive_tool = is_sensitive or (tool_name in self.sensitive_tools)

        needs_approval = global_require_confirmation or tool_specific_require or is_sensitive_tool

        if needs_approval:
            reason = (
                f"Tool '{tool_name}' is sensitive"
                if is_sensitive_tool
                else f"Tool '{tool_name}' requires explicit confirmation"
            )
            approval_id = str(uuid.uuid4())
            task.pending_approval = {
                "approval_id": approval_id,
                "tool_name": tool_name,
                "tool_args": args,
                "is_sensitive": is_sensitive_tool,
                "requested_at": datetime.now(UTC).isoformat(),
            }
            # Pause task in WAITING_APPROVAL
            self.transition_task(
                task_id,
                TaskState.WAITING_APPROVAL,
                reason=f"Paused for approval of tool '{tool_name}'",
            )
            return ToolExecutionDecision(
                allowed=False,
                needs_approval=True,
                reason=reason,
                tool_name=tool_name,
                approval_id=approval_id,
                tool_args=args,
            )

        return ToolExecutionDecision(
            allowed=True,
            needs_approval=False,
            reason=f"Tool '{tool_name}' allowed for direct execution",
            tool_name=tool_name,
            tool_args=args,
        )

    def approve_tool(
        self,
        task_id: str,
        reviewer: str = "user",
        notes: str | None = None,
    ) -> OrchestratedTask:
        """Approve pending tool call and resume execution."""
        task = self._tasks.get(task_id)
        if not task:
            raise KeyError(f"Task with ID {task_id} not found")

        if task.state != TaskState.WAITING_APPROVAL:
            raise ValueError(
                f"Task is in {task.state.value}, cannot approve tool execution"
            )

        tool_info = task.pending_approval or {}
        tool_nm = tool_info.get("tool_name", "")
        task.pending_approval = None
        return self.transition_task(
            task_id,
            TaskState.EXECUTING,
            reason=f"Tool '{tool_nm}' approved by {reviewer}. {notes or ''}".strip(),
        )

    def reject_tool(
        self,
        task_id: str,
        reviewer: str = "user",
        reason: str | None = None,
        cancel_task: bool = False,
    ) -> OrchestratedTask:
        """Reject pending tool call, failing or cancelling task."""
        task = self._tasks.get(task_id)
        if not task:
            raise KeyError(f"Task with ID {task_id} not found")

        if task.state != TaskState.WAITING_APPROVAL:
            raise ValueError(
                f"Task is in {task.state.value}, cannot reject tool execution"
            )

        tool_info = task.pending_approval or {}
        tool_nm = tool_info.get("tool_name", "")
        task.pending_approval = None
        r_msg = reason or "No reason provided"
        rejection_reason = f"Tool '{tool_nm}' rejected by {reviewer}: {r_msg}"

        if cancel_task:
            return self.cancel_task(task_id, reason=rejection_reason)
        return self.fail_task(task_id, error=rejection_reason)

    # ------------------------------------------------------------------
    # User Input Gating
    # ------------------------------------------------------------------

    def request_user_input(
        self,
        task_id: str,
        prompt_message: str,
    ) -> OrchestratedTask:
        """Pause task in WAITING_INPUT state for user clarification."""
        task = self._tasks.get(task_id)
        if not task:
            raise KeyError(f"Task with ID {task_id} not found")

        task.pending_input = prompt_message
        return self.transition_task(
            task_id,
            TaskState.WAITING_INPUT,
            reason=f"Waiting for user input: {prompt_message}",
        )

    def provide_user_input(
        self,
        task_id: str,
        user_input: str | dict[str, Any],
        resume_to: TaskState = TaskState.EXECUTING,
    ) -> OrchestratedTask:
        """Provide input and resume task."""
        task = self._tasks.get(task_id)
        if not task:
            raise KeyError(f"Task with ID {task_id} not found")

        if task.state != TaskState.WAITING_INPUT:
            raise ValueError(f"Task is in {task.state.value}, not waiting for input")

        task.pending_input = None
        return self.transition_task(
            task_id,
            resume_to,
            reason="User input received, resuming execution",
        )

    # ------------------------------------------------------------------
    # Rollback Mechanism for Temporary Changes
    # ------------------------------------------------------------------

    def register_rollback_action(
        self,
        task_id: str,
        action: Any,
    ) -> None:
        """Register a cleanup/rollback action (callable, dict, or filepath)."""
        task = self._tasks.get(task_id)
        if not task:
            raise KeyError(f"Task with ID {task_id} not found")
        task.rollback_actions.append(action)

    def record_temporary_change(
        self,
        task_id: str,
        change_type: str,
        details: dict[str, Any],
        revert_fn: Callable[[], Any] | None = None,
    ) -> None:
        """Record a temporary change with an optional revert callable."""
        task = self._tasks.get(task_id)
        if not task:
            raise KeyError(f"Task with ID {task_id} not found")

        action = {
            "type": change_type,
            "details": details,
            "revert_fn": revert_fn,
            "recorded_at": datetime.now(UTC).isoformat(),
        }
        task.rollback_actions.append(action)

    def _execute_rollback(self, task: OrchestratedTask) -> None:
        """Execute all registered rollback actions in LIFO order."""
        if not task.rollback_actions or task.rolled_back:
            task.rolled_back = True
            return

        actions = list(reversed(task.rollback_actions))
        for action in actions:
            log_entry: dict[str, Any] = {
                "timestamp": datetime.now(UTC).isoformat(),
                "status": "success",
            }

            try:
                if callable(action):
                    if inspect.iscoroutinefunction(action):
                        # In sync context, if it's a coroutine, record notice
                        log_entry["action_type"] = "coroutine_callable"
                    else:
                        action()
                        log_entry["action_type"] = "callable"

                elif isinstance(action, dict):
                    log_entry["action_type"] = action.get("type", "dict_action")
                    log_entry["details"] = action.get("details", {})
                    revert_fn = action.get("revert_fn")
                    if callable(revert_fn):
                        revert_fn()

                elif isinstance(action, str) and os.path.exists(action):
                    # Action is a file path created during task
                    try:
                        os.remove(action)
                        log_entry["action_type"] = "file_cleanup"
                        log_entry["file_removed"] = action
                    except OSError as exc:
                        log_entry["status"] = "error"
                        log_entry["error"] = str(exc)

                else:
                    log_entry["action_type"] = "unsupported_action"
                    log_entry["repr"] = repr(action)

            except Exception as exc:
                logger.error("orchestration.rollback_action_failed", error=str(exc))
                log_entry["status"] = "error"
                log_entry["error"] = str(exc)

            task.rollback_log.append(log_entry)

        task.rolled_back = True
        logger.info(
            "orchestration.rollback_completed",
            task_id=task.task_id,
            reverted_count=len(task.rollback_log),
        )
