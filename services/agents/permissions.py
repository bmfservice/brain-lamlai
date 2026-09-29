"""Permission enforcement for the agent hierarchy.

Rules the spec requires to be enforced at the orchestration layer, not by
prompt (spec §37):

* specialist cannot bypass leader
* leader cannot close root task
* specialist cannot close department task
* only Braind (executive) can finalize root task
* only leader can create/assign subtasks
* only executive can create/assign department tasks

Every guarded action calls ``require`` with the acting agent's level and the
permission it needs; a violation raises ``PermissionDenied`` which the caller
turns into an audit ``PERMISSION_DENIED`` event.
"""

from __future__ import annotations

from services.agents.domain import (
    AgentLevel,
    Permission,
    TaskType,
    permissions_for,
)


class PermissionDenied(Exception):
    """Raised when an agent attempts an action outside its granted permissions."""

    def __init__(self, level: AgentLevel, permission: Permission, reason: str | None = None):
        self.level = level
        self.permission = permission
        self.reason = reason
        msg = f"{level.value} may not {permission.value}"
        if reason:
            msg = f"{msg}: {reason}"
        super().__init__(msg)


def has_permission(level: AgentLevel, permission: Permission) -> bool:
    """Return True if the level is granted the permission."""
    return permission in permissions_for(level)


def require(level: AgentLevel, permission: Permission, reason: str | None = None) -> None:
    """Raise ``PermissionDenied`` unless ``level`` holds ``permission``."""
    if not has_permission(level, permission):
        raise PermissionDenied(level, permission, reason)


def require_can_finalize(level: AgentLevel, task_type: TaskType) -> None:
    """Only the executive may finalize a ROOT task.

    Enforces both "leader cannot close root task" and "specialist cannot close
    department task": closing a ROOT requires FINALIZE_ROOT (executive only),
    and closing a DEPARTMENT requires CONSOLIDATE_DEPARTMENT (leader only).
    """
    if task_type == TaskType.ROOT:
        require(level, Permission.FINALIZE_ROOT, "only Braind can finalize the root task")
    elif task_type == TaskType.DEPARTMENT:
        require(
            level,
            Permission.CONSOLIDATE_DEPARTMENT,
            "only a department leader can close a department task",
        )


def require_can_assign(parent_level: AgentLevel, child_type: TaskType) -> None:
    """Guard the two legal assignment edges of the hierarchy.

    executive -> department task, leader -> subtask. Anything else (e.g. a
    specialist creating a task, or the executive assigning a subtask directly)
    is denied — this is what stops Braind bypassing a leader and stops a
    specialist bypassing its leader.
    """
    if child_type == TaskType.DEPARTMENT:
        require(parent_level, Permission.ASSIGN_TO_LEADER, "department tasks come from Braind only")
    elif child_type == TaskType.SUBTASK:
        require(parent_level, Permission.ASSIGN_TO_SPECIALIST, "subtasks come from a leader only")
    else:
        raise PermissionDenied(
            parent_level,
            Permission.ASSIGN_TO_SPECIALIST,
            f"cannot assign a {child_type.value} task",
        )
