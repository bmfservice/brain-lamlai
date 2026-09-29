"""Task state machine — the only place a task state may legally change.

Every transition in the workflow is checked here. An illegal transition
raises ``IllegalTransition`` rather than silently corrupting task state, which
is what enforces the lifecycle described in spec §33 at the application layer
(not merely in prompts).
"""

from __future__ import annotations

from services.agents.domain import (
    TERMINAL_STATES,
    TaskState,
    TaskType,
)


class IllegalTransition(Exception):
    """Raised when a task is asked to move between incompatible states."""

    def __init__(
        self,
        task_type: TaskType | str,
        current: TaskState | str,
        target: TaskState | str,
    ):
        self.task_type = task_type
        self.current = current
        self.target = target
        c_val = current.value if isinstance(current, TaskState) else str(current)
        t_val = target.value if isinstance(target, TaskState) else str(target)
        tt_val = task_type.value if isinstance(task_type, TaskType) else str(task_type)
        super().__init__(
            f"Illegal {tt_val} transition: {c_val} -> {t_val}"
        )


# The 11 canonical task states transition table (Brain Second spec)
CANONICAL_TRANSITIONS: dict[TaskState, frozenset[TaskState]] = {
    TaskState.PENDING: frozenset({TaskState.PLANNING, TaskState.CANCELLED, TaskState.FAILED}),
    TaskState.PLANNING: frozenset(
        {TaskState.PLANNED, TaskState.WAITING_INPUT, TaskState.FAILED, TaskState.CANCELLED}
    ),
    TaskState.PLANNED: frozenset(
        {TaskState.EXECUTING, TaskState.WAITING_APPROVAL, TaskState.CANCELLED, TaskState.FAILED}
    ),
    TaskState.EXECUTING: frozenset(
        {
            TaskState.WAITING_APPROVAL,
            TaskState.WAITING_INPUT,
            TaskState.VERIFYING,
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.CANCELLED,
        }
    ),
    TaskState.WAITING_APPROVAL: frozenset(
        {TaskState.EXECUTING, TaskState.PLANNING, TaskState.FAILED, TaskState.CANCELLED}
    ),
    TaskState.WAITING_INPUT: frozenset(
        {TaskState.PLANNING, TaskState.EXECUTING, TaskState.FAILED, TaskState.CANCELLED}
    ),
    TaskState.VERIFYING: frozenset(
        {TaskState.COMPLETED, TaskState.REPAIRING, TaskState.FAILED, TaskState.CANCELLED}
    ),
    TaskState.REPAIRING: frozenset(
        {TaskState.EXECUTING, TaskState.VERIFYING, TaskState.FAILED, TaskState.CANCELLED}
    ),
    TaskState.COMPLETED: frozenset(),
    TaskState.FAILED: frozenset(),
    TaskState.CANCELLED: frozenset(),
}

# Root task traverses the full executive lifecycle.
_ROOT_TRANSITIONS: dict[TaskState, frozenset[TaskState]] = {
    TaskState.CREATED: frozenset({TaskState.ANALYZING, TaskState.FAILED, TaskState.CANCELLED}),
    TaskState.ANALYZING: frozenset(
        {TaskState.PLANNED, TaskState.FAILED, TaskState.WAITING_FOR_USER, TaskState.CANCELLED}
    ),
    TaskState.PLANNED: frozenset({TaskState.DISPATCHED, TaskState.FAILED, TaskState.CANCELLED}),
    TaskState.DISPATCHED: frozenset({TaskState.IN_PROGRESS, TaskState.FAILED, TaskState.CANCELLED}),
    TaskState.IN_PROGRESS: frozenset(
        {TaskState.REVIEWING, TaskState.BLOCKED, TaskState.FAILED, TaskState.CANCELLED}
    ),
    TaskState.REVIEWING: frozenset(
        {
            TaskState.WAITING_FOR_USER,
            TaskState.REWORK_REQUIRED,
            TaskState.FINAL_VALIDATION,
            TaskState.FAILED,
            TaskState.CANCELLED,
        }
    ),
    TaskState.WAITING_FOR_USER: frozenset(
        {TaskState.CONTINUING, TaskState.FAILED, TaskState.REWORK_REQUIRED, TaskState.CANCELLED}
    ),
    TaskState.CONTINUING: frozenset(
        {
            TaskState.DISPATCHED,
            TaskState.IN_PROGRESS,
            TaskState.REVIEWING,
            TaskState.FINAL_VALIDATION,
            TaskState.FAILED,
            TaskState.CANCELLED,
        }
    ),
    TaskState.REWORK_REQUIRED: frozenset(
        {TaskState.DISPATCHED, TaskState.FAILED, TaskState.CANCELLED}
    ),
    TaskState.FINAL_VALIDATION: frozenset(
        {
            TaskState.COMPLETED,
            TaskState.REWORK_REQUIRED,
            TaskState.WAITING_FOR_USER,
            TaskState.FAILED,
            TaskState.CANCELLED,
        }
    ),
    TaskState.BLOCKED: frozenset(
        {TaskState.IN_PROGRESS, TaskState.WAITING_FOR_USER, TaskState.FAILED, TaskState.CANCELLED}
    ),
    TaskState.COMPLETED: frozenset(),
    TaskState.FAILED: frozenset(),
    TaskState.CANCELLED: frozenset(),
}

# Department and subtask share a simpler executor lifecycle: no user gate,
# no final validation (those belong to the root task owned by Braind).
_WORKER_TRANSITIONS: dict[TaskState, frozenset[TaskState]] = {
    TaskState.CREATED: frozenset(
        {
            TaskState.ANALYZING,
            TaskState.DISPATCHED,
            TaskState.IN_PROGRESS,
            TaskState.FAILED,
            TaskState.CANCELLED,
        }
    ),
    TaskState.ANALYZING: frozenset(
        {TaskState.PLANNED, TaskState.IN_PROGRESS, TaskState.FAILED, TaskState.CANCELLED}
    ),
    TaskState.PLANNED: frozenset(
        {TaskState.DISPATCHED, TaskState.IN_PROGRESS, TaskState.FAILED, TaskState.CANCELLED}
    ),
    TaskState.DISPATCHED: frozenset(
        {TaskState.IN_PROGRESS, TaskState.BLOCKED, TaskState.FAILED, TaskState.CANCELLED}
    ),
    TaskState.IN_PROGRESS: frozenset(
        {
            TaskState.REVIEWING,
            TaskState.COMPLETED,
            TaskState.BLOCKED,
            TaskState.FAILED,
            TaskState.CANCELLED,
        }
    ),
    TaskState.REVIEWING: frozenset(
        {
            TaskState.COMPLETED,
            TaskState.REWORK_REQUIRED,
            TaskState.BLOCKED,
            TaskState.FAILED,
            TaskState.CANCELLED,
        }
    ),
    TaskState.REWORK_REQUIRED: frozenset(
        {
            TaskState.ANALYZING,
            TaskState.DISPATCHED,
            TaskState.IN_PROGRESS,
            TaskState.FAILED,
            TaskState.CANCELLED,
        }
    ),
    TaskState.BLOCKED: frozenset({TaskState.IN_PROGRESS, TaskState.FAILED, TaskState.CANCELLED}),
    TaskState.COMPLETED: frozenset(),
    TaskState.FAILED: frozenset(),
    TaskState.CANCELLED: frozenset(),
}


def _coerce_task_state(state: TaskState | str) -> TaskState:
    if isinstance(state, TaskState):
        return state
    try:
        return TaskState(state)
    except ValueError:
        return TaskState[state.upper()]


def _coerce_task_type(task_type: TaskType | str) -> TaskType:
    if isinstance(task_type, TaskType):
        return task_type
    try:
        return TaskType(task_type)
    except ValueError:
        return TaskType[task_type.upper()]


def _table(task_type: TaskType) -> dict[TaskState, frozenset[TaskState]]:
    if task_type == TaskType.CANONICAL:
        return CANONICAL_TRANSITIONS
    if task_type == TaskType.ROOT:
        return _ROOT_TRANSITIONS
    return _WORKER_TRANSITIONS


def can_transition(
    task_type: TaskType | str,
    current: TaskState | str,
    target: TaskState | str,
) -> bool:
    """Return True if ``current -> target`` is legal for this task type."""
    tt = _coerce_task_type(task_type)
    curr = _coerce_task_state(current)
    tgt = _coerce_task_state(target)
    if curr == tgt:
        return True

    table = _table(tt)
    if curr in table:
        return tgt in table.get(curr, frozenset())

    # Fallback to canonical table if state belongs to canonical machine
    if curr in CANONICAL_TRANSITIONS:
        return tgt in CANONICAL_TRANSITIONS.get(curr, frozenset())
    return False


def assert_transition(
    task_type: TaskType | str,
    current: TaskState | str,
    target: TaskState | str,
) -> None:
    """Raise ``IllegalTransition`` unless the move is legal.

    A no-op self-transition (current == target) is always allowed so idempotent
    re-saves do not blow up.
    """
    tt = _coerce_task_type(task_type)
    curr = _coerce_task_state(current)
    tgt = _coerce_task_state(target)

    if curr == tgt:
        return
    if not can_transition(tt, curr, tgt):
        raise IllegalTransition(tt, curr, tgt)


def is_terminal(state: TaskState | str) -> bool:
    """Return True if no further transition is possible from ``state``."""
    s = _coerce_task_state(state)
    return s in TERMINAL_STATES


class TaskStateMachine:
    """Explicit state machine managing the 11 canonical task states."""

    @classmethod
    def can_transition(
        cls,
        current: TaskState | str,
        target: TaskState | str,
        task_type: TaskType | str = TaskType.CANONICAL,
    ) -> bool:
        return can_transition(task_type, current, target)

    @classmethod
    def assert_transition(
        cls,
        current: TaskState | str,
        target: TaskState | str,
        task_type: TaskType | str = TaskType.CANONICAL,
    ) -> None:
        assert_transition(task_type, current, target)

    @classmethod
    def is_terminal(cls, state: TaskState | str) -> bool:
        return is_terminal(state)

    @classmethod
    def get_legal_targets(
        cls,
        current: TaskState | str,
        task_type: TaskType | str = TaskType.CANONICAL,
    ) -> frozenset[TaskState]:
        tt = _coerce_task_type(task_type)
        curr = _coerce_task_state(current)
        table = _table(tt)
        return table.get(curr, frozenset())

