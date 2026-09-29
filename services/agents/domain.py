"""Domain contracts for the hierarchical multi-agent office.

This module is the single source of truth for the workflow's vocabulary:
task states, the legal transitions between them, agent levels/permissions,
audit actions, and knowledge-provenance types. Routing and lifecycle logic
must import these enums rather than comparing raw strings, so a new department
or agent needs no orchestration change (config-first, per the spec).
"""

from __future__ import annotations

from enum import StrEnum


class AgentLevel(StrEnum):
    """Vertical position in the hierarchy.

    Level drives routing and permission checks; role names and agent ids do
    not. A new department only needs an EXECUTIVE-less pair of LEADER +
    SPECIALIST rows to slot in.
    """

    EXECUTIVE = "executive"  # Braind
    LEADER = "leader"  # Department tactical orchestrator
    SPECIALIST = "specialist"  # Executor


class TaskType(StrEnum):
    """Where a task sits in the task tree."""

    ROOT = "root"  # User request, owned by Braind
    DEPARTMENT = "department"  # Braind -> Leader
    SUBTASK = "subtask"  # Leader -> Specialist
    CANONICAL = "canonical"  # Canonical orchestrator task lifecycle


class TaskState(StrEnum):
    """Explicit lifecycle state shared by every task in the tree.

    Supports both the 11 canonical states from the Brain Second spec and
    legacy aliases mapped seamlessly.
    """

    # 11 Canonical task states from the Brain Second spec
    PENDING = "pending"
    PLANNING = "planning"
    PLANNED = "planned"
    EXECUTING = "executing"
    WAITING_APPROVAL = "waiting_approval"
    WAITING_INPUT = "waiting_input"
    VERIFYING = "verifying"
    REPAIRING = "repairing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    # Legacy workflow states (for backward compatibility with hierarchical office)
    CREATED = "created"
    ANALYZING = "analyzing"
    DISPATCHED = "dispatched"
    IN_PROGRESS = "in_progress"
    REVIEWING = "reviewing"
    WAITING_FOR_USER = "waiting_for_user"
    CONTINUING = "continuing"
    FINAL_VALIDATION = "final_validation"
    BLOCKED = "blocked"
    REWORK_REQUIRED = "rework_required"


#: The 11 canonical task states defined in the Brain Second specification.
CANONICAL_STATES: frozenset[TaskState] = frozenset(
    {
        TaskState.PENDING,
        TaskState.PLANNING,
        TaskState.PLANNED,
        TaskState.EXECUTING,
        TaskState.WAITING_APPROVAL,
        TaskState.WAITING_INPUT,
        TaskState.VERIFYING,
        TaskState.REPAIRING,
        TaskState.COMPLETED,
        TaskState.FAILED,
        TaskState.CANCELLED,
    }
)

#: Mapping from legacy/workflow states to canonical 11 states.
LEGACY_TO_CANONICAL: dict[TaskState, TaskState] = {
    TaskState.CREATED: TaskState.PENDING,
    TaskState.ANALYZING: TaskState.PLANNING,
    TaskState.PLANNED: TaskState.PLANNED,
    TaskState.DISPATCHED: TaskState.EXECUTING,
    TaskState.IN_PROGRESS: TaskState.EXECUTING,
    TaskState.REVIEWING: TaskState.VERIFYING,
    TaskState.WAITING_FOR_USER: TaskState.WAITING_APPROVAL,
    TaskState.CONTINUING: TaskState.EXECUTING,
    TaskState.FINAL_VALIDATION: TaskState.VERIFYING,
    TaskState.BLOCKED: TaskState.WAITING_INPUT,
    TaskState.REWORK_REQUIRED: TaskState.REPAIRING,
    TaskState.COMPLETED: TaskState.COMPLETED,
    TaskState.FAILED: TaskState.FAILED,
    TaskState.CANCELLED: TaskState.CANCELLED,
    TaskState.PENDING: TaskState.PENDING,
    TaskState.PLANNING: TaskState.PLANNING,
    TaskState.EXECUTING: TaskState.EXECUTING,
    TaskState.WAITING_APPROVAL: TaskState.WAITING_APPROVAL,
    TaskState.WAITING_INPUT: TaskState.WAITING_INPUT,
    TaskState.VERIFYING: TaskState.VERIFYING,
    TaskState.REPAIRING: TaskState.REPAIRING,
}


def to_canonical_state(state: TaskState | str) -> TaskState:
    """Convert any task state or string to its canonical 11-state enum."""
    if isinstance(state, str):
        try:
            state_enum = TaskState(state)
        except ValueError:
            state_enum = TaskState[state.upper()]
    else:
        state_enum = state
    return LEGACY_TO_CANONICAL.get(state_enum, state_enum)


#: States from which no further transition is allowed.
TERMINAL_STATES: frozenset[TaskState] = frozenset(
    {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED}
)


class AgentStatus(StrEnum):
    """Live execution status of an agent, derived from task lifecycle.

    Never set cosmetically — the orchestrator sets it as it moves an agent
    through real work so the UI reflects true state (spec §25).
    """

    IDLE = "idle"
    THINKING = "thinking"
    ANALYZING = "analyzing"
    WORKING = "working"
    WAITING = "waiting"
    BLOCKED = "blocked"
    REVIEWING = "reviewing"
    REPORTING = "reporting"
    COMPLETED = "completed"
    ERROR = "error"


class ReportStatus(StrEnum):
    """Status a specialist/department report may carry."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    REVIEW_REQUIRED = "review_required"
    COMPLETED = "completed"
    FAILED = "failed"


class ReviewDecision(StrEnum):
    """Outcome of a Braind executive review."""

    CONTINUE = "continue"
    REWORK = "rework"
    ASK_USER = "ask_user"
    COMPLETE = "complete"
    FAILED = "failed"


class KnowledgeType(StrEnum):
    """Provenance of a piece of information Braind may keep.

    Braind must not treat every agent output as absolute truth (spec §16);
    long-term memory records carry one of these so confidence is explicit.
    """

    FACT = "fact"
    OBSERVATION = "observation"
    INFERENCE = "inference"
    ASSUMPTION = "assumption"
    USER_CONFIRMED_FACT = "user_confirmed_fact"
    AGENT_REPORT = "agent_report"
    VERIFIED_DATA = "verified_data"


class AuditAction(StrEnum):
    """Auditable workflow events (spec §24)."""

    TASK_CREATED = "task_created"
    TASK_ASSIGNED = "task_assigned"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_BLOCKED = "task_blocked"
    TASK_REWORKED = "task_reworked"
    REPORT_CREATED = "report_created"
    REPORT_SUBMITTED = "report_submitted"
    BRAIND_REVIEW_STARTED = "braind_review_started"
    BRAIND_REVIEW_COMPLETED = "braind_review_completed"
    USER_CONFIRMATION_REQUIRED = "user_confirmation_required"
    USER_CONFIRMED = "user_confirmed"
    MEMORY_UPDATED = "memory_updated"
    FINAL_VALIDATION = "final_validation"
    ROOT_TASK_COMPLETED = "root_task_completed"
    STATE_TRANSITION = "state_transition"
    PERMISSION_DENIED = "permission_denied"


class Permission(StrEnum):
    """Discrete capabilities enforced at the orchestration layer (spec §37)."""

    ANALYZE_ROOT = "analyze_root"  # Only executive
    CREATE_DEPARTMENT_TASK = "create_department_task"  # Only executive
    ASSIGN_TO_LEADER = "assign_to_leader"  # Only executive
    FINALIZE_ROOT = "finalize_root"  # Only executive
    REQUEST_USER_CONFIRMATION = "request_user_confirmation"  # Only executive
    CREATE_SUBTASK = "create_subtask"  # Only leader
    ASSIGN_TO_SPECIALIST = "assign_to_specialist"  # Only leader
    CONSOLIDATE_DEPARTMENT = "consolidate_department"  # Only leader
    REQUEST_REWORK = "request_rework"  # Executive + leader
    EXECUTE_SUBTASK = "execute_subtask"  # Only specialist
    SUBMIT_REPORT = "submit_report"  # Leader + specialist


#: Fixed permission grants by level. Config-first: adding a department reuses
#: these grants via the agent's level, so no orchestration rewrite is needed.
LEVEL_PERMISSIONS: dict[AgentLevel, frozenset[Permission]] = {
    AgentLevel.EXECUTIVE: frozenset(
        {
            Permission.ANALYZE_ROOT,
            Permission.CREATE_DEPARTMENT_TASK,
            Permission.ASSIGN_TO_LEADER,
            Permission.FINALIZE_ROOT,
            Permission.REQUEST_USER_CONFIRMATION,
            Permission.REQUEST_REWORK,
        }
    ),
    AgentLevel.LEADER: frozenset(
        {
            Permission.CREATE_SUBTASK,
            Permission.ASSIGN_TO_SPECIALIST,
            Permission.CONSOLIDATE_DEPARTMENT,
            Permission.REQUEST_REWORK,
            Permission.SUBMIT_REPORT,
        }
    ),
    AgentLevel.SPECIALIST: frozenset(
        {
            Permission.EXECUTE_SUBTASK,
            Permission.SUBMIT_REPORT,
        }
    ),
}


def permissions_for(level: AgentLevel) -> frozenset[Permission]:
    """Return the permission set granted to an agent level."""
    return LEVEL_PERMISSIONS.get(level, frozenset())
