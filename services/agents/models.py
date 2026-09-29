"""Data models for multi-agent system."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from services.agents.domain import AgentLevel, TaskState, TaskType


class AgentRole(StrEnum):
    """Agent role types."""

    CEO = "CEO"
    LEADER = "LEADER"
    SPECIALIST = "SPECIALIST"


class TaskStatus(StrEnum):
    """Task status types."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentProfile:
    """Agent profile with capabilities and metadata."""

    agent_id: str
    name: str
    role: AgentRole
    department: str
    specialization: str | None = None

    # Visual
    avatar: str = "👤"
    color: str = "#6B7280"  # Gray default

    # AI Behavior
    system_prompt: str | None = None
    temperature: float = 0.7

    # Capabilities
    can_delegate: bool = False
    can_use_tools: bool = False
    available_tools: list[str] = field(default_factory=list)

    # Hierarchy metadata (config-first routing — see services/agents/domain.py)
    level: str | None = None  # executive | leader | specialist; derived from role if None
    parent_agent: str | None = None  # agent_id of the supervising agent
    capabilities: list[str] = field(default_factory=list)  # domain skills for capability routing

    # UI Position
    position: dict[str, int] = field(default_factory=dict)  # {"x": 100, "y": 200}

    # Status
    status: str = "active"  # active, busy, offline
    current_task: str | None = None  # task_id the agent is actively working
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def agent_level(self) -> AgentLevel:
        """Resolve the agent's hierarchy level.

        Uses the explicit ``level`` column when present, otherwise derives it
        from ``role`` so pre-existing seeded rows keep working.
        """
        from services.agents.domain import AgentLevel

        if self.level:
            return AgentLevel(self.level)
        return {
            AgentRole.CEO: AgentLevel.EXECUTIVE,
            AgentRole.LEADER: AgentLevel.LEADER,
            AgentRole.SPECIALIST: AgentLevel.SPECIALIST,
        }[self.role]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role.value if isinstance(self.role, AgentRole) else self.role,
            "department": self.department,
            "specialization": self.specialization,
            "avatar": self.avatar,
            "color": self.color,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "can_delegate": self.can_delegate,
            "can_use_tools": self.can_use_tools,
            "available_tools": self.available_tools,
            "level": self.agent_level.value,
            "parent_agent": self.parent_agent,
            "capabilities": self.capabilities,
            "position": self.position,
            "status": self.status,
            "current_task": self.current_task,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentProfile:
        """Create from dictionary."""
        # Convert string role to enum
        if isinstance(data.get("role"), str):
            data["role"] = AgentRole(data["role"])

        # Convert ISO strings to datetime
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])

        # Keep only recognised fields so extra DB columns don't break construction.
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class AgentTask:
    """Task assigned to an agent."""

    task_id: str
    task_description: str
    assigned_to: str  # agent_id
    assigned_by: str | None = None  # agent_id
    parent_task_id: str | None = None  # For sub-tasks
    user_request: str | None = None  # Original user request (only for root tasks)

    status: TaskStatus = TaskStatus.PENDING
    priority: int = 1  # 1=low, 2=medium, 3=high, 4=urgent

    # Hierarchy + lifecycle (see services/agents/domain.py + state_machine.py)
    task_type: str = "subtask"  # root | department | subtask
    state: str = "created"  # TaskState value; the authoritative lifecycle marker
    department: str | None = None
    root_task_id: str | None = None
    objective: str | None = None
    instructions: str | None = None
    expected_output: str | None = None
    required_output: str | None = None
    acceptance_criteria: list[str] = field(default_factory=list)
    input_data: dict[str, Any] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    confirmation_required: bool = False
    rework_count: int = 0

    result: dict[str, Any] | None = None
    error: str | None = None

    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def task_type_enum(self) -> TaskType:
        from services.agents.domain import TaskType

        return TaskType(self.task_type)

    @property
    def state_enum(self) -> TaskState:
        from services.agents.domain import TaskState

        return TaskState(self.state)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "assigned_to": self.assigned_to,
            "assigned_by": self.assigned_by,
            "parent_task_id": self.parent_task_id,
            "user_request": self.user_request,
            "status": self.status.value if isinstance(self.status, TaskStatus) else self.status,
            "priority": self.priority,
            "task_type": self.task_type,
            "state": self.state,
            "department": self.department,
            "root_task_id": self.root_task_id,
            "objective": self.objective,
            "instructions": self.instructions,
            "expected_output": self.expected_output,
            "required_output": self.required_output,
            "acceptance_criteria": self.acceptance_criteria,
            "input_data": self.input_data,
            "constraints": self.constraints,
            "dependencies": self.dependencies,
            "confirmation_required": self.confirmation_required,
            "rework_count": self.rework_count,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentTask:
        """Create from dictionary."""
        data = dict(data)
        # Convert string status to enum
        if isinstance(data.get("status"), str):
            data["status"] = TaskStatus(data["status"])

        # Convert ISO strings to datetime
        for field_name in ["created_at", "started_at", "completed_at"]:
            if isinstance(data.get(field_name), str):
                data[field_name] = datetime.fromisoformat(data[field_name])

        # Normalise JSON columns that may arrive as None from the DB.
        for json_field in ["acceptance_criteria", "constraints", "dependencies"]:
            if data.get(json_field) is None and json_field in data:
                data[json_field] = []
        if data.get("input_data") is None and "input_data" in data:
            data["input_data"] = {}

        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

