"""Persistence layer for the internal hierarchical agent team.

This is the single source of truth (spec §34): every task, report, audit event,
durable run checkpoint and memory record is read and written here. State
changes go through the state machine, and each significant mutation writes an
audit event, so the stored data always reflects real execution — chat, the
workflow API and every other braind capability read from these rows; nothing
is simulated and there is no separate office screen to render.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import text

from core.logging import get_logger
from services.agents.contracts import DepartmentReport, SpecialistReport
from services.agents.domain import (
    AuditAction,
    KnowledgeType,
    TaskState,
)
from services.agents.models import AgentTask, TaskStatus
from services.agents.state_machine import assert_transition

logger = get_logger(__name__)


# Map the coarse legacy TaskStatus (kept for the existing `status` column and
# older API consumers) onto the fine-grained lifecycle state.
_STATE_TO_STATUS: dict[TaskState, TaskStatus] = {
    TaskState.CREATED: TaskStatus.PENDING,
    TaskState.ANALYZING: TaskStatus.IN_PROGRESS,
    TaskState.PLANNED: TaskStatus.IN_PROGRESS,
    TaskState.DISPATCHED: TaskStatus.IN_PROGRESS,
    TaskState.IN_PROGRESS: TaskStatus.IN_PROGRESS,
    TaskState.REVIEWING: TaskStatus.IN_PROGRESS,
    TaskState.WAITING_FOR_USER: TaskStatus.IN_PROGRESS,
    TaskState.CONTINUING: TaskStatus.IN_PROGRESS,
    TaskState.FINAL_VALIDATION: TaskStatus.IN_PROGRESS,
    TaskState.REWORK_REQUIRED: TaskStatus.IN_PROGRESS,
    TaskState.BLOCKED: TaskStatus.IN_PROGRESS,
    TaskState.COMPLETED: TaskStatus.COMPLETED,
    TaskState.FAILED: TaskStatus.FAILED,
}


def _dumps(value: Any) -> str | None:
    return json.dumps(value, ensure_ascii=False) if value is not None else None


def _loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return default
    return value


class AgentRepository:
    """Data access for tasks, reports, audit events and memory."""

    def __init__(self, session_factory):
        self.session_factory = session_factory

    # ------------------------------------------------------------------ tasks
    async def create_task(self, task: AgentTask) -> AgentTask:
        """Insert a new task and emit a TASK_CREATED audit event."""
        task.status = _STATE_TO_STATUS.get(task.state_enum, TaskStatus.PENDING)
        async with self.session_factory() as session:
            await session.execute(
                text(
                    """INSERT INTO agent_tasks (
                        task_id, parent_task_id, root_task_id, user_request,
                        task_description, assigned_to, assigned_by, status, state,
                        task_type, department, priority, objective, instructions,
                        expected_output, required_output, acceptance_criteria,
                        input_data, constraints, dependencies, confirmation_required,
                        rework_count
                    ) VALUES (
                        :task_id, :parent_task_id, :root_task_id, :user_request,
                        :task_description, :assigned_to, :assigned_by, :status, :state,
                        :task_type, :department, :priority, :objective, :instructions,
                        :expected_output, :required_output, :acceptance_criteria,
                        :input_data, :constraints, :dependencies, :confirmation_required,
                        :rework_count
                    )"""
                ),
                {
                    "task_id": task.task_id,
                    "parent_task_id": task.parent_task_id,
                    "root_task_id": task.root_task_id,
                    "user_request": task.user_request,
                    "task_description": task.task_description,
                    "assigned_to": task.assigned_to,
                    "assigned_by": task.assigned_by,
                    "status": task.status.value,
                    "state": task.state,
                    "task_type": task.task_type,
                    "department": task.department,
                    "priority": task.priority,
                    "objective": task.objective,
                    "instructions": task.instructions,
                    "expected_output": task.expected_output,
                    "required_output": task.required_output,
                    "acceptance_criteria": _dumps(task.acceptance_criteria),
                    "input_data": _dumps(task.input_data),
                    "constraints": _dumps(task.constraints),
                    "dependencies": _dumps(task.dependencies),
                    "confirmation_required": task.confirmation_required,
                    "rework_count": task.rework_count,
                },
            )
            await session.commit()

        await self.audit(
            AuditAction.TASK_CREATED,
            actor=task.assigned_by or "user",
            task_id=task.task_id,
            parent_task_id=task.parent_task_id,
            root_task_id=task.root_task_id,
            new_status=task.state,
            reason=f"{task.task_type} task created for {task.assigned_to}",
        )
        return task

    async def get_task(self, task_id: str) -> AgentTask | None:
        async with self.session_factory() as session:
            result = await session.execute(
                text("SELECT * FROM agent_tasks WHERE task_id = :task_id"),
                {"task_id": task_id},
            )
            row = result.fetchone()
            if not row:
                return None
            return self._row_to_task(dict(row._mapping))

    async def get_children(self, parent_task_id: str) -> list[AgentTask]:
        async with self.session_factory() as session:
            result = await session.execute(
                text(
                    "SELECT * FROM agent_tasks WHERE parent_task_id = :pid ORDER BY created_at ASC"
                ),
                {"pid": parent_task_id},
            )
            return [self._row_to_task(dict(r._mapping)) for r in result.fetchall()]

    async def get_tree(self, root_task_id: str) -> list[AgentTask]:
        """Return every task in a root's tree (root included)."""
        async with self.session_factory() as session:
            result = await session.execute(
                text(
                    """SELECT * FROM agent_tasks
                       WHERE root_task_id = :rid OR task_id = :rid
                       ORDER BY created_at ASC"""
                ),
                {"rid": root_task_id},
            )
            return [self._row_to_task(dict(r._mapping)) for r in result.fetchall()]

    async def transition(
        self,
        task: AgentTask,
        target: TaskState,
        actor: str,
        reason: str | None = None,
        audit_action: AuditAction | None = None,
    ) -> AgentTask:
        """Move a task to ``target`` after validating the transition.

        Raises ``IllegalTransition`` (from the state machine) on an illegal
        move, so no caller can corrupt the lifecycle. Writes the new state plus
        the derived coarse status, and emits an audit event.
        """
        current = task.state_enum
        assert_transition(task.task_type_enum, current, target)

        new_status = _STATE_TO_STATUS.get(target, TaskStatus.IN_PROGRESS)
        now = datetime.now()
        started = now if target == TaskState.IN_PROGRESS and task.started_at is None else None
        completed = now if target in (TaskState.COMPLETED, TaskState.FAILED) else None

        sets = ["state = :state", "status = :status"]
        params: dict[str, Any] = {
            "task_id": task.task_id,
            "state": target.value,
            "status": new_status.value,
        }
        if started is not None:
            sets.append("started_at = :started_at")
            params["started_at"] = started
        if completed is not None:
            sets.append("completed_at = :completed_at")
            params["completed_at"] = completed

        async with self.session_factory() as session:
            await session.execute(
                text(f"UPDATE agent_tasks SET {', '.join(sets)} WHERE task_id = :task_id"),
                params,
            )
            await session.commit()

        task.state = target.value
        task.status = new_status
        if started:
            task.started_at = started
        if completed:
            task.completed_at = completed

        await self.audit(
            audit_action or AuditAction.STATE_TRANSITION,
            actor=actor,
            task_id=task.task_id,
            parent_task_id=task.parent_task_id,
            root_task_id=task.root_task_id or task.task_id,
            previous_status=current.value,
            new_status=target.value,
            reason=reason,
        )
        return task

    async def set_result(self, task_id: str, result: dict[str, Any]) -> None:
        async with self.session_factory() as session:
            await session.execute(
                text("UPDATE agent_tasks SET result = :result WHERE task_id = :task_id"),
                {"task_id": task_id, "result": _dumps(result)},
            )
            await session.commit()

    async def set_error(self, task_id: str, error: str) -> None:
        async with self.session_factory() as session:
            await session.execute(
                text("UPDATE agent_tasks SET error = :error WHERE task_id = :task_id"),
                {"task_id": task_id, "error": error},
            )
            await session.commit()

    async def increment_rework(self, task_id: str) -> None:
        async with self.session_factory() as session:
            await session.execute(
                text(
                    "UPDATE agent_tasks SET rework_count = rework_count + 1 "
                    "WHERE task_id = :task_id"
                ),
                {"task_id": task_id},
            )
            await session.commit()

    def _row_to_task(self, row: dict[str, Any]) -> AgentTask:
        row = dict(row)
        row["acceptance_criteria"] = _loads(row.get("acceptance_criteria"), [])
        row["constraints"] = _loads(row.get("constraints"), [])
        row["dependencies"] = _loads(row.get("dependencies"), [])
        row["input_data"] = _loads(row.get("input_data"), {})
        row["result"] = _loads(row.get("result"), None)
        return AgentTask.from_dict(row)

    # ---------------------------------------------------------------- reports
    async def save_specialist_report(
        self, report: SpecialistReport, root_task_id: str, department_task_id: str
    ) -> str:
        report_id = str(uuid.uuid4())
        await self._insert_report(
            report_id,
            task_id=report.task_id,
            root_task_id=root_task_id,
            department_task_id=department_task_id,
            agent_id=report.agent_id,
            report_type="specialist",
            status=report.status.value,
            confidence=report.confidence,
            payload=report.to_dict(),
        )
        return report_id

    async def save_department_report(self, report: DepartmentReport, root_task_id: str) -> str:
        report_id = str(uuid.uuid4())
        await self._insert_report(
            report_id,
            task_id=report.task_id,
            root_task_id=root_task_id,
            department_task_id=report.task_id,
            agent_id=report.leader_id,
            report_type="department",
            status=report.status.value,
            confidence=report.confidence,
            payload=report.to_dict(),
        )
        return report_id

    async def _insert_report(
        self,
        report_id: str,
        *,
        task_id: str,
        root_task_id: str,
        department_task_id: str | None,
        agent_id: str,
        report_type: str,
        status: str,
        confidence: float,
        payload: dict[str, Any],
    ) -> None:
        async with self.session_factory() as session:
            await session.execute(
                text(
                    """INSERT INTO agent_reports (
                        report_id, task_id, root_task_id, department_task_id,
                        agent_id, report_type, status, confidence, payload
                    ) VALUES (
                        :report_id, :task_id, :root_task_id, :department_task_id,
                        :agent_id, :report_type, :status, :confidence, :payload
                    )"""
                ),
                {
                    "report_id": report_id,
                    "task_id": task_id,
                    "root_task_id": root_task_id,
                    "department_task_id": department_task_id,
                    "agent_id": agent_id,
                    "report_type": report_type,
                    "status": status,
                    "confidence": confidence,
                    "payload": _dumps(payload),
                },
            )
            await session.commit()
        await self.audit(
            AuditAction.REPORT_SUBMITTED,
            actor=agent_id,
            task_id=task_id,
            root_task_id=root_task_id,
            reason=f"{report_type} report submitted",
        )

    async def get_reports_for_root(self, root_task_id: str) -> list[dict[str, Any]]:
        async with self.session_factory() as session:
            result = await session.execute(
                text(
                    """SELECT * FROM agent_reports WHERE root_task_id = :rid
                       ORDER BY created_at ASC"""
                ),
                {"rid": root_task_id},
            )
            out = []
            for r in result.fetchall():
                d = dict(r._mapping)
                d["payload"] = _loads(d.get("payload"), {})
                if d.get("created_at"):
                    d["created_at"] = d["created_at"].isoformat()
                out.append(d)
            return out

    async def get_department_reports(self, root_task_id: str) -> list[DepartmentReport]:
        reports = await self.get_reports_for_root(root_task_id)
        return [
            DepartmentReport.from_dict(r["payload"])
            for r in reports
            if r.get("report_type") == "department"
        ]

    # ----------------------------------------------------------------- audit
    async def audit(
        self,
        action: AuditAction,
        *,
        actor: str | None = None,
        task_id: str | None = None,
        parent_task_id: str | None = None,
        root_task_id: str | None = None,
        previous_status: str | None = None,
        new_status: str | None = None,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        async with self.session_factory() as session:
            await session.execute(
                text(
                    """INSERT INTO agent_audit_events (
                        event_id, action, actor, task_id, parent_task_id, root_task_id,
                        previous_status, new_status, reason, event_metadata
                    ) VALUES (
                        :event_id, :action, :actor, :task_id, :parent_task_id, :root_task_id,
                        :previous_status, :new_status, :reason, :metadata
                    )"""
                ),
                {
                    "event_id": str(uuid.uuid4()),
                    "action": action.value,
                    "actor": actor,
                    "task_id": task_id,
                    "parent_task_id": parent_task_id,
                    "root_task_id": root_task_id,
                    "previous_status": previous_status,
                    "new_status": new_status,
                    "reason": reason,
                    "metadata": _dumps(metadata),
                },
            )
            await session.commit()

    async def get_audit_trail(self, root_task_id: str, limit: int = 500) -> list[dict[str, Any]]:
        async with self.session_factory() as session:
            result = await session.execute(
                text(
                    """SELECT * FROM agent_audit_events
                       WHERE root_task_id = :rid OR task_id = :rid
                       ORDER BY created_at ASC LIMIT :limit"""
                ),
                {"rid": root_task_id, "limit": limit},
            )
            out = []
            for r in result.fetchall():
                d = dict(r._mapping)
                d["event_metadata"] = _loads(d.get("event_metadata"), None)
                if d.get("created_at"):
                    d["created_at"] = d["created_at"].isoformat()
                out.append(d)
            return out

    # ---------------------------------------------------------------- memory
    async def save_memory(
        self,
        *,
        knowledge_type: KnowledgeType,
        content: str,
        root_task_id: str | None = None,
        task_id: str | None = None,
        department: str | None = None,
        agent_id: str | None = None,
        source: str | None = None,
        confidence: float = 0.5,
        evidence: list[str] | None = None,
    ) -> str:
        memory_id = str(uuid.uuid4())
        async with self.session_factory() as session:
            await session.execute(
                text(
                    """INSERT INTO agent_memory (
                        memory_id, root_task_id, task_id, department, agent_id,
                        knowledge_type, content, source, confidence, evidence
                    ) VALUES (
                        :memory_id, :root_task_id, :task_id, :department, :agent_id,
                        :knowledge_type, :content, :source, :confidence, :evidence
                    )"""
                ),
                {
                    "memory_id": memory_id,
                    "root_task_id": root_task_id,
                    "task_id": task_id,
                    "department": department,
                    "agent_id": agent_id,
                    "knowledge_type": knowledge_type.value,
                    "content": content,
                    "source": source,
                    "confidence": confidence,
                    "evidence": _dumps(evidence or []),
                },
            )
            await session.commit()
        await self.audit(
            AuditAction.MEMORY_UPDATED,
            actor=agent_id or "ceo",
            task_id=task_id,
            root_task_id=root_task_id,
            reason=f"memory[{knowledge_type.value}] stored",
        )
        return memory_id
