"""Structured data contracts exchanged in the workflow core (spec §35).

These dataclasses give every inter-agent message and report an explicit schema
so nothing flows through the orchestrator as free-form dict soup. They also
carry the full context chain (root/department/subtask/agent ids) so any record
can be traced back to the root task (spec §36).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from services.agents.domain import ReportStatus, ReviewDecision


@dataclass
class ContextChain:
    """Trace identifiers carried by every message so work is fully traceable."""

    root_task_id: str
    sender_agent_id: str
    receiver_agent_id: str
    department_task_id: str | None = None
    subtask_id: str | None = None
    parent_agent_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_task_id": self.root_task_id,
            "department_task_id": self.department_task_id,
            "subtask_id": self.subtask_id,
            "parent_agent_id": self.parent_agent_id,
            "sender_agent_id": self.sender_agent_id,
            "receiver_agent_id": self.receiver_agent_id,
        }


@dataclass
class AgentTaskMessage:
    """A task instruction handed down the hierarchy (Braind→Leader, Leader→Specialist)."""

    task_id: str
    context: ContextChain
    objective: str
    instructions: str
    expected_output: str
    acceptance_criteria: list[str] = field(default_factory=list)
    input_data: dict[str, Any] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    priority: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "context": self.context.to_dict(),
            "objective": self.objective,
            "instructions": self.instructions,
            "expected_output": self.expected_output,
            "acceptance_criteria": self.acceptance_criteria,
            "input_data": self.input_data,
            "constraints": self.constraints,
            "dependencies": self.dependencies,
            "priority": self.priority,
        }


@dataclass
class SpecialistReport:
    """Report a specialist returns to its leader (spec §12)."""

    task_id: str
    agent_id: str
    work_performed: str
    status: ReportStatus
    data_used: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    results: str = ""
    output_artifacts: list[dict[str, Any]] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    confidence: float = 0.5
    recommendation: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "work_performed": self.work_performed,
            "status": self.status.value,
            "data_used": self.data_used,
            "findings": self.findings,
            "results": self.results,
            "output_artifacts": self.output_artifacts,
            "assumptions": self.assumptions,
            "problems": self.problems,
            "missing_information": self.missing_information,
            "confidence": self.confidence,
            "recommendation": self.recommendation,
            "citations": self.citations,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpecialistReport:
        status = data.get("status", "completed")
        return cls(
            task_id=data["task_id"],
            agent_id=data["agent_id"],
            work_performed=data.get("work_performed", ""),
            status=ReportStatus(status) if isinstance(status, str) else status,
            data_used=data.get("data_used", []),
            findings=data.get("findings", []),
            results=data.get("results", ""),
            output_artifacts=data.get("output_artifacts", []),
            assumptions=data.get("assumptions", []),
            problems=data.get("problems", []),
            missing_information=data.get("missing_information", []),
            confidence=data.get("confidence", 0.5),
            recommendation=data.get("recommendation", ""),
            citations=data.get("citations", []),
        )


@dataclass
class DepartmentReport:
    """Consolidated report a leader submits to Braind (spec §14)."""

    task_id: str
    department: str
    leader_id: str
    objective: str
    status: ReportStatus
    work_breakdown: list[str] = field(default_factory=list)
    specialists_involved: list[str] = field(default_factory=list)
    work_completed: list[str] = field(default_factory=list)
    key_findings: list[str] = field(default_factory=list)
    data_used: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    missing_data: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    final_result: str = ""
    supporting_artifacts: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.5
    recommendation: str = ""
    specialist_reports: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "department": self.department,
            "leader_id": self.leader_id,
            "objective": self.objective,
            "status": self.status.value,
            "work_breakdown": self.work_breakdown,
            "specialists_involved": self.specialists_involved,
            "work_completed": self.work_completed,
            "key_findings": self.key_findings,
            "data_used": self.data_used,
            "problems": self.problems,
            "missing_data": self.missing_data,
            "conflicts": self.conflicts,
            "risks": self.risks,
            "final_result": self.final_result,
            "supporting_artifacts": self.supporting_artifacts,
            "confidence": self.confidence,
            "recommendation": self.recommendation,
            "specialist_reports": self.specialist_reports,
            "citations": self.citations,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DepartmentReport:
        status = data.get("status", "completed")
        return cls(
            task_id=data["task_id"],
            department=data.get("department", "unknown"),
            leader_id=data.get("leader_id", ""),
            objective=data.get("objective", ""),
            status=ReportStatus(status) if isinstance(status, str) else status,
            work_breakdown=data.get("work_breakdown", []),
            specialists_involved=data.get("specialists_involved", []),
            work_completed=data.get("work_completed", []),
            key_findings=data.get("key_findings", []),
            data_used=data.get("data_used", []),
            problems=data.get("problems", []),
            missing_data=data.get("missing_data", []),
            conflicts=data.get("conflicts", []),
            risks=data.get("risks", []),
            final_result=data.get("final_result", ""),
            supporting_artifacts=data.get("supporting_artifacts", []),
            confidence=data.get("confidence", 0.5),
            recommendation=data.get("recommendation", ""),
            specialist_reports=data.get("specialist_reports", []),
            citations=data.get("citations", []),
        )


@dataclass
class BrainReviewMessage:
    """Result of Braind's executive review of department reports (spec §15)."""

    root_task_id: str
    decision: ReviewDecision
    requirement_coverage: dict[str, bool] = field(default_factory=dict)
    consistency_issues: list[str] = field(default_factory=list)
    data_quality: dict[str, list[str]] = field(default_factory=dict)
    outstanding_items: list[str] = field(default_factory=list)
    reason: str = ""
    approval_question: str | None = None
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_task_id": self.root_task_id,
            "decision": self.decision.value,
            "requirement_coverage": self.requirement_coverage,
            "consistency_issues": self.consistency_issues,
            "data_quality": self.data_quality,
            "outstanding_items": self.outstanding_items,
            "reason": self.reason,
            "approval_question": self.approval_question,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class UserConfirmationMessage:
    """A user's answer to a Braind confirmation gate (spec §18)."""

    root_task_id: str
    request_id: str
    decision: str
    notes: str | None = None
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_task_id": self.root_task_id,
            "request_id": self.request_id,
            "decision": self.decision,
            "notes": self.notes,
            "created_at": self.created_at.isoformat(),
        }
