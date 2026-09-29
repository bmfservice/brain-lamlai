"""Specialist worker — the executor at the bottom of the hierarchy (spec §11-12).

A specialist receives a subtask from its leader, executes it within its own
domain using its available tools, and returns a structured SpecialistReport. It
cannot delegate, cannot talk to the user, cannot bypass its leader, and cannot
declare the overall project done — those are enforced structurally by the
permission layer and by the fact that only the leader/engine advance parent
tasks. Every state change is audited.
"""

from __future__ import annotations

import contextlib
from typing import Any

from core.logging import get_logger
from services.agents.contracts import SpecialistReport
from services.agents.domain import (
    AuditAction,
    Permission,
    ReportStatus,
    TaskState,
)
from services.agents.llm_json import extract_json
from services.agents.models import AgentProfile, AgentTask
from services.agents.permissions import require
from services.agents.repository import AgentRepository

logger = get_logger(__name__)


class SpecialistWorker:
    """Executes one subtask and produces a structured report."""

    def __init__(
        self,
        repo: AgentRepository,
        vllm,
        tool_registry,
        profile: AgentProfile,
        agent_registry=None,
    ):
        self.repo = repo
        self.vllm = vllm
        self.tool_registry = tool_registry
        self.profile = profile
        self.agent_registry = agent_registry

    async def execute(
        self, root: AgentTask, dept_task: AgentTask, subtask: AgentTask,
        rework_note: str | None = None,
    ) -> SpecialistReport:
        require(self.profile.agent_level, Permission.EXECUTE_SUBTASK)
        logger.info("specialist.execute", agent=self.profile.name, task=subtask.task_id)
        await self.agent_registry_status("busy")

        await self.repo.transition(
            subtask, TaskState.IN_PROGRESS, actor=self.profile.agent_id,
            reason="specialist executing", audit_action=AuditAction.TASK_STARTED,
        )

        try:
            work = await self._do_work(subtask, rework_note)
            report = self._to_report(subtask, work)
        except Exception as exc:  # noqa: BLE001
            logger.error("specialist.failed", agent=self.profile.name, error=str(exc))
            await self.repo.set_error(subtask.task_id, str(exc))
            await self.repo.transition(
                subtask, TaskState.FAILED, actor=self.profile.agent_id,
                reason=f"execution error: {exc}", audit_action=AuditAction.TASK_FAILED,
            )
            await self.agent_registry_status("active")
            return SpecialistReport(
                task_id=subtask.task_id, agent_id=self.profile.agent_id,
                work_performed="", status=ReportStatus.FAILED, problems=[str(exc)], confidence=0.0,
            )

        # Persist the report and set the subtask's terminal-ish state. The leader
        # (not the specialist) decides completion vs rework, but the specialist
        # records its own honest status.
        require(self.profile.agent_level, Permission.SUBMIT_REPORT)
        await self.repo.save_specialist_report(report, root.task_id, dept_task.task_id)
        await self.repo.set_result(subtask.task_id, report.to_dict())

        target = {
            ReportStatus.COMPLETED: TaskState.COMPLETED,
            ReportStatus.REVIEW_REQUIRED: TaskState.REVIEWING,
            ReportStatus.BLOCKED: TaskState.BLOCKED,
            ReportStatus.FAILED: TaskState.FAILED,
        }.get(report.status, TaskState.REVIEWING)
        audit = {
            TaskState.COMPLETED: AuditAction.TASK_COMPLETED,
            TaskState.BLOCKED: AuditAction.TASK_BLOCKED,
            TaskState.FAILED: AuditAction.TASK_FAILED,
        }.get(target, AuditAction.REPORT_SUBMITTED)
        await self.repo.transition(
            subtask, target, actor=self.profile.agent_id,
            reason=f"report status={report.status.value}", audit_action=audit,
        )

        await self.agent_registry_status("active")
        return report

    async def _do_work(self, subtask: AgentTask, rework_note: str | None) -> dict[str, Any]:
        criteria = "; ".join(subtask.acceptance_criteria) or "produce a usable result"
        tools = ", ".join(self.profile.available_tools) or "none"
        rework = f"\n\nIMPORTANT — this is a rework attempt. {rework_note}\n" if rework_note else ""
        prompt = f"""You are {self.profile.name}, a {self.profile.specialization} specialist.

Do the assigned subtask within your domain. Be concrete and produce the output.
{rework}
Objective: {subtask.objective}
Instructions: {subtask.instructions}
Expected output: {subtask.expected_output}
Acceptance criteria: {criteria}
Available tools: {tools}

Respond ONLY with JSON:
{{
  "work_performed": "what you did",
  "results": "the actual output/deliverable text",
  "findings": ["..."],
  "data_used": ["..."],
  "assumptions": ["..."],
  "problems": ["..."],
  "missing_information": ["..."],
  "confidence": 0.0-1.0,
  "recommendation": "next step suggestion",
  "status": "completed|blocked|review_required"
}}"""
        raw = await self._call_llm(prompt)
        parsed = extract_json(raw)
        if not isinstance(parsed, dict):
            # Model returned prose; wrap it honestly rather than pretend structure.
            return {
                "work_performed": "produced a free-form response",
                "results": raw.strip(),
                "confidence": 0.4,
                "status": "review_required",
            }
        return parsed

    def _to_report(self, subtask: AgentTask, work: dict[str, Any]) -> SpecialistReport:
        raw_status = str(work.get("status", "completed")).lower()
        try:
            status = ReportStatus(raw_status)
        except ValueError:
            status = ReportStatus.COMPLETED
        # A specialist must not self-certify against criteria it clearly missed:
        # if it flagged missing info, downgrade to review_required for the leader.
        if status == ReportStatus.COMPLETED and work.get("missing_information"):
            status = ReportStatus.REVIEW_REQUIRED
        return SpecialistReport(
            task_id=subtask.task_id,
            agent_id=self.profile.agent_id,
            work_performed=work.get("work_performed", ""),
            status=status,
            data_used=work.get("data_used", []),
            findings=work.get("findings", []),
            results=work.get("results", ""),
            output_artifacts=work.get("output_artifacts", []),
            assumptions=work.get("assumptions", []),
            problems=work.get("problems", []),
            missing_information=work.get("missing_information", []),
            confidence=float(work.get("confidence", 0.5) or 0.5),
            recommendation=work.get("recommendation", ""),
        )

    async def agent_registry_status(self, status: str) -> None:
        # Best-effort UI status; failures here must not break execution.
        if not self.agent_registry:
            return
        with contextlib.suppress(Exception):
            await self.agent_registry.update_agent_status(self.profile.agent_id, status)

    async def _call_llm(self, prompt: str) -> str:
        result = await self.vllm.chat(
            messages=[{"role": "user", "content": prompt}],
            system=self.profile.system_prompt or "",
            temperature=self.profile.temperature,
            max_tokens=1024,
        )
        return str(result)
