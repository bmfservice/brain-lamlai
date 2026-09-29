"""Department leader — tactical orchestrator for one department (spec §10-14).

The leader does NOT forward the department task verbatim to a specialist. It:

1. analyses the task (LEADER ANALYSIS),
2. produces a SUBTASK PLAN of concrete, executable instructions,
3. assigns each subtask to a capable specialist (permission-guarded),
4. runs the specialists and collects structured reports,
5. reviews each report against its acceptance criteria and issues rework when a
   subtask falls short,
6. consolidates everything into a DEPARTMENT REPORT and submits it to Braind.

All state changes go through the repository/state-machine and are audited.
"""

from __future__ import annotations

import uuid
from typing import Any

from core.logging import get_logger
from services.agents.contracts import DepartmentReport, SpecialistReport
from services.agents.domain import (
    AgentLevel,
    AuditAction,
    Permission,
    ReportStatus,
    TaskState,
    TaskType,
)
from services.agents.llm_json import extract_json
from services.agents.models import AgentProfile, AgentTask
from services.agents.permissions import require, require_can_assign
from services.agents.repository import AgentRepository
from services.agents.worker import SpecialistWorker

logger = get_logger(__name__)

MAX_SUBTASK_REWORK = 1


class DepartmentLeader:
    """Runs one department task end to end and reports to Braind."""

    def __init__(
        self, repo: AgentRepository, vllm, tool_registry, agent_registry, profile: AgentProfile
    ):
        self.repo = repo
        self.vllm = vllm
        self.tool_registry = tool_registry
        self.agent_registry = agent_registry
        self.profile = profile

    async def run(self, root: AgentTask, dept_task: AgentTask) -> DepartmentReport:
        logger.info("leader.run", leader=self.profile.name, dept=dept_task.department)
        await self.agent_registry.update_agent_status(self.profile.agent_id, "busy")

        # STAGE 4 — analyse + build subtask plan.
        await self.repo.transition(
            dept_task, TaskState.ANALYZING, actor=self.profile.agent_id,
            reason="leader analysing department task", audit_action=AuditAction.TASK_STARTED,
        )
        team = await self._team()
        plan = await self._plan_subtasks(dept_task, team)
        await self.repo.transition(
            dept_task, TaskState.PLANNED, actor=self.profile.agent_id, reason="subtask plan ready"
        )

        # STAGE 5 — create + assign specialist tasks (permission-guarded).
        subtasks = await self._create_subtasks(root, dept_task, plan, team)
        await self.repo.transition(
            dept_task, TaskState.DISPATCHED, actor=self.profile.agent_id,
            reason="subtasks dispatched",
        )
        await self.repo.transition(
            dept_task, TaskState.IN_PROGRESS, actor=self.profile.agent_id,
            reason="specialists working",
        )

        # STAGE 6-7 — execute specialists, collect reports, review + rework.
        reports = await self._execute_subtasks(root, dept_task, subtasks)

        # STAGE 8 — consolidate into a department report.
        await self.repo.transition(
            dept_task, TaskState.REVIEWING, actor=self.profile.agent_id,
            reason="leader consolidating",
        )
        dept_report = await self._consolidate(root, dept_task, plan, reports)

        # A department with an unresolved blocked/failed subtask is reported as
        # blocked, not completed — Braind decides the next step (spec §29).
        final_state = TaskState.COMPLETED
        if dept_report.status in (ReportStatus.BLOCKED, ReportStatus.FAILED):
            final_state = (
                TaskState.BLOCKED
                if dept_report.status == ReportStatus.BLOCKED
                else TaskState.FAILED
            )

        require(self.profile.agent_level, Permission.CONSOLIDATE_DEPARTMENT)
        await self.repo.save_department_report(dept_report, root.task_id)
        await self.repo.set_result(dept_task.task_id, dept_report.to_dict())

        if final_state == TaskState.COMPLETED:
            await self.repo.transition(
                dept_task, TaskState.COMPLETED, actor=self.profile.agent_id,
                reason="department report submitted", audit_action=AuditAction.TASK_COMPLETED,
            )
        elif final_state == TaskState.BLOCKED:
            await self.repo.transition(
                dept_task, TaskState.BLOCKED, actor=self.profile.agent_id,
                reason="department blocked", audit_action=AuditAction.TASK_BLOCKED,
            )
        else:
            await self.repo.transition(
                dept_task, TaskState.FAILED, actor=self.profile.agent_id,
                reason="department failed", audit_action=AuditAction.TASK_FAILED,
            )

        await self.agent_registry.update_agent_status(self.profile.agent_id, "active")
        return dept_report

    # -------------------------------------------------------------- team + plan
    async def _team(self) -> list[AgentProfile]:
        agents = await self.agent_registry.get_agents_by_department(self.profile.department)
        return [
            a for a in agents
            if a.agent_level == AgentLevel.SPECIALIST and a.agent_id != self.profile.agent_id
        ]

    async def _plan_subtasks(
        self, dept_task: AgentTask, team: list[AgentProfile]
    ) -> dict[str, Any]:
        team_desc = "\n".join(
            f"- id={a.agent_id} name={a.name} specialization={a.specialization} "
            f"capabilities={a.capabilities or a.available_tools}"
            for a in team
        )
        criteria = "; ".join(dept_task.acceptance_criteria) or "produce the expected output"
        prompt = f"""You are {self.profile.name}, the {self.profile.department} department leader.

Analyse this department task and break it into concrete, executable subtasks —
one per specialist. Do NOT write vague instructions like "help with marketing".
Each subtask must state exactly what to analyse, what data to use, what output to
produce, and how completion is judged.

Department objective: {dept_task.objective or dept_task.task_description}
Expected output: {dept_task.expected_output or 'a clear deliverable'}
Acceptance criteria: {criteria}

Your specialists:
{team_desc or '(none configured)'}

Respond ONLY with JSON:
{{
  "analysis": "short analysis of what the task truly requires",
  "subtasks": [
    {{
      "assigned_to": "<specialist agent_id from the list>",
      "objective": "what this specialist must achieve",
      "instructions": "step-by-step, specific instructions",
      "expected_output": "the concrete artifact/result",
      "acceptance_criteria": ["criterion 1", "criterion 2"]
    }}
  ]
}}"""
        raw = await self._call_llm(prompt)
        plan = extract_json(raw)
        if not isinstance(plan, dict) or not plan.get("subtasks"):
            plan = self._fallback_plan(dept_task, team)
        # Keep only subtasks assigned to a real member of this team.
        valid_ids = {a.agent_id for a in team}
        plan["subtasks"] = [
            s for s in plan.get("subtasks", []) if s.get("assigned_to") in valid_ids
        ]
        if not plan["subtasks"]:
            plan = self._fallback_plan(dept_task, team)
        return plan

    def _fallback_plan(self, dept_task: AgentTask, team: list[AgentProfile]) -> dict[str, Any]:
        """Deterministic plan if the model output is unusable: give the whole
        objective to the first specialist. Marked so the source is honest."""
        if not team:
            return {"analysis": "no specialists available", "subtasks": [], "fallback": True}
        first = team[0]
        return {
            "analysis": "fallback: single-specialist assignment",
            "fallback": True,
            "subtasks": [
                {
                    "assigned_to": first.agent_id,
                    "objective": dept_task.objective or dept_task.task_description,
                    "instructions": dept_task.instructions or dept_task.task_description,
                    "expected_output": dept_task.expected_output or "a deliverable",
                    "acceptance_criteria": (
                        dept_task.acceptance_criteria or ["produces a usable result"]
                    ),
                }
            ],
        }

    async def _create_subtasks(
        self, root: AgentTask, dept_task: AgentTask, plan: dict[str, Any], team: list[AgentProfile]
    ) -> list[AgentTask]:
        require(self.profile.agent_level, Permission.CREATE_SUBTASK)
        created: list[AgentTask] = []
        for spec in plan["subtasks"]:
            require_can_assign(self.profile.agent_level, TaskType.SUBTASK)
            subtask = AgentTask(
                task_id=str(uuid.uuid4()),
                task_description=spec.get("objective", ""),
                assigned_to=spec["assigned_to"],
                assigned_by=self.profile.agent_id,
                parent_task_id=dept_task.task_id,
                root_task_id=root.task_id,
                task_type=TaskType.SUBTASK.value,
                state=TaskState.CREATED.value,
                department=dept_task.department,
                objective=spec.get("objective", ""),
                instructions=spec.get("instructions", ""),
                expected_output=spec.get("expected_output", ""),
                required_output=spec.get("expected_output", ""),
                acceptance_criteria=spec.get("acceptance_criteria", []),
                input_data=dept_task.input_data,
                priority=dept_task.priority,
            )
            await self.repo.create_task(subtask)
            await self.repo.audit(
                AuditAction.TASK_ASSIGNED, actor=self.profile.agent_id,
                task_id=subtask.task_id, parent_task_id=dept_task.task_id,
                root_task_id=root.task_id, reason=f"assigned to {spec['assigned_to']}",
            )
            created.append(subtask)
        return created

    # ------------------------------------------------------------- execution
    async def _execute_subtasks(
        self, root: AgentTask, dept_task: AgentTask, subtasks: list[AgentTask]
    ) -> list[SpecialistReport]:
        reports: list[SpecialistReport] = []
        for subtask in subtasks:
            profile = await self.agent_registry.get_agent(subtask.assigned_to)
            if not profile:
                await self.repo.transition(
                    subtask, TaskState.FAILED, actor=self.profile.agent_id,
                    reason="specialist profile missing", audit_action=AuditAction.TASK_FAILED,
                )
                continue
            worker = SpecialistWorker(
                self.repo, self.vllm, self.tool_registry, profile, self.agent_registry
            )
            report = await worker.execute(root, dept_task, subtask)

            # STAGE 7 — leader reviews the report against acceptance criteria.
            report = await self._review_and_maybe_rework(root, dept_task, subtask, worker, report)
            reports.append(report)
        return reports

    async def _review_and_maybe_rework(
        self, root: AgentTask, dept_task: AgentTask, subtask: AgentTask,
        worker: SpecialistWorker, report: SpecialistReport,
    ) -> SpecialistReport:
        rounds = 0
        while rounds < MAX_SUBTASK_REWORK and self._needs_rework(report):
            rounds += 1
            fresh = await self.repo.get_task(subtask.task_id)
            await self.repo.transition(
                fresh, TaskState.REWORK_REQUIRED, actor=self.profile.agent_id,
                reason=f"acceptance criteria unmet (round {rounds})",
                audit_action=AuditAction.TASK_REWORKED,
            )
            await self.repo.increment_rework(subtask.task_id)
            fresh = await self.repo.get_task(subtask.task_id)
            await self.repo.transition(
                fresh, TaskState.DISPATCHED, actor=self.profile.agent_id,
                reason="re-dispatch for rework",
            )
            report = await worker.execute(
                root, dept_task, fresh, rework_note=self._rework_note(report)
            )

        # Leader acceptance decision (STAGE 7): the specialist leaves a
        # review_required subtask in REVIEWING. Once the rework budget is spent
        # the leader — the acceptance authority for its own subtasks — must move
        # it to a terminal state so the executive's final validation can trust
        # the tree. A blocked/failed report stays blocked/failed; anything else
        # the leader accepts as COMPLETED.
        fresh = await self.repo.get_task(subtask.task_id)
        if fresh and fresh.state_enum == TaskState.REVIEWING:
            if report.status == ReportStatus.BLOCKED:
                await self.repo.transition(
                    fresh, TaskState.BLOCKED, actor=self.profile.agent_id,
                    reason="leader could not resolve subtask",
                    audit_action=AuditAction.TASK_BLOCKED,
                )
            elif report.status == ReportStatus.FAILED:
                await self.repo.transition(
                    fresh, TaskState.FAILED, actor=self.profile.agent_id,
                    reason="leader rejected subtask", audit_action=AuditAction.TASK_FAILED,
                )
            else:
                await self.repo.transition(
                    fresh, TaskState.COMPLETED, actor=self.profile.agent_id,
                    reason="leader accepted subtask report",
                    audit_action=AuditAction.TASK_COMPLETED,
                )
        return report

    @staticmethod
    def _needs_rework(report: SpecialistReport) -> bool:
        return (
            report.status == ReportStatus.REVIEW_REQUIRED
            or (report.status == ReportStatus.COMPLETED and report.confidence < 0.35)
        )

    @staticmethod
    def _rework_note(report: SpecialistReport) -> str:
        problems = "; ".join(report.problems) or "insufficient detail / low confidence"
        missing = "; ".join(report.missing_information)
        note = f"Previous attempt fell short: {problems}."
        if missing:
            note += f" Address missing info: {missing}."
        return note

    # ----------------------------------------------------------- consolidation
    async def _consolidate(
        self, root: AgentTask, dept_task: AgentTask, plan: dict[str, Any],
        reports: list[SpecialistReport],
    ) -> DepartmentReport:
        specialists = [r.agent_id for r in reports]
        blocked = [r for r in reports if r.status == ReportStatus.BLOCKED]
        failed = [r for r in reports if r.status == ReportStatus.FAILED]

        status = ReportStatus.COMPLETED
        if failed:
            status = ReportStatus.FAILED
        elif blocked:
            status = ReportStatus.BLOCKED

        avg_conf = round(sum(r.confidence for r in reports) / len(reports), 3) if reports else 0.0

        prompt = f"""You are {self.profile.name}, {self.profile.department} leader.
Consolidate your specialists' reports into a department report for Braind (the CEO).

Department objective: {dept_task.objective or dept_task.task_description}

Specialist reports (JSON):
{[r.to_dict() for r in reports]}

Respond ONLY with JSON:
{{
  "final_result": "the department's combined deliverable/answer",
  "key_findings": ["..."],
  "work_completed": ["..."],
  "problems": ["..."],
  "missing_data": ["..."],
  "conflicts": ["..."],
  "risks": ["..."],
  "recommendation": "what Braind should do with this"
}}"""
        raw = await self._call_llm(prompt)
        parsed = extract_json(raw) or {}

        return DepartmentReport(
            task_id=dept_task.task_id,
            department=dept_task.department or self.profile.department,
            leader_id=self.profile.agent_id,
            objective=dept_task.objective or dept_task.task_description,
            status=status,
            work_breakdown=[s.get("objective", "") for s in plan.get("subtasks", [])],
            specialists_involved=specialists,
            work_completed=parsed.get("work_completed", [r.work_performed for r in reports]),
            key_findings=parsed.get("key_findings", []),
            data_used=sorted({d for r in reports for d in r.data_used}),
            problems=parsed.get("problems", [p for r in reports for p in r.problems]),
            missing_data=parsed.get(
                "missing_data", [m for r in reports for m in r.missing_information]
            ),
            conflicts=parsed.get("conflicts", []),
            risks=parsed.get("risks", []),
            final_result=parsed.get("final_result", ""),
            supporting_artifacts=[a for r in reports for a in r.output_artifacts],
            confidence=avg_conf,
            recommendation=parsed.get("recommendation", ""),
            specialist_reports=[r.to_dict() for r in reports],
        )

    async def _call_llm(self, prompt: str) -> str:
        result = await self.vllm.chat(
            messages=[{"role": "user", "content": prompt}],
            system=self.profile.system_prompt or "",
            temperature=self.profile.temperature,
            max_tokens=1024,
        )
        return str(result)
