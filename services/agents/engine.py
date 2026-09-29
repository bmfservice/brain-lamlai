"""Hierarchical orchestration engine — the real workflow driver.

This replaces the deadlocked poll-based coordinator. Instead of an agent
delegating a task and then polling the database for a completion that nothing
produces, the engine drives every stage of the workflow directly and in order:

    USER -> BRAIND(analyze, plan) -> LEADER(decompose) -> SPECIALIST(execute)
         -> LEADER(review + consolidate) -> BRAIND(review + learn + decide)
         -> [ASK_USER gate] -> FINAL_VALIDATION -> COMPLETED

Every stage:
* writes real task rows and lifecycle-state transitions (single source of truth),
* is guarded by the permission layer (specialist cannot bypass leader, only
  Braind finalizes the root, etc.),
* emits audit events, and
* uses structured contracts, not free-form dicts.

Runs are executed on a background asyncio task so the HTTP request returns a
root_task_id immediately and the UI polls real state. A run pauses at the
user-confirmation gate and is resumed by ``resume_after_confirmation``.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from core.logging import get_logger
from services.agents.brain import BrainExecutive
from services.agents.department import DepartmentLeader
from services.agents.domain import (
    AuditAction,
    ReviewDecision,
    TaskState,
    TaskType,
)
from services.agents.models import AgentTask
from services.agents.permissions import require_can_assign
from services.agents.repository import AgentRepository

logger = get_logger(__name__)

MAX_REWORK_ROUNDS = 2


class OrchestrationEngine:
    """Drives a user request through the full agent hierarchy."""

    def __init__(self, session_factory, vllm, tool_registry, agent_registry, approval_service):
        self.session_factory = session_factory
        self.vllm = vllm
        self.tool_registry = tool_registry
        self.agent_registry = agent_registry
        self.approval_service = approval_service
        self.repo = AgentRepository(session_factory)
        self.brain = BrainExecutive(self.repo, vllm, agent_registry, approval_service)
        # Track in-flight background runs so we don't garbage-collect the task.
        self._runs: dict[str, asyncio.Task] = {}

    # --------------------------------------------------------------- entrypoints
    async def submit_request(self, user_request: str, priority: int = 2) -> str:
        """Create the root task and launch the workflow in the background.

        Returns the root task id immediately; the UI polls task/tree/audit
        endpoints for real progress.
        """
        ceo = await self.agent_registry.get_ceo()
        if not ceo:
            raise RuntimeError("No executive (CEO) agent configured")

        root_id = str(uuid.uuid4())
        root = AgentTask(
            task_id=root_id,
            task_description=user_request,
            user_request=user_request,
            assigned_to=ceo.agent_id,
            assigned_by=None,
            parent_task_id=None,
            root_task_id=root_id,
            task_type=TaskType.ROOT.value,
            state=TaskState.CREATED.value,
            priority=priority,
        )
        await self.repo.create_task(root)

        run = asyncio.create_task(self._run_root(root_id))
        self._runs[root_id] = run
        run.add_done_callback(lambda _t, rid=root_id: self._runs.pop(rid, None))
        return root_id

    async def resume_after_confirmation(
        self, root_task_id: str, decision: str, notes: str | None
    ) -> None:
        """Resume a run paused at the user-confirmation gate (spec §18)."""
        root = await self.repo.get_task(root_task_id)
        if not root or root.state_enum != TaskState.WAITING_FOR_USER:
            logger.warning("engine.resume_no_gate", root=root_task_id)
            return

        await self.repo.audit(
            AuditAction.USER_CONFIRMED,
            actor="user",
            task_id=root_task_id,
            root_task_id=root_task_id,
            reason=f"user decision: {decision}",
            metadata={"notes": notes},
        )
        await self.brain.record_user_decision(root_task_id, decision, notes)

        # A rejection sends the work back for rework; anything else continues.
        if decision.lower() in {"reject", "rejected", "no", "rework"}:
            await self.repo.transition(
                root, TaskState.REWORK_REQUIRED, actor="ceo",
                reason="user rejected results", audit_action=AuditAction.TASK_REWORKED,
            )
        else:
            await self.repo.transition(
                root, TaskState.CONTINUING, actor="ceo", reason="user confirmed",
            )

        run = asyncio.create_task(self._run_root(root_task_id, resumed=True))
        self._runs[root_task_id] = run
        run.add_done_callback(lambda _t, rid=root_task_id: self._runs.pop(rid, None))

    # ------------------------------------------------------------------- driver
    async def _run_root(self, root_task_id: str, resumed: bool = False) -> None:
        """Execute the full workflow for a root task.

        Wrapped so any unexpected error fails the root task cleanly with an
        audit trail rather than vanishing on a background task.
        """
        try:
            await self._drive(root_task_id, resumed)
        except Exception as exc:  # noqa: BLE001 - background boundary
            logger.error("engine.run_failed", root=root_task_id, error=str(exc))
            root = await self.repo.get_task(root_task_id)
            if root and root.state_enum not in (TaskState.COMPLETED, TaskState.FAILED):
                try:
                    await self.repo.transition(
                        root, TaskState.FAILED, actor="ceo", reason=f"engine error: {exc}",
                        audit_action=AuditAction.TASK_FAILED,
                    )
                except Exception:  # noqa: BLE001
                    await self.repo.set_error(root_task_id, str(exc))

    async def _drive(self, root_task_id: str, resumed: bool) -> None:
        root = await self.repo.get_task(root_task_id)
        if root is None:
            return

        # STAGE 2 — Braind analysis + executive plan (skip if resuming past it).
        if not resumed:
            await self.repo.transition(
                root, TaskState.ANALYZING, actor="ceo",
                reason="Braind analysing request", audit_action=AuditAction.TASK_STARTED,
            )
            plan = await self.brain.analyze(root)
            await self.repo.transition(
                root, TaskState.PLANNED, actor="ceo", reason="executive plan ready"
            )

            # STAGE 3 — Braind -> Leaders: create department tasks.
            dept_tasks = await self._create_department_tasks(root, plan)
            if not dept_tasks:
                await self.repo.transition(
                    root, TaskState.FAILED, actor="ceo",
                    reason="no department could handle the request",
                    audit_action=AuditAction.TASK_FAILED,
                )
                return
            await self.repo.transition(
                root, TaskState.DISPATCHED, actor="ceo", reason="department tasks dispatched"
            )

        await self._execute_and_review(root_task_id)

    async def _execute_and_review(self, root_task_id: str) -> None:
        """Run department tasks, review, and loop on rework up to a cap."""
        for round_no in range(MAX_REWORK_ROUNDS + 1):
            root = await self.repo.get_task(root_task_id)
            if root is None or root.state_enum in (TaskState.COMPLETED, TaskState.FAILED):
                return

            await self.repo.transition(
                root, TaskState.IN_PROGRESS, actor="ceo", reason="departments executing"
            )

            # STAGE 4-8 — each leader decomposes, specialists execute, leader consolidates.
            # Only run departments that still have pending work: freshly created
            # ones, or ones Braind explicitly reopened for rework. A department
            # left BLOCKED/FAILED is not silently re-run (that would corrupt its
            # lifecycle) — Braind decides its fate via the review below.
            dept_tasks = [
                t for t in await self.repo.get_children(root_task_id)
                if t.task_type_enum == TaskType.DEPARTMENT
                and t.state_enum in (TaskState.CREATED, TaskState.REWORK_REQUIRED)
            ]
            await asyncio.gather(*(self._run_department(root, dt) for dt in dept_tasks))

            # STAGE 9 — Braind executive review of all department reports.
            root = await self.repo.get_task(root_task_id)
            await self.repo.transition(
                root, TaskState.REVIEWING, actor="ceo",
                reason="Braind executive review", audit_action=AuditAction.BRAIND_REVIEW_STARTED,
            )
            review = await self.brain.review(root)
            # STAGE 10 — learning with provenance.
            await self.brain.learn(root, review)
            await self.repo.audit(
                AuditAction.BRAIND_REVIEW_COMPLETED, actor="ceo",
                task_id=root_task_id, root_task_id=root_task_id,
                reason=f"decision={review.decision.value}", metadata=review.to_dict(),
            )

            decision = review.decision

            if decision == ReviewDecision.ASK_USER:
                await self._enter_user_gate(root, review)
                return  # paused; resumed via resume_after_confirmation

            if decision == ReviewDecision.REWORK and round_no < MAX_REWORK_ROUNDS:
                await self.repo.transition(
                    root, TaskState.REWORK_REQUIRED, actor="ceo",
                    reason=review.reason or "rework required",
                    audit_action=AuditAction.TASK_REWORKED,
                )
                await self._mark_departments_for_rework(root_task_id, review)
                # transition back into the execute loop
                root = await self.repo.get_task(root_task_id)
                await self.repo.transition(
                    root, TaskState.DISPATCHED, actor="ceo", reason="re-dispatch after rework"
                )
                continue

            if decision == ReviewDecision.FAILED:
                await self.repo.transition(
                    root, TaskState.FAILED, actor="ceo",
                    reason=review.reason or "review failed",
                    audit_action=AuditAction.TASK_FAILED,
                )
                return

            # CONTINUE / COMPLETE -> final validation.
            await self._finalize(root, review)
            return

        # Exhausted rework rounds without a clean review -> escalate to user.
        root = await self.repo.get_task(root_task_id)
        if root and root.state_enum not in (
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.WAITING_FOR_USER,
        ):
            review = await self.brain.review(root)
            await self._enter_user_gate(root, review)

    # ------------------------------------------------------------- stage helpers
    async def _create_department_tasks(
        self, root: AgentTask, plan: dict[str, Any]
    ) -> list[AgentTask]:
        """STAGE 3: turn the executive plan into department tasks for leaders."""
        ceo = await self.agent_registry.get_ceo()
        created: list[AgentTask] = []
        for dept_plan in plan.get("departments", []):
            department = dept_plan.get("department")
            leader = await self.agent_registry.get_leader(department)
            if not leader:
                logger.warning("engine.no_leader", department=department)
                continue

            # Permission gate: only the executive may assign a department task.
            require_can_assign(ceo.agent_level, TaskType.DEPARTMENT)

            dept_task = AgentTask(
                task_id=str(uuid.uuid4()),
                task_description=dept_plan.get("objective", root.task_description),
                assigned_to=leader.agent_id,
                assigned_by=ceo.agent_id,
                parent_task_id=root.task_id,
                root_task_id=root.task_id,
                task_type=TaskType.DEPARTMENT.value,
                state=TaskState.CREATED.value,
                department=department,
                objective=dept_plan.get("objective", ""),
                instructions=dept_plan.get("instructions", ""),
                expected_output=dept_plan.get("expected_output", ""),
                required_output=dept_plan.get("expected_output", ""),
                acceptance_criteria=dept_plan.get("acceptance_criteria", []),
                input_data=dept_plan.get("available_data", {}),
                constraints=dept_plan.get("constraints", []),
                priority=root.priority,
            )
            await self.repo.create_task(dept_task)
            await self.repo.audit(
                AuditAction.TASK_ASSIGNED, actor=ceo.agent_id,
                task_id=dept_task.task_id, parent_task_id=root.task_id,
                root_task_id=root.task_id, reason=f"assigned to {leader.name} ({department})",
            )
            created.append(dept_task)
        return created

    async def _run_department(self, root: AgentTask, dept_task: AgentTask) -> None:
        """STAGE 4-8 for one department, isolated so one failure can't sink siblings."""
        leader_profile = await self.agent_registry.get_agent(dept_task.assigned_to)
        if not leader_profile:
            await self.repo.transition(
                dept_task, TaskState.FAILED, actor="ceo", reason="leader profile missing",
                audit_action=AuditAction.TASK_FAILED,
            )
            return
        leader = DepartmentLeader(
            self.repo, self.vllm, self.tool_registry, self.agent_registry, leader_profile
        )
        try:
            await leader.run(root, dept_task)
        except Exception as exc:  # noqa: BLE001
            logger.error("engine.department_failed", dept=dept_task.department, error=str(exc))
            fresh = await self.repo.get_task(dept_task.task_id)
            if fresh and fresh.state_enum not in (TaskState.COMPLETED, TaskState.FAILED):
                await self.repo.transition(
                    fresh, TaskState.FAILED, actor=leader_profile.agent_id,
                    reason=f"department error: {exc}", audit_action=AuditAction.TASK_FAILED,
                )

    async def _mark_departments_for_rework(self, root_task_id: str, review) -> None:
        for dt in await self.repo.get_children(root_task_id):
            if dt.task_type_enum != TaskType.DEPARTMENT:
                continue
            if dt.state_enum == TaskState.COMPLETED:
                # Re-open for another pass.
                await self.repo.transition(
                    dt, TaskState.REWORK_REQUIRED, actor="ceo",
                    reason="Braind requested rework", audit_action=AuditAction.TASK_REWORKED,
                )
                await self.repo.increment_rework(dt.task_id)

    async def _enter_user_gate(self, root: AgentTask, review) -> None:
        """STAGE 11/18: pause the run and raise a user-confirmation request."""
        await self.repo.transition(
            root, TaskState.WAITING_FOR_USER, actor="ceo",
            reason=review.reason or "user decision required",
            audit_action=AuditAction.USER_CONFIRMATION_REQUIRED,
        )
        await self.brain.request_user_confirmation(root, review)

    async def _finalize(self, root: AgentTask, review) -> None:
        """STAGE 12/21: final executive validation, then COMPLETE (executive only)."""
        await self.repo.transition(
            root, TaskState.FINAL_VALIDATION, actor="ceo",
            reason="final executive validation", audit_action=AuditAction.FINAL_VALIDATION,
        )
        validation = await self.brain.final_validation(root)
        if not validation.get("passed", False):
            # Unmet requirement: do not complete. Rework if we can, else ask user.
            root = await self.repo.get_task(root.task_id)
            if root.rework_count < MAX_REWORK_ROUNDS:
                await self.repo.transition(
                    root, TaskState.REWORK_REQUIRED, actor="ceo",
                    reason=validation.get("reason", "requirements unmet"),
                    audit_action=AuditAction.TASK_REWORKED,
                )
                await self.repo.increment_rework(root.task_id)
                await self.repo.transition(
                    root, TaskState.DISPATCHED, actor="ceo",
                    reason="re-dispatch after failed validation",
                )
                await self._execute_and_review(root.task_id)
                return
            await self._enter_user_gate(root, review)
            return

        final_report = await self.brain.build_executive_report(root, validation)
        await self.repo.set_result(root.task_id, final_report)
        root = await self.repo.get_task(root.task_id)
        await self.repo.transition(
            root, TaskState.COMPLETED, actor="ceo",
            reason="all requirements met", audit_action=AuditAction.ROOT_TASK_COMPLETED,
        )
