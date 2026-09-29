"""Braind executive — the top-level orchestrator (spec §2, §8-21).

Braind never plays a specialist. It:

* analyses the user request into an executive plan of department tasks,
* reviews department reports (requirement coverage, consistency, data quality,
  completeness) and decides CONTINUE / REWORK / ASK_USER / COMPLETE / FAILED,
* learns from results with explicit provenance (not treating every agent output
  as absolute truth),
* runs a final validation before allowing COMPLETE,
* is the only party permitted to finalize the root task.

The engine calls these methods in order; this class holds the executive
reasoning, the repository holds the state.
"""

from __future__ import annotations

import json
from typing import Any

from core.logging import get_logger
from services.agents.contracts import BrainReviewMessage
from services.agents.domain import (
    KnowledgeType,
    Permission,
    ReviewDecision,
    TaskState,
)
from services.agents.llm_json import extract_json
from services.agents.models import AgentTask
from services.agents.permissions import require
from services.agents.repository import AgentRepository

logger = get_logger(__name__)


class BrainExecutive:
    """Executive reasoning for Braind."""

    def __init__(self, repo: AgentRepository, vllm, agent_registry, approval_service):
        self.repo = repo
        self.vllm = vllm
        self.agent_registry = agent_registry
        self.approval_service = approval_service

    async def _profile(self):
        return await self.agent_registry.get_ceo()

    # ------------------------------------------------------------- STAGE 2/8
    async def analyze(self, root: AgentTask) -> dict[str, Any]:
        """Produce the executive plan: which departments, and each one's task."""
        ceo = await self._profile()
        require(ceo.agent_level, Permission.ANALYZE_ROOT)

        departments = await self._known_departments()
        dept_lines = "\n".join(f"- {d}" for d in departments)
        prompt = f"""You are Braind, the executive/CEO. Analyse this user request and
decide which departments must act. Do NOT assign to individuals — only to
departments. For each department, write a clear department-level task.

User request: {root.user_request}

Available departments:
{dept_lines}

Respond ONLY with JSON:
{{
  "objective": "the user's overall goal",
  "final_deliverable": "what the final output should be",
  "departments": [
    {{
      "department": "<one of the available departments>",
      "objective": "department-level objective",
      "instructions": "what this department must do",
      "expected_output": "the concrete result expected",
      "acceptance_criteria": ["..."],
      "available_data": {{}},
      "constraints": ["..."]
    }}
  ],
  "open_questions": ["things that are unclear but may be resolvable internally"]
}}"""
        raw = await self._call_llm(ceo, prompt)
        plan = extract_json(raw)
        if not isinstance(plan, dict) or not plan.get("departments"):
            plan = self._fallback_plan(root, departments)
        # Keep only real departments.
        plan["departments"] = [
            d for d in plan.get("departments", []) if d.get("department") in departments
        ]
        if not plan["departments"]:
            plan = self._fallback_plan(root, departments)

        await self.repo.set_result(root.task_id, {"executive_plan": plan})
        return plan

    async def _known_departments(self) -> list[str]:
        agents = await self.agent_registry.load_agents()
        return sorted(
            {a.department for a in agents if a.department and a.department != "executive"}
        )

    def _fallback_plan(self, root: AgentTask, departments: list[str]) -> dict[str, Any]:
        target = (
            "marketing"
            if "marketing" in departments
            else (departments[0] if departments else "marketing")
        )
        return {
            "objective": root.user_request or root.task_description,
            "final_deliverable": "a completed response to the request",
            "fallback": True,
            "departments": [
                {
                    "department": target,
                    "objective": root.user_request or root.task_description,
                    "instructions": "Handle the request end to end within your domain.",
                    "expected_output": "a usable deliverable",
                    "acceptance_criteria": ["addresses the user request"],
                    "available_data": {},
                    "constraints": [],
                }
            ],
            "open_questions": [],
        }

    # ------------------------------------------------------------- STAGE 9
    async def review(self, root: AgentTask) -> BrainReviewMessage:
        """Executive review of the department reports; returns a decision."""
        ceo = await self._profile()
        dept_reports = await self.repo.get_department_reports(root.task_id)
        tree = await self.repo.get_tree(root.task_id)

        # Deterministic completeness signal: any non-terminal or failed/blocked
        # department is an automatic non-complete (spec §9/§21).
        dept_tasks = [t for t in tree if t.task_type == "department"]
        unresolved = [
            t for t in dept_tasks
            if t.state_enum in (TaskState.BLOCKED, TaskState.FAILED, TaskState.REWORK_REQUIRED)
        ]
        incomplete = [t for t in dept_tasks if t.state_enum != TaskState.COMPLETED]

        reports_json = json.dumps([r.to_dict() for r in dept_reports], ensure_ascii=False)[:6000]
        prompt = f"""You are Braind, the executive/CEO. Review the department reports for
the user request. Do not rubber-stamp. Check requirement coverage, consistency
between departments, data quality, and completeness.

User request: {root.user_request}

Department reports (JSON): {reports_json}

Decide ONE decision:
- "complete": every requirement is met, no blockers, ready to finalise
- "rework": something is fixable internally by another department pass
- "ask_user": missing info or a decision that genuinely needs the user
- "continue": more internal work can proceed
- "failed": cannot be completed

Respond ONLY with JSON:
{{
  "decision": "complete|rework|ask_user|continue|failed",
  "reason": "why",
  "requirement_coverage": {{"requirement": true/false}},
  "consistency_issues": ["..."],
  "data_quality": {{"verified": ["..."], "assumed": ["..."], "missing": ["..."]}},
  "outstanding_items": ["..."],
  "approval_question": "only if decision is ask_user"
}}"""
        raw = await self._call_llm(ceo, prompt)
        parsed = extract_json(raw) or {}

        try:
            decision = ReviewDecision(str(parsed.get("decision", "continue")).lower())
        except ValueError:
            decision = ReviewDecision.CONTINUE

        # Override the model when hard signals contradict a "complete".
        if unresolved and decision == ReviewDecision.COMPLETE:
            decision = ReviewDecision.REWORK
        if incomplete and decision == ReviewDecision.COMPLETE:
            decision = ReviewDecision.CONTINUE
        if any(t.state_enum == TaskState.FAILED for t in dept_tasks) and decision not in (
            ReviewDecision.ASK_USER, ReviewDecision.FAILED, ReviewDecision.REWORK
        ):
            decision = ReviewDecision.REWORK

        return BrainReviewMessage(
            root_task_id=root.task_id,
            decision=decision,
            requirement_coverage=parsed.get("requirement_coverage", {}),
            consistency_issues=parsed.get("consistency_issues", []),
            data_quality=parsed.get("data_quality", {}),
            outstanding_items=parsed.get("outstanding_items", [t.department for t in unresolved]),
            reason=parsed.get("reason", ""),
            approval_question=parsed.get("approval_question"),
        )

    # ------------------------------------------------------------- STAGE 10
    async def learn(self, root: AgentTask, review: BrainReviewMessage) -> None:
        """Store learnings with provenance — agent reports are not facts."""
        dept_reports = await self.repo.get_department_reports(root.task_id)
        for rep in dept_reports:
            if rep.final_result:
                await self.repo.save_memory(
                    knowledge_type=KnowledgeType.AGENT_REPORT,
                    content=rep.final_result[:2000],
                    root_task_id=root.task_id,
                    task_id=rep.task_id,
                    department=rep.department,
                    agent_id=rep.leader_id,
                    source=f"department_report:{rep.department}",
                    confidence=rep.confidence,
                    evidence=rep.key_findings[:5],
                )
        # Data the review explicitly marked verified is kept at higher trust.
        for item in review.data_quality.get("verified", []):
            await self.repo.save_memory(
                knowledge_type=KnowledgeType.VERIFIED_DATA,
                content=item,
                root_task_id=root.task_id,
                agent_id="ceo",
                source="executive_review",
                confidence=0.85,
            )

    # ------------------------------------------------------------- STAGE 18
    async def request_user_confirmation(self, root: AgentTask, review: BrainReviewMessage) -> str:
        """Raise a structured approval request the UI shows at the gate."""
        ceo = await self._profile()
        require(ceo.agent_level, Permission.REQUEST_USER_CONFIRMATION)
        dept_reports = await self.repo.get_department_reports(root.task_id)
        action_data = {
            "root_task_id": root.task_id,
            "user_request": root.user_request,
            "question": review.approval_question or "Should we proceed?",
            "reason": review.reason,
            "known": review.data_quality.get("verified", []),
            "missing": review.data_quality.get("missing", []),
            "outstanding": review.outstanding_items,
            "department_summaries": {r.department: r.final_result for r in dept_reports},
        }
        req = await self.approval_service.request_approval(
            agent_id="ceo",
            action_type="workflow_user_confirmation",
            action_data=action_data,
            task_id=root.task_id,
            priority=3,
        )
        return req.request_id

    async def record_user_decision(
        self, root_task_id: str, decision: str, notes: str | None
    ) -> None:
        await self.repo.save_memory(
            knowledge_type=KnowledgeType.USER_CONFIRMED_FACT,
            content=f"User decision: {decision}. {notes or ''}".strip(),
            root_task_id=root_task_id,
            agent_id="user",
            source="user_confirmation",
            confidence=1.0,
        )

    # ------------------------------------------------------------- STAGE 12
    async def final_validation(self, root: AgentTask) -> dict[str, Any]:
        """Compare user request against the produced work before completing."""
        tree = await self.repo.get_tree(root.task_id)
        dept_tasks = [t for t in tree if t.task_type == "department"]
        subtasks = [t for t in tree if t.task_type == "subtask"]

        all_depts_done = bool(dept_tasks) and all(
            t.state_enum == TaskState.COMPLETED for t in dept_tasks
        )
        all_subs_done = all(
            t.state_enum in (TaskState.COMPLETED,) for t in subtasks
        ) if subtasks else True
        dept_reports = await self.repo.get_department_reports(root.task_id)
        has_deliverable = any(r.final_result for r in dept_reports)

        passed = all_depts_done and all_subs_done and has_deliverable
        reason = ""
        if not all_depts_done:
            reason = "not all departments completed"
        elif not all_subs_done:
            reason = "not all subtasks completed"
        elif not has_deliverable:
            reason = "no final deliverable produced"

        return {
            "passed": passed,
            "reason": reason,
            "departments_complete": all_depts_done,
            "subtasks_complete": all_subs_done,
            "has_deliverable": has_deliverable,
        }

    async def build_executive_report(
        self, root: AgentTask, validation: dict[str, Any]
    ) -> dict[str, Any]:
        """STAGE 22: the final executive report + deliverable shown to the user."""
        tree = await self.repo.get_tree(root.task_id)
        dept_reports = await self.repo.get_department_reports(root.task_id)
        audit = await self.repo.get_audit_trail(root.task_id)

        leaders = sorted({r.leader_id for r in dept_reports})
        specialists = sorted({s for r in dept_reports for s in r.specialists_involved})
        departments = sorted({r.department for r in dept_reports})
        deliverable = "\n\n".join(
            f"## {r.department}\n{r.final_result}" for r in dept_reports if r.final_result
        )

        plan_objective = (
            (root.result or {}).get("executive_plan", {}).get("objective", root.user_request)
            if root.result
            else root.user_request
        )
        return {
            "status": "completed",
            "user_request": root.user_request,
            "objective": plan_objective,
            "final_deliverable": deliverable,
            "departments_involved": departments,
            "leaders_involved": leaders,
            "specialists_involved": specialists,
            "tasks_created": len([t for t in tree if t.task_type == "department"]),
            "subtasks_created": len([t for t in tree if t.task_type == "subtask"]),
            "department_reports": [r.to_dict() for r in dept_reports],
            "validation": validation,
            "audit_event_count": len(audit),
            "remaining_risks": sorted({risk for r in dept_reports for risk in r.risks}),
        }

    async def _call_llm(self, ceo, prompt: str) -> str:
        result = await self.vllm.chat(
            messages=[{"role": "user", "content": prompt}],
            system=ceo.system_prompt or "You are Braind, the executive orchestrator.",
            temperature=ceo.temperature,
            max_tokens=1024,
        )
        return str(result)
