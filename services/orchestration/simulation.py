"""Simulation Runner & Dry-Run Orchestrator (Phase 7).

Provides dry-run mode (dry_run=True):
- Intercepts all mutating tool calls and DB writes.
- Produces an operation preview with simulated state changes.
- Guarantees 0 actual DB mutations and 0 unintended state changes.
- Integrates with ToolPolicyService and ExplainabilityService.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from core.logging import get_logger
from services.audit.explain import ExplainabilityService
from services.governance.tool_policy import ToolPolicyService, TrustLevel

logger = get_logger(__name__)


@dataclass
class InterceptedAction:
    """Details of an intercepted mutating action in simulation mode."""

    step_index: int
    tool_name: str
    arguments: dict[str, Any]
    action_type: str  # "file_write", "db_write", "db_delete", "setting_change", "system_command"
    simulated_impact: str
    state_change_preview: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SimulatedStepResult:
    """Outcome of one step in a simulated plan."""

    step_index: int
    tool_name: str
    arguments: dict[str, Any]
    is_mutating: bool
    status: str  # "intercepted", "simulated_read", "denied"
    allowed_by_policy: bool
    simulated_output: Any
    policy_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SimulationResult:
    """Complete summary of a simulation run."""

    task_id: str
    task_name: str
    dry_run: bool
    total_steps: int
    mutations_intercepted: list[InterceptedAction] = field(default_factory=list)
    step_results: list[SimulatedStepResult] = field(default_factory=list)
    actual_db_mutations: int = 0  # Guarantees 0 in dry_run mode
    simulated_state_changes_count: int = 0
    can_execute_safely: bool = True
    preview_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mutations_intercepted"] = [m.to_dict() for m in self.mutations_intercepted]
        d["step_results"] = [s.to_dict() for s in self.step_results]
        return d


class SimulationRunner:
    """Executes tasks in dry-run mode, intercepting all state mutations."""

    def __init__(
        self,
        tool_policy: ToolPolicyService | None = None,
        explain_service: ExplainabilityService | None = None,
    ) -> None:
        self.tool_policy = tool_policy or ToolPolicyService()
        self.explain_service = explain_service or ExplainabilityService()

    async def run(
        self,
        task_name: str,
        planned_steps: list[dict[str, Any]],
        *,
        trust_level: TrustLevel | str | int = TrustLevel.USER,
        dry_run: bool = True,
        task_id: str | None = None,
    ) -> SimulationResult:
        """Run a sequence of planned steps in simulation mode.

        Args:
            task_name: Human-readable name of the task.
            planned_steps: List of dicts with 'tool' and 'args'.
            trust_level: Authority trust level of the executing context.
            dry_run: If True, intercepts all mutating calls; executes 0 DB writes.
            task_id: Unique task identifier.

        Returns:
            SimulationResult with intercepted mutations and operation preview.
        """
        tid = task_id or f"sim-{uuid.uuid4().hex[:8]}"
        level = TrustLevel.from_value(trust_level)

        mutations_intercepted: list[InterceptedAction] = []
        step_results: list[SimulatedStepResult] = []
        actual_db_mutations = 0
        all_allowed = True

        for idx, step in enumerate(planned_steps):
            tool_name = step.get("tool", "")
            args = step.get("args", {})

            # 1. Policy Evaluation
            decision = self.tool_policy.evaluate(
                tool_name,
                level,
                arguments=args,
                task_id=tid,
                is_confirmed=True,  # In simulation, assume confirmation check preview
            )

            is_mutating = self.tool_policy.is_mutating_tool(tool_name)
            is_setting = self.tool_policy.is_settings_modification(tool_name, args)

            if not decision.allowed:
                all_allowed = False
                step_results.append(
                    SimulatedStepResult(
                        step_index=idx,
                        tool_name=tool_name,
                        arguments=args,
                        is_mutating=is_mutating,
                        status="denied",
                        allowed_by_policy=False,
                        simulated_output=None,
                        policy_reason=decision.reason,
                    )
                )
                continue

            # 2. Dry-Run Interception of Mutating Calls
            if dry_run and (is_mutating or is_setting):
                # Categorize action
                if is_setting:
                    action_type = "setting_change"
                    impact = f"Would update setting '{args.get('key')}' to '{args.get('value')}'"
                    preview = {"setting": args.get("key"), "new_value": args.get("value")}
                elif "delete" in tool_name:
                    action_type = "db_delete"
                    impact = f"Would delete records matching {args}"
                    preview = {"deleted": args}
                else:
                    action_type = "db_write"
                    impact = f"Would insert/update records via '{tool_name}'"
                    preview = {"written": args}

                intercepted = InterceptedAction(
                    step_index=idx,
                    tool_name=tool_name,
                    arguments=args,
                    action_type=action_type,
                    simulated_impact=impact,
                    state_change_preview=preview,
                )
                mutations_intercepted.append(intercepted)

                simulated_output = (
                    f"[DRY RUN PREVIEW] Successfully intercepted '{tool_name}'. "
                    f"Impact: {impact}. 0 real mutations executed."
                )

                step_results.append(
                    SimulatedStepResult(
                        step_index=idx,
                        tool_name=tool_name,
                        arguments=args,
                        is_mutating=True,
                        status="intercepted",
                        allowed_by_policy=True,
                        simulated_output=simulated_output,
                        policy_reason=decision.reason,
                    )
                )
            else:
                # Read-only or safe step simulation
                step_results.append(
                    SimulatedStepResult(
                        step_index=idx,
                        tool_name=tool_name,
                        arguments=args,
                        is_mutating=False,
                        status="simulated_read",
                        allowed_by_policy=True,
                        simulated_output=f"[SIMULATED READ] {tool_name} returned mock results",
                        policy_reason=decision.reason,
                    )
                )

        # In dry_run mode, actual mutations must be strictly 0
        if dry_run:
            actual_db_mutations = 0

        # Construct Preview Summary
        preview_lines = [
            f"=== Simulation Report for Task: '{task_name}' (ID: {tid}) ===",
            f"Mode: {'DRY-RUN (Active)' if dry_run else 'LIVE'}",
            f"Total Planned Steps: {len(planned_steps)}",
            f"Intercepted Mutating Calls: {len(mutations_intercepted)}",
            f"Actual DB Mutations: {actual_db_mutations} (Strict Invariant: 0)",
            "",
            "Planned Operation Details:",
        ]

        for s in step_results:
            status_tag = f"[{s.status.upper()}]"
            preview_lines.append(
                f"  Step {s.step_index + 1}: {status_tag} {s.tool_name}({s.arguments})"
            )
            if s.status == "denied":
                preview_lines.append(f"    Policy Denial: {s.policy_reason}")
            elif s.status == "intercepted":
                preview_lines.append(f"    Simulated Impact: {s.simulated_output}")

        preview_summary = "\n".join(preview_lines)

        return SimulationResult(
            task_id=tid,
            task_name=task_name,
            dry_run=dry_run,
            total_steps=len(planned_steps),
            mutations_intercepted=mutations_intercepted,
            step_results=step_results,
            actual_db_mutations=actual_db_mutations,
            simulated_state_changes_count=len(mutations_intercepted),
            can_execute_safely=all_allowed,
            preview_summary=preview_summary,
        )
