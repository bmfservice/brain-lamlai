"""Automatic Repair Loop (Brain Second Phase 3).

Injects verification errors back into agent prompt and re-executes up to
`max_retry_count` times (read from the task's settings snapshot).
Enforces `repair_must_reverify` after every repair attempt.
"""

from __future__ import annotations

import contextlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from core.logging import get_logger
from services.agents.domain import TaskState
from services.settings.snapshot import SettingsSnapshot, create_default_snapshot
from services.verification.engine import VerificationResult, VerificationService
from services.verification.quality_gate import QualityGate, QualityGateDecision

logger = get_logger(__name__)


@dataclass
class RepairAttempt:
    """Record of a single attempt in the repair loop."""

    attempt_number: int
    feedback_injected: list[str]
    deliverable: Any
    verification_result: VerificationResult
    gate_decision: QualityGateDecision
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_number": self.attempt_number,
            "feedback_injected": self.feedback_injected,
            "verification_status": self.verification_result.status.value,
            "confidence_score": self.verification_result.confidence_score,
            "gate_action": self.gate_decision.action.value,
            "gate_passed": self.gate_decision.passed,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RepairResult:
    """Final result of the repair loop."""

    success: bool
    attempts_count: int
    final_deliverable: Any
    final_verification: VerificationResult | None = None
    final_gate_decision: QualityGateDecision | None = None
    attempts: list[RepairAttempt] = field(default_factory=list)
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "attempts_count": self.attempts_count,
            "error_message": self.error_message,
            "attempts": [a.to_dict() for a in self.attempts],
            "final_gate_passed": (
                self.final_gate_decision.passed if self.final_gate_decision else False
            ),
        }


class RepairLoop:
    """Manages iterative repair and re-verification of deliverables."""

    def __init__(
        self,
        verification_service: VerificationService | None = None,
        quality_gate: QualityGate | None = None,
    ) -> None:
        self.verification_service = verification_service or VerificationService()
        self.quality_gate = quality_gate or QualityGate()

    def run(
        self,
        task: Any,
        execute_fn: Callable[..., Any],
        *,
        initial_deliverable: Any = None,
        settings_snapshot: SettingsSnapshot | None = None,
        orchestration_service: Any = None,
        rag_chunks: Any = None,
        referenced_files: Any = None,
        expected_schema: Any = None,
        objective: str = "",
    ) -> RepairResult:
        """Run the repair loop synchronously."""
        snapshot = self._resolve_snapshot(task, settings_snapshot)
        max_retries = int(
            snapshot.get("max_repair_attempts", snapshot.get("max_retry_count", 2))
        )

        attempts: list[RepairAttempt] = []
        current_deliverable = initial_deliverable
        last_errors: list[str] = []

        # 1. Check initial deliverable if provided
        if current_deliverable is not None:
            v_res = self.verification_service.verify(
                deliverable=current_deliverable,
                objective=objective,
                rag_chunks=rag_chunks,
                referenced_files=referenced_files,
                expected_schema=expected_schema,
                settings_snapshot=snapshot,
            )
            q_res = self.quality_gate.evaluate(v_res, snapshot)
            if q_res.passed:
                return RepairResult(
                    success=True,
                    attempts_count=0,
                    final_deliverable=current_deliverable,
                    final_verification=v_res,
                    final_gate_decision=q_res,
                    attempts=[],
                )
            last_errors = q_res.critical_failures + q_res.errors

        # 2. Iterate up to max_retries
        for attempt in range(1, max_retries + 1):
            logger.info("repair_loop.attempt_start", attempt=attempt, max_retries=max_retries)

            # Move task to REPAIRING
            if orchestration_service and hasattr(task, "task_id"):
                with contextlib.suppress(Exception):
                    orchestration_service.transition_task(
                        task.task_id,
                        TaskState.REPAIRING,
                        reason=f"Repair loop attempt {attempt}/{max_retries}",
                    )

            # Inject errors into execution callable
            feedback_messages = list(last_errors)
            current_deliverable = self._invoke_executor(
                execute_fn,
                task=task,
                error_feedback=feedback_messages,
                attempt=attempt,
            )

            # Move task to VERIFYING (repair_must_reverify)
            if orchestration_service and hasattr(task, "task_id"):
                with contextlib.suppress(Exception):
                    orchestration_service.transition_task(
                        task.task_id,
                        TaskState.VERIFYING,
                        reason=f"Re-verifying after repair attempt {attempt}",
                    )

            # Re-verify
            v_res = self.verification_service.verify(
                deliverable=current_deliverable,
                objective=objective,
                rag_chunks=rag_chunks,
                referenced_files=referenced_files,
                expected_schema=expected_schema,
                settings_snapshot=snapshot,
            )
            q_res = self.quality_gate.evaluate(v_res, snapshot)

            attempt_record = RepairAttempt(
                attempt_number=attempt,
                feedback_injected=feedback_messages,
                deliverable=current_deliverable,
                verification_result=v_res,
                gate_decision=q_res,
            )
            attempts.append(attempt_record)

            if q_res.passed:
                logger.info("repair_loop.attempt_success", attempt=attempt)
                if orchestration_service and hasattr(task, "task_id"):
                    with contextlib.suppress(Exception):
                        res_val = (
                            current_deliverable
                            if isinstance(current_deliverable, dict)
                            else {"result": current_deliverable}
                        )
                        orchestration_service.complete_task(
                            task.task_id,
                            result=res_val,
                        )

                return RepairResult(
                    success=True,
                    attempts_count=attempt,
                    final_deliverable=current_deliverable,
                    final_verification=v_res,
                    final_gate_decision=q_res,
                    attempts=attempts,
                )

            last_errors = q_res.critical_failures + q_res.errors

        # Retries exhausted
        error_msg = (
            f"Deliverable failed quality gate after {max_retries} repair attempt(s). "
            f"Unresolved errors: {'; '.join(last_errors[:2])}"
        )
        logger.warning("repair_loop.failed_exhausted", error=error_msg)

        if orchestration_service and hasattr(task, "task_id"):
            with contextlib.suppress(Exception):
                orchestration_service.fail_task(task.task_id, error=error_msg)

        return RepairResult(
            success=False,
            attempts_count=max_retries,
            final_deliverable=current_deliverable,
            final_verification=attempts[-1].verification_result if attempts else None,
            final_gate_decision=attempts[-1].gate_decision if attempts else None,
            attempts=attempts,
            error_message=error_msg,
        )

    async def run_async(
        self,
        task: Any,
        execute_fn: Callable[..., Any],
        *,
        initial_deliverable: Any = None,
        settings_snapshot: SettingsSnapshot | None = None,
        orchestration_service: Any = None,
        rag_chunks: Any = None,
        referenced_files: Any = None,
        expected_schema: Any = None,
        objective: str = "",
    ) -> RepairResult:
        """Run the repair loop asynchronously."""
        snapshot = self._resolve_snapshot(task, settings_snapshot)
        max_retries = int(
            snapshot.get("max_repair_attempts", snapshot.get("max_retry_count", 2))
        )

        attempts: list[RepairAttempt] = []
        current_deliverable = initial_deliverable
        last_errors: list[str] = []

        if current_deliverable is not None:
            v_res = await self.verification_service.verify_async(
                deliverable=current_deliverable,
                objective=objective,
                rag_chunks=rag_chunks,
                referenced_files=referenced_files,
                expected_schema=expected_schema,
                settings_snapshot=snapshot,
            )
            q_res = self.quality_gate.evaluate(v_res, snapshot)
            if q_res.passed:
                return RepairResult(
                    success=True,
                    attempts_count=0,
                    final_deliverable=current_deliverable,
                    final_verification=v_res,
                    final_gate_decision=q_res,
                    attempts=[],
                )
            last_errors = q_res.critical_failures + q_res.errors

        for attempt in range(1, max_retries + 1):
            if orchestration_service and hasattr(task, "task_id"):
                with contextlib.suppress(Exception):
                    orchestration_service.transition_task(
                        task.task_id,
                        TaskState.REPAIRING,
                        reason=f"Repair loop attempt {attempt}/{max_retries}",
                    )

            feedback_messages = list(last_errors)
            current_deliverable = await self._invoke_executor_async(
                execute_fn,
                task=task,
                error_feedback=feedback_messages,
                attempt=attempt,
            )

            if orchestration_service and hasattr(task, "task_id"):
                with contextlib.suppress(Exception):
                    orchestration_service.transition_task(
                        task.task_id,
                        TaskState.VERIFYING,
                        reason=f"Re-verifying after repair attempt {attempt}",
                    )

            v_res = await self.verification_service.verify_async(
                deliverable=current_deliverable,
                objective=objective,
                rag_chunks=rag_chunks,
                referenced_files=referenced_files,
                expected_schema=expected_schema,
                settings_snapshot=snapshot,
            )
            q_res = self.quality_gate.evaluate(v_res, snapshot)

            attempt_record = RepairAttempt(
                attempt_number=attempt,
                feedback_injected=feedback_messages,
                deliverable=current_deliverable,
                verification_result=v_res,
                gate_decision=q_res,
            )
            attempts.append(attempt_record)

            if q_res.passed:
                if orchestration_service and hasattr(task, "task_id"):
                    with contextlib.suppress(Exception):
                        res_val = (
                            current_deliverable
                            if isinstance(current_deliverable, dict)
                            else {"result": current_deliverable}
                        )
                        orchestration_service.complete_task(
                            task.task_id,
                            result=res_val,
                        )

                return RepairResult(
                    success=True,
                    attempts_count=attempt,
                    final_deliverable=current_deliverable,
                    final_verification=v_res,
                    final_gate_decision=q_res,
                    attempts=attempts,
                )

            last_errors = q_res.critical_failures + q_res.errors

        error_msg = (
            f"Deliverable failed quality gate after {max_retries} repair attempt(s). "
            f"Unresolved errors: {'; '.join(last_errors[:2])}"
        )

        if orchestration_service and hasattr(task, "task_id"):
            with contextlib.suppress(Exception):
                orchestration_service.fail_task(task.task_id, error=error_msg)

        return RepairResult(
            success=False,
            attempts_count=max_retries,
            final_deliverable=current_deliverable,
            final_verification=attempts[-1].verification_result if attempts else None,
            final_gate_decision=attempts[-1].gate_decision if attempts else None,
            attempts=attempts,
            error_message=error_msg,
        )

    def _resolve_snapshot(
        self,
        task: Any,
        settings_snapshot: SettingsSnapshot | None,
    ) -> SettingsSnapshot:
        if settings_snapshot is not None:
            return settings_snapshot
        if hasattr(task, "settings_snapshot") and task.settings_snapshot is not None:
            return task.settings_snapshot
        return create_default_snapshot()

    def _invoke_executor(
        self,
        execute_fn: Callable[..., Any],
        task: Any,
        error_feedback: list[str],
        attempt: int,
    ) -> Any:
        sig = inspect.signature(execute_fn)
        kwargs: dict[str, Any] = {}
        params = sig.parameters

        if "error_feedback" in params:
            kwargs["error_feedback"] = error_feedback
        if "attempt" in params:
            kwargs["attempt"] = attempt

        if len(params) == 0:
            return execute_fn()
        if len(params) == 1 and not kwargs:
            return execute_fn(task)
        if "task" in params:
            kwargs["task"] = task
            return execute_fn(**kwargs)
        return execute_fn(task, **kwargs)

    async def _invoke_executor_async(
        self,
        execute_fn: Callable[..., Any],
        task: Any,
        error_feedback: list[str],
        attempt: int,
    ) -> Any:
        if inspect.iscoroutinefunction(execute_fn):
            sig = inspect.signature(execute_fn)
            kwargs: dict[str, Any] = {}
            if "error_feedback" in sig.parameters:
                kwargs["error_feedback"] = error_feedback
            if "attempt" in sig.parameters:
                kwargs["attempt"] = attempt
            if len(sig.parameters) == 0:
                return await execute_fn()
            if len(sig.parameters) == 1 and not kwargs:
                return await execute_fn(task)
            if "task" in sig.parameters:
                kwargs["task"] = task
                return await execute_fn(**kwargs)
            return await execute_fn(task, **kwargs)

        return self._invoke_executor(execute_fn, task, error_feedback, attempt)
