"""Tests for Phase 3: Quality Gate & Repair Loop (tests/test_quality_gate.py).

Includes Test Scenario #4: Quality Gate blocks critical failure and triggers RepairLoop.
"""

from __future__ import annotations

from services.agents.domain import TaskState
from services.orchestration.service import TaskOrchestrationService
from services.settings.snapshot import create_default_snapshot
from services.verification.engine import (
    Layer1Result,
    Layer2Result,
    VerificationError,
    VerificationResult,
    VerificationStatus,
)
from services.verification.quality_gate import (
    QualityGate,
    QualityGateAction,
)
from services.verification.repair import RepairLoop

# ----------------------------------------------------------------------
# Scenario #4: Quality Gate blocks critical failure and triggers RepairLoop
# ----------------------------------------------------------------------


def test_scenario_4_quality_gate_blocks_critical_failure_regardless_of_score():
    """Scenario #4 (Part A): Critical failure MUST block even if score is high."""
    gate = QualityGate()

    # Construct verification result with high Layer 2 score, but critical Layer 1 citation failure
    crit_err = VerificationError(
        type="hallucinated_citation",
        message="Hallucinated citation: chunk_fake_123 not in RAG",
        is_critical=True,
    )
    l1 = Layer1Result(passed=False, citation_valid=False, errors=[crit_err], score=0.5)
    l2 = Layer2Result(passed=True, completeness_score=0.98, coherence_score=0.95)

    v_result = VerificationResult(
        is_valid=False,
        status=VerificationStatus.UNVERIFIED,
        confidence_score=0.90,  # Simulated high confidence
        layer1=l1,
        layer2=l2,
        errors=[crit_err],
        critical_errors=[crit_err],
    )

    # Evaluate against settings where critical_failure_blocks_regardless_of_score=True
    snapshot = create_default_snapshot(
        overrides={"quality_gate_critical_failure_blocks": True}
    )

    decision = gate.evaluate(v_result, snapshot)

    assert not decision.passed
    assert decision.action == QualityGateAction.BLOCK
    assert decision.requires_repair is True
    assert len(decision.critical_failures) == 1
    assert "Critical quality failure blocked" in decision.reason


def test_scenario_4_repair_loop_injects_errors_and_resolves_deliverable():
    """Scenario #4 (Part B): Quality gate failure triggers RepairLoop, which injects errors."""
    orch_service = TaskOrchestrationService()
    task = orch_service.create_task(
        "Generate research summary",
        settings_overrides={"max_retry_count": 3},
    )
    orch_service.start_planning(task.task_id)
    orch_service.plan_task(task.task_id)
    orch_service.start_execution(task.task_id)

    rag_context = [
        {"id": "real_chunk_1", "text": "Brain Second architecture uses 11 canonical states."}
    ]

    # Simulation: Agent initially generates hallucinated citation "chunk_fake_99".
    # On repair attempt, agent receives the error feedback and corrects it to "real_chunk_1".
    generation_history = []

    def mock_agent_executor(task_obj, error_feedback=None, attempt=0):
        generation_history.append({"attempt": attempt, "feedback": error_feedback})
        if error_feedback:
            # Agent fixes the issue after receiving error feedback
            return {
                "summary": "Corrected summary citing genuine context.",
                "citations": ["real_chunk_1"],
            }
        else:
            # Initial bad deliverable
            return {
                "summary": "Initial summary with fake citation.",
                "citations": ["chunk_fake_99"],
            }

    repair_loop = RepairLoop()

    # Initial deliverable has hallucinated citation
    initial_output = mock_agent_executor(task)

    result = repair_loop.run(
        task=task,
        execute_fn=mock_agent_executor,
        initial_deliverable=initial_output,
        rag_chunks=rag_context,
        orchestration_service=orch_service,
    )

    assert result.success is True
    assert result.attempts_count == 1
    assert len(generation_history) == 2  # initial + 1 repair attempt
    fb_str = str(generation_history[1]["feedback"])
    assert "hallucinated_citation" in fb_str or "chunk_fake_99" in fb_str
    assert task.state == TaskState.COMPLETED
    assert result.final_deliverable["citations"] == ["real_chunk_1"]


def test_scenario_4_repair_loop_bounds_retries_to_max_retry_count():
    """Scenario #4 (Part C): Repair loop never retries infinitely; halts at max_retry_count."""
    orch_service = TaskOrchestrationService()
    max_retries = 2
    task = orch_service.create_task(
        "Persistent failing task",
        settings_overrides={"max_retry_count": max_retries},
    )
    orch_service.start_planning(task.task_id)
    orch_service.plan_task(task.task_id)
    orch_service.start_execution(task.task_id)

    rag_context = [{"id": "chunk_alpha", "text": "Valid facts."}]
    attempts_called = []

    # Agent that keeps hallucinating citations repeatedly
    def unfixable_agent_executor(task_obj, error_feedback=None, attempt=0):
        attempts_called.append(attempt)
        return {
            "summary": "Still using bad citation.",
            "citations": [f"fake_chunk_{attempt}"],
        }

    repair_loop = RepairLoop()
    initial_bad = unfixable_agent_executor(task, attempt=0)

    result = repair_loop.run(
        task=task,
        execute_fn=unfixable_agent_executor,
        initial_deliverable=initial_bad,
        rag_chunks=rag_context,
        orchestration_service=orch_service,
    )

    assert result.success is False
    assert result.attempts_count == max_retries
    assert len(result.attempts) == max_retries
    assert task.state == TaskState.FAILED
    assert "failed quality gate after 2 repair attempt(s)" in result.error_message


# ----------------------------------------------------------------------
# Quality Gate Policy Settings
# ----------------------------------------------------------------------


def test_quality_gate_disabled_setting():
    """When quality_gate_enabled is False, gate passes unconditionally."""
    gate = QualityGate()
    v_res = VerificationResult(
        is_valid=False,
        status=VerificationStatus.UNVERIFIED,
        confidence_score=0.1,
        layer1=Layer1Result(passed=False, score=0.1),
        layer2=Layer2Result(passed=False, completeness_score=0.1, coherence_score=0.1),
    )

    snapshot = create_default_snapshot(overrides={"quality_gate_enabled": False})
    decision = gate.evaluate(v_res, snapshot)

    assert decision.passed is True
    assert decision.action == QualityGateAction.PASS


def test_quality_gate_low_confidence_triggers_repair():
    """Confidence below min_confidence triggers REPAIR action."""
    gate = QualityGate()
    l1 = Layer1Result(passed=True, score=0.7)
    l2 = Layer2Result(passed=True, completeness_score=0.7, coherence_score=0.7)

    v_res = VerificationResult(
        is_valid=False,
        status=VerificationStatus.PARTIALLY_VERIFIED,
        confidence_score=0.70,  # Below default min 0.80
        layer1=l1,
        layer2=l2,
    )

    snapshot = create_default_snapshot(
        overrides={"quality_gate_min_confidence": 0.85}
    )
    decision = gate.evaluate(v_res, snapshot)

    assert not decision.passed
    assert decision.action == QualityGateAction.REPAIR
    assert decision.requires_repair is True
