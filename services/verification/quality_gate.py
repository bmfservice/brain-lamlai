"""Quality Gate — policy enforcement for verified deliverables (Brain Second Phase 3).

Enforces quality rules defined in SettingsSnapshot:
- quality_gate_enabled (default True)
- quality_gate_critical_failure_blocks / critical_failure_blocks_regardless_of_score (default True)
- quality_gate_min_confidence / min_verification_confidence (default 0.8)
- min_overall_quality_score (default 0.7)
- return_with_warning_allowed (default True)

Crucial contract:
A critical failure (e.g. hallucinated citation or missing file) MUST BLOCK the deliverable
regardless of how high the average score or confidence is (Scenario #4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.logging import get_logger
from services.settings.snapshot import SettingsSnapshot, create_default_snapshot
from services.verification.engine import VerificationResult

logger = get_logger(__name__)


class QualityGateAction(StrEnum):
    """Action dictated by the Quality Gate."""

    PASS = "pass"
    REPAIR = "repair"
    BLOCK = "block"
    WARNING = "warning"


@dataclass
class QualityGateDecision:
    """Outcome and enforcement decision produced by the Quality Gate."""

    passed: bool
    action: QualityGateAction
    confidence_score: float
    reason: str
    critical_failures: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    requires_repair: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "action": self.action.value,
            "confidence_score": self.confidence_score,
            "reason": self.reason,
            "critical_failures": self.critical_failures,
            "errors": self.errors,
            "requires_repair": self.requires_repair,
        }


class QualityGate:
    """Enforces quality rules and gates deliverables before completion."""

    def __init__(self, default_settings: SettingsSnapshot | None = None) -> None:
        self.default_settings = default_settings

    def evaluate(
        self,
        verification_result: VerificationResult,
        settings: SettingsSnapshot | dict[str, Any] | None = None,
    ) -> QualityGateDecision:
        """Evaluate a verification result against settings and return enforcement decision."""
        # Resolve settings
        snapshot = self._resolve_settings(settings)

        # 1. Is Quality Gate enabled?
        gate_enabled = bool(snapshot.get("quality_gate_enabled", True))
        if not gate_enabled:
            return QualityGateDecision(
                passed=True,
                action=QualityGateAction.PASS,
                confidence_score=verification_result.confidence_score,
                reason="Quality Gate is disabled by configuration",
                requires_repair=False,
            )

        crit_failure_blocks = bool(
            snapshot.get("quality_gate_critical_failure_blocks", True)
        )
        min_confidence = float(
            snapshot.get("quality_gate_min_confidence", 0.8)
        )
        min_quality_score = float(
            snapshot.get("min_overall_quality_score", 0.7)
        )
        return_with_warning = bool(
            snapshot.get("return_with_warning_allowed", True)
        )

        critical_messages = [e.message for e in verification_result.critical_errors]
        all_messages = [e.message for e in verification_result.errors]

        # 2. Rule: Critical failure blocks REGARDLESS of confidence / average score
        if critical_messages and crit_failure_blocks:
            reason = (
                f"Critical quality failure blocked: {len(critical_messages)} critical error(s) "
                f"found ({'; '.join(critical_messages[:2])})"
            )
            logger.warning("quality_gate.blocked_critical", reason=reason)
            return QualityGateDecision(
                passed=False,
                action=QualityGateAction.BLOCK,
                confidence_score=verification_result.confidence_score,
                reason=reason,
                critical_failures=critical_messages,
                errors=all_messages,
                requires_repair=True,
            )

        # 3. Rule: Confidence score must meet minimum threshold
        if verification_result.confidence_score < min_confidence:
            reason = (
                f"Confidence score {verification_result.confidence_score:.2f} is below "
                f"minimum required threshold {min_confidence:.2f}"
            )
            logger.info("quality_gate.repair_required_low_confidence", reason=reason)
            return QualityGateDecision(
                passed=False,
                action=QualityGateAction.REPAIR,
                confidence_score=verification_result.confidence_score,
                reason=reason,
                critical_failures=critical_messages,
                errors=all_messages,
                requires_repair=True,
            )

        # 4. Rule: Layer 2 quality score must meet minimum threshold
        l2_quality = verification_result.layer2.quality_score
        if l2_quality < min_quality_score:
            reason = (
                f"Layer 2 quality score {l2_quality:.2f} is below "
                f"minimum required threshold {min_quality_score:.2f}"
            )
            return QualityGateDecision(
                passed=False,
                action=QualityGateAction.REPAIR,
                confidence_score=verification_result.confidence_score,
                reason=reason,
                critical_failures=critical_messages,
                errors=all_messages,
                requires_repair=True,
            )

        # 5. Non-critical warnings
        if all_messages and return_with_warning:
            reason = f"Passed with warnings: {'; '.join(all_messages[:2])}"
            return QualityGateDecision(
                passed=True,
                action=QualityGateAction.WARNING,
                confidence_score=verification_result.confidence_score,
                reason=reason,
                errors=all_messages,
                requires_repair=False,
            )

        # 6. Full Pass
        return QualityGateDecision(
            passed=True,
            action=QualityGateAction.PASS,
            confidence_score=verification_result.confidence_score,
            reason="All quality gate criteria satisfied",
            requires_repair=False,
        )

    def _resolve_settings(
        self,
        settings: SettingsSnapshot | dict[str, Any] | None,
    ) -> SettingsSnapshot:
        if isinstance(settings, SettingsSnapshot):
            return settings
        if isinstance(settings, dict):
            return create_default_snapshot(overrides=settings)
        if self.default_settings is not None:
            return self.default_settings
        return create_default_snapshot()
