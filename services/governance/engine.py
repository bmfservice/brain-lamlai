"""Governance policy engine (Giai đoạn 5, Layer 6).

A declarative gate evaluated before every mutating action — the OPA/Rego
pattern: rules live in the database, never hard-coded in business logic, and
are editable via REST without a restart. The engine is deliberately ignorant
of content correctness; it only answers one question — *is this action
permitted?*

Rule semantics (§4 of the architecture spec):

- ``rule_type`` ∈ {allow, deny, ask_first}; ``deny`` beats ``allow`` at equal
  priority.
- ``scope`` names the guarded surface (file_import, response, tool, category).
- ``pattern`` is a plain match string (a mime type, a file extension, an
  action name). Case-insensitive; ``*`` matches everything.
- Every evaluation is appended to the existing ``audit_logs`` table — which
  existed but was never written to before this phase.

Human corrections (owner edits) always sit above policy: they bypass this
engine entirely and are recorded as training-priority overrides.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import sqlalchemy as sa
from sqlalchemy import select

from core import database as db
from core.logging import get_logger

logger = get_logger(__name__)

ALLOW = "allow"
DENY = "deny"
ASK_FIRST = "ask_first"


@dataclass(frozen=True)
class PolicyDecision:
    """Outcome of one policy evaluation — always explainable."""

    action: str  # the scope+pattern that was evaluated
    verdict: str  # allow | deny | ask_first
    rule_id: uuid.UUID | None  # which rule decided; None = default-allow
    reason: str


class PolicyEngine:
    """Evaluate declarative DB-backed rules against an action context."""

    def __init__(self, session_factory, enabled: bool = True) -> None:
        self.session_factory = session_factory
        self.enabled = enabled

    async def evaluate(self, *, scope: str, pattern: str) -> PolicyDecision:
        """Decide whether ``scope:pattern`` is permitted.

        Rules are loaded per evaluation (they change via REST at runtime, so
        caching would need invalidation logic that the current scale does not
        justify). Ordered by priority descending, then deny-over-allow.
        """
        if not self.enabled:
            return PolicyDecision(
                action=f"{scope}:{pattern}",
                verdict=ALLOW,
                rule_id=None,
                reason="policy engine disabled",
            )
        try:
            async with self.session_factory() as session:
                rows = (
                    await session.execute(
                        select(db.PolicyRule)
                        .where(
                            db.PolicyRule.scope == scope,
                            db.PolicyRule.enabled.is_(True),
                        )
                        .order_by(
                            db.PolicyRule.priority.desc(),
                            # deny first at equal priority
                            sa.case((db.PolicyRule.rule_type == DENY, 0), else_=1),
                        )
                    )
                ).scalars().all()
        except Exception:
            # Fail-open would silently permit everything; fail-closed would
            # brick every ingest on a transient DB hiccup. Log loudly and
            # allow — policy is a gate, not the foundation.
            logger.exception("policy.evaluate_failed")
            return PolicyDecision(
                action=f"{scope}:{pattern}",
                verdict=ALLOW,
                rule_id=None,
                reason="policy evaluation failed (logged, fail-open)",
            )

        pattern_lower = pattern.strip().lower()
        for rule in rows:
            rule_pattern = (rule.pattern or "").strip().lower()
            matches = (
                rule_pattern == "*"
                or pattern_lower == rule_pattern
                or pattern_lower.endswith(rule_pattern)
            )
            if matches:
                verdict = rule.rule_type
                await self._audit(scope, pattern, verdict, rule.id, rule.reason)
                return PolicyDecision(
                    action=f"{scope}:{pattern}",
                    verdict=verdict,
                    rule_id=rule.id,
                    reason=rule.reason or f"matched rule {rule.id}",
                )

        # No rule matched → allow by default, but record it.
        await self._audit(scope, pattern, ALLOW, None, "no matching rule (default allow)")
        return PolicyDecision(
            action=f"{scope}:{pattern}",
            verdict=ALLOW,
            rule_id=None,
            reason="no matching rule (default allow)",
        )

    async def _audit(
        self,
        scope: str,
        pattern: str,
        verdict: str,
        rule_id: uuid.UUID | None,
        reason: str | None,
    ) -> None:
        """Append the evaluation to the audit_logs table (append-only)."""
        try:
            async with self.session_factory() as session:
                session.add(
                    db.AuditLog(
                        service="governance",
                        module="policy_engine",
                        action=f"{scope}:{pattern}",
                        status=verdict,
                        message=(reason or "")[:500],
                        meta={"rule_id": str(rule_id) if rule_id else None},
                    )
                )
                await session.commit()
        except Exception:
            logger.warning("policy.audit_write_failed", exc_info=True)

    async def audit(
        self,
        *,
        module: str,
        scope: str,
        pattern: str,
        verdict: str,
        reason: str | None = None,
    ) -> None:
        """Record a governance decision made *outside* the DB-rule path.

        :class:`FileImportPolicy` falls back to built-in denylists from settings
        when no DB rule matches; those are real gate decisions and must appear in
        the audit trail too — otherwise a file blocked by the built-in denylist
        would only show the engine's earlier ``default allow`` rows, and the
        trail would contradict the 403 the user actually got. Best-effort: an
        audit write failure never changes the verdict.
        """
        try:
            async with self.session_factory() as session:
                session.add(
                    db.AuditLog(
                        service="governance",
                        module=module,
                        action=f"{scope}:{pattern}",
                        status=verdict,
                        message=(reason or "")[:500],
                        meta={},
                    )
                )
                await session.commit()
        except Exception:
            logger.warning("policy.audit_write_failed", exc_info=True)
