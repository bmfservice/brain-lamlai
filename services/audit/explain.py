"""Audit & Explainability Service (Phase 7).

Explains:
1. Why a setting resolved to a particular value:
   - Scans the seven-scope Priority Hierarchy in order:
     SYSTEM_HARD (0) > SECURITY (1) > PROJECT (2) > TASK (3) >
     USER_PREFERENCE (4) > TEMPORARY (5) > LEARNED (6) > DEFAULT
   - Identifies the winning scope, overridden candidate values, and rationale.
   - Enforces HR-NO-CREDENTIAL-LEAK: sensitive settings are masked ('***').
2. Why an action or tool call was allowed/denied:
   - Explains trust levels, matching policy rules, hard rules, and confirmation gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from core.logging import get_logger
from services.governance.tool_policy import TrustLevel
from services.settings.catalog import CATALOG
from services.settings.schema import SettingScope, SettingSpec

logger = get_logger(__name__)

# Catalog index by key for fast spec lookup
_CATALOG_INDEX: dict[str, SettingSpec] = {spec.key: spec for spec in CATALOG}


@dataclass
class ScopeEvaluation:
    """Evaluation record of one setting scope."""

    scope: str
    scope_ref: str | None
    value: Any
    is_winner: bool
    status: str  # "winner", "overridden", "unset"


@dataclass
class SettingExplanation:
    """Human-readable and structured explanation of setting resolution."""

    key: str
    resolved_value: Any
    winning_scope: str
    winning_scope_ref: str | None
    category: str | None
    rationale: str | None
    spec_default: Any
    is_sensitive: bool
    evaluations: list[ScopeEvaluation] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ActionExplanation:
    """Explanation of why an action or tool call was permitted or blocked."""

    action: str
    verdict: str  # "allow" | "deny" | "ask_first"
    allowed: bool
    trust_level: str | None = None
    reason: str = ""
    factors: list[str] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExplainabilityService:
    """Explains setting resolutions and governance decisions for auditability."""

    def __init__(self, settings_store=None) -> None:
        self.settings_store = settings_store

    def explain_setting(
        self,
        key: str,
        *,
        scope_values: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> SettingExplanation:
        """Explain why a setting resolved to its current value.

        Scans the 7-scope hierarchy:
          SYSTEM_HARD -> SECURITY -> PROJECT -> TASK -> USER_PREFERENCE ->
          TEMPORARY -> LEARNED -> CATALOG DEFAULT

        Args:
            key: The setting key to explain.
            scope_values: Optional mapping of {scope_name: value} for offline evaluation.
            context: Context containing optional project_id, task_id, user_id.

        Returns:
            SettingExplanation with winning scope, candidate comparisons, and rationale.
        """
        spec = _CATALOG_INDEX.get(key)
        spec_default = spec.default_value if spec else None
        category = str(spec.category) if spec else None
        rationale = spec.rationale if spec else None
        is_sensitive = getattr(spec, "is_sensitive", False) if spec else False

        provided = scope_values or {}
        evaluations: list[ScopeEvaluation] = []
        winner_scope: str | None = None
        winner_scope_ref: str | None = None
        winner_value: Any = None

        ctx = context or {}

        # Scan scopes in strict Priority Hierarchy order (declaration order)
        for scope_enum in SettingScope:
            scope_name = str(scope_enum)
            scope_ref = None
            if scope_enum == SettingScope.PROJECT:
                scope_ref = ctx.get("project_id")
            elif scope_enum == SettingScope.TASK:
                scope_ref = ctx.get("task_id")
            elif scope_enum == SettingScope.USER_PREFERENCE:
                scope_ref = ctx.get("user_id")

            # Check if this scope has a value
            has_val = scope_name in provided
            raw_val = provided.get(scope_name) if has_val else None

            # Enforce HR-NO-CREDENTIAL-LEAK on display values
            display_val = "***" if (is_sensitive and raw_val is not None) else raw_val

            if has_val and winner_scope is None:
                winner_scope = scope_name
                winner_scope_ref = scope_ref
                winner_value = raw_val
                evaluations.append(
                    ScopeEvaluation(
                        scope=scope_name,
                        scope_ref=scope_ref,
                        value=display_val,
                        is_winner=True,
                        status="winner",
                    )
                )
            elif has_val:
                evaluations.append(
                    ScopeEvaluation(
                        scope=scope_name,
                        scope_ref=scope_ref,
                        value=display_val,
                        is_winner=False,
                        status="overridden",
                    )
                )
            else:
                evaluations.append(
                    ScopeEvaluation(
                        scope=scope_name,
                        scope_ref=scope_ref,
                        value=None,
                        is_winner=False,
                        status="unset",
                    )
                )

        # Fallback to catalog default if unset at all scopes
        if winner_scope is None:
            winner_scope = "catalog_default"
            winner_value = spec_default
            display_winner = "***" if (is_sensitive and winner_value is not None) else winner_value
        else:
            display_winner = "***" if (is_sensitive and winner_value is not None) else winner_value

        # Build natural-language explanation
        explanation_parts = []
        if winner_scope == "catalog_default":
            explanation_parts.append(
                f"Setting '{key}' resolved to default value '{display_winner}' "
                "from catalog (no explicit override set at any scope)."
            )
        else:
            explanation_parts.append(
                f"Setting '{key}' resolved to '{display_winner}' via scope '{winner_scope}'."
            )

        overridden = [e for e in evaluations if e.status == "overridden"]
        if overridden:
            overridden_desc = ", ".join(f"{e.scope} ('{e.value}')" for e in overridden)
            explanation_parts.append(f"Higher-precedence scope overrode: {overridden_desc}.")

        if rationale:
            explanation_parts.append(f"Rationale: {rationale}")

        return SettingExplanation(
            key=key,
            resolved_value=winner_value,
            winning_scope=winner_scope,
            winning_scope_ref=winner_scope_ref,
            category=category,
            rationale=rationale,
            spec_default="***" if is_sensitive else spec_default,
            is_sensitive=is_sensitive,
            evaluations=evaluations,
            explanation=" ".join(explanation_parts),
        )

    def explain_action(
        self,
        action: str,
        *,
        verdict: str = "allow",
        trust_level: TrustLevel | str | int | None = None,
        tool_name: str | None = None,
        is_mutating: bool = False,
        reason: str | None = None,
    ) -> ActionExplanation:
        """Explain why an action or tool call was permitted or blocked.

        Factors analyzed:
        - Trust level authority: SYSTEM(5) > USER(4) > MEMORY(3) > DOCUMENT(2) > EXTERNAL(1)
        - Mutating vs read-only constraints
        - Human confirmation requirement
        """
        factors = []
        t_level = None
        if trust_level is not None:
            t_level = TrustLevel.from_value(trust_level)
            factors.append(f"Trust level: {t_level.name} ({t_level.value})")

        if is_mutating:
            factors.append("Action is state-mutating")
        else:
            factors.append("Action is read-only")

        allowed = verdict == "allow"

        # Construct narrative
        if not allowed:
            if t_level is not None and t_level <= TrustLevel.DOCUMENT and is_mutating:
                explanation = (
                    f"Action '{action}' was DENIED because trust level {t_level.name} "
                    "cannot execute mutating tools without explicit human confirmation. "
                    "External and document contents are treated as data, never instructions."
                )
                factors.append("HR-PROMPT-INJECTION-DEFENSE constraint")
            elif verdict == "ask_first":
                explanation = (
                    f"Action '{action}' requires user confirmation before proceeding "
                    "(high-impact or destructive tool policy)."
                )
                factors.append("Confirmation required by policy")
            else:
                explanation = f"Action '{action}' was DENIED: {reason or 'Denied by policy'}."
        else:
            explanation = (
                f"Action '{action}' was ALLOWED. "
                + (f"Caller holds sufficient trust ({t_level.name}). " if t_level else "")
                + (f"Reason: {reason}" if reason else "Complies with all policy rules.")
            )

        return ActionExplanation(
            action=action,
            verdict=verdict,
            allowed=allowed,
            trust_level=t_level.name if t_level else None,
            reason=reason or ("Permitted" if allowed else "Blocked"),
            factors=factors,
            explanation=explanation,
        )
