"""The settings catalog (Brain Second Phase 1a).

Declarative data, not logic: one ``SettingSpec`` per setting key. This is where
the "no hardcoding" rule is honoured — behavior defaults live here and are
seeded into ``setting_definitions``/``setting_values``, so running code reads a
value from the store instead of branching on a literal.

Defaults follow the user's stated precedence, applied key by key:

    ACCURACY > SPEED
    CORRECTNESS > TOKEN SAVING
    VERIFICATION > GUESSING

Every spec carries ``source`` (which spec section it came from) and ``rationale``
(why this default), so a reviewer can audit a value instead of trusting it.
Keys for the five categories Appendix A left empty (Verification, Quality Gate,
Memory, Workflow Rules, Security) are marked ``source='derived_phaseN'`` and
their rationale names the Definition-of-Done item they serve.

Two keys appear under both Core Behavior and Behavior Rules in the spec
(``when_missing_information``, ``when_uncertain``). Per Phase 0 decision Q6 they
are declared ONCE, with ``category=BEHAVIOR_RULES`` and
``also_in_categories=(CORE_BEHAVIOR,)`` so the Phase 6 UI can list them in both
sidebars without two contradictory values existing.

``SettingSpec.__post_init__`` validates every ``default_value`` against its own
spec at import time, so a mistyped default fails here loudly rather than at the
first seed (Phase 1a risk R13).
"""

from __future__ import annotations

from services.settings.schema import (
    DataType,
    RiskLevel,
    SettingCategory,
    SettingSpec,
    build_index,
)

# Behavior-rule actions allowed by Appendix A (verbatim list).
_ACTIONS: tuple[str, ...] = (
    "ask",
    "stop",
    "retry",
    "search_memory",
    "search_knowledge",
    "use_tool",
    "use_best_effort",
    "warn_user",
    "escalate",
)

# Reasoning depth vocabulary from Appendix A.
_DEPTH: tuple[str, ...] = ("low", "normal", "deep", "critical")

# --- spec helpers (keep ~88 declarations readable) --------------------------


def _b(key, cat, default, desc, *, source="appendix_a", rationale=None, **kw) -> SettingSpec:
    return SettingSpec(
        key=key,
        category=cat,
        data_type=DataType.BOOL,
        default_value=default,
        description=desc,
        source=source,
        rationale=rationale,
        **kw,
    )


def _i(key, cat, default, desc, *, source="appendix_a", rationale=None, **kw) -> SettingSpec:
    return SettingSpec(
        key=key,
        category=cat,
        data_type=DataType.INT,
        default_value=default,
        description=desc,
        source=source,
        rationale=rationale,
        **kw,
    )


def _f(key, cat, default, desc, *, source="appendix_a", rationale=None, **kw) -> SettingSpec:
    return SettingSpec(
        key=key,
        category=cat,
        data_type=DataType.FLOAT,
        default_value=default,
        description=desc,
        source=source,
        rationale=rationale,
        **kw,
    )


def _s(key, cat, default, desc, *, source="appendix_a", rationale=None, **kw) -> SettingSpec:
    return SettingSpec(
        key=key,
        category=cat,
        data_type=DataType.STR,
        default_value=default,
        description=desc,
        source=source,
        rationale=rationale,
        **kw,
    )


def _e(
    key, cat, default, allowed, desc, *, source="appendix_a", rationale=None, **kw
) -> SettingSpec:
    return SettingSpec(
        key=key,
        category=cat,
        data_type=DataType.ENUM,
        default_value=default,
        allowed_values=allowed,
        description=desc,
        source=source,
        rationale=rationale,
        **kw,
    )


def _j(key, cat, default, desc, *, source="appendix_a", rationale=None, **kw) -> SettingSpec:
    return SettingSpec(
        key=key,
        category=cat,
        data_type=DataType.JSON,
        default_value=default,
        description=desc,
        source=source,
        rationale=rationale,
        **kw,
    )


def _sl(
    key, cat, default, allowed, desc, *, source="appendix_a", rationale=None, **kw
) -> SettingSpec:
    return SettingSpec(
        key=key,
        category=cat,
        data_type=DataType.STRING_LIST,
        default_value=default,
        allowed_values=allowed,
        description=desc,
        source=source,
        rationale=rationale,
        **kw,
    )


_CB = SettingCategory.CORE_BEHAVIOR
_RS = SettingCategory.REASONING
_TE = SettingCategory.TASK_EXECUTION
_VR = SettingCategory.VERIFICATION
_QG = SettingCategory.QUALITY_GATE
_LN = SettingCategory.LEARNING
_MM = SettingCategory.MEMORY
_BR = SettingCategory.BEHAVIOR_RULES
_TL = SettingCategory.TOOLS
_WR = SettingCategory.WORKFLOW_RULES
_SC = SettingCategory.SECURITY

_ACC = "ACCURACY > SPEED"
_COR = "CORRECTNESS > TOKEN SAVING"
_VER = "VERIFICATION > GUESSING"

# --- Core Behavior / Identity (Appendix A) ---------------------------------

CORE_BEHAVIOR: tuple[SettingSpec, ...] = (
    _s(
        "identity",
        _CB,
        "BILL",
        "Name the system uses for itself.",
        rationale="Stable identity; the user named the system BILL.",
    ),
    _s(
        "primary_role",
        _CB,
        "Local-first AI second brain",
        "The system's primary role statement.",
        rationale="Matches the product's stated purpose.",
    ),
    _s(
        "primary_goal",
        _CB,
        "Answer from learned knowledge with verifiable evidence",
        "The goal that ranks every other trade-off.",
        rationale=_VER,
    ),
    _j(
        "inviolable_principles",
        _CB,
        ["No fabricated evidence", "No silent data loss", "No unverified claims presented as fact"],
        "Principles the system will not trade away for convenience.",
        risk_level=RiskLevel.CRITICAL,
        is_critical=True,
        rationale="Mirrors the System Hard Rules in hard_rules.py.",
    ),
    _e(
        "processing_style",
        _CB,
        "deliberate",
        ("deliberate", "balanced", "fast"),
        "Default deliberation style for a task.",
        rationale=_ACC,
    ),
    _e(
        "proactivity_level",
        _CB,
        "suggest",
        ("passive", "suggest", "proactive", "autonomous"),
        "How much the system acts without being asked.",
        rationale="Suggest-by-default keeps a human in the loop.",
    ),
    _e(
        "caution_level",
        _CB,
        "high",
        ("low", "medium", "high", "paranoid"),
        "How cautiously ambiguous or risky input is treated.",
        rationale=_ACC,
    ),
    _b(
        "allow_assumption",
        _CB,
        False,
        "May the system proceed on an unstated assumption?",
        rationale=_VER,
    ),
    _i(
        "max_assumption_count",
        _CB,
        0,
        "Maximum assumptions allowed per task.",
        min_value=0,
        max_value=10,
        rationale="Zero while allow_assumption is False.",
    ),
    _b(
        "ask_before_high_risk_assumption",
        _CB,
        True,
        "Ask the user before relying on a high-impact assumption.",
        risk_level=RiskLevel.HIGH,
        rationale=_VER,
    ),
    _e(
        "when_data_conflict",
        _CB,
        "escalate",
        _ACTIONS,
        "What to do when two stored facts contradict.",
        rationale="Never silently pick one; surface the conflict (Phase 4).",
    ),
    _e(
        "when_user_request_invalid",
        _CB,
        "warn_user",
        _ACTIONS,
        "What to do when a request is impossible or self-contradictory.",
        rationale="Tell the user rather than fabricate a result.",
    ),
    _e(
        "when_self_error_detected",
        _CB,
        "retry",
        _ACTIONS,
        "What to do when the system detects its own mistake.",
        rationale="Repair before answering (Phase 3 repair loop).",
    ),
)

# --- Reasoning (Appendix A) -------------------------------------------------

REASONING: tuple[SettingSpec, ...] = (
    _e(
        "reasoning_depth",
        _RS,
        "deep",
        _DEPTH,
        "How deeply to reason before acting.",
        rationale=_ACC,
    ),
    _b(
        "require_task_plan_before_execution",
        _RS,
        True,
        "Produce a plan before executing a task.",
        rationale=_ACC,
    ),
    _b(
        "require_requirement_extraction",
        _RS,
        True,
        "Extract an explicit requirement list from the request.",
        rationale=_VER,
    ),
    _b(
        "require_constraint_extraction",
        _RS,
        True,
        "Extract constraints (limits, must-nots) from the request.",
        rationale=_ACC,
    ),
    _b(
        "require_dependency_analysis",
        _RS,
        True,
        "Analyse what a task depends on before running it.",
        rationale=_ACC,
    ),
    _b(
        "require_alternative_analysis",
        _RS,
        False,
        "Consider alternative approaches before committing.",
        rationale="Off by default: costs tokens; correctness does not need it.",
    ),
    _b(
        "require_risk_analysis",
        _RS,
        True,
        "Assess risk before acting.",
        risk_level=RiskLevel.HIGH,
        rationale=_ACC,
    ),
    _b(
        "require_contradiction_check",
        _RS,
        True,
        "Check the answer for internal contradictions.",
        rationale=_VER,
    ),
    _b(
        "require_second_pass_review",
        _RS,
        True,
        "Re-review the result with a second pass.",
        rationale=_ACC,
    ),
    _b(
        "require_final_self_review",
        _RS,
        True,
        "Self-review the final answer before returning it.",
        rationale=_VER,
    ),
    _b(
        "prefer_evidence_over_inference",
        _RS,
        True,
        "Prefer cited evidence to plausible inference.",
        rationale=_VER,
    ),
    _b(
        "prefer_existing_knowledge_over_guessing",
        _RS,
        True,
        "Search learned knowledge before guessing.",
        rationale=_VER,
    ),
    _b(
        "require_source_validation",
        _RS,
        True,
        "Validate the source of a claim before relying on it.",
        rationale=_VER,
    ),
    _b(
        "require_cross_check",
        _RS,
        True,
        "Cross-check important claims against a second source.",
        rationale=_COR,
    ),
    _b(
        "require_independent_verification",
        _RS,
        True,
        "Verify a claim by a method independent of how it was produced.",
        rationale="Breaks the LLM-judging-its-own-output loop (Phase 1a R6).",
    ),
    _b(
        "require_cause_effect_validation",
        _RS,
        False,
        "Validate claimed cause-effect relationships.",
        rationale="Off by default; enable for analytical tasks.",
    ),
    _b(
        "require_edge_case_analysis",
        _RS,
        True,
        "Consider edge cases before declaring a result complete.",
        rationale=_ACC,
    ),
    _b(
        "require_failure_scenario_analysis",
        _RS,
        True,
        "Consider how the result could be wrong.",
        rationale=_VER,
    ),
)

# --- Task Execution (Appendix A) -------------------------------------------

TASK_EXECUTION: tuple[SettingSpec, ...] = (
    _b("require_plan_before_action", _TE, True, "Plan before taking action.", rationale=_ACC),
    _b(
        "require_pre_execution_validation",
        _TE,
        True,
        "Validate preconditions before executing.",
        rationale=_VER,
    ),
    _b(
        "require_post_execution_validation",
        _TE,
        True,
        "Validate results after executing.",
        rationale=_VER,
    ),
    _b(
        "auto_retry_on_failure",
        _TE,
        True,
        "Retry a failed step automatically.",
        rationale="Bounded by max_retry_count.",
    ),
    _i(
        "max_retry_count",
        _TE,
        2,
        "Maximum automatic retries per step.",
        min_value=0,
        max_value=5,
        rationale="Bounded so a failing step cannot loop forever (Phase 2 DoD).",
    ),
    _b(
        "retry_with_different_strategy",
        _TE,
        True,
        "On retry, change strategy rather than repeat the same call.",
        rationale=_ACC,
    ),
    _b(
        "stop_on_critical_error",
        _TE,
        True,
        "Stop immediately on a critical error.",
        risk_level=RiskLevel.HIGH,
        rationale=_COR,
    ),
    _b(
        "allow_partial_result",
        _TE,
        False,
        "May a partially complete result be returned?",
        rationale=_COR,
    ),
    _b(
        "allow_incomplete_result_to_user",
        _TE,
        False,
        "May an incomplete result reach the user as if complete?",
        risk_level=RiskLevel.HIGH,
        rationale="Forbids fake completion (Master Rule 1/7).",
    ),
    _b("require_complete_result", _TE, True, "Require a complete result.", rationale=_COR),
    _b("require_dependency_check", _TE, True, "Check dependencies before acting.", rationale=_ACC),
    _b("require_environment_check", _TE, True, "Check the environment is ready.", rationale=_COR),
    _b(
        "require_permission_check",
        _TE,
        True,
        "Check permissions/policy before acting.",
        rationale="Governance gate (Phase 5).",
    ),
    _b("require_impact_analysis", _TE, True, "Analyse impact before a change.", rationale=_ACC),
    _b(
        "require_rollback_plan",
        _TE,
        True,
        "Have a rollback plan before a change.",
        risk_level=RiskLevel.HIGH,
        rationale="Destructive-op guard (Phase 5).",
    ),
    _b(
        "require_backup_before_destructive_operation",
        _TE,
        True,
        "Back up before any destructive operation.",
        risk_level=RiskLevel.CRITICAL,
        is_critical=True,
        rationale=_COR,
    ),
    _b(
        "require_confirmation_for_dangerous_operation",
        _TE,
        True,
        "Require explicit user confirmation for a dangerous operation.",
        risk_level=RiskLevel.CRITICAL,
        is_critical=True,
        rationale="Human-in-the-loop for irreversible actions (Phase 5).",
    ),
)

# --- Verification (derived: Appendix A leaves this category empty — Phase 3) -

VERIFICATION: tuple[SettingSpec, ...] = (
    _e(
        "verification_level",
        _VR,
        "strict",
        ("minimal", "standard", "strict", "critical"),
        "How thoroughly a result is verified before it may be returned.",
        source="derived_phase3",
        rationale=_VER + " (Phase 3 VerificationReport).",
    ),
    _b(
        "require_deterministic_validation_first",
        _VR,
        True,
        "Run deterministic validators before any LLM reviewer.",
        source="derived_phase3",
        rationale="Deterministic evidence beats LLM opinion (Phase 3 scope).",
    ),
    _b(
        "allow_unverified_response",
        _VR,
        False,
        "May an unverified response be returned at all?",
        source="derived_phase3",
        rationale="If True it must still be labelled UNVERIFIED, never VERIFIED.",
    ),
    _b(
        "tool_result_must_be_checked",
        _VR,
        True,
        "Cross-check a tool's real result instead of trusting the model's claim.",
        source="derived_phase3",
        rationale="Phase 3 mandatory test: tool fails but model says success.",
    ),
    _sl(
        "verification_status_vocab",
        _VR,
        ["verified", "verified_with_high_confidence", "partially_verified", "unverified"],
        ["verified", "verified_with_high_confidence", "partially_verified", "unverified"],
        "The only honesty labels a response may carry.",
        source="derived_phase3",
        rationale="Master Rule 6 vocabulary; 'exactly 100%' is not allowed.",
    ),
    _f(
        "min_verification_confidence",
        _VR,
        0.8,
        "Confidence below which a result is not called verified.",
        min_value=0.0,
        max_value=1.0,
        source="derived_phase3",
        rationale=_VER,
    ),
)

# --- Quality Gate (derived: Appendix A empty — Phase 3) ---------------------

QUALITY_GATE: tuple[SettingSpec, ...] = (
    _b(
        "quality_gate_enabled",
        _QG,
        True,
        "Run the quality gate on every result.",
        source="derived_phase3",
        rationale=_VER,
    ),
    _b(
        "critical_failure_blocks_regardless_of_score",
        _QG,
        True,
        "A critical failure blocks even if the average score is high.",
        risk_level=RiskLevel.CRITICAL,
        is_critical=True,
        source="derived_phase3",
        rationale="Phase 3: critical failure must BLOCK regardless of mean.",
    ),
    _f(
        "min_overall_quality_score",
        _QG,
        0.7,
        "Minimum overall quality score to approve.",
        min_value=0.0,
        max_value=1.0,
        source="derived_phase3",
        rationale=_ACC,
    ),
    _i(
        "max_repair_attempts",
        _QG,
        2,
        "Maximum repair-loop attempts before giving up.",
        min_value=0,
        max_value=5,
        source="derived_phase3",
        rationale="Bounded loop; never infinite (Phase 3 DoD).",
    ),
    _b(
        "repair_must_reverify",
        _QG,
        True,
        "Re-verify after every repair attempt.",
        source="derived_phase3",
        rationale="A repair is not trusted just because it ran.",
    ),
    _b(
        "return_with_warning_allowed",
        _QG,
        True,
        "May the final gate return a result carrying an explicit warning?",
        source="derived_phase3",
        rationale="RETURN_WITH_WARNING is honest; silently passing is not.",
    ),
)

# --- Learning (Appendix A) --------------------------------------------------

LEARNING: tuple[SettingSpec, ...] = (
    _b(
        "enable_learning",
        _LN,
        True,
        "May the system learn at all?",
        rationale="Overlaps core.config chat_learn_enabled; this is the canonical "
        "behavioral switch (Phase 0 overlap table).",
    ),
    _e(
        "learning_mode",
        _LN,
        "verified_only",
        ("off", "passive", "verified_only", "aggressive"),
        "What the system is allowed to learn.",
        rationale=_VER,
    ),
    _b(
        "auto_learn_from_conversation",
        _LN,
        True,
        "Learn from chat turns.",
        rationale="Gated by the learning pipeline, not stored raw (Phase 4).",
    ),
    _b(
        "auto_learn_from_successful_task", _LN, True, "Learn from successful tasks.", rationale=_VER
    ),
    _b(
        "learn_from_failed_task",
        _LN,
        True,
        "Learn from failed tasks (failure memory).",
        rationale="Failure memory is explicitly required by Phase 4.",
    ),
    _b(
        "learn_from_user_correction",
        _LN,
        True,
        "Learn from a user correction.",
        rationale="A correction is the highest-value learning signal.",
    ),
    _b(
        "learn_from_verified_facts_only",
        _LN,
        True,
        "Only store facts that passed verification.",
        risk_level=RiskLevel.HIGH,
        rationale=_VER,
    ),
    _f(
        "minimum_confidence_to_learn",
        _LN,
        0.8,
        "Confidence floor for storing a learned item.",
        min_value=0.0,
        max_value=1.0,
        rationale=_VER,
    ),
    _b(
        "require_verification_before_learning",
        _LN,
        True,
        "Verify a candidate before learning it.",
        rationale=_VER,
    ),
    _b(
        "require_user_approval_before_permanent_learning",
        _LN,
        True,
        "Ask the user before making a learning permanent.",
        risk_level=RiskLevel.HIGH,
        rationale="No silent self-modification (Phase 4).",
    ),
    _b(
        "prevent_contradictory_memory",
        _LN,
        True,
        "Refuse to store a fact contradicting verified memory.",
        risk_level=RiskLevel.HIGH,
        rationale=_VER,
    ),
    _b(
        "auto_merge_similar_memory",
        _LN,
        False,
        "Automatically merge near-duplicate memories.",
        rationale="Off: merging without evidence violates the no-silent-overwrite rule.",
    ),
    _e(
        "memory_conflict_strategy",
        _LN,
        "quarantine",
        ("quarantine", "keep_both", "ask_user", "prefer_newer", "prefer_verified"),
        "How a memory conflict is resolved.",
        rationale="Quarantine, never auto-overwrite (Phase 4 DoD).",
    ),
    _b(
        "forget_low_confidence_memory",
        _LN,
        False,
        "Drop memories that stay below the confidence floor.",
        rationale="Off by default: forgetting is destructive; quarantine instead.",
    ),
    _f(
        "memory_decay_rate",
        _LN,
        0.0,
        "Per-day confidence decay for unused memory (0 disables).",
        min_value=0.0,
        max_value=1.0,
        rationale="Reuses the consolidation weight-decay law; 0 keeps memory stable.",
    ),
    _f(
        "memory_importance_threshold",
        _LN,
        0.5,
        "Importance floor for long-term retention.",
        min_value=0.0,
        max_value=1.0,
        rationale=_ACC,
    ),
    _i(
        "max_learning_items_per_task",
        _LN,
        10,
        "Cap learned items per task so one task cannot flood memory.",
        min_value=0,
        max_value=100,
        rationale="Bounds RAM/CPU and table growth.",
    ),
)

# --- Memory (derived: Appendix A empty — Phase 4) ---------------------------

MEMORY: tuple[SettingSpec, ...] = (
    _b(
        "verified_memory_protected",
        _MM,
        True,
        "Verified knowledge may not be overwritten without stronger evidence.",
        risk_level=RiskLevel.HIGH,
        source="derived_phase4",
        rationale="Phase 4 DoD: conflict must not auto-overwrite verified memory.",
    ),
    _i(
        "memory_retention_days",
        _MM,
        0,
        "Days to retain short-term memory (0 = until consolidated).",
        min_value=0,
        max_value=3650,
        source="derived_phase4",
        rationale="Expiry support Phase 4 metadata (expires_at).",
    ),
    _b(
        "conflict_check_on_write",
        _MM,
        True,
        "Check for conflicts at write time, not only in the nightly scan.",
        source="derived_phase4",
        rationale="Phase 4 write-time detection; bounded to entity-overlap set (R7).",
    ),
    _b(
        "strategy_memory_enabled",
        _MM,
        True,
        "Record success/failure strategy memory for reuse.",
        source="derived_phase4",
        rationale="Phase 4 success/failure strategy memory.",
    ),
    _i(
        "strategy_min_uses_before_trust",
        _MM,
        3,
        "Successful repetitions before a strategy's reliability rises.",
        min_value=1,
        max_value=100,
        source="derived_phase4",
        rationale="Reliability must be earned by repeated outcomes, not asserted.",
    ),
)

# --- Behavior Rules (Appendix A) -------------------------------------------
# when_missing_information / when_uncertain are declared HERE (their canonical
# category) and surfaced under Core Behavior via also_in_categories (Q6).

BEHAVIOR_RULES: tuple[SettingSpec, ...] = (
    _e(
        "when_missing_information",
        _BR,
        "ask",
        _ACTIONS,
        "What to do when required information is missing.",
        also_in_categories=(_CB,),
        rationale="Ask rather than assume (_VER).",
    ),
    _e(
        "when_uncertain",
        _BR,
        "ask",
        _ACTIONS,
        "What to do when the system is uncertain.",
        also_in_categories=(_CB,),
        rationale="Ask rather than guess (_VER).",
    ),
    _e(
        "when_conflict_detected",
        _BR,
        "escalate",
        _ACTIONS,
        "What to do when a conflict is detected.",
        rationale="Surface it; never silently resolve (Phase 4).",
    ),
    _e(
        "when_task_failed",
        _BR,
        "warn_user",
        _ACTIONS,
        "What to do when a task fails.",
        rationale="Honest failure, not fake success.",
    ),
    _e(
        "when_tool_failed",
        _BR,
        "retry",
        _ACTIONS,
        "What to do when a tool call fails.",
        rationale="Bounded retry, then escalate; never claim success (Phase 3 test 2).",
    ),
    _e(
        "when_requirement_impossible",
        _BR,
        "warn_user",
        _ACTIONS,
        "What to do when a requirement cannot be met.",
        rationale="Tell the user; do not fabricate a result.",
    ),
    _e(
        "when_user_request_ambiguous",
        _BR,
        "ask",
        _ACTIONS,
        "What to do when the request is ambiguous.",
        rationale="Clarify before acting.",
    ),
    _e(
        "when_potential_data_loss",
        _BR,
        "stop",
        _ACTIONS,
        "What to do when an action may lose data.",
        risk_level=RiskLevel.CRITICAL,
        is_critical=True,
        rationale="Stop and confirm; aligns with HR-NO-PERMANENT-DELETE.",
    ),
    _e(
        "when_potential_security_risk",
        _BR,
        "stop",
        _ACTIONS,
        "What to do on a potential security risk.",
        risk_level=RiskLevel.CRITICAL,
        is_critical=True,
        rationale="Stop; security scope is top-tier.",
    ),
    _e(
        "when_memory_conflict",
        _BR,
        "escalate",
        _ACTIONS,
        "What to do on a memory conflict.",
        rationale="Quarantine/ask, never auto-overwrite.",
    ),
    _e(
        "when_low_confidence",
        _BR,
        "warn_user",
        _ACTIONS,
        "What to do when confidence is low.",
        rationale="Label and warn; never present low confidence as verified (_VER).",
    ),
)

# --- Tools (Appendix A) ----------------------------------------------------
# These are the *default* policy applied to any tool. Per-tool overrides are
# stored as setting_values rows with scope=PROJECT and scope_ref=<tool_name>
# (Phase 5 wires them); the catalog deliberately does NOT enumerate the 15
# registered tool names, because that list lives in code (api/main.py) and
# copying it here would both hardcode it and drift out of sync.

_DANGER: tuple[str, ...] = ("safe", "low", "medium", "high", "destructive")
_TRUST: tuple[str, ...] = ("system", "user", "memory", "document", "external")

TOOLS: tuple[SettingSpec, ...] = (
    _b(
        "tool_allow",
        _TL,
        True,
        "Default: is a tool allowed to run at all?",
        rationale="Per-tool override via scope_ref (Phase 5).",
    ),
    _b(
        "tool_require_confirmation",
        _TL,
        False,
        "Default: require user confirmation before a tool runs.",
        rationale="Only dangerous tools default this on, via per-tool override.",
    ),
    _i(
        "tool_max_calls",
        _TL,
        8,
        "Default max calls of one tool per task.",
        min_value=1,
        max_value=100,
        rationale="Bounds cost; AgentLoop max_steps is separate.",
    ),
    _i(
        "tool_timeout_seconds",
        _TL,
        30,
        "Default per-call timeout in seconds.",
        min_value=1,
        max_value=600,
        rationale="Matches outbound_fetch_timeout_seconds order.",
    ),
    _i(
        "tool_retry_count",
        _TL,
        1,
        "Default retries for a failed tool call.",
        min_value=0,
        max_value=5,
        rationale=_COR,
    ),
    _b(
        "tool_validate_result",
        _TL,
        True,
        "Default: validate a tool's real result before trusting it.",
        risk_level=RiskLevel.HIGH,
        rationale="Phase 3 ToolResultValidator (test 2).",
    ),
    _b(
        "tool_allow_parallel_execution",
        _TL,
        True,
        "Default: may independent tool calls run in parallel?",
        rationale="Throughput; safe for reads.",
    ),
    _b(
        "tool_require_evidence",
        _TL,
        True,
        "Default: a tool result must carry evidence to be used as fact.",
        rationale=_VER,
    ),
    _e(
        "tool_default_trust_level",
        _TL,
        "document",
        _TRUST,
        "Default trust level assigned to tool output.",
        risk_level=RiskLevel.HIGH,
        rationale="Tool output is 'document' trust — may not change settings (Phase 5).",
    ),
    _sl(
        "tool_allowed_task_types",
        _TL,
        [],
        [],
        "Default task types a tool may run in (empty = all).",
        rationale="Empty default = unrestricted; tightened per tool in Phase 5.",
    ),
    _sl(
        "tool_blocked_task_types",
        _TL,
        [],
        [],
        "Default task types a tool is blocked from (empty = none).",
        rationale="Empty default; Phase 5 adds per-tool blocks.",
    ),
    _e(
        "tool_default_danger_level",
        _TL,
        "low",
        _DANGER,
        "Default danger classification for a tool.",
        rationale="destructive/high tools get confirmation via per-tool override.",
    ),
)

# --- Workflow Rules (derived: Appendix A empty — Phase 5 rule engine) -------

WORKFLOW_RULES: tuple[SettingSpec, ...] = (
    _b(
        "rule_engine_enabled",
        _WR,
        True,
        "Evaluate IF-condition-THEN-action rules during a task.",
        source="derived_phase5",
        rationale="Phase 5 declarative rule engine.",
    ),
    _i(
        "max_rules_per_evaluation",
        _WR,
        50,
        "Cap rules evaluated per task so a rule set cannot blow up latency.",
        min_value=1,
        max_value=1000,
        source="derived_phase5",
        rationale="Bounds CPU (user constraint: limit server RAM/CPU).",
    ),
    _i(
        "rule_evaluation_timeout_ms",
        _WR,
        200,
        "Timeout for evaluating the whole rule set on one task.",
        min_value=10,
        max_value=5000,
        source="derived_phase5",
        rationale="Rules must be cheap and deterministic, not an LLM call.",
    ),
    _b(
        "log_triggered_rules",
        _WR,
        True,
        "Log which rules fired for a task (explainability).",
        source="derived_phase5",
        rationale="Phase 5 DoD + Phase 7 explainability.",
    ),
    _b(
        "risk_affects_verification_depth",
        _WR,
        True,
        "Let task risk raise the verification level.",
        source="derived_phase5",
        rationale="Phase 2 risk classification drives Phase 3 depth (scenario 5).",
    ),
)

# --- Security (derived: Appendix A empty — Phase 5) ------------------------

SECURITY: tuple[SettingSpec, ...] = (
    _b(
        "treat_retrieved_content_as_data",
        _SC,
        True,
        "Retrieved/tool/external content is data, never instructions.",
        risk_level=RiskLevel.CRITICAL,
        is_critical=True,
        source="derived_phase5",
        rationale="Primary prompt-injection defense (Phase 5).",
    ),
    _b(
        "allow_external_content_to_change_settings",
        _SC,
        False,
        "May content from a document/tool/external source change settings?",
        risk_level=RiskLevel.CRITICAL,
        is_critical=True,
        source="derived_phase5",
        rationale="MUST stay False; backed by hard_rules + structural isolation.",
    ),
    _b(
        "allow_retrieved_content_to_trigger_tools",
        _SC,
        False,
        "May retrieved content directly trigger a tool/side effect?",
        risk_level=RiskLevel.CRITICAL,
        is_critical=True,
        source="derived_phase5",
        rationale="Blocks injection-driven side effects.",
    ),
    _b(
        "injection_detection_enabled",
        _SC,
        True,
        "Scan inbound external content for injection patterns.",
        source="derived_phase5",
        rationale="Defense in depth, not the only layer.",
    ),
    _sl(
        "trust_source_order",
        _SC,
        ["system", "user", "memory", "document", "external"],
        _TRUST,
        "Trust ordering of the five content sources (most to least trusted).",
        risk_level=RiskLevel.HIGH,
        source="derived_phase5",
        rationale="The 5 authority sources from Phase 5; order drives conflict resolution.",
    ),
    _b(
        "mask_sensitive_settings_in_logs",
        _SC,
        True,
        "Never write is_sensitive setting values to logs/audit.",
        risk_level=RiskLevel.CRITICAL,
        is_critical=True,
        source="derived_phase5",
        rationale="Enforces HR-NO-CREDENTIAL-LEAK.",
    ),
)

# --- Step 0 shared runtime foundation --------------------------------------

STEP0_FOUNDATION: tuple[SettingSpec, ...] = (
    _f(
        "selforg_merge_threshold",
        _LN,
        0.9,
        "Minimum model confidence required to merge taxonomy categories.",
        min_value=0.0,
        max_value=1.0,
        source="approved_step0_plan",
        rationale="Keeps model-directed category merging configurable and auditable.",
    ),
    _f(
        "selforg_split_dispersion_threshold",
        _LN,
        0.7,
        "Minimum semantic dispersion that triggers category split evaluation.",
        min_value=0.0,
        max_value=1.0,
        source="approved_step0_plan",
        rationale="Prevents a fixed split threshold from living in self-organization logic.",
    ),
    _b(
        "selforg_enabled",
        _LN,
        True,
        "Enable non-blocking background knowledge-store reorganization.",
        source="approved_step0_plan",
        rationale="Allows the owner to disable self-organization at runtime without code changes.",
    ),
    _i(
        "selforg_frequency",
        _LN,
        3600,
        "Interval in seconds between background self-organization evaluations.",
        min_value=1,
        source="approved_step0_plan",
        rationale=(
            "Makes background scheduling runtime-configurable instead of embedding "
            "an interval in code."
        ),
    ),
    _i(
        "gpu_reserve_free_mib",
        _TE,
        5120,
        "Minimum free VRAM in MiB that must remain available on each GPU.",
        min_value=5120,
        source="approved_step0_plan",
        rationale="Enforces the project requirement to preserve at least 5 GiB free VRAM per GPU.",
    ),
    _i(
        "gpu_monitor_interval_seconds",
        _TE,
        5,
        "Interval in seconds between runtime GPU resource samples.",
        min_value=1,
        source="approved_step0_plan",
        rationale=(
            "Makes resource-monitor polling configurable and avoids a timing literal "
            "in the guard."
        ),
    ),
    _i(
        "max_parallel_tasks",
        _TE,
        4,
        "Maximum number of independently executable tasks running concurrently.",
        min_value=1,
        source="approved_step0_plan",
        rationale="Provides the configurable concurrency bound required for parallel execution.",
    ),
    _i(
        "embed_batch_size",
        _TE,
        16,
        "Requested embedding batch size before runtime VRAM adaptation.",
        min_value=1,
        source="approved_step0_plan",
        rationale=(
            "Lets the resource guard reduce a user-configured batch size without "
            "hardcoded business logic."
        ),
    ),
    _f(
        "taxonomy_acceptance_confidence",
        _LN,
        0.8,
        "Minimum model confidence for accepting a learned taxonomy proposal.",
        min_value=0.0,
        max_value=1.0,
        source="approved_step0_plan",
        rationale=(
            "Keeps taxonomy acceptance policy owner-configurable while preserving "
            "empty-start behavior."
        ),
    ),
    _f(
        "taxonomy_ambiguity_confidence",
        _LN,
        0.5,
        "Confidence boundary below which taxonomy proposals require ambiguity review.",
        min_value=0.0,
        max_value=1.0,
        source="approved_step0_plan",
        rationale="Routes uncertain learned categories to review using a configurable policy.",
    ),
    _f(
        "taxonomy_propose_threshold",
        _LN,
        0.6,
        "Minimum model confidence required to propose a new taxonomy term.",
        min_value=0.0,
        max_value=1.0,
        source="approved_step0_plan",
        rationale=(
            "Keeps model-driven taxonomy term proposal threshold configurable and "
            "prevents random noise from creating proposed terms."
        ),
    ),
    _i(
        "taxonomy_promote_min_hits",
        _LN,
        3,
        "Minimum occurrences before a proposed taxonomy term promotes to active.",
        min_value=1,
        source="approved_step0_plan",
        rationale=(
            "Requires repeated observations before auto-promoting proposed terms to "
            "active vocabulary."
        ),
    ),
    _f(
        "consolidation_confidence_threshold",
        _LN,
        0.85,
        "Minimum consolidated-model confidence before accepting a relation without escalation.",
        min_value=0.0,
        max_value=1.0,
        source="approved_step0_plan",
        rationale=(
            "Makes model-versus-escalation routing configurable rather than "
            "fixed in relation logic."
        ),
    ),
    _b(
        "agent_workflows_enabled",
        _TE,
        True,
        "Enable durable multi-agent workflow execution.",
        source="approved_step0_plan",
        rationale=(
            "Provides the required runtime kill switch for real database-backed "
            "agent execution."
        ),
    ),
    _i(
        "agent_max_steps",
        _TE,
        32,
        "Maximum execution steps allowed for one durable agent run.",
        min_value=1,
        source="approved_step0_plan",
        rationale="Moves the agent step budget into settings snapshots instead of engine literals.",
    ),
    _i(
        "agent_max_model_calls",
        _TE,
        16,
        "Maximum model calls allowed for one durable agent run.",
        min_value=1,
        source="approved_step0_plan",
        rationale="Makes the per-run model-call budget configurable and enforceable.",
    ),
    _i(
        "agent_timeout_seconds",
        _TE,
        900,
        "Maximum wall-clock execution time for one durable agent run.",
        min_value=1,
        source="approved_step0_plan",
        rationale="Moves the agent time budget into the settings service for runtime governance.",
    ),
)

# --- assembly --------------------------------------------------------------

# All specs, in declaration order. ``build_index`` fails loudly on a duplicate
# key, so a catalog typo is caught at import time.
CATALOG: tuple[SettingSpec, ...] = (
    CORE_BEHAVIOR
    + REASONING
    + TASK_EXECUTION
    + VERIFICATION
    + QUALITY_GATE
    + LEARNING
    + MEMORY
    + BEHAVIOR_RULES
    + TOOLS
    + WORKFLOW_RULES
    + SECURITY
    + STEP0_FOUNDATION
)

SPECS_BY_KEY: dict[str, SettingSpec] = build_index(CATALOG)


def specs_in_category(category: SettingCategory) -> list[SettingSpec]:
    """Specs whose primary category OR also_in_categories includes ``category``.

    This is what the Phase 6 UI sidebar reads so a Q6 key appears under both of
    its sections while still resolving to one stored value.
    """
    out = []
    for spec in CATALOG:
        if spec.category is category or category in spec.also_in_categories:
            out.append(spec)
    return sorted(out, key=lambda s: (s.sort_order, s.key))
