"""Simulation Mode & Explainability Tests (Phase 7).

Covers:
- Scenario #7: Simulation mode executes complex task and produces operation preview
  with 0 actual state changes / 0 DB mutations.
- Settings explainability: walks the 7-scope hierarchy, explains winning scope,
  overridden values, rationale, and masks sensitive credentials (HR-NO-CREDENTIAL-LEAK).
- Action explainability: explains why actions were allowed/denied based on trust level
  and governance constraints.
"""

from __future__ import annotations

import asyncio

from services.audit.explain import ExplainabilityService
from services.governance.tool_policy import TrustLevel
from services.orchestration.simulation import SimulationRunner


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Test Scenario #7: Simulation Mode Executes Complex Task With 0 State Changes
# ---------------------------------------------------------------------------


def test_scenario_7_simulation_mode_zero_state_changes():
    """Scenario #7: Complex multi-step task executed in dry-run mode.
    Mutating tools and DB writes are intercepted and previewed.
    Invariant: actual_db_mutations == 0.
    """
    runner = SimulationRunner()

    # Complex multi-step task involving read, write, delete, and settings change
    planned_steps = [
        {"tool": "search_knowledge", "args": {"query": "legacy database entries"}},
        {"tool": "read_document", "args": {"doc_id": "doc-99"}},
        {"tool": "write_database_record", "args": {"table": "users", "name": "Alice"}},
        {"tool": "delete_outdated_data", "args": {"before_date": "2024-01-01"}},
        {"tool": "update_setting", "args": {"key": "allow_assumption", "value": True}},
    ]

    result = _run(
        runner.run(
            task_name="Migrate Legacy User Knowledge",
            planned_steps=planned_steps,
            trust_level=TrustLevel.USER,
            dry_run=True,
        )
    )

    # Invariant: 0 actual DB mutations
    assert result.dry_run is True
    assert result.actual_db_mutations == 0
    assert result.total_steps == 5

    # 3 mutating actions were intercepted (write, delete, setting update)
    assert len(result.mutations_intercepted) == 3
    assert result.simulated_state_changes_count == 3

    # Check intercepted details
    intercepted_tools = [m.tool_name for m in result.mutations_intercepted]
    assert "write_database_record" in intercepted_tools
    assert "delete_outdated_data" in intercepted_tools
    assert "update_setting" in intercepted_tools

    # Read-only steps were simulated cleanly without interception
    read_steps = [s for s in result.step_results if s.status == "simulated_read"]
    assert len(read_steps) == 2
    assert read_steps[0].tool_name == "search_knowledge"
    assert read_steps[1].tool_name == "read_document"

    # Preview summary contains human-readable report
    assert "Simulation Report for Task: 'Migrate Legacy User Knowledge'" in result.preview_summary
    assert "Actual DB Mutations: 0 (Strict Invariant: 0)" in result.preview_summary
    assert "[DRY RUN PREVIEW]" in result.preview_summary


def test_simulation_denies_unauthorized_step_for_document_trust():
    """Simulation accurately reflects policy denials for low trust levels."""
    runner = SimulationRunner()

    planned_steps = [
        {"tool": "search_knowledge", "args": {"query": "test"}},
        {"tool": "delete_user_data", "args": {"id": "123"}},
    ]

    # Running with DOCUMENT trust level
    result = _run(
        runner.run(
            task_name="Document-Triggered Task",
            planned_steps=planned_steps,
            trust_level=TrustLevel.DOCUMENT,
            dry_run=True,
        )
    )

    assert result.can_execute_safely is False
    assert result.step_results[0].status == "simulated_read"
    assert result.step_results[0].allowed_by_policy is True

    # Mutating tool is denied for DOCUMENT trust level
    assert result.step_results[1].status == "denied"
    assert result.step_results[1].allowed_by_policy is False
    assert "cannot trigger mutating tools" in result.step_results[1].policy_reason


# ---------------------------------------------------------------------------
# Settings Explainability Tests
# ---------------------------------------------------------------------------


def test_settings_explainability_hierarchy_resolution():
    """Explainability service explains winning scope, overridden values, and rationale."""
    explainer = ExplainabilityService()

    # Provide values at PROJECT and USER_PREFERENCE
    scope_values = {
        "user_preference": "ask",
        "project": "escalate",
        "temporary": "retry",
    }

    explanation = explainer.explain_setting("when_data_conflict", scope_values=scope_values)

    # PROJECT (rank 2) beats USER_PREFERENCE (rank 4) and TEMPORARY (rank 5)
    assert explanation.winning_scope == "project"
    assert explanation.resolved_value == "escalate"
    assert explanation.key == "when_data_conflict"
    assert "Never silently pick one" in (explanation.rationale or "")

    # Check that USER_PREFERENCE and TEMPORARY are marked as overridden
    statuses = {e.scope: e.status for e in explanation.evaluations}
    assert statuses["project"] == "winner"
    assert statuses["user_preference"] == "overridden"
    assert statuses["temporary"] == "overridden"
    assert statuses["system_hard"] == "unset"

    # Explanation narrative
    assert "resolved to 'escalate' via scope 'project'" in explanation.explanation
    assert "Higher-precedence scope overrode: user_preference ('ask')" in explanation.explanation


def test_settings_explainability_catalog_default_fallback():
    """When no scope has a value, explainability explains catalog default resolution."""
    explainer = ExplainabilityService()

    explanation = explainer.explain_setting("tool_max_calls", scope_values={})

    assert explanation.winning_scope == "catalog_default"
    assert explanation.resolved_value == 8
    assert "resolved to default value '8' from catalog" in explanation.explanation


def test_settings_explainability_masks_sensitive_values():
    """Sensitive settings (e.g. secrets/tokens) are masked as '***' (HR-NO-CREDENTIAL-LEAK)."""
    explainer = ExplainabilityService()

    # Even if an audit or user queries a sensitive key, values are masked
    explanation = explainer.explain_setting(
        "admin_token",
        scope_values={"system_hard": "super_secret_token_12345"},
    )

    # Sensitive values must be masked
    if explanation.is_sensitive:
        for eval_rec in explanation.evaluations:
            if eval_rec.value is not None:
                assert eval_rec.value == "***"
        assert "super_secret_token" not in explanation.explanation


# ---------------------------------------------------------------------------
# Action Explainability Tests
# ---------------------------------------------------------------------------


def test_action_explainability_allowed_and_denied():
    """Explains why an action was allowed or denied with clear factors."""
    explainer = ExplainabilityService()

    # Explain allowed read action
    exp_allow = explainer.explain_action(
        "read_document",
        verdict="allow",
        trust_level=TrustLevel.USER,
        is_mutating=False,
    )
    assert exp_allow.allowed is True
    assert exp_allow.verdict == "allow"
    assert "Action 'read_document' was ALLOWED" in exp_allow.explanation
    assert "Action is read-only" in exp_allow.factors

    # Explain denied mutating action from DOCUMENT context
    exp_deny = explainer.explain_action(
        "delete_database",
        verdict="deny",
        trust_level=TrustLevel.DOCUMENT,
        is_mutating=True,
    )
    assert exp_deny.allowed is False
    assert exp_deny.verdict == "deny"
    assert "cannot execute mutating tools" in exp_deny.explanation
    assert "HR-PROMPT-INJECTION-DEFENSE constraint" in exp_deny.factors
