"""System Hard Rules (Brain Second Phase 1a).

These live in CODE, not in a database row, on purpose. The Priority Hierarchy's
top level is ``SYSTEM_HARD``; if hard rules were stored as ordinary setting
values, a sufficiently privileged write (or a learned-behavior row that somehow
got promoted) could rewrite them. As module constants they cannot be edited at
runtime at all, which is what makes the mandatory test — "Learned Behavior never
overrides a System Hard Rule" — pass *structurally* rather than by a guard that
could itself be bypassed.

Each rule names the code that enforces it, so the list is an index into real
checks rather than aspirational prose. ``assert_not_violated`` is called by
``SettingsStore.set_value`` before any write, so no scope (including PROJECT or
an admin API call) can persist a setting value that contradicts a hard rule.

The rules are drawn from the user's standing blueprint, not invented here:
no hardcoding, no fake verification, no permanent deletion of knowledge, no
credential leakage, no Chinese-model usage.
"""

from __future__ import annotations

from dataclasses import dataclass

from services.settings.schema import HardRuleViolation

__all__ = ["HardRule", "SYSTEM_HARD_RULES", "assert_not_violated"]


@dataclass(frozen=True)
class HardRule:
    """One inviolable rule: what it states and where it is enforced."""

    id: str
    statement: str
    enforcement: str  # the module/mechanism that actually upholds it


SYSTEM_HARD_RULES: tuple[HardRule, ...] = (
    HardRule(
        id="HR-NO-HARDCODE",
        statement=(
            "Knowledge, categories and behavior come from data the system learned "
            "or from declared settings — never from values hardcoded in business "
            "logic. Settings defaults live in the catalog and the DB, not in "
            "branches of running code."
        ),
        enforcement=(
            "settings catalog + setting_definitions are the source of truth; "
            "core/taxonomy.coerce maps model output onto closed vocabularies "
            "instead of letting new categories be invented inline."
        ),
    ),
    HardRule(
        id="HR-NO-FAKE-VERIFICATION",
        statement=(
            "No verification function may return success without a real check "
            "against actual data or results. A validator that always passes is a "
            "defect, not a convenience."
        ),
        enforcement=(
            "Phase 3 validators must each have a test proving they detect a "
            "failure; deterministic checks run before any LLM reviewer."
        ),
    ),
    HardRule(
        id="HR-NO-PERMANENT-DELETE",
        statement=(
            "Knowledge is never permanently deleted by automated passes. "
            "Contradicted, duplicated or ambiguous data is moved to quarantine "
            "(reversible), and only verifiably corrupt data is removed."
        ),
        enforcement=(
            "quarantined_knowledge table written by "
            "services/consolidation/job.py; GraphService.deactivate_edge marks "
            "lifecycle='superseded' rather than deleting rows."
        ),
    ),
    HardRule(
        id="HR-NO-CREDENTIAL-LEAK",
        statement=(
            "Secrets (database password, admin token, API keys) are never written "
            "into knowledge, logs, audit rows or model context."
        ),
        enforcement=(
            "settings marked is_sensitive are masked by the API/UI; audit and "
            "history store only the setting key and value shape, never .env "
            "secrets; Phase 7 audit excludes sensitive keys."
        ),
    ),
    HardRule(
        id="HR-NO-CN-MODEL",
        statement=(
            "Inference and embedding models must not be Chinese-origin models. "
            "Model choice is configuration, but it may not select a banned source."
        ),
        enforcement=(
            "model ids come from .env (LLM_MODEL/EMBED_MODEL); this rule is the "
            "policy basis for rejecting a banned model at config-change time."
        ),
    ),
    HardRule(
        id="HR-HARD-RULES-IMMUTABLE",
        statement=(
            "The set of System Hard Rules and their precedence cannot be lowered "
            "by any setting write, learned behavior or runtime request. They are "
            "module constants, editable only by changing this file."
        ),
        enforcement="services/settings/hard_rules.py is code, not a DB row.",
    ),
)


# Keys whose *value* could be used to weaken a hard rule. The store refuses a
# write that would set any of these to a hard-rule-contradicting value. The map
# is key -> (rule_id, predicate(value) that means "violates the rule").
#
# This is deliberately small and explicit: a hard rule that cannot be expressed
# as a concrete value check is enforced by the mechanism named in its
# ``enforcement`` field, not faked here with a ``return True``.
def _is_cn_model(value: object) -> bool:
    """True if a model id looks Chinese-origin (banned by HR-NO-CN-MODEL).

    Matches the well-known CN model families by name fragment. Conservative: an
    unknown vendor is allowed (the rule bans a specific origin, not novelty).
    """
    if not isinstance(value, str):
        return False
    lowered = value.lower()
    banned_fragments = ("qwen", "yi-", "deepseek", "glm", "chatglm", "baichuan", "ernie", "hunyuan")
    return any(frag in lowered for frag in banned_fragments)


_GUARDS = {
    # A model setting may not be pointed at a banned-origin model.
    "llm_model": ("HR-NO-CN-MODEL", _is_cn_model),
    "embed_model": ("HR-NO-CN-MODEL", _is_cn_model),
}


def assert_not_violated(key: str, value: object) -> None:
    """Raise :class:`HardRuleViolation` if writing ``value`` to ``key`` would
    breach a System Hard Rule.

    Called by ``SettingsStore.set_value`` before persisting. Keys with no guard
    pass through — the hard rules that bind them are upheld by the enforcement
    mechanism named in the rule, not by a value check.
    """
    guard = _GUARDS.get(key)
    if guard is None:
        return
    rule_id, predicate = guard
    if predicate(value):
        rule = next((r for r in SYSTEM_HARD_RULES if r.id == rule_id), None)
        raise HardRuleViolation(
            f"{key}={value!r} violates {rule_id}"
            + (f": {rule.statement}" if rule else "")
        )
