"""Unit tests — settings schema validation (Brain Second Phase 1a).

Không DB: chứng minh validate, SettingScope precedence, catalog invariants,
hard_rules.assert_not_violated.
"""

from __future__ import annotations

import re

import pytest

from services.settings.catalog import CATALOG
from services.settings.hard_rules import SYSTEM_HARD_RULES, assert_not_violated
from services.settings.schema import (
    DataType,
    HardRuleViolation,
    SettingCategory,
    SettingScope,
    SettingSpec,
    SettingValueError,
    validate,
)

# --- validate per data_type -------------------------------------------------


def test_validate_bool():
    spec = SettingSpec(
        key="test_bool",
        category=SettingCategory.CORE_BEHAVIOR,
        data_type=DataType.BOOL,
        default_value=True,
        description="test",
    )
    assert validate(spec, True) is True
    assert validate(spec, "true") is True
    assert validate(spec, "FALSE") is False
    assert validate(spec, 1) is True
    assert validate(spec, 0) is False
    with pytest.raises(SettingValueError, match="expected bool"):
        validate(spec, "yes")


def test_validate_int_rejects_bool():
    """bool là subclass của int trong Python → phải reject riêng."""
    spec = SettingSpec(
        key="test_int",
        category=SettingCategory.TASK_EXECUTION,
        data_type=DataType.INT,
        default_value=5,
        description="test",
    )
    assert validate(spec, 42) == 42
    with pytest.raises(SettingValueError, match="expected int"):
        validate(spec, True)  # bool rejected
    with pytest.raises(SettingValueError, match="expected int"):
        validate(spec, 3.14)


def test_validate_float():
    spec = SettingSpec(
        key="test_float",
        category=SettingCategory.LEARNING,
        data_type=DataType.FLOAT,
        default_value=0.5,
        description="test",
    )
    assert validate(spec, 3.14) == 3.14
    assert validate(spec, 7) == 7.0  # int → float OK
    with pytest.raises(SettingValueError, match="expected float"):
        validate(spec, True)


def test_validate_str():
    spec = SettingSpec(
        key="test_str",
        category=SettingCategory.CORE_BEHAVIOR,
        data_type=DataType.STR,
        default_value="hello",
        description="test",
    )
    assert validate(spec, "world") == "world"
    with pytest.raises(SettingValueError, match="expected str"):
        validate(spec, 123)


def test_validate_enum_via_normalize_label():
    """Enum dùng core.textanalysis.normalize_label → case/separator-insensitive."""
    spec = SettingSpec(
        key="test_enum",
        category=SettingCategory.REASONING,
        data_type=DataType.ENUM,
        default_value="normal",
        allowed_values=("low", "normal", "deep"),
        description="test",
    )
    assert validate(spec, "DEEP") == "deep"
    assert validate(spec, "Low") == "low"
    with pytest.raises(SettingValueError, match="not in allowed"):
        validate(spec, "ultra")


def test_validate_string_list():
    spec = SettingSpec(
        key="test_list",
        category=SettingCategory.TOOLS,
        data_type=DataType.STRING_LIST,
        default_value=["a", "b"],
        allowed_values=("a", "b", "c"),
        description="test",
    )
    assert validate(spec, ["b", "c"]) == ["b", "c"]
    with pytest.raises(SettingValueError, match="disallowed list items"):
        validate(spec, ["a", "x"])
    with pytest.raises(SettingValueError, match="expected a list"):
        validate(spec, "not-a-list")


def test_validate_json_round_trip():
    spec = SettingSpec(
        key="test_json",
        category=SettingCategory.CORE_BEHAVIOR,
        data_type=DataType.JSON,
        default_value={"a": 1},
        description="test",
    )
    assert validate(spec, [1, 2, 3]) == [1, 2, 3]
    assert validate(spec, None) is None
    # Non-serializable → raise
    with pytest.raises(SettingValueError, match="not JSON-serializable"):
        validate(spec, {"bad": lambda: None})


# --- bounds + validation_rules ----------------------------------------------


def test_bounds_min_max():
    spec = SettingSpec(
        key="bounded",
        category=SettingCategory.TASK_EXECUTION,
        data_type=DataType.INT,
        default_value=5,
        min_value=0,
        max_value=10,
        description="test",
    )
    assert validate(spec, 7) == 7
    with pytest.raises(SettingValueError, match="< min"):
        validate(spec, -1)
    with pytest.raises(SettingValueError, match="> max"):
        validate(spec, 20)


def test_validation_rules_pattern():
    spec = SettingSpec(
        key="pattern_test",
        category=SettingCategory.CORE_BEHAVIOR,
        data_type=DataType.STR,
        default_value="abc",
        description="test",
        validation_rules={"pattern": r"[a-z]{3}"},
    )
    assert validate(spec, "xyz") == "xyz"
    with pytest.raises(SettingValueError, match="does not match"):
        validate(spec, "ABC")


def test_validation_rules_gt_gte():
    spec = SettingSpec(
        key="gt_test",
        category=SettingCategory.LEARNING,
        data_type=DataType.FLOAT,
        default_value=0.5,
        description="test",
        validation_rules={"gt": 0.0, "lte": 1.0},
    )
    assert validate(spec, 0.5) == 0.5
    with pytest.raises(SettingValueError, match="fails gt"):
        validate(spec, 0.0)
    with pytest.raises(SettingValueError, match="fails lte"):
        validate(spec, 1.1)


def test_validation_rules_max_len():
    spec = SettingSpec(
        key="len_test",
        category=SettingCategory.CORE_BEHAVIOR,
        data_type=DataType.STR,
        default_value="ok",
        description="test",
        validation_rules={"max_len": 10},
    )
    assert validate(spec, "short") == "short"
    with pytest.raises(SettingValueError, match="exceeds max_len"):
        validate(spec, "a" * 20)


def test_validation_rules_min_max_items():
    spec = SettingSpec(
        key="items_test",
        category=SettingCategory.TOOLS,
        data_type=DataType.STRING_LIST,
        default_value=["a"],
        allowed_values=("a", "b", "c"),
        description="test",
        validation_rules={"min_items": 1, "max_items": 3},
    )
    assert validate(spec, ["a", "b"]) == ["a", "b"]
    with pytest.raises(SettingValueError, match="fewer than min_items"):
        validate(spec, [])
    with pytest.raises(SettingValueError, match="more than max_items"):
        validate(spec, ["a", "b", "c", "a"])


def test_validation_rules_unknown_rule_raises():
    """Unknown rule name → raise, so typo can't disable a guard.

    Caught even earlier than validate(): SettingSpec.__post_init__ validates the
    default_value, so constructing a spec with a bad rule name raises immediately.
    """
    with pytest.raises(SettingValueError, match="unknown validation rule"):
        SettingSpec(
            key="bad_rule",
            category=SettingCategory.CORE_BEHAVIOR,
            data_type=DataType.INT,
            default_value=5,
            description="test",
            validation_rules={"typo_gt": 0},
        )


# --- SettingScope precedence ------------------------------------------------


def test_scope_rank_order():
    """Declaration order = precedence: SYSTEM_HARD rank 0, LEARNED rank 6."""
    assert SettingScope.SYSTEM_HARD.rank == 0
    assert SettingScope.SECURITY.rank == 1
    assert SettingScope.PROJECT.rank == 2
    assert SettingScope.TASK.rank == 3
    assert SettingScope.USER_PREFERENCE.rank == 4
    assert SettingScope.TEMPORARY.rank == 5
    assert SettingScope.LEARNED.rank == 6

    # Resolver 1b sẽ duyệt theo rank tăng → SYSTEM_HARD thắng, LEARNED thua
    scopes = list(SettingScope)
    assert scopes[0] is SettingScope.SYSTEM_HARD
    assert scopes[-1] is SettingScope.LEARNED


# --- Catalog invariants -----------------------------------------------------


def test_catalog_keys_unique_snake_case():
    """Mọi key là snake_case duy nhất (không duplicate, không CamelCase)."""
    keys = [spec.key for spec in CATALOG]
    assert len(keys) == len(set(keys)), "duplicate keys found"
    for key in keys:
        assert re.fullmatch(r"[a-z_][a-z0-9_]*", key), f"{key} not snake_case"


def test_catalog_defaults_self_validate():
    """Mọi default_value tự pass validate của chính nó (chống default sai type).

    Đây là mitigate R13: ~100 key hand-written dễ có lỗi. SettingSpec.__post_init__
    đã gọi validate(self, self.default_value) nên import thành công = pass.
    Test này chỉ double-check lại tất cả.
    """
    for spec in CATALOG:
        # __post_init__ đã validate, nhưng re-run để chắc
        coerced = validate(spec, spec.default_value)
        assert coerced is not None  # không raise


def test_catalog_q6_keys_appear_once():
    """Q6: when_missing_information, when_uncertain xuất hiện 1 lần với also_in_categories.

    Không nhân đôi thành 2 row, không mâu thuẫn 2 giá trị.
    """
    q6_keys = {"when_missing_information", "when_uncertain"}
    found = {spec.key: spec for spec in CATALOG if spec.key in q6_keys}
    assert len(found) == 2

    for spec in found.values():
        # Category chính = BEHAVIOR_RULES
        assert spec.category is SettingCategory.BEHAVIOR_RULES
        # also_in_categories chứa CORE_BEHAVIOR
        assert SettingCategory.CORE_BEHAVIOR in spec.also_in_categories


def test_catalog_every_key_has_source_and_rationale():
    """Q7: mọi key có source (appendix_a|derived_phase*) và rationale không rỗng."""
    for spec in CATALOG:
        assert spec.source, f"{spec.key} missing source"
        assert spec.rationale, f"{spec.key} missing rationale"


def test_catalog_derived_keys_marked():
    """5 category thiếu (Verification, Quality Gate, Memory, Workflow Rules, Security)
    có key source='derived_phase*'.
    """
    derived = [s for s in CATALOG if s.source.startswith("derived_phase")]
    assert len(derived) > 0, "no derived keys found"

    # Derived keys phải thuộc 1 trong 5 category thiếu (hoặc Tools — đã có trong spec)
    derived_cats = {
        SettingCategory.VERIFICATION,
        SettingCategory.QUALITY_GATE,
        SettingCategory.MEMORY,
        SettingCategory.WORKFLOW_RULES,
        SettingCategory.SECURITY,
    }
    for spec in derived:
        assert (
            spec.category in derived_cats
        ), f"{spec.key} derived but category={spec.category} not in expected set"


# --- Hard rules -------------------------------------------------------------


def test_hard_rules_defined():
    """6 System Hard Rules tồn tại với id/statement/enforcement."""
    assert len(SYSTEM_HARD_RULES) == 6
    ids = {r.id for r in SYSTEM_HARD_RULES}
    assert "HR-NO-HARDCODE" in ids
    assert "HR-NO-FAKE-VERIFICATION" in ids
    assert "HR-NO-PERMANENT-DELETE" in ids
    assert "HR-NO-CREDENTIAL-LEAK" in ids
    assert "HR-NO-CN-MODEL" in ids
    assert "HR-HARD-RULES-IMMUTABLE" in ids


def test_hard_rule_cn_model_blocks():
    """HR-NO-CN-MODEL chặn model Trung Quốc (Qwen/DeepSeek/GLM/...)."""
    # Catalog không có llm_model/embed_model (thuộc .env), nhưng hard_rules._GUARDS
    # có guard cho chúng. Test assert_not_violated trực tiếp.
    with pytest.raises(HardRuleViolation, match="HR-NO-CN-MODEL"):
        assert_not_violated("llm_model", "Qwen/Qwen2-7B-Instruct")

    with pytest.raises(HardRuleViolation, match="HR-NO-CN-MODEL"):
        assert_not_violated("embed_model", "deepseek-ai/deepseek-coder-1.3b")

    # Các model khác OK
    assert_not_violated("llm_model", "anthropic/claude-sonnet-4-20250514")
    assert_not_violated("llm_model", "google/gemma-2-9b-it")


def test_hard_rule_unknown_key_passes():
    """Key không có guard → pass (enforcement ở chỗ khác, không phải value check)."""
    assert_not_violated("reasoning_depth", "deep")  # không raise
    assert_not_violated("unknown_key_xyz", "anything")  # không raise
