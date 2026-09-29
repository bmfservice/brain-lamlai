"""Typed setting schema (Brain Second Phase 1a).

One ``SettingSpec`` describes one setting key: its type, bounds, allowed values,
risk, restart semantics and provenance. This is the *typed* alternative to the
"dictionary of string→object" the spec forbids — every consumer (store, resolver
in 1b, API in 1c, UI in Phase 6) reads the same declared shape instead of
guessing what a value means.

Two vocabularies here are load-bearing and deliberately closed:

- ``SettingScope`` is the seven-level Priority Hierarchy. Its declaration ORDER
  *is* the precedence (``rank`` 0 = highest). Phase 1b's resolver walks scopes
  in this order and takes the first that has a value, so reordering the enum
  silently re-prioritizes the whole system. It must never be reordered.
- ``SettingCategory`` is the 11 UI sidebar sections from the spec.

Validation is real, not a stub: an out-of-range, wrong-typed or disallowed value
raises ``SettingValueError`` with the reason, and the caller stores nothing.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.textanalysis import normalize_label

__all__ = [
    "DataType",
    "HardRuleViolation",
    "RiskLevel",
    "SettingCategory",
    "SettingScope",
    "SettingSpec",
    "SettingValueError",
    "build_index",
    "validate",
]


class DataType(StrEnum):
    """The value types a setting may declare."""

    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    STR = "str"
    ENUM = "enum"
    JSON = "json"
    STRING_LIST = "string_list"


class RiskLevel(StrEnum):
    """How dangerous changing this setting is (drives the Phase 6 risk badge)."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SettingScope(StrEnum):
    """The seven-level Priority Hierarchy, in DESCENDING precedence.

    Declaration order is the contract: ``rank`` is the position, and the Phase 1b
    resolver scans scopes in this exact order, taking the first with a value.
    ``SYSTEM_HARD`` always wins; ``LEARNED`` always loses — which is what makes
    the mandatory "Learned Behavior cannot override System Hard Rules" test pass
    structurally rather than by a special case.
    """

    SYSTEM_HARD = "system_hard"
    SECURITY = "security"
    PROJECT = "project"
    TASK = "task"
    USER_PREFERENCE = "user_preference"
    TEMPORARY = "temporary"
    LEARNED = "learned"

    @property
    def rank(self) -> int:
        """0 = highest precedence. Derived from declaration order, never hardcoded."""
        return list(type(self)).index(self)


class SettingCategory(StrEnum):
    """The 11 settings categories (Phase 6 sidebar sections)."""

    CORE_BEHAVIOR = "core_behavior"
    REASONING = "reasoning"
    TASK_EXECUTION = "task_execution"
    VERIFICATION = "verification"
    QUALITY_GATE = "quality_gate"
    LEARNING = "learning"
    MEMORY = "memory"
    BEHAVIOR_RULES = "behavior_rules"
    TOOLS = "tools"
    WORKFLOW_RULES = "workflow_rules"
    SECURITY = "security"


class SettingValueError(ValueError):
    """A value failed schema validation. Carries a human-readable reason."""


class HardRuleViolation(SettingValueError):
    """A value would violate a System Hard Rule (see ``hard_rules.py``).

    Subclasses ``SettingValueError`` so callers that only catch validation errors
    still stop the write — a hard-rule breach is the strongest kind of invalid.
    """


@dataclass(frozen=True)
class SettingSpec:
    """Immutable definition of one setting key."""

    key: str
    category: SettingCategory
    data_type: DataType
    default_value: Any
    description: str
    also_in_categories: tuple[SettingCategory, ...] = ()
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: tuple[str, ...] | None = None
    detailed_description: str | None = None
    risk_level: RiskLevel = RiskLevel.LOW
    requires_restart: bool = False
    is_sensitive: bool = False
    is_critical: bool = False
    validation_rules: Mapping[str, Any] = field(default_factory=dict)
    source: str = "appendix_a"
    rationale: str | None = None
    sort_order: int = 0

    def __post_init__(self) -> None:
        # Catch authoring mistakes at import time, not at first seed: an enum
        # without allowed_values, a numeric bound on a non-numeric type, or a
        # default that fails its own validation are all catalog bugs.
        if self.data_type is DataType.ENUM and not self.allowed_values:
            raise ValueError(f"{self.key}: ENUM requires allowed_values")
        if self.data_type is DataType.STRING_LIST and self.allowed_values is None:
            raise ValueError(f"{self.key}: STRING_LIST requires allowed_values (even if empty)")
        if (self.min_value is not None or self.max_value is not None) and self.data_type not in (
            DataType.INT,
            DataType.FLOAT,
        ):
            raise ValueError(f"{self.key}: min/max_value only valid on INT/FLOAT")
        validate(self, self.default_value)

    # --- serialization to/from the DB row --------------------------------

    def to_row(self) -> dict[str, Any]:
        """Render as a dict matching ``SettingDefinition`` columns."""
        return {
            "key": self.key,
            "category": str(self.category),
            "also_in_categories": [str(c) for c in self.also_in_categories],
            "data_type": str(self.data_type),
            "default_value": _wrap(self.default_value),
            "min_value": self.min_value,
            "max_value": self.max_value,
            "allowed_values": (
                list(self.allowed_values) if self.allowed_values is not None else None
            ),
            "description": self.description,
            "detailed_description": self.detailed_description,
            "risk_level": str(self.risk_level),
            "requires_restart": self.requires_restart,
            "is_sensitive": self.is_sensitive,
            "is_critical": self.is_critical,
            "validation_rules": dict(self.validation_rules),
            "source": self.source,
            "rationale": self.rationale,
            "sort_order": self.sort_order,
        }

    @classmethod
    def from_row(cls, row: Any) -> SettingSpec:
        """Rebuild a spec from a ``SettingDefinition`` ORM row."""
        return cls(
            key=row.key,
            category=SettingCategory(row.category),
            data_type=DataType(row.data_type),
            default_value=_unwrap(row.default_value),
            description=row.description,
            also_in_categories=tuple(
                SettingCategory(c) for c in (row.also_in_categories or [])
            ),
            min_value=row.min_value,
            max_value=row.max_value,
            allowed_values=(
                tuple(row.allowed_values) if row.allowed_values is not None else None
            ),
            detailed_description=row.detailed_description,
            risk_level=RiskLevel(row.risk_level),
            requires_restart=bool(row.requires_restart),
            is_sensitive=bool(row.is_sensitive),
            is_critical=bool(row.is_critical),
            validation_rules=dict(row.validation_rules or {}),
            source=row.source,
            rationale=row.rationale,
            sort_order=int(row.sort_order or 0),
        )


# --- value validation -------------------------------------------------------


def _wrap(value: Any) -> Any:
    """JSONB columns reject a bare Python ``None``/scalar ambiguity, so every
    stored value is wrapped as ``{"v": value}``. One convention, applied on both
    write (``to_row``/store) and read (``_unwrap``), keeps round-trips exact."""
    return {"v": value}


def _unwrap(stored: Any) -> Any:
    """Reverse :func:`_wrap`. Tolerates a legacy bare value (no ``v`` key)."""
    if isinstance(stored, Mapping) and "v" in stored:
        return stored["v"]
    return stored


def validate(spec: SettingSpec, raw: Any) -> Any:
    """Validate ``raw`` against ``spec`` and return the coerced value.

    Raises :class:`SettingValueError` with a specific reason on any mismatch —
    never silently "fixes" the value. ``raw`` may arrive wrapped (from the DB) or
    bare (from a caller); both are accepted.
    """
    value = _unwrap(raw)
    dt = spec.data_type

    if dt is DataType.BOOL:
        if isinstance(value, bool):
            coerced: Any = value
        elif isinstance(value, str) and value.lower() in ("true", "false"):
            coerced = value.lower() == "true"
        elif isinstance(value, (int, float)) and value in (0, 1):
            coerced = bool(value)
        else:
            raise SettingValueError(f"{spec.key}: expected bool, got {type(value).__name__}")
    elif dt is DataType.INT:
        if isinstance(value, bool) or not isinstance(value, int):
            # bool is a subclass of int in Python; reject it explicitly so a
            # true/false never sneaks into an integer setting.
            raise SettingValueError(f"{spec.key}: expected int, got {type(value).__name__}")
        coerced = value
    elif dt is DataType.FLOAT:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SettingValueError(f"{spec.key}: expected float, got {type(value).__name__}")
        coerced = float(value)
    elif dt is DataType.STR:
        if not isinstance(value, str):
            raise SettingValueError(f"{spec.key}: expected str, got {type(value).__name__}")
        coerced = value
    elif dt is DataType.ENUM:
        if not isinstance(value, str):
            raise SettingValueError(f"{spec.key}: enum value must be a str")
        allowed = frozenset(v.lower() for v in (spec.allowed_values or ()))
        # Pure string normalization (case/whitespace/separator) from
        # core.textanalysis.normalize_label. It deliberately does NOT go through
        # the knowledge taxonomy, which is now a DB-backed open vocabulary.
        normalized = normalize_label(value)
        if normalized not in allowed:
            raise SettingValueError(
                f"{spec.key}: {value!r} not in allowed {sorted(allowed)}"
            )
        coerced = normalized
    elif dt is DataType.STRING_LIST:
        if not isinstance(value, (list, tuple)):
            raise SettingValueError(f"{spec.key}: expected a list of str")
        items = list(value)
        if not all(isinstance(x, str) for x in items):
            raise SettingValueError(f"{spec.key}: every list item must be a str")
        if spec.allowed_values is not None:
            allowed = frozenset(v.lower() for v in spec.allowed_values)
            bad = [x for x in items if x.lower() not in allowed]
            if bad:
                raise SettingValueError(f"{spec.key}: disallowed list items {bad}")
        coerced = items
    elif dt is DataType.JSON:
        # Any JSON-serializable value is structurally valid; reject only what
        # cannot round-trip through JSONB.
        try:
            json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise SettingValueError(f"{spec.key}: not JSON-serializable ({exc})") from exc
        coerced = value
    else:  # pragma: no cover - StrEnum is closed; unreachable unless extended
        raise SettingValueError(f"{spec.key}: unknown data_type {dt}")

    coerced = _apply_bounds(spec, coerced)
    coerced = _apply_rules(spec, coerced)
    return coerced


def _apply_bounds(spec: SettingSpec, value: Any) -> Any:
    """Enforce min_value/max_value on numeric types."""
    if spec.data_type in (DataType.INT, DataType.FLOAT):
        if spec.min_value is not None and value < spec.min_value:
            raise SettingValueError(f"{spec.key}: {value} < min {spec.min_value}")
        if spec.max_value is not None and value > spec.max_value:
            raise SettingValueError(f"{spec.key}: {value} > max {spec.max_value}")
    return value


def _apply_rules(spec: SettingSpec, value: Any) -> Any:
    """Enforce the declarative ``validation_rules`` (pattern/gt/gte/lt/lte/
    max_len/min_items/max_items). Unknown rule names are rejected rather than
    ignored, so a typo cannot silently disable a guard."""
    rules = spec.validation_rules or {}
    for name, expected in rules.items():
        if name == "pattern":
            if not isinstance(value, str) or not re.fullmatch(expected, value):
                raise SettingValueError(f"{spec.key}: {value!r} does not match /{expected}/")
        elif name in ("gt", "gte", "lt", "lte"):
            cmp_ok = {
                "gt": lambda a, b: a > b,
                "gte": lambda a, b: a >= b,
                "lt": lambda a, b: a < b,
                "lte": lambda a, b: a <= b,
            }[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SettingValueError(f"{spec.key}: rule {name} needs a number")
            if not cmp_ok(value, expected):
                raise SettingValueError(f"{spec.key}: {value} fails {name} {expected}")
        elif name == "max_len":
            if not isinstance(value, (str, Sequence)) or len(value) > expected:
                raise SettingValueError(f"{spec.key}: length exceeds max_len {expected}")
        elif name == "min_items":
            if not isinstance(value, Sequence) or len(value) < expected:
                raise SettingValueError(f"{spec.key}: fewer than min_items {expected}")
        elif name == "max_items":
            if not isinstance(value, Sequence) or len(value) > expected:
                raise SettingValueError(f"{spec.key}: more than max_items {expected}")
        else:
            raise SettingValueError(f"{spec.key}: unknown validation rule {name!r}")
    return value


def build_index(specs: Sequence[SettingSpec]) -> dict[str, SettingSpec]:
    """Index specs by key, failing loudly on a duplicate (a catalog bug)."""
    index: dict[str, SettingSpec] = {}
    for spec in specs:
        if spec.key in index:
            raise ValueError(f"duplicate setting key in catalog: {spec.key}")
        index[spec.key] = spec
    return index
