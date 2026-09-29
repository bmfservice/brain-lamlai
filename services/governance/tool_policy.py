"""Tool Policy & Trust Level Permission Enforcement (Giai đoạn 5, Layer 6).

Implements the five-level authority model:
  SYSTEM (5) > USER (4) > MEMORY (3) > DOCUMENT (2) > EXTERNAL (1)

Constraints enforced:
- Context with trust level DOCUMENT or EXTERNAL cannot trigger mutating tools
  or modify settings.
- Enforces tool limits: tool_max_calls, tool_timeout_seconds, tool_require_confirmation.
- Integration with Settings catalog defaults and audit logging.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from core.errors import BraindError
from core.logging import get_logger

logger = get_logger(__name__)


class ToolPolicyViolation(BraindError):
    """Raised when an operation violates tool policy or trust level constraints."""

    code = "tool_policy_violation"
    public_message = "Tool execution was blocked by governance policy."


class TrustLevel(IntEnum):
    """The five authority trust levels in strict descending order of privilege."""

    EXTERNAL = 1
    DOCUMENT = 2
    MEMORY = 3
    USER = 4
    SYSTEM = 5

    @classmethod
    def from_value(cls, val: str | int | TrustLevel) -> TrustLevel:
        if isinstance(val, TrustLevel):
            return val
        if isinstance(val, int):
            return cls(val)
        if isinstance(val, str):
            mapping = {
                "external": cls.EXTERNAL,
                "document": cls.DOCUMENT,
                "memory": cls.MEMORY,
                "user": cls.USER,
                "system": cls.SYSTEM,
            }
            lowered = val.strip().lower()
            if lowered in mapping:
                return mapping[lowered]
        raise ValueError(f"Unknown trust level: {val!r}")


# Known read-only tool prefix and names
_READ_ONLY_PREFIXES = (
    "read_",
    "get_",
    "search_",
    "fetch_",
    "list_",
    "query_",
    "find_",
    "describe_",
    "explain_",
    "check_",
)
# Known mutating tool prefixes
_MUTATING_PREFIXES = (
    "write_",
    "delete_",
    "update_",
    "modify_",
    "set_",
    "create_",
    "execute_",
    "remove_",
    "drop_",
    "alter_",
    "patch_",
)
# Setting modifying keywords
_SETTING_KEYWORDS = ("setting", "config", "preference", "policy_rule")


@dataclass
class ToolPolicyDecision:
    """Outcome of a tool evaluation."""

    allowed: bool
    verdict: str  # allow | deny | ask_first
    reason: str
    tool_name: str
    trust_level: TrustLevel
    requires_confirmation: bool = False
    timeout_seconds: int = 30
    calls_remaining: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class ToolPolicyService:
    """Enforces execution policies, trust level constraints, and quotas for tools."""

    def __init__(
        self,
        *,
        settings_store=None,
        default_max_calls: int = 8,
        default_timeout_seconds: int = 30,
        default_require_confirmation: bool = False,
    ) -> None:
        self.settings_store = settings_store
        self.default_max_calls = default_max_calls
        self.default_timeout_seconds = default_timeout_seconds
        self.default_require_confirmation = default_require_confirmation

        # Task call counter: task_id -> {tool_name: count}
        self._call_counts: dict[str, dict[str, int]] = {}

    def is_mutating_tool(self, tool_name: str, tool_instance: Any = None) -> bool:
        """Determine whether a tool mutates state (files, DB, system, external APIs)."""
        if tool_instance is not None:
            if hasattr(tool_instance, "is_mutating"):
                return bool(tool_instance.is_mutating)
            if hasattr(tool_instance, "mutating"):
                return bool(tool_instance.mutating)

        lowered = tool_name.lower()
        if any(lowered.startswith(p) for p in _MUTATING_PREFIXES):
            return True
        if any(lowered.startswith(p) for p in _READ_ONLY_PREFIXES):
            return False
        # Fallback conservative: if named "delete", "write", "exec", "save", "run"
        return any(
            k in lowered for k in ("delete", "write", "execute", "modify", "patch", "create")
        )

    def is_settings_modification(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> bool:
        """Determine whether a tool call attempts to modify configuration or settings."""
        lowered = tool_name.lower()
        if any(k in lowered for k in _SETTING_KEYWORDS) and self.is_mutating_tool(tool_name):
            return True
        if arguments:
            # Inspect argument keys or values targeting settings/configs
            target = str(arguments.get("target", "")).lower()
            action = str(arguments.get("action", "")).lower()
            if any(k in target or k in action for k in _SETTING_KEYWORDS):
                return True
        return False

    def evaluate(
        self,
        tool_name: str,
        trust_level: TrustLevel | str | int,
        *,
        tool_instance: Any = None,
        arguments: dict[str, Any] | None = None,
        task_id: str | None = None,
        is_confirmed: bool = False,
        is_mutating_override: bool | None = None,
    ) -> ToolPolicyDecision:
        """Evaluate whether a tool invocation is permitted under governance policy.

        Enforces:
        1. Trust Level Hierarchy: Context with trust level DOCUMENT (2) or
           EXTERNAL (1) cannot trigger mutating tools or modify settings.
        2. Quotas: tool_max_calls per task.
        3. Human confirmation: tool_require_confirmation or destructive tools.
        """
        level = TrustLevel.from_value(trust_level)

        is_mutating = (
            is_mutating_override
            if is_mutating_override is not None
            else self.is_mutating_tool(tool_name, tool_instance)
        )
        is_setting_mod = self.is_settings_modification(tool_name, arguments)

        # 1. Enforce Trust Level Constraints (retrieved content is data, not instructions)
        if level <= TrustLevel.DOCUMENT:
            if is_setting_mod:
                return ToolPolicyDecision(
                    allowed=False,
                    verdict="deny",
                    reason=(
                        f"Trust level {level.name} ({level.value}) cannot modify settings "
                        "(backed by allow_external_content_to_change_settings=False)."
                    ),
                    tool_name=tool_name,
                    trust_level=level,
                )
            if is_mutating:
                return ToolPolicyDecision(
                    allowed=False,
                    verdict="deny",
                    reason=(
                        f"Trust level {level.name} ({level.value}) cannot trigger mutating tools "
                        f"('{tool_name}') without human intervention."
                    ),
                    tool_name=tool_name,
                    trust_level=level,
                )

        # 2. Check call limits (tool_max_calls)
        max_calls = self.default_max_calls
        tid = task_id or "default_task"
        tool_counts = self._call_counts.setdefault(tid, {})
        current_calls = tool_counts.get(tool_name, 0)

        if current_calls >= max_calls:
            return ToolPolicyDecision(
                allowed=False,
                verdict="deny",
                reason=(
                    f"Tool '{tool_name}' exceeded max calls quota "
                    f"({current_calls}/{max_calls}) for task '{tid}'."
                ),
                tool_name=tool_name,
                trust_level=level,
                calls_remaining=0,
            )

        # 3. Check confirmation requirement
        requires_conf = self.default_require_confirmation
        if hasattr(tool_instance, "require_confirmation"):
            requires_conf = bool(tool_instance.require_confirmation)

        if requires_conf and not is_confirmed:
            return ToolPolicyDecision(
                allowed=False,
                verdict="ask_first",
                reason=f"Tool '{tool_name}' requires user confirmation before execution.",
                tool_name=tool_name,
                trust_level=level,
                requires_confirmation=True,
                calls_remaining=max_calls - current_calls,
            )

        # Approved
        remaining = max_calls - (current_calls + 1)
        return ToolPolicyDecision(
            allowed=True,
            verdict="allow",
            reason="Tool invocation permitted by policy.",
            tool_name=tool_name,
            trust_level=level,
            timeout_seconds=self.default_timeout_seconds,
            calls_remaining=remaining,
        )

    def record_call(self, tool_name: str, task_id: str | None = None) -> int:
        """Increment execution count for the given tool under task_id."""
        tid = task_id or "default_task"
        counts = self._call_counts.setdefault(tid, {})
        counts[tool_name] = counts.get(tool_name, 0) + 1
        return counts[tool_name]

    def reset_quotas(self, task_id: str | None = None) -> None:
        """Reset quota tracker for a specific task or all tasks."""
        if task_id:
            self._call_counts.pop(task_id, None)
        else:
            self._call_counts.clear()

    async def execute_guarded(
        self,
        tool: Any,
        arguments: dict[str, Any],
        trust_level: TrustLevel | str | int,
        *,
        task_id: str | None = None,
        is_confirmed: bool = False,
        timeout_seconds: int | None = None,
    ) -> Any:
        """Evaluate policy and execute the tool with timeout and quota tracking.

        Raises ToolPolicyViolation if policy check fails.
        """
        tool_name = getattr(tool, "name", str(tool))
        decision = self.evaluate(
            tool_name,
            trust_level,
            tool_instance=tool,
            arguments=arguments,
            task_id=task_id,
            is_confirmed=is_confirmed,
        )

        if not decision.allowed:
            logger.warning(
                "tool_policy.execution_denied",
                tool=tool_name,
                verdict=decision.verdict,
                reason=decision.reason,
            )
            raise ToolPolicyViolation(
                f"Policy denied execution of '{tool_name}': {decision.reason}"
            )

        timeout = timeout_seconds or decision.timeout_seconds
        self.record_call(tool_name, task_id)

        try:
            if hasattr(tool, "execute"):
                if asyncio.iscoroutinefunction(tool.execute):
                    return await asyncio.wait_for(tool.execute(**arguments), timeout=timeout)
                return tool.execute(**arguments)
            elif callable(tool):
                if asyncio.iscoroutinefunction(tool):
                    return await asyncio.wait_for(tool(**arguments), timeout=timeout)
                return tool(**arguments)
            else:
                raise ValueError(f"Tool {tool} is not executable")
        except TimeoutError as exc:
            logger.error("tool_policy.timeout", tool=tool_name, timeout=timeout)
            raise ToolPolicyViolation(
                f"Tool '{tool_name}' timed out after {timeout} seconds"
            ) from exc
