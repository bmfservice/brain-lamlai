"""Braind Core - Tool abstraction + registry.

Dify-style tool system: each tool declares a name, description, JSON schema
and an async execute() method. The agent runtime discovers tools from the
registry and passes their schemas to the LLM for native function calling.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.errors import BraindError
from core.logging import get_logger

logger = get_logger(__name__)


class ToolError(BraindError):
    """Raised when a tool fails to execute."""

    code = "tool_error"
    public_message = "A tool could not complete the request."


class BaseTool(ABC):
    """Base class for all Braind tools (Dify-style)."""

    #: Tool name shown to the LLM (kebab-case).
    name: str = ""
    #: Short human-readable description used by the LLM to pick a tool.
    description: str = ""
    #: JSON schema of the tool's arguments.
    parameters: dict[str, Any] = {}

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """Run the tool and return a JSON-serializable result."""

    # --- Schema helpers -------------------------------------------------

    @classmethod
    def get_schema(cls) -> dict[str, Any]:
        """Return the OpenAI-style function schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": cls.name,
                "description": cls.description,
                "parameters": cls.parameters,
            },
        }

    @classmethod
    def get_name(cls) -> str:
        return cls.name


class ToolRegistry:
    """Registry that holds all available tools and offers schema lists."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        if not tool.name:
            raise ValueError("Tool must have a name")
        self._tools[tool.name] = tool
        logger.debug("tool.registered", name=tool.name)

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def all(self) -> list[BaseTool]:
        return list(self._tools.values())

    def schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI-style tool schemas for the LLM."""
        return [t.get_schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return sorted(self._tools.keys())

    async def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool by name, converting exceptions to ToolError."""
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"Unknown tool: {name}")
        try:
            return await tool.execute(**arguments)
        except ToolError:
            raise
        except Exception as exc:
            logger.error("tool.exec_error", name=name, error=str(exc), exc_info=True)
            raise ToolError(f"{name} failed: {exc}") from exc


# Module-level default registry (services wire tools into it at startup).
_registry = ToolRegistry()


def register_tool(tool: BaseTool) -> ToolRegistry:
    """Register a tool on the global registry."""
    _registry.register(tool)
    return _registry


def get_registry() -> ToolRegistry:
    """Return the global tool registry."""
    return _registry
