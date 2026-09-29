"""Braind Core - Agent runtime (Dify-style ReAct / tool-calling loop).

The loop sends the user query + available tool schemas to the LLM.
When the model returns a tool_call, we execute it and feed the result back
as a tool message. The loop continues until the model produces a final
natural-language answer (no more tool_calls).

Supports both:
- Native OpenAI-compatible vLLM tool calling
- Fallback text-based ReAct parsing if the model refuses native format.
"""

from __future__ import annotations

import json
import re
from typing import Any

from core.agent.tool import ToolError, ToolRegistry
from core.logging import get_logger
from core.vllm import VLLMClient

logger = get_logger(__name__)


class AgentLoop:
    """Single-turn / multi-turn agent executor."""

    def __init__(
        self,
        vllm: VLLMClient,
        registry: ToolRegistry,
        *,
        max_steps: int = 6,
        temperature: float = 0.2,
    ) -> None:
        self.vllm = vllm
        self.registry = registry
        self.max_steps = max_steps
        self.temperature = temperature

    async def run(
        self,
        message: str,
        *,
        system: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """
        Execute the agent loop and return the final answer + trace.

        Returns:
            {
                "answer": str,
                "steps": list[dict],      # [{thought, tool, args, observation}]
                "tool_calls": list[dict], # OpenAI function calls with IDs
            }
        """
        tools_schema = self.registry.schemas()
        if not tools_schema:
            # No tools → just answer directly
            answer = await self.vllm.chat(
                [{"role": "user", "content": message}],
                system=system or "You are a helpful assistant.",
                temperature=self.temperature,
            )
            return {"answer": answer, "steps": [], "tool_calls": []}

        history = conversation_history or []
        steps: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []

        # Build the initial messages
        messages = history + [{"role": "user", "content": message}]

        for _step in range(self.max_steps):
            # vLLM returns JSON containing OpenAI-compatible tool calls when tools are passed.
            resp = await self.vllm.chat(
                messages,
                system=system,
                temperature=self.temperature,
                tools=tools_schema,
            )

            # ``VLLMClient.chat`` serializes the first OpenAI message so the
            # loop can preserve its tool-call IDs for follow-up tool messages.
            parsed: dict[str, Any] = {}
            try:
                data = json.loads(resp) if isinstance(resp, str) else resp
                if isinstance(data, dict) and ("content" in data or "tool_calls" in data):
                    parsed = data
            except (TypeError, ValueError):
                parsed = {}

            content = parsed.get("content", resp if isinstance(resp, str) else "")
            native_calls = parsed.get("tool_calls", [])

            if native_calls:
                for tc in native_calls:
                    tool_calls.append(tc)
                    name = tc.get("function", {}).get("name", "")
                    raw_args = tc.get("function", {}).get("arguments", {})
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args or {}
                    except (TypeError, ValueError):
                        logger.warning("agent.tool_args_unparsable", tool=name, raw=str(raw_args))
                        args = {}
                    if not isinstance(args, dict):
                        logger.warning("agent.tool_args_not_object", tool=name)
                        args = {}
                    obs = await self._safe_execute(name, args)
                    steps.append(
                        {
                            "thought": f"Tool call #{len(tool_calls)}: {name}",
                            "tool": name,
                            "args": args,
                            "observation": obs,
                        }
                    )
                    tool_call_id = str(tc.get("id") or f"call_{len(tool_calls)}")
                    messages.append({"role": "assistant", "content": content, "tool_calls": [tc]})
                    messages.append(
                        {
                            "role": "tool",
                            "content": json.dumps(obs, ensure_ascii=False),
                            "tool_call_id": tool_call_id,
                        }
                    )
                continue

            # Fallback: try to parse ReAct text format
            react = self._parse_react(content)
            if react:
                tool_calls.append({"function": {"name": react["tool"], "arguments": react["args"]}})
                obs = await self._safe_execute(react["tool"], react["args"])
                steps.append(
                    {
                        "thought": react.get("thought", ""),
                        "tool": react["tool"],
                        "args": react["args"],
                        "observation": obs,
                    }
                )
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(obs, ensure_ascii=False),
                        "name": react["tool"],
                    }
                )
                continue

            # No tool call → final answer
            return {
                "answer": content,
                "steps": steps,
                "tool_calls": tool_calls,
            }

        # Max steps reached
        return {
            "answer": (
                "Tôi đã thử nhiều bước nhưng chưa hoàn thành. "
                "Bạn có thể cung cấp thêm thông tin không?"
            ),
            "steps": steps,
            "tool_calls": tool_calls,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_react(self, text: str) -> dict[str, Any] | None:
        """Very simple ReAct parser: Thought / Action / Action Input."""
        m = re.search(
            r"Action\s*:\s*([a-z0-9_\-]+)\s*Action Input\s*:\s*(\{.*?\})",
            text,
            re.I | re.S,
        )
        if not m:
            return None
        try:
            args = json.loads(m.group(2))
        except (TypeError, ValueError):
            args = {}
        return {"tool": m.group(1).strip(), "args": args, "thought": text}

    async def _safe_execute(self, name: str, args: dict[str, Any]) -> Any:
        try:
            return await self.registry.execute(name, args)
        except ToolError as exc:
            return {"error": str(exc)}
