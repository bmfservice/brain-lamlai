"""Braind API - Agent routes (Dify-style tool registry + agent chat).

Exposes:
- GET /api/agent/tools — list registered tools + their schemas
- POST /api/agent/chat — run the agent (returns steps + tool_calls like Dify)
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request

from api.deps import get_agent
from core.errors import ServiceUnavailableError, ValidationError
from services.agent.service import AgentService

router = APIRouter(prefix="/api/agent", tags=["agent"])


@router.get("/tools")
async def list_tools(request: Request) -> dict[str, Any]:
    """List all registered tools and their JSON schemas (Dify-style registry)."""
    registry = getattr(request.app.state, "tools", None)
    if registry is None:
        raise ServiceUnavailableError("Tool registry has not been initialized")
    tools = []
    for t in registry.all():
        tools.append(
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
        )
    return {"count": len(tools), "tools": tools}


@router.post("/chat")
async def agent_chat(
    payload: dict[str, Any],
    agent: Annotated[AgentService, Depends(get_agent)],
) -> dict[str, Any]:
    """Run the Dify-style agent on a user message.

    Request body:
    {
      "message": "tìm tin tức AI mới nhất",
      "conversation_id": "optional-uuid",
      "system": "optional override system prompt"
    }
    """
    message = payload.get("message", "").strip()
    if not message:
        raise ValidationError("A message is required")
    conversation_id = payload.get("conversation_id")
    system = payload.get("system")
    result = await agent.chat(message, conversation_id=conversation_id, system=system)
    return result
