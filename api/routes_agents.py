"""API routes for multi-agent system (read-only registry views).

The hierarchical workflow itself is driven by ``api/routes_workflow.py`` and the
``OrchestrationEngine``. These endpoints expose the agent registry (roster and
departments) for chat and internal orchestration consumers alongside live workflow state.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/agents", tags=["agents"])


@router.get("/list")
async def list_agents(request: Request):
    """List all agents in the system."""
    agent_registry = request.app.state.agent_registry

    try:
        agents = await agent_registry.load_agents()
        return {
            "agents": [agent.to_dict() for agent in agents],
            "total": len(agents),
        }
    except Exception as exc:
        logger.error("agents.list_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/departments")
async def list_departments(request: Request):
    """List agents grouped by department."""
    agent_registry = request.app.state.agent_registry

    try:
        agents = await agent_registry.load_agents()

        departments: dict[str, list] = {}
        for agent in agents:
            departments.setdefault(agent.department, []).append(agent.to_dict())

        return {"departments": departments}
    except Exception as exc:
        logger.error("agents.departments_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{agent_id}")
async def get_agent(agent_id: str, request: Request):
    """Get details of a specific agent."""
    agent_registry = request.app.state.agent_registry

    try:
        agent = await agent_registry.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        return agent.to_dict()
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("agents.get_failed", agent_id=agent_id, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc

