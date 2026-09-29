"""API routes for the hierarchical agent workflow.

These expose the real orchestration engine: submit a request (returns a root
task id immediately), read the live task tree / audit trail / reports, and
answer the user-confirmation gate. All execution runs against real database
state — nothing here is simulated.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from core.logging import get_logger
from services.agents.domain import TaskState, to_canonical_state
from services.agents.repository import AgentRepository

logger = get_logger(__name__)

router = APIRouter(prefix="/api/workflow", tags=["workflow"])


class SubmitRequest(BaseModel):
    request: str = Field(..., description="User request to execute through the hierarchy")
    priority: int = Field(2, ge=1, le=4)


class ConfirmRequest(BaseModel):
    decision: str = Field(..., description="approve | reject | or a chosen option")
    notes: str | None = None


def _repo(request: Request) -> AgentRepository:
    return AgentRepository(request.app.state.session_factory)


@router.post("/submit")
async def submit(body: SubmitRequest, request: Request) -> dict[str, Any]:
    """Create the root task and launch the workflow in the background."""
    orchestrator = request.app.state.orchestrator
    try:
        root_id = await orchestrator.submit_request(body.request, body.priority)
        return {"root_task_id": root_id, "status": "accepted"}
    except Exception as exc:
        logger.error("workflow.submit_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/root/{root_task_id}")
async def get_root(root_task_id: str, request: Request) -> dict[str, Any]:
    """Full snapshot of one workflow: root + tree + reports + audit."""
    repo = _repo(request)
    root = await repo.get_task(root_task_id)
    if not root:
        raise HTTPException(status_code=404, detail="Root task not found")
    tree = await repo.get_tree(root_task_id)
    reports = await repo.get_reports_for_root(root_task_id)
    audit = await repo.get_audit_trail(root_task_id)
    return {
        "root": root.to_dict(),
        "tree": [t.to_dict() for t in tree],
        "reports": reports,
        "audit": audit,
        "final_result": root.result,
    }


@router.get("/root/{root_task_id}/tree")
async def get_tree(root_task_id: str, request: Request) -> dict[str, Any]:
    """The task tree (parent<->child) so the UI can render the hierarchy."""
    repo = _repo(request)
    tree = await repo.get_tree(root_task_id)
    if not tree:
        raise HTTPException(status_code=404, detail="Root task not found")
    return {"tree": [t.to_dict() for t in tree], "total": len(tree)}


@router.get("/root/{root_task_id}/audit")
async def get_audit(root_task_id: str, request: Request, limit: int = 500) -> dict[str, Any]:
    """The audit trail for real-time activity display."""
    repo = _repo(request)
    events = await repo.get_audit_trail(root_task_id, limit)
    return {"events": events, "total": len(events)}


@router.post("/root/{root_task_id}/confirm")
async def confirm(root_task_id: str, body: ConfirmRequest, request: Request) -> dict[str, Any]:
    """Answer the user-confirmation gate and resume the paused workflow."""
    orchestrator = request.app.state.orchestrator
    repo = _repo(request)
    root = await repo.get_task(root_task_id)
    if not root:
        raise HTTPException(status_code=404, detail="Root task not found")
    if to_canonical_state(root.state) not in (
        TaskState.WAITING_FOR_USER,
        TaskState.WAITING_APPROVAL,
        TaskState.WAITING_INPUT,
    ):
        raise HTTPException(
            status_code=409, detail=f"Task is not waiting for user (state={root.state})"
        )
    try:
        await orchestrator.resume_after_confirmation(root_task_id, body.decision, body.notes)
        return {"status": "resumed", "root_task_id": root_task_id, "decision": body.decision}
    except Exception as exc:
        logger.error("workflow.confirm_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/active")
async def list_active(request: Request, limit: int = 50) -> dict[str, Any]:
    """List recent root tasks with their live state for the dashboard."""
    session_factory = request.app.state.session_factory
    from sqlalchemy import text

    async with session_factory() as session:
        result = await session.execute(
            text(
                """SELECT task_id, user_request, state, status, created_at, confirmation_required
                   FROM agent_tasks WHERE task_type = 'root'
                   ORDER BY created_at DESC LIMIT :limit"""
            ),
            {"limit": limit},
        )
        rows = [dict(r._mapping) for r in result.fetchall()]
    for r in rows:
        if r.get("created_at"):
            r["created_at"] = r["created_at"].isoformat()
    return {"roots": rows, "total": len(rows)}
