"""API routes for approval workflow."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/approvals", tags=["approvals"])


class ApproveRequest(BaseModel):
    """Approve/reject request body."""

    reviewer: str
    notes: str | None = None


@router.get("/pending")
async def get_pending_approvals(limit: int = 50, request: Request = None):
    """Get pending approval requests."""
    approval_service = request.app.state.approval_service

    try:
        requests = await approval_service.get_pending_approvals(limit)
        return {
            "approvals": [r.to_dict() for r in requests],
            "total": len(requests),
        }
    except Exception as exc:
        logger.error("approvals.get_pending_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{request_id}")
async def get_approval(request_id: str, request: Request):
    """Get specific approval request."""
    approval_service = request.app.state.approval_service

    try:
        approval = await approval_service.get_request(request_id)
        if not approval:
            raise HTTPException(status_code=404, detail="Approval not found")
        return approval.to_dict()
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("approvals.get_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/{request_id}/approve")
async def approve_request(request_id: str, body: ApproveRequest, request: Request):
    """Approve an approval request."""
    approval_service = request.app.state.approval_service

    try:
        await approval_service.approve(
            request_id, body.reviewer, body.notes
        )
        return {"status": "approved", "request_id": request_id}
    except Exception as exc:
        logger.error("approvals.approve_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/{request_id}/reject")
async def reject_request(request_id: str, body: ApproveRequest, request: Request):
    """Reject an approval request."""
    approval_service = request.app.state.approval_service

    try:
        await approval_service.reject(
            request_id, body.reviewer, body.notes
        )
        return {"status": "rejected", "request_id": request_id}
    except Exception as exc:
        logger.error("approvals.reject_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc
