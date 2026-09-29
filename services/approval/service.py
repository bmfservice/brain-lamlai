"""Approval workflow service."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import text

from core.logging import get_logger
from services.approval.models import ApprovalRequest, ApprovalStatus

logger = get_logger(__name__)


class ApprovalService:
    """Manages approval workflow."""

    def __init__(self, session_factory):
        self.session_factory = session_factory

    async def request_approval(
        self,
        agent_id: str,
        action_type: str,
        action_data: dict[str, Any],
        task_id: str | None = None,
        priority: int = 2,
        expires_in_hours: int = 24,
    ) -> ApprovalRequest:
        """Create approval request."""
        request = ApprovalRequest(
            request_id=str(uuid.uuid4()),
            agent_id=agent_id,
            action_type=action_type,
            action_data=action_data,
            task_id=task_id,
            priority=priority,
            expires_at=datetime.now() + timedelta(hours=expires_in_hours),
        )

        # Save to database
        async with self.session_factory() as session:
            await session.execute(
                text(
                    """INSERT INTO approval_requests 
                       (request_id, task_id, agent_id, action_type, action_data, 
                        status, priority, expires_at)
                       VALUES (:request_id, :task_id, :agent_id, :action_type, :action_data,
                               :status, :priority, :expires_at)"""
                ),
                {
                    "request_id": request.request_id,
                    "task_id": request.task_id,
                    "agent_id": request.agent_id,
                    "action_type": request.action_type,
                    "action_data": json.dumps(action_data),  # Serialize to JSON string
                    "status": request.status.value,
                    "priority": request.priority,
                    "expires_at": request.expires_at,
                },
            )
            await session.commit()

        logger.info(
            "approval.requested",
            request_id=request.request_id,
            agent_id=agent_id,
            action_type=action_type,
        )

        return request

    async def get_pending_approvals(self, limit: int = 50) -> list[ApprovalRequest]:
        """Get pending approval requests."""
        async with self.session_factory() as session:
            result = await session.execute(
                text(
                    """SELECT * FROM approval_requests 
                       WHERE status = 'pending' 
                       ORDER BY priority DESC, requested_at ASC
                       LIMIT :limit"""
                ),
                {"limit": limit},
            )
            rows = result.fetchall()

            requests = []
            for row in rows:
                row_dict = dict(row._mapping)
                requests.append(self._row_to_request(row_dict))

            return requests

    async def approve(
        self, request_id: str, reviewer: str, notes: str | None = None
    ) -> bool:
        """Approve a request."""
        async with self.session_factory() as session:
            await session.execute(
                text(
                    """UPDATE approval_requests 
                       SET status = 'approved', reviewed_at = NOW(), 
                           reviewed_by = :reviewer, review_notes = :notes
                       WHERE request_id = :request_id"""
                ),
                {"request_id": request_id, "reviewer": reviewer, "notes": notes},
            )

            # Add audit log
            await session.execute(
                text(
                    """INSERT INTO approval_audit 
                       (audit_id, request_id, action, actor, notes)
                       VALUES (:audit_id, :request_id, 'approved', :actor, :notes)"""
                ),
                {
                    "audit_id": str(uuid.uuid4()),
                    "request_id": request_id,
                    "actor": reviewer,
                    "notes": notes,
                },
            )

            await session.commit()

        logger.info("approval.approved", request_id=request_id, reviewer=reviewer)
        return True

    async def reject(
        self, request_id: str, reviewer: str, notes: str | None = None
    ) -> bool:
        """Reject a request."""
        async with self.session_factory() as session:
            await session.execute(
                text(
                    """UPDATE approval_requests 
                       SET status = 'rejected', reviewed_at = NOW(), 
                           reviewed_by = :reviewer, review_notes = :notes
                       WHERE request_id = :request_id"""
                ),
                {"request_id": request_id, "reviewer": reviewer, "notes": notes},
            )

            # Add audit log
            await session.execute(
                text(
                    """INSERT INTO approval_audit 
                       (audit_id, request_id, action, actor, notes)
                       VALUES (:audit_id, :request_id, 'rejected', :actor, :notes)"""
                ),
                {
                    "audit_id": str(uuid.uuid4()),
                    "request_id": request_id,
                    "actor": reviewer,
                    "notes": notes,
                },
            )

            await session.commit()

        logger.info("approval.rejected", request_id=request_id, reviewer=reviewer)
        return True

    async def get_request(self, request_id: str) -> ApprovalRequest | None:
        """Get approval request by ID."""
        async with self.session_factory() as session:
            result = await session.execute(
                text("SELECT * FROM approval_requests WHERE request_id = :request_id"),
                {"request_id": request_id},
            )
            row = result.fetchone()

            if not row:
                return None

            return self._row_to_request(dict(row._mapping))

    def _row_to_request(self, row_dict: dict) -> ApprovalRequest:
        """Convert database row to ApprovalRequest."""
        return ApprovalRequest(
            request_id=row_dict["request_id"],
            agent_id=row_dict["agent_id"],
            action_type=row_dict["action_type"],
            action_data=row_dict["action_data"],
            task_id=row_dict.get("task_id"),
            status=ApprovalStatus(row_dict["status"]),
            priority=row_dict["priority"],
            requested_at=row_dict["requested_at"],
            reviewed_at=row_dict.get("reviewed_at"),
            reviewed_by=row_dict.get("reviewed_by"),
            review_notes=row_dict.get("review_notes"),
            expires_at=row_dict.get("expires_at"),
        )
