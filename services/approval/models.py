"""Approval system data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class ApprovalStatus(StrEnum):
    """Approval request status."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class ApprovalRequest:
    """Approval request from an agent."""

    request_id: str
    agent_id: str
    action_type: str  # 'social_post', 'email_campaign', etc.
    action_data: dict[str, Any]
    task_id: str | None = None
    status: ApprovalStatus = ApprovalStatus.PENDING
    priority: int = 2
    requested_at: datetime = field(default_factory=datetime.now)
    reviewed_at: datetime | None = None
    reviewed_by: str | None = None
    review_notes: str | None = None
    expires_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "agent_id": self.agent_id,
            "action_type": self.action_type,
            "action_data": self.action_data,
            "task_id": self.task_id,
            "status": self.status.value if isinstance(self.status, ApprovalStatus) else self.status,
            "priority": self.priority,
            "requested_at": self.requested_at.isoformat() if self.requested_at else None,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "reviewed_by": self.reviewed_by,
            "review_notes": self.review_notes,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }
