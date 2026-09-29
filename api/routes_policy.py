"""Governance policy rules API (Giai đoạn 5, Layer 6).

CRUD for the declarative rule set. Edits take effect on the next evaluation —
no restart, because the engine reads rules per decision.
"""

from __future__ import annotations

from typing import Annotated, Any

import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.deps import get_session_factory
from core.database import PolicyRule

router = APIRouter(prefix="/api/policy", tags=["policy"])

_RULE_TYPES = {"allow", "deny", "ask_first"}
_SCOPES = {"file_import", "response", "tool", "category"}


class PolicyRuleIn(BaseModel):
    """Create/update payload for one rule."""

    scope: str = Field(..., description="file_import | response | tool | category")
    rule_type: str = Field(..., description="allow | deny | ask_first")
    pattern: str = Field(..., min_length=1, max_length=255, description="mime, extension, or '*'")
    reason: str | None = None
    priority: int = 100
    enabled: bool = True


@router.get("/rules", response_model=dict)
async def list_rules(
    scope: str | None = None,
    session_factory: Annotated[Any, Depends(get_session_factory)] = None,
) -> dict:
    """List policy rules, optionally filtered by scope."""
    from sqlalchemy import select

    async with session_factory() as session:
        stmt = select(PolicyRule).order_by(PolicyRule.priority.desc(), PolicyRule.created_at)
        if scope:
            stmt = stmt.where(PolicyRule.scope == scope)
        rows = (await session.execute(stmt)).scalars().all()
        return {"rules": [_rule_payload(r) for r in rows]}


@router.post("/rules", response_model=dict)
async def create_rule(
    rule_in: PolicyRuleIn,
    session_factory: Annotated[Any, Depends(get_session_factory)] = None,
) -> dict:
    """Create one rule. Takes effect on the next policy evaluation."""
    _validate(rule_in)
    async with session_factory() as session:
        rule = PolicyRule(
            scope=rule_in.scope,
            rule_type=rule_in.rule_type,
            pattern=rule_in.pattern.strip().lower(),
            reason=rule_in.reason,
            priority=rule_in.priority,
            enabled=rule_in.enabled,
        )
        session.add(rule)
        await session.commit()
        return {"rule": _rule_payload(rule)}


@router.put("/rules/{rule_id}", response_model=dict)
async def update_rule(
    rule_id: str,
    rule_in: PolicyRuleIn,
    session_factory: Annotated[Any, Depends(get_session_factory)] = None,
) -> dict:
    """Update one rule (bumps its version)."""
    _validate(rule_in)
    from sqlalchemy import select

    async with session_factory() as session:
        rule = (
            await session.execute(select(PolicyRule).where(PolicyRule.id == rule_id))
        ).scalar_one_or_none()
        if rule is None:
            raise HTTPException(404, "rule not found")
        rule.scope = rule_in.scope
        rule.rule_type = rule_in.rule_type
        rule.pattern = rule_in.pattern.strip().lower()
        rule.reason = rule_in.reason
        rule.priority = rule_in.priority
        rule.enabled = rule_in.enabled
        rule.version += 1
        await session.commit()
        return {"rule": _rule_payload(rule)}


@router.delete("/rules/{rule_id}")
async def delete_rule(
    rule_id: str,
    session_factory: Annotated[Any, Depends(get_session_factory)] = None,
) -> dict:
    """Delete one rule."""
    async with session_factory() as session:
        result = await session.execute(
            sa.delete(PolicyRule).where(PolicyRule.id == rule_id)
        )
        await session.commit()
        if result.rowcount == 0:
            raise HTTPException(404, "rule not found")
        return {"deleted": rule_id}


def _validate(rule_in: PolicyRuleIn) -> None:
    if rule_in.scope not in _SCOPES:
        raise HTTPException(400, f"scope must be one of {sorted(_SCOPES)}")
    if rule_in.rule_type not in _RULE_TYPES:
        raise HTTPException(400, f"rule_type must be one of {sorted(_RULE_TYPES)}")


def _rule_payload(rule: PolicyRule) -> dict:
    return {
        "id": str(rule.id),
        "scope": rule.scope,
        "rule_type": rule.rule_type,
        "pattern": rule.pattern,
        "reason": rule.reason,
        "priority": rule.priority,
        "version": rule.version,
        "enabled": rule.enabled,
    }

