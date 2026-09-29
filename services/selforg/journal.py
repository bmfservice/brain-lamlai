"""Append-only audit journaling and rollback for self-organization (CLAUDE.md §8, §9).

Every restructuring action (merge, split, reparent, quarantine, link reorg) is
recorded in the `reorg_operations` table with full `before_state` and `after_state`
snapshots so any automated or manual pass is completely auditable and reversible.
Rollback creates a new operation with `rollback_of_id` set (preserving the append-only audit trail).
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any

import sqlalchemy as sa
from sqlalchemy import select, update

from core import database as db
from core.logging import get_logger

logger = get_logger(__name__)

# Operation types
OP_MERGE_TERMS = "merge_terms"
OP_SPLIT_CATEGORY = "split_category"
OP_REPARENT_NODE = "reparent_node"
OP_QUARANTINE_ITEM = "quarantine_item"
OP_ROLLBACK = "rollback"

# Status values
STATUS_COMPLETED = "completed"
STATUS_ROLLED_BACK = "rolled_back"
STATUS_FAILED = "failed"
STATUS_PENDING = "pending"


class ReorgJournal:
    """Append-only journal for knowledge-store reorganization operations."""

    def __init__(self, session_factory) -> None:
        self.session_factory = session_factory

    async def record_operation(
        self,
        *,
        operation_type: str,
        target_type: str,
        target_id: str,
        before_state: dict[str, Any] | None = None,
        after_state: dict[str, Any] | None = None,
        rationale: str | None = None,
        status: str = STATUS_COMPLETED,
        rollback_of_id: uuid.UUID | None = None,
    ) -> uuid.UUID:
        """Record a reorg operation in the append-only `reorg_operations` log."""
        op_id = uuid.uuid4()
        now = datetime.datetime.now(datetime.UTC)

        async with self.session_factory() as session:
            entry = db.ReorgOperation(
                id=op_id,
                operation_type=operation_type,
                status=status,
                target_type=target_type,
                target_id=str(target_id),
                before_state=before_state,
                after_state=after_state,
                rationale=rationale,
                rollback_of_id=rollback_of_id,
                created_at=now,
                completed_at=now if status == STATUS_COMPLETED else None,
            )
            session.add(entry)
            await session.commit()

        logger.info(
            "selforg.journal.recorded",
            op_id=str(op_id),
            operation_type=operation_type,
            target_type=target_type,
            target_id=target_id,
            status=status,
        )
        return op_id

    async def get_operation(self, op_id: uuid.UUID | str) -> dict[str, Any] | None:
        """Fetch a single reorg operation by ID."""
        target_uuid = uuid.UUID(str(op_id)) if isinstance(op_id, str) else op_id

        async with self.session_factory() as session:
            stmt = select(db.ReorgOperation).where(db.ReorgOperation.id == target_uuid)
            op = (await session.execute(stmt)).scalar_one_or_none()
            if op is None:
                return None
            return {
                "id": str(op.id),
                "operation_type": op.operation_type,
                "status": op.status,
                "target_type": op.target_type,
                "target_id": op.target_id,
                "before_state": op.before_state,
                "after_state": op.after_state,
                "rationale": op.rationale,
                "rollback_of_id": str(op.rollback_of_id) if op.rollback_of_id else None,
                "created_at": op.created_at.isoformat() if op.created_at else None,
                "completed_at": op.completed_at.isoformat() if op.completed_at else None,
            }

    async def list_operations(
        self,
        *,
        limit: int = 50,
        operation_type: str | None = None,
        target_type: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """List historical reorg operations with optional filtering."""
        async with self.session_factory() as session:
            stmt = select(db.ReorgOperation).order_by(db.ReorgOperation.created_at.desc())
            if operation_type:
                stmt = stmt.where(db.ReorgOperation.operation_type == operation_type)
            if target_type:
                stmt = stmt.where(db.ReorgOperation.target_type == target_type)
            if status:
                stmt = stmt.where(db.ReorgOperation.status == status)
            stmt = stmt.limit(limit)

            rows = (await session.execute(stmt)).scalars().all()
            return [
                {
                    "id": str(r.id),
                    "operation_type": r.operation_type,
                    "status": r.status,
                    "target_type": r.target_type,
                    "target_id": r.target_id,
                    "before_state": r.before_state,
                    "after_state": r.after_state,
                    "rationale": r.rationale,
                    "rollback_of_id": str(r.rollback_of_id) if r.rollback_of_id else None,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                }
                for r in rows
            ]

    async def rollback(self, op_id: uuid.UUID | str, *, reason: str | None = None) -> uuid.UUID:
        """Roll back a previous reorg operation, restoring before_state.

        Creates an append-only compensating reorg operation referencing `rollback_of_id`.
        """
        target_uuid = uuid.UUID(str(op_id)) if isinstance(op_id, str) else op_id

        async with self.session_factory() as session:
            stmt = select(db.ReorgOperation).where(db.ReorgOperation.id == target_uuid)
            orig_op = (await session.execute(stmt)).scalar_one_or_none()
            if orig_op is None:
                raise ValueError(f"Reorg operation {op_id} not found")

            if orig_op.status == STATUS_ROLLED_BACK:
                raise ValueError(f"Reorg operation {op_id} was already rolled back")

            before = orig_op.before_state or {}
            op_type = orig_op.operation_type
            target_type = orig_op.target_type

            # Apply state restoration depending on target type and operation type
            if op_type == OP_MERGE_TERMS and target_type == "taxonomy_terms":
                # Restore source term status and clear merged_into_id
                source_id_str = before.get("source_term_id")
                if source_id_str:
                    source_uuid = uuid.UUID(source_id_str)
                    prev_status = before.get("source_status", "active")
                    prev_merged = before.get("source_merged_into_id")
                    await session.execute(
                        update(db.TaxonomyTerm)
                        .where(db.TaxonomyTerm.id == source_uuid)
                        .values(
                            status=prev_status,
                            merged_into_id=uuid.UUID(prev_merged) if prev_merged else None,
                        )
                    )
                # Restore reparented child terms if any
                reparented = before.get("reparented_children", [])
                for child in reparented:
                    child_id = uuid.UUID(child["id"])
                    await session.execute(
                        update(db.TaxonomyTerm)
                        .where(db.TaxonomyTerm.id == child_id)
                        .values(
                            parent_id=uuid.UUID(child["original_parent_id"])
                            if child.get("original_parent_id")
                            else None
                        )
                    )

            elif op_type == OP_SPLIT_CATEGORY and target_type == "taxonomy_terms":
                # If sub-categories were created, set their status to 'archived' or 'proposed'
                created_ids = (orig_op.after_state or {}).get("created_subterm_ids", [])
                for sub_id_str in created_ids:
                    sub_uuid = uuid.UUID(sub_id_str)
                    await session.execute(
                        update(db.TaxonomyTerm)
                        .where(db.TaxonomyTerm.id == sub_uuid)
                        .values(status="archived")
                    )

            elif op_type == OP_REPARENT_NODE and target_type == "taxonomy_terms":
                node_id_str = before.get("term_id")
                if node_id_str:
                    node_uuid = uuid.UUID(node_id_str)
                    old_parent = before.get("original_parent_id")
                    await session.execute(
                        update(db.TaxonomyTerm)
                        .where(db.TaxonomyTerm.id == node_uuid)
                        .values(
                            parent_id=uuid.UUID(old_parent) if old_parent else None
                        )
                    )

            # Mark original operation as rolled back
            orig_op.status = STATUS_ROLLED_BACK
            await session.commit()

        # Record compensating append-only rollback operation
        rollback_rationale = (
            f"Rollback of operation {op_id}: {reason}"
            if reason
            else f"Rollback of operation {op_id}"
        )
        rollback_op_id = await self.record_operation(
            operation_type=OP_ROLLBACK,
            target_type=orig_op.target_type,
            target_id=orig_op.target_id,
            before_state=orig_op.after_state,
            after_state=orig_op.before_state,
            rationale=rollback_rationale,
            status=STATUS_COMPLETED,
            rollback_of_id=orig_op.id,
        )

        logger.info("selforg.journal.rollback_completed", op_id=str(op_id), rollback_op_id=str(rollback_op_id))
        return rollback_op_id
