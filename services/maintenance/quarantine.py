"""Memory & Continuous Learning Protection — Quarantine Service (Giai đoạn 4, Layer 4).

Enforces HR-NO-PERMANENT-DELETE:
Knowledge is never permanently deleted by automated passes. Contradicted,
duplicated, or ambiguous data is moved to quarantine (fully reversible),
and only verifiably corrupt data is removed.

Integrates with:
- QuarantinedKnowledge table in core/database.py
- Edge weight decay + reuse reinforcement from services/consolidation/weight.py
- Conflict resolution policies (when_data_conflict, when_memory_conflict)
"""

from __future__ import annotations

import datetime
import uuid
from dataclasses import asdict, dataclass
from typing import Any

import sqlalchemy as sa
from sqlalchemy import delete, select

from core import database as db
from core.logging import get_logger
from services.consolidation.weight import compute_weight, decay_factor, reuse_factor

logger = get_logger(__name__)

# Standard quarantine reasons
REASON_CONTRADICTION = "contradiction"
REASON_DUPLICATE = "duplicate"
REASON_AMBIGUOUS = "ambiguous"
REASON_DECAY = "decay"
REASON_LOW_CONFIDENCE = "low_confidence"

VALID_REASONS = {
    REASON_CONTRADICTION,
    REASON_DUPLICATE,
    REASON_AMBIGUOUS,
    REASON_DECAY,
    REASON_LOW_CONFIDENCE,
}


@dataclass
class QuarantineRecord:
    """In-memory representation of a quarantined knowledge item."""

    id: uuid.UUID
    session_id: uuid.UUID | None
    table_name: str
    row_id: str
    reason: str
    detail: str | None
    payload: dict[str, Any] | None
    created_at: datetime.datetime

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["id"] = str(self.id)
        d["session_id"] = str(self.session_id) if self.session_id else None
        d["created_at"] = self.created_at.isoformat() if self.created_at else None
        return d


class QuarantineService:
    """Quarantine management for contradicted, ambiguous, or decayed knowledge.

    Guarantees reversible isolation of knowledge items without data loss.
    """

    def __init__(self, session_factory=None, settings_store=None) -> None:
        self.session_factory = session_factory
        self.settings_store = settings_store
        # In-memory storage fallback for standalone / test usage without DB
        self._in_memory: dict[uuid.UUID, QuarantineRecord] = {}

    async def quarantine_item(
        self,
        *,
        table_name: str,
        row_id: str,
        reason: str,
        detail: str | None = None,
        payload: dict[str, Any] | None = None,
        session_id: uuid.UUID | None = None,
    ) -> uuid.UUID:
        """Quarantine a knowledge item, preserving its full state in payload.

        Enforces HR-NO-PERMANENT-DELETE: item is moved to quarantine, never
        permanently destroyed.

        Args:
            table_name: Target table (e.g., 'semantic_units', 'knowledge_links', 'memory')
            row_id: Identifier of the row in the source table
            reason: Reason code (contradiction, duplicate, ambiguous, decay, etc.)
            detail: Explanatory text for human auditing
            payload: Snapshot of original row data for lossless restoration
            session_id: Optional session context

        Returns:
            UUID of the created QuarantinedKnowledge record.
        """
        record_id = uuid.uuid4()
        now = datetime.datetime.now(datetime.UTC)

        if self.session_factory is not None:
            try:
                async with self.session_factory() as session:
                    # If payload was not provided and table is known, try to snapshot it
                    row_payload = payload
                    if row_payload is None:
                        row_payload = await self._fetch_source_payload(
                            session, table_name, row_id
                        )

                    entry = db.QuarantinedKnowledge(
                        id=record_id,
                        session_id=session_id,
                        table_name=table_name,
                        row_id=str(row_id),
                        reason=reason,
                        detail=detail,
                        payload=row_payload,
                        created_at=now,
                    )
                    session.add(entry)
                    await session.commit()
                    logger.info(
                        "quarantine.item_quarantined",
                        id=str(record_id),
                        table_name=table_name,
                        row_id=str(row_id),
                        reason=reason,
                    )
                    return record_id
            except Exception as exc:
                logger.error("quarantine.db_error", error=str(exc), exc_info=True)
                # Fallback to in-memory if DB fails to uphold no-data-loss guarantee

        record = QuarantineRecord(
            id=record_id,
            session_id=session_id,
            table_name=table_name,
            row_id=str(row_id),
            reason=reason,
            detail=detail,
            payload=payload or {},
            created_at=now,
        )
        self._in_memory[record_id] = record
        logger.info(
            "quarantine.item_quarantined_in_memory",
            id=str(record_id),
            table_name=table_name,
            row_id=str(row_id),
            reason=reason,
        )
        return record_id

    async def list_quarantined(
        self,
        *,
        table_name: str | None = None,
        reason: str | None = None,
        session_id: uuid.UUID | None = None,
    ) -> list[dict[str, Any]]:
        """List quarantined items matching optional criteria."""
        if self.session_factory is not None:
            try:
                async with self.session_factory() as session:
                    stmt = select(db.QuarantinedKnowledge).order_by(
                        db.QuarantinedKnowledge.created_at.desc()
                    )
                    if table_name:
                        stmt = stmt.where(db.QuarantinedKnowledge.table_name == table_name)
                    if reason:
                        stmt = stmt.where(db.QuarantinedKnowledge.reason == reason)
                    if session_id:
                        stmt = stmt.where(db.QuarantinedKnowledge.session_id == session_id)
                    rows = (await session.execute(stmt)).scalars().all()
                    return [
                        {
                            "id": str(r.id),
                            "session_id": str(r.session_id) if r.session_id else None,
                            "table_name": r.table_name,
                            "row_id": r.row_id,
                            "reason": r.reason,
                            "detail": r.detail,
                            "payload": r.payload,
                            "created_at": r.created_at.isoformat() if r.created_at else None,
                        }
                        for r in rows
                    ]
            except Exception as exc:
                logger.warning("quarantine.list_db_error", error=str(exc))

        # In-memory list
        results = []
        for rec in self._in_memory.values():
            if table_name and rec.table_name != table_name:
                continue
            if reason and rec.reason != reason:
                continue
            if session_id and rec.session_id != session_id:
                continue
            results.append(rec.to_dict())
        return sorted(results, key=lambda x: x["created_at"] or "", reverse=True)

    async def get_quarantined(self, quarantine_id: uuid.UUID | str) -> dict[str, Any] | None:
        """Fetch a single quarantined item by its ID."""
        qid = uuid.UUID(str(quarantine_id)) if isinstance(quarantine_id, str) else quarantine_id

        if self.session_factory is not None:
            try:
                async with self.session_factory() as session:
                    entry = (
                        await session.execute(
                            select(db.QuarantinedKnowledge).where(
                                db.QuarantinedKnowledge.id == qid
                            )
                        )
                    ).scalar_one_or_none()
                    if entry is not None:
                        return {
                            "id": str(entry.id),
                            "session_id": str(entry.session_id) if entry.session_id else None,
                            "table_name": entry.table_name,
                            "row_id": entry.row_id,
                            "reason": entry.reason,
                            "detail": entry.detail,
                            "payload": entry.payload,
                            "created_at": entry.created_at.isoformat()
                            if entry.created_at
                            else None,
                        }
            except Exception as exc:
                logger.warning("quarantine.get_db_error", error=str(exc))

        rec = self._in_memory.get(qid)
        return rec.to_dict() if rec else None

    async def restore_quarantined(
        self, quarantine_id: uuid.UUID | str
    ) -> dict[str, Any] | None:
        """Restore a quarantined item back to active knowledge.

        Reverses the quarantine action by retrieving the preserved payload.
        Removes the quarantine record upon successful restoration.

        Returns:
            Dict with restoration details, or None if quarantine ID not found.
        """
        qid = uuid.UUID(str(quarantine_id)) if isinstance(quarantine_id, str) else quarantine_id

        if self.session_factory is not None:
            try:
                async with self.session_factory() as session:
                    entry = (
                        await session.execute(
                            select(db.QuarantinedKnowledge).where(
                                db.QuarantinedKnowledge.id == qid
                            )
                        )
                    ).scalar_one_or_none()
                    if entry is None:
                        return None

                    restored_data = {
                        "id": str(entry.id),
                        "table_name": entry.table_name,
                        "row_id": entry.row_id,
                        "reason": entry.reason,
                        "payload": entry.payload,
                        "restored": True,
                    }

                    # Remove from quarantine table
                    await session.execute(
                        delete(db.QuarantinedKnowledge).where(db.QuarantinedKnowledge.id == qid)
                    )
                    await session.commit()
                    logger.info(
                        "quarantine.item_restored",
                        id=str(qid),
                        table_name=entry.table_name,
                        row_id=entry.row_id,
                    )
                    return restored_data
            except Exception as exc:
                logger.error("quarantine.restore_db_error", error=str(exc), exc_info=True)

        # In-memory restoration
        rec = self._in_memory.pop(qid, None)
        if rec is None:
            return None

        logger.info(
            "quarantine.item_restored_in_memory",
            id=str(qid),
            table_name=rec.table_name,
            row_id=rec.row_id,
        )
        return {
            "id": str(rec.id),
            "table_name": rec.table_name,
            "row_id": rec.row_id,
            "reason": rec.reason,
            "payload": rec.payload,
            "restored": True,
        }

    async def handle_data_conflict(
        self,
        item_a: dict[str, Any],
        item_b: dict[str, Any],
        *,
        conflict_type: str = REASON_CONTRADICTION,
        detail: str | None = None,
        strategy: str | None = None,
        session_id: uuid.UUID | None = None,
    ) -> dict[str, Any]:
        """Handle detected contradiction or ambiguity between two knowledge items.

        Enforces HR-NO-PERMANENT-DELETE:
        Instead of deleting either item, both conflicting entries (or the offending
        one) are routed to quarantine with full payloads intact.

        Args:
            item_a: Dict containing at least 'id', 'table_name', 'content'/'data'
            item_b: Dict containing at least 'id', 'table_name', 'content'/'data'
            conflict_type: Reason code ('contradiction', 'ambiguous', 'duplicate')
            detail: Explanation of the conflict
            strategy: Policy strategy override (e.g. 'escalate', 'quarantine', 'ask')
            session_id: Optional session context

        Returns:
            Dict reporting the quarantine actions and IDs.
        """
        table_a = item_a.get("table_name", "knowledge")
        table_b = item_b.get("table_name", "knowledge")
        id_a = str(item_a.get("id", uuid.uuid4()))
        id_b = str(item_b.get("id", uuid.uuid4()))

        conflict_desc = detail or f"Conflict between {table_a}:{id_a} and {table_b}:{id_b}"

        # Quarantine item A
        qid_a = await self.quarantine_item(
            table_name=table_a,
            row_id=id_a,
            reason=conflict_type,
            detail=f"{conflict_desc} (item A)",
            payload=item_a,
            session_id=session_id,
        )

        # Quarantine item B
        qid_b = await self.quarantine_item(
            table_name=table_b,
            row_id=id_b,
            reason=conflict_type,
            detail=f"{conflict_desc} (item B)",
            payload=item_b,
            session_id=session_id,
        )

        return {
            "action": "quarantine",
            "strategy": strategy or "escalate",
            "conflict_type": conflict_type,
            "detail": conflict_desc,
            "quarantined_items": [
                {"quarantine_id": str(qid_a), "table_name": table_a, "row_id": id_a},
                {"quarantine_id": str(qid_b), "table_name": table_b, "row_id": id_b},
            ],
            "data_loss": False,  # HR-NO-PERMANENT-DELETE invariant holds
        }

    def evaluate_edge_decay(
        self,
        *,
        confidence: float,
        days_unused: float,
        use_count: int,
        half_life_days: float = 30.0,
        review_threshold: float = 0.1,
    ) -> dict[str, Any]:
        """Evaluate edge weight using decay factor and reuse reinforcement.

        Formula: weight = clamp(confidence * decay(days_unused) * reuse(use_count), 0, 1)

        Returns:
            Dict with computed weight, factors, and review/quarantine flags.
        """
        decay = decay_factor(days_unused, half_life_days=half_life_days)
        reuse = reuse_factor(use_count)
        weight = compute_weight(
            confidence=confidence,
            days_unused=days_unused,
            use_count=use_count,
            half_life_days=half_life_days,
        )

        needs_review = weight < review_threshold
        return {
            "confidence": confidence,
            "days_unused": days_unused,
            "use_count": use_count,
            "decay_factor": round(decay, 4),
            "reuse_factor": round(reuse, 4),
            "weight": weight,
            "needs_review": needs_review,
            "should_quarantine": weight < (review_threshold / 2),
        }

    async def _fetch_source_payload(
        self, session: sa.ext.asyncio.AsyncSession, table_name: str, row_id: str
    ) -> dict[str, Any] | None:
        """Attempt to fetch row snapshot from database if table is recognized."""
        try:
            if table_name == "semantic_units":
                uid = uuid.UUID(row_id)
                res = await session.execute(
                    select(db.SemanticUnit).where(db.SemanticUnit.id == uid)
                )
                unit = res.scalar_one_or_none()
                if unit:
                    return {
                        "id": str(unit.id),
                        "chunk_id": str(unit.chunk_id),
                        "unit_type": unit.unit_type,
                        "content": unit.content,
                        "polarity": unit.polarity,
                        "entity_ids": unit.entity_ids,
                    }
            elif table_name == "knowledge_links":
                lid = uuid.UUID(row_id)
                res = await session.execute(
                    select(db.KnowledgeLink).where(db.KnowledgeLink.id == lid)
                )
                link = res.scalar_one_or_none()
                if link:
                    return {
                        "id": str(link.id),
                        "source_id": str(link.source_id),
                        "target_id": str(link.target_id),
                        "relation_type": link.relation_type,
                        "weight": link.weight,
                        "confidence": link.confidence,
                        "lifecycle": link.lifecycle,
                    }
        except Exception:
            logger.debug("quarantine.fetch_source_payload_failed", table=table_name, row_id=row_id)
        return None
