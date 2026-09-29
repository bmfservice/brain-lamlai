"""Knowledge-graph human-override API (Giai đoạn 5, Layer 4.5 / requirement 9).

A human ruling on an edge is *ground truth* and outranks both reasoning tiers.
This module lets a person correct or suppress a relation between two nodes and
makes that correction effective **immediately** — on the very next decision for
that pair — without waiting for tonight's consolidation retrain.

Each override does four things atomically:

1. Records a ``relation_decisions`` row with ``human_override=True`` (weighted
   10× at the next distillation, so the model learns the correction durably).
2. Updates the in-memory override registry on the shared classifier, so the next
   ingest of a similar pair is ruled by the human, not the LLM/model.
3. Mutates the actual edge: ``PUT`` upserts it as ``human-override``/active;
   ``DELETE`` marks it superseded (never destroyed) and suppresses the pair.
4. Appends to the ``audit_logs`` trail (who/what/when).
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.deps import get_graph, get_relation_classifier, get_session_factory
from core.database import AuditLog, RelationDecision
from core.logging import get_logger
from services.consolidation.features import REJECT_LABEL, featurize
from services.relation.validator import VALID_EDGE_TYPES

logger = get_logger(__name__)

router = APIRouter(prefix="/api/graph", tags=["graph"])

# A human may assert any taxonomy type, or REJECT to suppress the pair.
_ALLOWED_LABELS = VALID_EDGE_TYPES | {REJECT_LABEL}


class EdgeOverrideIn(BaseModel):
    """Payload asserting/correcting the relation on one node pair."""

    relation_type: str = Field(..., description="taxonomy type, or REJECT to suppress")
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    reason: str | None = Field(None, max_length=1000)
    evidence: list[str] = Field(
        default_factory=lambda: ["human-override"],
        description="evidence labels; an edge can never be stored without some",
    )


def _validate_label(label: str) -> str:
    upper = (label or "").strip().upper()
    if upper not in _ALLOWED_LABELS:
        raise HTTPException(400, f"relation_type must be one of VALID_EDGE_TYPES or {REJECT_LABEL}")
    return upper


def _validate_uuid(value: str, field: str) -> uuid.UUID:
    """Reject a malformed node id with 400 instead of a DB cast error."""
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(400, f"{field} must be a UUID") from None


@router.put("/edges/{source_id}/{target_id}", response_model=dict)
async def override_edge(
    source_id: str,
    target_id: str,
    body: EdgeOverrideIn,
    graph: Annotated[Any, Depends(get_graph)],
    session_factory: Annotated[Any, Depends(get_session_factory)],
    classifier: Annotated[Any, Depends(get_relation_classifier)] = None,
) -> dict:
    """Assert a human ruling on the relation between two nodes.

    Takes effect on the next decision for this pair immediately. Setting
    ``relation_type`` to ``REJECT`` suppresses the relation (the edge, if any,
    is marked superseded) just as ``DELETE`` does, but keeps it overridable.
    """
    label = _validate_label(body.relation_type)
    _validate_uuid(source_id, "source_id")
    _validate_uuid(target_id, "target_id")
    if source_id == target_id:
        raise HTTPException(400, "source and target must differ")

    async with session_factory() as session:
        # 1. Durable training signal (ground truth, weighted at next distillation).
        features = featurize(
            proposed_type=label, signals={}, sentence=body.reason or ""
        )
        session.add(
            RelationDecision(
                source_id=source_id,
                target_id=target_id,
                features=features,
                label=label,
                decided_by="human",
                confidence=float(body.confidence),
                rationale=body.reason,
                human_override=True,
            )
        )

        # 2. Mutate the actual edge.
        if label == REJECT_LABEL:
            changed = await graph.deactivate_edge(
                session,
                source_id=source_id,
                target_id=target_id,
                relation_type=None,  # suppress every type on this pair
            )
        else:
            await graph.upsert_edge(
                session,
                source_id=source_id,
                target_id=target_id,
                relation_type=label,
                confidence=body.confidence,
                evidence=body.evidence or ["human-override"],
                reason=body.reason or "human override",
                model_version="human-override",
            )
            changed = 1

        # 3. Audit trail.
        await _audit(
            session,
            action=f"override:{source_id}->{target_id}",
            status=label,
            message=body.reason,
            meta={"confidence": body.confidence, "edges_changed": changed},
        )
        await session.commit()

    # 4. Immediate effect: update the shared in-memory override registry AFTER
    #    commit, so a rolled-back write never leaves a phantom override live.
    if classifier is not None:
        classifier.set_override(source_id, target_id, label, body.confidence)

    logger.info(
        "graph.edge_override",
        source=source_id,
        target=target_id,
        label=label,
        edges_changed=changed,
    )
    return {
        "ok": True,
        "source_id": source_id,
        "target_id": target_id,
        "relation_type": label,
        "confidence": body.confidence,
        "decided_by": "human-override",
        "edges_changed": changed,
        "effective": "immediate",
    }


@router.delete("/edges/{source_id}/{target_id}", response_model=dict)
async def suppress_edge(
    source_id: str,
    target_id: str,
    reason: str | None = None,
    graph: Annotated[Any, Depends(get_graph)] = None,
    session_factory: Annotated[Any, Depends(get_session_factory)] = None,
    classifier: Annotated[Any, Depends(get_relation_classifier)] = None,
) -> dict:
    """Suppress the relation on a pair (mark superseded; never delete the row).

    Records a ``REJECT`` human override so the next ingest does not silently
    recreate the edge the person just removed.
    """
    _validate_uuid(source_id, "source_id")
    _validate_uuid(target_id, "target_id")
    async with session_factory() as session:
        features = featurize(proposed_type=REJECT_LABEL, signals={}, sentence=reason or "")
        session.add(
            RelationDecision(
                source_id=source_id,
                target_id=target_id,
                features=features,
                label=REJECT_LABEL,
                decided_by="human",
                confidence=1.0,
                rationale=reason,
                human_override=True,
            )
        )
        changed = await graph.deactivate_edge(
            session,
            source_id=source_id,
            target_id=target_id,
            relation_type=None,
        )
        await _audit(
            session,
            action=f"suppress:{source_id}->{target_id}",
            status=REJECT_LABEL,
            message=reason,
            meta={"edges_changed": changed},
        )
        await session.commit()

    if classifier is not None:
        classifier.set_override(source_id, target_id, REJECT_LABEL, 1.0)

    logger.info("graph.edge_suppressed", source=source_id, target=target_id, edges_changed=changed)
    return {
        "ok": True,
        "source_id": source_id,
        "target_id": target_id,
        "relation_type": REJECT_LABEL,
        "edges_changed": changed,
        "effective": "immediate",
    }


@router.get("/overrides", response_model=dict)
async def list_overrides(
    classifier: Annotated[Any, Depends(get_relation_classifier)] = None,
) -> dict:
    """Return the live in-memory override registry (the pairs a human ruled on)."""
    if classifier is None:
        return {"overrides": [], "count": 0}
    return {
        "overrides": [
            {"source_id": s, "target_id": t, "relation_type": label, "confidence": conf}
            for s, t, label, conf in classifier.overrides()
        ],
        "count": classifier.override_count(),
    }


async def _audit(session, *, action: str, status: str, message: str | None, meta: dict) -> None:
    """Append one override event to the audit trail (same table governance uses)."""
    session.add(
        AuditLog(
            service="graph",
            module="human_override",
            action=action[:100],
            status=status[:50],
            message=(message or "")[:500],
            meta=meta,
        )
    )
